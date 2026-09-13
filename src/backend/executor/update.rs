//! Implements UPDATE with SET assignments, optional WHERE conditions, and RETURNING support.
//!
//! Supports:
//!   UPDATE table SET col = val;
//!   UPDATE table SET col = val WHERE other = x;
//!   UPDATE table SET col1 = val1, col2 = val2 WHERE id > 5;
//!   UPDATE table SET col = val WHERE (a = 1 AND b = 2) OR c = 3;
//!
//! UPDATE is implemented using delete + insert semantics for latest-version wins.
//! Rows with the SLOT_FLAG_DELETED bit set are invisible and are never updated.

use std::io;

use crate::catalog::types::Catalog;
use crate::backend::log::operation_log::current_timestamp_iso;
use crate::page::{Page, PAGE_HEADER_SIZE, ITEM_ID_SIZE, SLOT_FLAG_DELETED};
use crate::backend::page::page_lock::PageWriteLock;
use serde_json::{Value, json};
use crate::types::row::deserialize_nullable_row;
use crate::types::value::{DataValue, OrderedF32, OrderedF64};
use crate::types::datatype::DataType;
use crate::types::row::serialize_nullable_typed_row;

use super::delete::ColumnValue;

/// (page_num, slot_index) pair that uniquely identifies a stored tuple.
#[derive(Debug, Clone, Copy)]
struct TuplePointer {
    page_id:    u32,
    slot_index: u16,
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Arithmetic operator used in a SET expression (e.g. `salary = salary * 1.10`).
#[derive(Debug, Clone)]
pub enum ArithOp {
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
}

/// Right-hand side of a SET assignment — either a literal or an expression
/// that references the column's current value.
///
/// Examples:
///   `age = 25`           → `SetExpr::Literal(Int(25))`
///   `age = age + 1`      → `SetExpr::Expr { col: "age", op: Add, rhs: Int(1) }`
///   `salary = salary * 1.10` → `SetExpr::Expr { col: "salary", op: Mul, rhs_f: 1.10 }`
///   `tag = tag`          → `SetExpr::Column("tag")`
///   `tag = NULL`         → `SetExpr::Null`
#[derive(Debug, Clone)]
pub enum SetExpr {
    /// A constant value.
    Literal(ColumnValue),
    /// A reference to another column (or self) in the current row.
    Column(String),
    /// SQL NULL literal.
    Null,
    /// `<src_col> <op> <rhs>` evaluated against the current row.
    /// `rhs_f` is used for floating-point multipliers (Mul / Div);
    /// `rhs_i` is used for integer Add / Sub.
    Expr {
        src_col: String,
        op:      ArithOp,
        rhs_i:   i64,   // used for Add/Sub
        rhs_f:   f64,   // used for Mul/Div
    },
}

/// One `col = <expr>` assignment from the SET clause.
#[derive(Debug, Clone)]
pub struct SetAssignment {
    pub column: String,
    pub expr:   SetExpr,
}

/// Result returned by the UPDATE entry points.
pub struct UpdateResult {
    /// How many rows were modified.
    pub updated_count:   usize,
    /// The rows **after** update (only populated when `returning = true`).
    pub returning_rows:  Vec<Vec<(String, String)>>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn values_to_column_values(
    columns: &[crate::catalog::types::Column],
    values: &[Option<DataValue>],
) -> Vec<(String, ColumnValue)> {
    columns
        .iter()
        .zip(values.iter())
        .map(|(col, val)| {
            let cv = match val {
                Some(DataValue::Int(n)) => ColumnValue::Int(*n),
                Some(DataValue::SmallInt(n)) => ColumnValue::Int(*n as i32),
                Some(DataValue::Varchar(s)) | Some(DataValue::Char(s)) => {
                    ColumnValue::Text(s.clone())
                }
                Some(other) => ColumnValue::Text(format!("{}", other)),
                None => ColumnValue::Text("NULL".to_string()),
            };
            (col.name.clone(), cv)
        })
        .collect()
}

/// Apply `assignments` directly to typed row values, preserving all types natively.
fn apply_assignments_typed(
    columns: &[crate::catalog::types::Column],
    mut values: Vec<Option<DataValue>>,
    assignments: &[SetAssignment],
) -> Vec<Option<DataValue>> {
    for asgn in assignments {
        let Some(target_idx) = columns.iter().position(|c| c.name.eq_ignore_ascii_case(&asgn.column)) else {
            continue;
        };
        let target_type = &columns[target_idx].data_type;

        let new_val: Option<DataValue> = match &asgn.expr {
            SetExpr::Null => None,
            SetExpr::Literal(cv) => match cv {
                ColumnValue::Int(n) => match target_type {
                    DataType::SmallInt => Some(DataValue::SmallInt(*n as i16)),
                    DataType::Int => Some(DataValue::Int(*n)),
                    DataType::BigInt => Some(DataValue::BigInt(*n as i64)),
                    DataType::Real => Some(DataValue::Real(OrderedF32(*n as f32))),
                    DataType::DoublePrecision => Some(DataValue::DoublePrecision(OrderedF64(*n as f64))),
                    DataType::Numeric { scale, .. } => {
                        let factor = 10_i128.pow(*scale as u32);
                        Some(DataValue::Numeric(crate::types::value::NumericValue {
                            unscaled: *n as i128 * factor,
                            scale: *scale,
                        }))
                    }
                    _ => Some(DataValue::Int(*n)),
                },
                ColumnValue::Text(s) => match target_type {
                    DataType::Char(n) => {
                        let mut padded = s.clone();
                        if padded.len() < *n as usize {
                            padded.push_str(&" ".repeat(*n as usize - padded.len()));
                        }
                        Some(DataValue::Char(padded))
                    }
                    DataType::Character(n) => {
                        let mut padded = s.clone();
                        if padded.len() < *n as usize {
                            padded.push_str(&" ".repeat(*n as usize - padded.len()));
                        }
                        Some(DataValue::Char(padded))
                    }
                    DataType::Varchar(_) => Some(DataValue::Varchar(s.clone())),
                    _ => super::create_index::parse_string_to_value(target_type, s).ok(),
                },
                ColumnValue::List(_) => None,
            },
            SetExpr::Column(src_col) => {
                if let Some(src_idx) = columns.iter().position(|c| c.name.eq_ignore_ascii_case(src_col)) {
                    values[src_idx].clone()
                } else {
                    continue;
                }
            }
            SetExpr::Expr { src_col, op, rhs_i, rhs_f } => {
                let src_val = columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(src_col))
                    .and_then(|idx| values.get(idx).cloned().flatten());

                match src_val {
                    None => None, // SQL semantics: arithmetic on NULL produces NULL!
                    Some(DataValue::Int(n)) => {
                        let result = match op {
                            ArithOp::Add => (n as i64 + rhs_i) as i32,
                            ArithOp::Sub => (n as i64 - rhs_i) as i32,
                            ArithOp::Mul => (n as f64 * rhs_f) as i32,
                            ArithOp::Div => if *rhs_f == 0.0 { n } else { (n as f64 / rhs_f) as i32 },
                        };
                        Some(DataValue::Int(result))
                    }
                    Some(DataValue::SmallInt(n)) => {
                        let result = match op {
                            ArithOp::Add => (n as i64 + rhs_i) as i16,
                            ArithOp::Sub => (n as i64 - rhs_i) as i16,
                            ArithOp::Mul => (n as f64 * rhs_f) as i16,
                            ArithOp::Div => if *rhs_f == 0.0 { n } else { (n as f64 / rhs_f) as i16 },
                        };
                        Some(DataValue::SmallInt(result))
                    }
                    Some(DataValue::BigInt(n)) => {
                        let result = match op {
                            ArithOp::Add => n + rhs_i,
                            ArithOp::Sub => n - rhs_i,
                            ArithOp::Mul => (n as f64 * rhs_f) as i64,
                            ArithOp::Div => if *rhs_f == 0.0 { n } else { (n as f64 / rhs_f) as i64 },
                        };
                        Some(DataValue::BigInt(result))
                    }
                    Some(DataValue::Real(r)) => {
                        let cur = r.0 as f64;
                        let result = match op {
                            ArithOp::Add => cur + *rhs_i as f64,
                            ArithOp::Sub => cur - *rhs_i as f64,
                            ArithOp::Mul => cur * rhs_f,
                            ArithOp::Div => if *rhs_f == 0.0 { cur } else { cur / rhs_f },
                        };
                        Some(DataValue::Real(OrderedF32(result as f32)))
                    }
                    Some(DataValue::DoublePrecision(d)) => {
                        let cur = d.0;
                        let result = match op {
                            ArithOp::Add => cur + *rhs_i as f64,
                            ArithOp::Sub => cur - *rhs_i as f64,
                            ArithOp::Mul => cur * rhs_f,
                            ArithOp::Div => if *rhs_f == 0.0 { cur } else { cur / rhs_f },
                        };
                        Some(DataValue::DoublePrecision(OrderedF64(result)))
                    }
                    Some(DataValue::Varchar(s)) => {
                        let result = match op {
                            ArithOp::Add => format!("{}{}", s, rhs_i),
                            ArithOp::Sub => {
                                let total_chars = s.chars().count();
                                let take_count = (total_chars as i64 - rhs_i).max(0) as usize;
                                s.chars().take(take_count).collect()
                            }
                            _ => s.clone(),
                        };
                        Some(DataValue::Varchar(result))
                    }
                    Some(DataValue::Char(s)) => {
                        let trimmed = s.trim_end();
                        let result = match op {
                            ArithOp::Add => format!("{}{}", trimmed, rhs_i),
                            ArithOp::Sub => {
                                let total_chars = trimmed.chars().count();
                                let take_count = (total_chars as i64 - rhs_i).max(0) as usize;
                                trimmed.chars().take(take_count).collect()
                            }
                            _ => trimmed.to_string(),
                        };
                        let width = if let DataType::Char(w) | DataType::Character(w) = target_type {
                            *w as usize
                        } else {
                            result.len()
                        };
                        let mut padded = result;
                        if padded.len() < width {
                            padded.push_str(&" ".repeat(width - padded.len()));
                        }
                        Some(DataValue::Char(padded))
                    }
                    Some(other) => Some(other),
                }
            }
        };

        values[target_idx] = new_val;
    }
    values
}

struct PendingUpdate {
    pointer: TuplePointer,
    /// The raw bytes of the OLD tuple (before update), for index key extraction.
    old_tuple_data: Vec<u8>,
    /// The raw bytes of the NEW tuple (after update).
    new_bytes: Vec<u8>,
    /// The decoded row BEFORE the update, for FK cascade propagation.
    old_decoded: Vec<(String, ColumnValue)>,
    /// The decoded row AFTER the update, for FK cascade propagation.
    updated_decoded: Vec<(String, ColumnValue)>,
}

fn update_log_details(
    updated_count: Option<usize>,
    error: Option<&str>,
) -> Value {
    json!({
        "timestamp": current_timestamp_iso(),
        "updated_count": updated_count,
        "error": error,
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SET clause parser
// ---------------------------------------------------------------------------

/// Parse a comma-separated SET clause string into `SetAssignment`s.
///
/// Handles literal, column-reference, and expression-based assignments:
///   `salary = 60000`               → `Literal(Int(60000))`
///   `name = 'Alice'`               → `Literal(Text("Alice"))`
///   `tag = tag`                    → `Column("tag")`
///   `tag = NULL`                   → `Null`
///   `age = age + 1`                → `Expr { src: age, op: Add, rhs_i: 1 }`
///   `salary = salary * 1.10`       → `Expr { src: salary, op: Mul, rhs_f: 1.10 }`
///   `score = score - 5, age = age + 1`  → two assignments
///
/// Returns `None` if the string is empty or no valid assignment could be parsed.
pub fn parse_set_clause(input: &str) -> Option<Vec<SetAssignment>> {
    let input = input.trim();
    if input.is_empty() { return None; }

    let mut assignments = Vec::new();

    for part in split_set_parts(input) {
        let part = part.trim();
        // Find the FIRST '=' (the assignment operator)
        let eq_pos = part.find('=')?;
        let col = part[..eq_pos].trim().to_string();
        let rhs = part[eq_pos + 1..].trim();

        if col.is_empty() || rhs.is_empty() { continue; }

        let expr = if rhs.starts_with('\'') && rhs.ends_with('\'') && rhs.len() >= 2 {
            let inner = &rhs[1..rhs.len() - 1];
            SetExpr::Literal(ColumnValue::Text(inner.replace("''", "'")))
        } else if rhs.starts_with('"') && rhs.ends_with('"') && rhs.len() >= 2 {
            let inner = &rhs[1..rhs.len() - 1];
            SetExpr::Literal(ColumnValue::Text(inner.replace("\"\"", "\"")))
        } else if rhs.eq_ignore_ascii_case("null") {
            SetExpr::Null
        } else if let Some((src, op, rhs_str)) = try_parse_arith_expr(rhs) {
            let rhs_f: f64 = rhs_str.parse().ok()?;
            let rhs_i: i64 = rhs_f as i64;
            SetExpr::Expr { src_col: src, op, rhs_i, rhs_f }
        } else if let Ok(n) = rhs.parse::<i32>() {
            SetExpr::Literal(ColumnValue::Int(n))
        } else if let Ok(n) = rhs.parse::<i64>() {
            if n >= i32::MIN as i64 && n <= i32::MAX as i64 {
                SetExpr::Literal(ColumnValue::Int(n as i32))
            } else {
                SetExpr::Literal(ColumnValue::Text(rhs.to_string()))
            }
        } else if rhs.parse::<f64>().is_ok() {
            SetExpr::Literal(ColumnValue::Text(rhs.to_string()))
        } else {
            // Unquoted string that is not numeric or NULL: column reference!
            SetExpr::Column(rhs.to_string())
        };

        assignments.push(SetAssignment { column: col, expr });
    }

    if assignments.is_empty() { None } else { Some(assignments) }
}

/// Split a SET clause by commas, but NOT commas inside parentheses or string literals.
fn split_set_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '\'' if !in_double_quote => in_single_quote = !in_single_quote,
            '"' if !in_single_quote => in_double_quote = !in_double_quote,
            '(' if !in_single_quote && !in_double_quote => depth += 1,
            ')' if !in_single_quote && !in_double_quote => depth = depth.saturating_sub(1),
            ',' if depth == 0 && !in_single_quote && !in_double_quote => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Try to parse an arithmetic expression like `age + 1`, `age+1`, or `salary * 1.10`.
/// Returns `(src_col, op, rhs_str)` or `None`.
fn try_parse_arith_expr(rhs: &str) -> Option<(String, ArithOp, String)> {
    if rhs.starts_with('\'') || rhs.starts_with('"') {
        return None;
    }

    // .trim() on both sides makes spaced and unspaced forms identical,
    // so we only need the bare operator symbol.
    // pos > 0 guard prevents treating a leading sign (e.g. "-5") as an expression.
    let ops: &[(&str, ArithOp)] = &[
        ("+", ArithOp::Add),
        ("-", ArithOp::Sub),
        ("*", ArithOp::Mul),
        ("/", ArithOp::Div),
    ];

    for (sym, op) in ops {
        if let Some(pos) = rhs.find(sym) {
            if pos == 0 { continue; } // leading sign — not an expression
            let src = rhs[..pos].trim().trim_matches('\'').to_string();
            let val = rhs[pos + sym.len()..].trim().trim_matches('\'').to_string();
            // src must look like a column name (non-numeric, non-empty)
            if !src.is_empty() && src.parse::<f64>().is_err() && !val.is_empty() {
                return Some((src, op.clone(), val));
            }
        }
    }
    None
}


/// Update rows identified by explicit heap pointers (page_id, slot_id).
///
/// This is the Volcano-aware UPDATE path: instead of scanning the heap and
/// evaluating WHERE conditions here, the caller (Volcano engine) has already
/// identified the matching rows. This function applies the SET assignments to
/// the rows at the given pointers (soft-delete + rewrite).
pub fn update_by_pointers(
    catalog: &Catalog,
    db_name: &str,
    table_name: &str,
    pointers: &[(u32, u32)],
    assignments: &[SetAssignment],
) -> io::Result<UpdateResult> {
    let db = catalog.databases.get(db_name).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("Database '{}' not found", db_name))
    })?;
    let table = db.tables.get(table_name).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("Table '{}' not found", table_name))
    })?;
    let columns = &table.columns;

    let path = format!("database/base/{}/{}.dat", db_name, table_name);
    // UPDATE rewrites pages via direct I/O — flush/evict cached pool state
    // first so the raw reads observe every prior insert.
    crate::backend::cache::quiesce_for_direct_io(std::path::Path::new(&path))
        .map_err(|e| io::Error::other(format!("Failed to flush cache: {}", e)))?;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| io::Error::other(format!("Failed to open table file: {}", e)))?;
    let file_identity = crate::table::file_identity_from_file(&file)?;

    let mut updated_count = 0usize;
    let mut returning_rows: Vec<Vec<(String, String)>> = Vec::new();
    let mut pending_updates: Vec<PendingUpdate> = Vec::new();

    for &(page_num, slot_idx) in pointers {
        let slot_index = slot_idx as u16;

        let mut page = Page::new();
        crate::disk::read_page(&mut file, &mut page, page_num)?;

        let base = (PAGE_HEADER_SIZE + slot_index as u32 * ITEM_ID_SIZE) as usize;
        let offset = u32::from_le_bytes(page.data[base..base + 4].try_into().unwrap());
        let length = u16::from_le_bytes(page.data[base + 4..base + 6].try_into().unwrap()) as u32;
        let flags = u16::from_le_bytes(page.data[base + 6..base + 8].try_into().unwrap());

        if (offset == 0 && length == 0) || (flags & SLOT_FLAG_DELETED != 0) {
            continue;
        }

        let tuple_data = page.data[offset as usize..(offset + length) as usize].to_vec();
        let schema: Vec<DataType> = columns.iter().map(|c| c.data_type.clone()).collect();
        let decoded = match deserialize_nullable_row(&schema, &tuple_data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let updated_values = apply_assignments_typed(columns, decoded.clone(), assignments);
        let new_bytes = match serialize_nullable_typed_row(&schema, &updated_values) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let old_decoded = values_to_column_values(columns, &decoded);
        let updated_decoded = values_to_column_values(columns, &updated_values);

        // Constraint validation
        let new_strings: Vec<String> = updated_values.iter().map(|val| {
            match val {
                Some(dv) => dv.to_string(),
                None => "NULL".to_string(),
            }
        }).collect();
        let new_values: Vec<&str> = new_strings.iter().map(|s| s.as_str()).collect();
        if let Err(e) = crate::backend::constraint::validate_row_update(
            catalog, db_name, table_name, &new_values,
            Some((page_num, slot_idx)),
        ) {
            log::warn!(
                "[UpdateByPointers] Skipping row due to constraint violation: {}", e
            );
            continue;
        }

        pending_updates.push(PendingUpdate {
            pointer: TuplePointer { page_id: page_num, slot_index },
            old_tuple_data: tuple_data,
            new_bytes,
            old_decoded,
            updated_decoded,
        });

        updated_count += 1;
    }

    // Phase 2: apply delete+insert for all pending updates
    if !pending_updates.is_empty() {
        for update in &pending_updates {
            let _page_lock = PageWriteLock::acquire(file_identity, update.pointer.page_id);
            let mut page = Page::new();
            crate::disk::read_page(&mut file, &mut page, update.pointer.page_id)?;
            let base = (PAGE_HEADER_SIZE + update.pointer.slot_index as u32 * ITEM_ID_SIZE) as usize;
            let mut flags = u16::from_le_bytes(page.data[base + 6..base + 8].try_into().unwrap());
            flags |= SLOT_FLAG_DELETED;
            page.data[base + 6..base + 8].copy_from_slice(&flags.to_le_bytes());
            crate::disk::write_page(&mut file, &mut page, update.pointer.page_id)?;
            let _ = crate::backend::visibility_map::vm_clear_page(db_name, table_name, update.pointer.page_id);
        }
        crate::table::increment_dead_tuple_count(&mut file, pending_updates.len() as u32)?;
    }

    if !pending_updates.is_empty() {
        for update in &pending_updates {
            let (new_page_id, new_slot_id) = crate::backend::executor::compaction_api::insert_raw_tuple(
                db_name, table_name, &update.new_bytes,
            )?;

            if let Err(e) = crate::backend::executor::create_index::update_index_on_update(
                db_name, table_name, columns,
                &update.old_tuple_data, &update.new_bytes,
                update.pointer.page_id, update.pointer.slot_index as u32,
                new_page_id, new_slot_id,
            ) {
                log::warn!("Failed to update index for updated tuple: {}", e);
            }

            returning_rows.push(
                update.updated_decoded.iter().map(|(col, val)| {
                    let s = match val {
                        ColumnValue::Int(n) => n.to_string(),
                        ColumnValue::Text(t) => t.clone(),
                        ColumnValue::List(_) => String::from("[list]"),
                    };
                    (col.clone(), s)
                }).collect()
            );
        }
    }

    // FK cascade propagation
    if !pending_updates.is_empty() {
        for update in &pending_updates {
            if let Err(e) = crate::backend::constraint::propagate_update_to_children(
                catalog, db_name, table_name,
                &update.old_decoded, &update.updated_decoded,
            ) {
                log::warn!(
                    "[UpdateByPointers] Failed to propagate FK cascade: {}", e
                );
            }
        }
    }

    let result = UpdateResult { updated_count, returning_rows };
    let details = update_log_details(Some(result.updated_count), None);
    let _ = crate::backend::log::operation_log::log_update(db_name, table_name, details, "success");

    Ok(result)
}