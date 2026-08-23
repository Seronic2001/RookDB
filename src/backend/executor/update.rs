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
#[derive(Debug, Clone)]
pub enum SetExpr {
    /// A constant value.
    Literal(ColumnValue),
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



fn decode_tuple(
    tuple_data: &[u8],
    columns: &[crate::catalog::types::Column],
) -> Vec<(String, ColumnValue)> {

    let schema: Vec<DataType> = columns
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    let decoded = match deserialize_nullable_row(&schema, tuple_data) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();

    for (col, value) in columns.iter().zip(decoded.iter()) {
        let converted = match value {
            Some(DataValue::Int(v)) => ColumnValue::Int(*v),

            Some(DataValue::Varchar(s))
            | Some(DataValue::Char(s)) => {
                ColumnValue::Text(s.clone())
            }

            Some(other) => ColumnValue::Text(format!("{:?}", other)),

            None => ColumnValue::Text("NULL".to_string()),
        };

        result.push((col.name.clone(), converted));
    }

    result
}


/// Re-encode a decoded tuple back into the real on-disk row format.
fn encode_tuple(
    decoded: &[(String, ColumnValue)],
    columns: &[crate::catalog::types::Column],
) -> Vec<u8> {

    // Build schema
    let schema: Vec<DataType> = columns
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // Build typed values in schema order
    let mut values: Vec<Option<DataValue>> = Vec::new();

    for col in columns {
        let val = decoded
            .iter()
            .find(|(name, _)| name == &col.name)
            .map(|(_, v)| v);

        let typed = match val {
            Some(ColumnValue::Int(n)) => {
                match &col.data_type {
                    DataType::SmallInt => Some(DataValue::SmallInt(*n as i16)),
                    DataType::Int => Some(DataValue::Int(*n)),
                    DataType::BigInt => Some(DataValue::BigInt(*n as i64)),
                    DataType::Real => Some(DataValue::Real(OrderedF32(*n as f32))),
                    DataType::DoublePrecision => Some(DataValue::DoublePrecision(OrderedF64(*n as f64))),
                    _ => Some(DataValue::Int(*n)),
                }
            }

            Some(ColumnValue::Text(s)) => {
                if s.eq_ignore_ascii_case("NULL") {
                    None
                } else {
                    // Use parse_string_to_value for proper type-aware parsing
                    super::create_index::parse_string_to_value(&col.data_type, s).ok()
                }
            }

            Some(ColumnValue::List(_)) => None,
            None => None,
        };

        values.push(typed);
    }

    serialize_nullable_typed_row(&schema, &values)
        .unwrap_or_default()
}

/// Apply `assignments` to a decoded row, returning the new row.
/// Arithmetic expressions are evaluated against the *current* column values.
fn apply_assignments(
    mut decoded: Vec<(String, ColumnValue)>,
    assignments: &[SetAssignment],
) -> Vec<(String, ColumnValue)> {
    for asgn in assignments {
        let new_val = match &asgn.expr {
            SetExpr::Literal(v) => v.clone(),
            SetExpr::Expr { src_col, op, rhs_i, rhs_f } => {
                // Look up current value of src_col
                let cur = decoded.iter().find(|(name, _)| name == src_col)
                    .map(|(_, v)| v.clone());
                match cur {
                    Some(ColumnValue::Int(n)) => {
                        let result = match op {
                            ArithOp::Add => (n as i64 + rhs_i) as i32,
                            ArithOp::Sub => (n as i64 - rhs_i) as i32,
                            ArithOp::Mul => (n as f64 * rhs_f) as i32,
                            ArithOp::Div => if *rhs_f == 0.0 { n } else { (n as f64 / rhs_f) as i32 },
                        };
                        ColumnValue::Int(result)
                    }
                    Some(ColumnValue::Text(s)) => {
                        // Text + int → append; Text * int → repeat
                        let result = match op {
                            ArithOp::Add => format!("{}{}", s, rhs_i),
                            ArithOp::Sub => s.chars().take(
                                (s.len() as i64 - rhs_i).max(0) as usize
                            ).collect(),
                            _ => s.clone(),
                        };
                        ColumnValue::Text(result)
                    }
                    _ => continue, // column not found, skip
                }
            }
        };
        if let Some(entry) = decoded.iter_mut().find(|(name, _)| *name == asgn.column) {
            entry.1 = new_val;
        }
    }
    decoded
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
/// Handles both literal and expression-based assignments:
///   `salary = 60000`               → `Literal(Int(60000))`
///   `name = 'Alice'`               → `Literal(Text("Alice"))`
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
        let rhs = part[eq_pos + 1..].trim().to_string();

        if col.is_empty() || rhs.is_empty() { continue; }

        // Try to detect arithmetic expression: `src_col OP value`
        // where OP is one of + - * /
        // e.g. "age + 1", "salary * 1.10", "score - 5"
        let expr = try_parse_arith_expr(&rhs)
            .and_then(|(src, op, rhs_str)| {
                let rhs_f: f64 = rhs_str.parse().ok()?;
                let rhs_i: i64 = rhs_f as i64;
                Some(SetExpr::Expr { src_col: src, op, rhs_i, rhs_f })
            })
            .unwrap_or_else(|| {
                // Literal value
                let v = rhs.trim_matches('\'').to_string();
                if let Ok(n) = v.parse::<i32>() {
                    SetExpr::Literal(ColumnValue::Int(n))
                } else {
                    SetExpr::Literal(ColumnValue::Text(v))
                }
            });

        assignments.push(SetAssignment { column: col, expr });
    }

    if assignments.is_empty() { None } else { Some(assignments) }
}

/// Split a SET clause by commas, but NOT commas inside parentheses.
fn split_set_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => { parts.push(&s[start..i]); start = i + 1; }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Try to parse an arithmetic expression like `age + 1`, `age+1`, or `salary * 1.10`.
/// Returns `(src_col, op, rhs_str)` or `None`.
fn try_parse_arith_expr(rhs: &str) -> Option<(String, ArithOp, String)> {
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
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to open table file: {}", e)))?;
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
        let decoded = decode_tuple(&tuple_data, columns);
        let updated_decoded = apply_assignments(decoded.clone(), assignments);
        let new_bytes = encode_tuple(&updated_decoded, columns);

        // Constraint validation
        let new_strings: Vec<String> = updated_decoded.iter().map(|(_name, val)| {
            match val {
                ColumnValue::Int(n) => n.to_string(),
                ColumnValue::Text(s) => s.clone(),
                ColumnValue::List(_) => "[list]".to_string(),
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
            old_decoded: decoded,
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