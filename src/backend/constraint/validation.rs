//! Validation helpers for constraint checking.
//!
//! Provides the core validation logic for NOT NULL, UNIQUE, FOREIGN KEY,
//! and CHECK constraints, called by the public API functions.

use crate::catalog::types::Column;
use crate::types::{DataValue, DataType};

use super::loaders;
use super::value_lookup;
use super::{ConstraintKind, RookError};

/// Check that no NOT NULL column receives a NULL value.
pub(crate) fn check_not_null(
    table_name: &str,
    columns: &[Column],
    values: &[&str],
) -> Result<(), RookError> {
    for (i, col) in columns.iter().enumerate() {
        if !col.nullable {
            let val = values.get(i).unwrap_or(&"");
            let trimmed = val.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") || trimmed.eq_ignore_ascii_case("NULL") {
                return Err(RookError::constraint(
                    ConstraintKind::NotNull,
                    table_name,
                    Some(&col.name),
                    format!(
                        "NOT NULL constraint violated: column '{}' cannot be null",
                        col.name
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Check UNIQUE constraints by looking up values in the B+ Tree index.
///
/// Scans `sys_indexes` for unique indexes on this table, then for each
/// indexed column, opens the B+ Tree and searches for the corresponding
/// value.  If found, the insert/update is rejected.
///
/// `exclude` names the row's own heap location; matches pointing at that
/// exact tuple are ignored so an UPDATE does not collide with itself (a row
/// always trivially equals its own UNIQUE values).
///
/// If no B+ Tree index exists for a UNIQUE column, falls back to a
/// sequential heap scan to check for duplicate values (SQL standard
/// requires UNIQUE enforcement regardless of index existence).
pub(crate) fn check_unique_insert(
    db_name: &str,
    table_name: &str,
    columns: &[Column],
    values: &[&str],
    exclude: Option<(u32, u32)>,
) -> Result<(), RookError> {
    // Load unique indexes from sys_indexes for this table
    // (load_unique_indexes internally resolves the table_id)
    let unique_indexes = match loaders::load_unique_indexes(db_name, table_name) {
        Ok(indexes) => indexes,
        Err(_) => return Ok(()),
    };

    for (col_pos, col) in columns.iter().enumerate() {
        // Check if this column has a UNIQUE constraint (either via index or column definition)
        let has_unique_via_index = unique_indexes.iter().any(|(_, idx_col)| idx_col.eq_ignore_ascii_case(&col.name));
        let has_unique_via_constraint = col.constraints.unique;
        let is_unique = has_unique_via_index || has_unique_via_constraint;

        if !is_unique {
            continue;
        }

        let raw_val = values.get(col_pos).unwrap_or(&"");
        let trimmed = raw_val.trim();

        // NULL values are not indexed in the B+ Tree — skip UNIQUE check for NULLs
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") || trimmed.eq_ignore_ascii_case("NULL") {
            continue;
        }

        // Try B+ Tree index first (fast path)
        // Find the index file that indexes this column
        let col_type = columns[col_pos].data_type.clone();
        let key_value = crate::backend::executor::create_index::parse_string_to_value(&col_type, trimmed)
            .map_err(|e| format!("Failed to parse value for UNIQUE check: {}", e))?;

        // Try to find a named index for this column
        let mut found_via_index = false;

        // Check if we have a named index on this column
        for (idx_name, idx_col) in &unique_indexes {
            if !idx_col.eq_ignore_ascii_case(&col.name) {
                continue;
            }
            let idx_path = std::path::PathBuf::from(format!(
                "database/base/{}/{}.{}.idx", db_name, table_name, idx_name
            ));
            if !idx_path.exists() {
                continue;
            }
            match crate::backend::index::btree::BTree::open(idx_path) {
                Ok(mut btree) => {
                    btree.set_key_type(col_type.clone());
                    match btree.search_all(&key_value) {
                        Ok(tids) => {
                            // A row never collides with itself: ignore the
                            // tuple this UPDATE is rewriting.
                            if tids.iter().any(|tid| Some(*tid) != exclude) {
                                found_via_index = true;
                                break;
                            }
                        }
                        Err(e) => {
                            log::warn!(
                                "[Constraint] BTree search error for UNIQUE check on '{}.{}': {}",
                                db_name, table_name, e
                            );
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "[Constraint] Failed to open index for UNIQUE check on '{}.{}': {}",
                        db_name, table_name, e
                    );
                }
            }
        }

        // Fallback to legacy single-index file
        if !found_via_index {
            let legacy_idx_path = std::path::PathBuf::from(format!(
                "database/base/{}/{}.idx", db_name, table_name
            ));
            if legacy_idx_path.exists() {
                match crate::backend::index::btree::BTree::open(legacy_idx_path) {
                    Ok(mut btree) => {
                        btree.set_key_type(col_type.clone());
                        match btree.search_all(&key_value) {
                            Ok(tids) if tids.iter().any(|tid| Some(*tid) != exclude) => {
                                found_via_index = true
                            }
                            Ok(_) => {}
                            Err(e) => {
                                log::warn!(
                                    "[Constraint] Legacy BTree search error for UNIQUE check on '{}.{}': {}",
                                    db_name, table_name, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "[Constraint] Failed to open legacy index for UNIQUE check on '{}.{}': {}",
                            db_name, table_name, e
                        );
                    }
                }
            }
        }

        if found_via_index {
            return Err(RookError::constraint(
                ConstraintKind::Unique,
                table_name,
                Some(&col.name),
                format!(
                    "UNIQUE constraint violated: value '{}' already exists for column '{}'",
                    trimmed, col.name
                ),
            ));
        }

        // If no index was available OR the index lookup didn't find a match,
        // but this column has a UNIQUE constraint, do a fallback heap scan
        // to ensure the value is truly unique.
        let heap_path = std::path::PathBuf::from(format!(
            "database/base/{}/{}.dat", db_name, table_name
        ));
        if !heap_path.exists() {
            continue;
        }

        if let Ok(heap) = crate::backend::heap::HeapManager::open(heap_path) {
            let schema_types: Vec<crate::types::DataType> = columns.iter().map(|c| c.data_type.clone()).collect();
            for result in heap.scan() {
                let (page_id, slot_id, raw_bytes) = match result {
                    Ok(triple) => triple,
                    Err(_) => continue,
                };
                // Skip the row's own location (self-match on UPDATE).
                if exclude == Some((page_id, slot_id)) {
                    continue;
                }

                let decoded = match crate::types::deserialize_nullable_row(&schema_types, &raw_bytes) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                if let Some(Some(existing_val)) = decoded.get(col_pos) {
                    use crate::types::Comparable;
                    if let Ok(cmp) = existing_val.compare(&key_value) {
                        if cmp == std::cmp::Ordering::Equal {
                            return Err(RookError::constraint(
                                ConstraintKind::Unique,
                                table_name,
                                Some(&col.name),
                                format!(
                                    "UNIQUE constraint violated: value '{}' already exists for column '{}'",
                                    trimmed, col.name
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check FOREIGN KEY constraints — verify referenced keys exist in parent tables.
pub(crate) fn check_foreign_key_insert(
    catalog: &crate::catalog::types::Catalog,
    db_name: &str,
    table_name: &str,
    columns: &[Column],
    values: &[&str],
) -> Result<(), RookError> {
    let (table_id, _) = match crate::backend::system_table::resolve_table_id(db_name, table_name) {
        Ok(ids) => ids,
        Err(_) => return Ok(()),
    };

    let foreign_keys = match loaders::load_foreign_keys(table_id) {
        Ok(fks) => fks,
        Err(_) => return Ok(()),
    };

    for fk in &foreign_keys {
        // fk: (child_col, parent_table, parent_col)
        let child_col = &fk.0;
        let parent_table = &fk.1;
        let parent_col = &fk.2;

        // Find child column value
        let col_pos = columns.iter().position(|c| c.name.eq_ignore_ascii_case(child_col))
            .ok_or_else(|| format!("FK column '{}' not found in table schema", child_col))?;

        let raw_val = values.get(col_pos).unwrap_or(&"");
        let trimmed = raw_val.trim();

        // NULL values in FK columns are allowed (SQL standard)
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") || trimmed.eq_ignore_ascii_case("NULL") {
            continue;
        }

        // Look up parent column type from catalog
        let parent_col_type = catalog.databases.get(db_name)
            .and_then(|db| db.tables.get(parent_table))
            .and_then(|t| t.columns.iter().find(|c| c.name.eq_ignore_ascii_case(parent_col)))
            .map(|c| c.data_type.clone())
            .unwrap_or(DataType::Varchar(255));

        // Check if the referenced value exists in the parent table
        if !value_lookup::value_exists_in_table(db_name, parent_table, parent_col, &parent_col_type, trimmed)? {
            return Err(RookError::constraint(
                ConstraintKind::ForeignKey,
                table_name,
                Some(child_col),
                format!(
                    "FOREIGN KEY constraint violated: value '{}' in column '{}' not found in parent table '{}' (column '{}')",
                    trimmed, child_col, parent_table, parent_col
                ),
            ));
        }
    }

    Ok(())
}

/// Check CHECK constraints loaded from `sys_constraints`.
///
/// Each CHECK expression (stored in the `columns` field of the constraint
/// row) is parsed with the real SQL grammar and evaluated against the new
/// row using the Volcano expression evaluator — full expression support
/// (arithmetic, AND/OR/NOT, IN, BETWEEN, LIKE, functions), replacing the
/// retired legacy DNF matcher.
///
/// Semantics follow SQL: the row passes when the predicate is TRUE;
/// NULL results are UNKNOWN and therefore pass.
pub(crate) fn check_constraints(
    db_name: &str,
    table_name: &str,
    columns: &[Column],
    values: &[&str],
) -> Result<(), RookError> {
    // Resolve table_id from system tables
    let (table_id, _) = match crate::backend::system_table::resolve_table_id(db_name, table_name) {
        Ok(ids) => ids,
        Err(_) => return Ok(()), // not in system tables yet
    };

    // Load CHECK constraints from sys_constraints
    let check_exprs = match loaders::load_check_constraints(table_id) {
        Ok(exprs) => exprs,
        Err(_) => return Ok(()),
    };

    if check_exprs.is_empty() {
        return Ok(());
    }

    // Build the candidate row once: a physical Tuple over the table's
    // columns, with values type-parsed from their raw string form.
    let tuple = build_check_tuple(columns, values);

    for expr in &check_exprs {
        let pred = compile_check_predicate(expr, columns)?;
        match crate::backend::executor::physical::expr::evaluate_predicate(&pred, &tuple) {
            Ok(Some(true)) => {}                       // satisfied
            Ok(Some(false)) => {
                return Err(RookError::constraint(
                    ConstraintKind::Check,
                    table_name,
                    None,
                    format!("CHECK constraint violated: '{}'", expr),
                ));
            }
            Ok(None) => {}                             // UNKNOWN (NULL) → pass, per SQL
            Err(e) => {
                return Err(RookError::Internal(format!(
                    "Failed to evaluate CHECK constraint '{}': {}", expr, e
                )));
            }
        }
    }

    Ok(())
}

/// Parse a CHECK expression with the real SQL grammar by embedding it in a
/// synthetic SELECT, then compile it into a physical `Predicate`.
fn compile_check_predicate(
    expr: &str,
    columns: &[Column],
) -> Result<crate::backend::executor::physical::expr::Predicate, RookError> {
    use crate::backend::executor::physical::expr::predicate_from_ast;

    let sql = format!("SELECT * FROM __check__ WHERE {}", expr);
    let select = rook_parser::parse_sql(&sql)
        .map_err(|e| RookError::Internal(format!(
            "Failed to parse CHECK constraint expression '{}': {}", expr, e
        )))?;
    let selection = match select {
        rook_ast::QueryPlan::Select(select) => select.selection,
        _ => None,
    };
    let node = selection.ok_or_else(|| RookError::Internal(format!(
        "CHECK constraint expression '{}' did not yield a predicate", expr
    )))?;

    let column_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
    predicate_from_ast(&node, &column_names)
        .map_err(|e| RookError::Internal(format!(
            "Failed to compile CHECK constraint '{}': {}", expr, e
        )))
}

/// Build a physical Tuple from raw insert/update strings, typed per column.
///
/// NULL (empty or literal "null", case-insensitive) becomes `None`, which
/// makes comparisons evaluate to UNKNOWN under three-valued logic.
fn build_check_tuple(columns: &[Column], values: &[&str]) -> crate::backend::executor::physical::tuple::Tuple {
    use crate::backend::executor::physical::tuple::{ColumnInfo as PhysColInfo, Tuple};

    let mut vals: Vec<Option<crate::types::DataValue>> = Vec::with_capacity(columns.len());
    let mut info: Vec<PhysColInfo> = Vec::with_capacity(columns.len());

    for (col, raw) in columns.iter().zip(values.iter()) {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
            vals.push(None);
        } else {
            vals.push(DataValue::parse_and_encode(&col.data_type, trimmed).ok().and_then(|bytes| {
                DataValue::from_bytes(&col.data_type, &bytes).ok()
            }));
        }
        info.push(PhysColInfo {
            name: col.name.clone(),
            data_type: col.data_type.clone(),
            table: None,
        });
    }

    Tuple::new(vals, info)
}

