//! Constraint Validator — Unified constraint enforcement for DML operations (Step 10).
//!
//! Validates NOT NULL, UNIQUE, and CHECK constraints at INSERT/UPDATE time before
//! data is written to the heap.  Loads constraint metadata from `sys_constraints`
//! and performs B+ Tree lookups for unique checks.
//!
//! # Usage
//!
//! ```ignore
//! use crate::backend::constraint::validate_row_insert;
//!
//! validate_row_insert(catalog, db, table, &str_values)?;
//! ```

pub mod validation;
pub mod loaders;
pub mod value_lookup;
pub mod fk_actions;

#[cfg(test)]
pub mod tests;

/// Validate all constraints before inserting a new row.
///
/// Checks (in order):
/// 1. NOT NULL — every `nullable = false` column must have a non-NULL value.
/// 2. UNIQUE — if a B+ Tree index exists for the column, the value must not
///    already exist in the index.
/// 3. FOREIGN KEY — referenced key must exist in the parent table.
/// 4. CHECK — any CHECK constraints registered in `sys_constraints` must
///    evaluate to `true` against the new row.
///
/// Returns `Ok(())` if all constraints pass, or `Err(message)` with a
/// user-visible description of the first violation found.
pub fn validate_row_insert(
    catalog: &crate::catalog::types::Catalog,
    db_name: &str,
    table_name: &str,
    values: &[&str],
) -> Result<(), String> {
    let db = catalog.databases.get(db_name)
        .ok_or_else(|| format!("Database '{}' not found", db_name))?;
    let table = db.tables.get(table_name)
        .ok_or_else(|| format!("Table '{}' not found", table_name))?;

    let columns = &table.columns;

    // 1. NOT NULL
    validation::check_not_null(columns, values)?;

    // 2. UNIQUE (via B+ Tree if index exists)
    validation::check_unique_insert(db_name, table_name, columns, values)?;

    // 3. FOREIGN KEY (parent key must exist)
    validation::check_foreign_key_insert(catalog, db_name, table_name, columns, values)?;

    // 4. CHECK constraints (loaded from sys_constraints)
    validation::check_constraints(db_name, table_name, columns, values)?;

    Ok(())
}

/// Validate FOREIGN KEY constraints before deleting from a parent table.
///
/// For each row being deleted, checks whether any child table has a row
/// referencing the deleted key value.
///
/// - RESTRICT mode (default): blocks the DELETE if child rows exist.
/// - CASCADE mode: automatically deletes all referencing child rows.
///
/// `column_values` is the decoded tuple as `Vec<(String, ColumnValue)>`
/// from the table being deleted from.
pub fn validate_row_delete(
    catalog: &crate::catalog::types::Catalog,
    db_name: &str,
    table_name: &str,
    column_values: &[(String, crate::backend::executor::delete::ColumnValue)],
) -> Result<(), String> {
    // Load FK constraints where OUR table is the parent (ref_table == our_name)
    let foreign_keys = match loaders::load_referencing_foreign_keys(db_name, table_name) {
        Ok(fks) => fks,
        Err(_) => return Ok(()),
    };

    if foreign_keys.is_empty() {
        return Ok(());
    }

    for fk in &foreign_keys {
        // fk: (child_table, child_col, parent_col, parent_table, action_type)
        let child_table = &fk.0;
        let child_col = &fk.1;
        let parent_col = &fk.2;
        let action_type = &fk.4;

        // Find the parent column value from the decoded tuple
        let parent_value_str = column_values.iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(parent_col))
            .map(|(_, val)| match val {
                crate::backend::executor::delete::ColumnValue::Int(n) => n.to_string(),
                crate::backend::executor::delete::ColumnValue::Text(s) => s.clone(),
                crate::backend::executor::delete::ColumnValue::List(_) => String::new(),
            })
            .unwrap_or_default();

        if parent_value_str.is_empty() || parent_value_str.eq_ignore_ascii_case("null") || parent_value_str.eq_ignore_ascii_case("NULL") {
            continue;
        }

        let action_upper = action_type.to_uppercase();
        if action_upper.contains("ON DELETE CASCADE") || action_upper == "FOREIGN KEY CASCADE" {
            // CASCADE: delete all child rows that reference this parent value
            let mut visited: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
            visited.insert((child_table.clone(), parent_value_str.clone()));
            fk_actions::cascade_delete_child_rows(
                catalog, db_name, child_table, child_col, &parent_value_str, &mut visited,
            )?;
        } else if action_upper.contains("ON DELETE SET NULL") || action_upper == "FOREIGN KEY SET NULL" {
            // SET NULL: set the FK column to NULL in all referencing child rows
            let mut visited: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
            visited.insert((child_table.clone(), parent_value_str.clone()));
            fk_actions::set_null_child_rows(catalog, db_name, child_table, child_col, &parent_value_str, &mut visited)?;
        } else {
            // RESTRICT (default): block the DELETE if child rows exist
            if value_lookup::child_has_referencing_row(db_name, child_table, child_col, &parent_value_str)? {
                return Err(format!(
                    "FOREIGN KEY constraint violated: cannot delete from '{}' because value '{}' is referenced by '{}' (column '{}' referencing '{}.{}')",
                    table_name, parent_value_str, child_table, child_col, table_name, parent_col
                ));
            }
        }
    }

    Ok(())
}

/// Propagate an UPDATE to child rows via ON UPDATE CASCADE / SET NULL.
///
/// Called after an UPDATE has been committed on a parent table.  Compares
/// old and new decoded tuples, and for each changed column that is referenced
/// by a FOREIGN KEY with an ON UPDATE action:
///
/// - **CASCADE**: updates child rows' FK column from the old value to the new value.
/// - **SET NULL**: sets child rows' FK column to NULL.
/// - **RESTRICT** (default): returns `Err` if any child row references the old value.
pub fn propagate_update_to_children(
    catalog: &crate::catalog::types::Catalog,
    db_name: &str,
    table_name: &str,
    old_decoded: &[(String, crate::backend::executor::delete::ColumnValue)],
    new_decoded: &[(String, crate::backend::executor::delete::ColumnValue)],
) -> Result<(), String> {
    let foreign_keys = match loaders::load_referencing_foreign_keys(db_name, table_name) {
        Ok(fks) => fks,
        Err(_) => return Ok(()),
    };

    if foreign_keys.is_empty() {
        return Ok(());
    }

    for fk in &foreign_keys {
        let child_table = &fk.0;
        let child_col = &fk.1;
        let parent_col = &fk.2;
        let action_type = &fk.4;

        // Find the OLD value of the parent column
        let old_val = old_decoded.iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(parent_col))
            .map(|(_, v)| match v {
                crate::backend::executor::delete::ColumnValue::Int(n) => n.to_string(),
                crate::backend::executor::delete::ColumnValue::Text(s) => s.clone(),
                _ => String::new(),
            })
            .unwrap_or_default();

        if old_val.is_empty() || old_val.eq_ignore_ascii_case("null") || old_val.eq_ignore_ascii_case("NULL") {
            continue;
        }

        // Find the NEW value of the parent column
        let new_val = new_decoded.iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(parent_col))
            .map(|(_, v)| match v {
                crate::backend::executor::delete::ColumnValue::Int(n) => n.to_string(),
                crate::backend::executor::delete::ColumnValue::Text(s) => s.clone(),
                _ => String::new(),
            })
            .unwrap_or_default();

        // Skip if the value didn't change
        if old_val == new_val {
            continue;
        }

        let action_upper = action_type.to_uppercase();

        if action_upper.contains("ON UPDATE CASCADE") {
            if !new_val.is_empty() && !new_val.eq_ignore_ascii_case("null") && !new_val.eq_ignore_ascii_case("NULL") {
                let mut visited: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
                visited.insert((child_table.clone(), old_val.clone()));
                fk_actions::update_child_rows_fk(
                    catalog, db_name, child_table, child_col,
                    &old_val, &new_val, &mut visited,
                )?;
            }
        } else if action_upper.contains("ON UPDATE SET NULL") {
            let mut visited: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
            visited.insert((child_table.clone(), old_val.clone()));
            fk_actions::set_null_child_rows(catalog, db_name, child_table, child_col, &old_val, &mut visited)?;
        } else {
            // RESTRICT (default): block the UPDATE if child rows reference the old value
            if value_lookup::child_has_referencing_row(db_name, child_table, child_col, &old_val)? {
                return Err(format!(
                    "FOREIGN KEY constraint violated: cannot update '{}' column '{}' because value '{}' is referenced by '{}' (column '{}')",
                    table_name, parent_col, old_val, child_table, child_col
                ));
            }
        }
    }

    Ok(())
}

/// Validate all constraints before updating a row.
///
/// Similar to `validate_row_insert`, but also handles the case where an
/// UPDATE might set a column to its existing value (which should not trigger
/// a false UNIQUE violation).  For simplicity, this implementation does **not**
/// exclude the current row's tuple from the UNIQUE check — a strict
/// no-duplicates policy is applied.
///
/// Returns `Ok(())` if all constraints pass, or `Err(message)`.
pub fn validate_row_update(
    catalog: &crate::catalog::types::Catalog,
    db_name: &str,
    table_name: &str,
    new_values: &[&str],
) -> Result<(), String> {
    let db = catalog.databases.get(db_name)
        .ok_or_else(|| format!("Database '{}' not found", db_name))?;
    let table = db.tables.get(table_name)
        .ok_or_else(|| format!("Table '{}' not found", table_name))?;

    let columns = &table.columns;

    // 1. NOT NULL (new values must not violate)
    validation::check_not_null(columns, new_values)?;

    // 2. UNIQUE (strict — no duplicate keys allowed even for self-update)
    validation::check_unique_insert(db_name, table_name, columns, new_values)?;

    // 3. FOREIGN KEY (parent key must exist)
    validation::check_foreign_key_insert(catalog, db_name, table_name, columns, new_values)?;

    // 4. CHECK constraints
    validation::check_constraints(db_name, table_name, columns, new_values)?;

    Ok(())
}
