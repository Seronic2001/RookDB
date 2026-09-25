//! Value lookup helpers for constraint checking.
//!
//! Provides functions to look up whether a value exists in a table column,
//! using B+ Tree index lookups (fast path) or sequential heap scans (fallback).

use super::loaders;
use crate::types::DataType;

/// Helper: try all index files (named + legacy) for a table and search for a value.
/// Returns `Some(true)` if found, `Some(false)` if not found (in any index),
/// or `None` if no indexes exist (caller should fall back to heap scan).
fn try_index_for_value(
    db_name: &str,
    table_name: &str,
    column_name: &str,
    col_type: &DataType,
    value_str: &str,
) -> Option<bool> {
    let key_value =
        crate::backend::executor::create_index::parse_string_to_value(col_type, value_str).ok()?;

    // Try named indexes: load all indexes from sys_indexes for this table
    if let Ok(indexes) =
        crate::backend::executor::create_index::load_table_indexes(db_name, table_name)
    {
        for (idx_name, idx_col, _is_unique) in &indexes {
            if !idx_col.eq_ignore_ascii_case(column_name) {
                continue;
            }
            let idx_path = std::path::PathBuf::from(format!(
                "database/base/{}/{}.{}.idx",
                db_name, table_name, idx_name
            ));
            if !idx_path.exists() {
                continue;
            }
            let search_res = crate::backend::cache::with_btree(
                &idx_path,
                || -> std::io::Result<crate::backend::index::btree::BTree> {
                    let mut bt = crate::backend::index::btree::BTree::open(idx_path.clone())?;
                    bt.set_key_type(col_type.clone());
                    Ok(bt)
                },
                |bt| {
                    bt.set_key_type(col_type.clone());
                    bt.search(&key_value).map_err(std::io::Error::other)
                },
            );
            match search_res {
                Ok(Some(_)) => return Some(true),
                Ok(None) => return Some(false),
                Err(e) => log::warn!(
                    "[Constraint] Cached BTree search error for FK lookup: {}",
                    e
                ),
            }
        }
    }

    // Fallback to legacy single-index file
    let legacy_idx =
        std::path::PathBuf::from(format!("database/base/{}/{}.idx", db_name, table_name));
    if legacy_idx.exists() {
        let search_res = crate::backend::cache::with_btree(
            &legacy_idx,
            || -> std::io::Result<crate::backend::index::btree::BTree> {
                let mut bt = crate::backend::index::btree::BTree::open(legacy_idx.clone())?;
                bt.set_key_type(col_type.clone());
                Ok(bt)
            },
            |bt| {
                bt.set_key_type(col_type.clone());
                bt.search(&key_value).map_err(std::io::Error::other)
            },
        );
        match search_res {
            Ok(Some(_)) => return Some(true),
            Ok(None) => return Some(false),
            Err(e) => log::warn!(
                "[Constraint] Cached legacy BTree search error for FK lookup: {}",
                e
            ),
        }
    }

    None // No index found — caller should fall back to heap scan
}

/// Check whether a specific value exists in a table column.
///
/// First tries a B+ Tree index lookup (for speed), falling back to a
/// sequential heap scan with proper deserialization if no index is available.
pub(crate) fn value_exists_in_table(
    db_name: &str,
    table_name: &str,
    column_name: &str,
    col_type: &DataType,
    value_str: &str,
) -> Result<bool, String> {
    // Try all B+ Tree indexes first (named + legacy)
    if let Some(result) = try_index_for_value(db_name, table_name, column_name, col_type, value_str)
    {
        return Ok(result);
    }

    // Fallback: sequential heap scan with proper deserialization
    scan_table_for_value(db_name, table_name, column_name, col_type, value_str)
}

/// Check whether a child table has any row whose given column matches the value.
///
/// Used by parent DELETE validation to detect referencing rows.
pub(crate) fn child_has_referencing_row(
    db_name: &str,
    child_table: &str,
    child_col: &str,
    value: &str,
) -> Result<bool, String> {
    // Load the child table's column schema from sys_columns to get the correct type
    let (schema, col_pos, col_type) =
        match loaders::load_table_schema_for_column(db_name, child_table, child_col) {
            Ok(Some(result)) => result,
            Ok(None) => {
                // Table not in system tables — infer type from value string
                let _inferred_type = if value.parse::<i32>().is_ok() {
                    DataType::Int
                } else {
                    DataType::Varchar(255)
                };
                // Fall back to string-based heap scan
                let heap_path = std::path::PathBuf::from(format!(
                    "database/base/{}/{}.dat",
                    db_name, child_table
                ));
                return scan_raw_heap_for_string(&heap_path, value);
            }
            Err(e) => {
                log::warn!(
                    "[Constraint] Error loading schema for child table FK lookup: {}",
                    e
                );
                return Ok(false);
            }
        };

    // Try all B+ Tree indexes (named + legacy)
    if let Some(result) = try_index_for_value(db_name, child_table, child_col, &col_type, value) {
        return Ok(result);
    }

    // Fallback: proper heap scan with deserialization
    let key_value =
        match crate::backend::executor::create_index::parse_string_to_value(&col_type, value) {
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "[Constraint] Failed to parse value for FK child heap scan: {}",
                    e
                );
                return Ok(false);
            }
        };

    let heap_path =
        std::path::PathBuf::from(format!("database/base/{}/{}.dat", db_name, child_table));
    if !heap_path.exists() {
        return Ok(false);
    }

    let heap = match crate::backend::heap::HeapManager::open(heap_path) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("[Constraint] Failed to open child heap: {}", e);
            return Ok(false);
        }
    };

    for result in heap.scan() {
        let (_page_id, _slot_id, raw_bytes) = match result {
            Ok(triple) => triple,
            Err(_) => continue,
        };

        let decoded = match crate::types::deserialize_nullable_row(&schema, &raw_bytes) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if let Some(Some(dv)) = decoded.get(col_pos) {
            use crate::types::Comparable;
            if let Ok(cmp) = dv.compare(&key_value) {
                use std::cmp::Ordering;
                if cmp == Ordering::Equal {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Scan a table heap with proper deserialization to find a matching value.
fn scan_table_for_value(
    db_name: &str,
    table_name: &str,
    column_name: &str,
    col_type: &DataType,
    value_str: &str,
) -> Result<bool, String> {
    // Load schema from sys_columns
    let schema = match loaders::load_table_schema(db_name, table_name) {
        Ok(Some(s)) => s,
        Ok(None) => return Ok(false),
        Err(e) => {
            log::warn!("[Constraint] Error loading schema: {}", e);
            return Ok(false);
        }
    };

    // Find the column position
    let col_pos = match schema
        .iter()
        .position(|(name, _)| name.eq_ignore_ascii_case(column_name))
    {
        Some(p) => p,
        None => {
            log::warn!(
                "[Constraint] Column '{}' not found in table schema for FK lookup",
                column_name
            );
            return Ok(false);
        }
    };

    let schema_types: Vec<DataType> = schema.iter().map(|(_, dt)| dt.clone()).collect();

    let key_value =
        match crate::backend::executor::create_index::parse_string_to_value(col_type, value_str) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("[Constraint] Failed to parse value for FK heap scan: {}", e);
                return Ok(false);
            }
        };

    let heap_path =
        std::path::PathBuf::from(format!("database/base/{}/{}.dat", db_name, table_name));
    if !heap_path.exists() {
        return Ok(false);
    }

    let heap = match crate::backend::heap::HeapManager::open(heap_path) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("[Constraint] Failed to open heap for FK scan: {}", e);
            return Ok(false);
        }
    };

    for result in heap.scan() {
        let (_page_id, _slot_id, raw_bytes) = match result {
            Ok(triple) => triple,
            Err(_) => continue,
        };

        let decoded = match crate::types::deserialize_nullable_row(&schema_types, &raw_bytes) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if let Some(Some(dv)) = decoded.get(col_pos) {
            use crate::types::Comparable;
            if let Ok(cmp) = dv.compare(&key_value) {
                use std::cmp::Ordering;
                if cmp == Ordering::Equal {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Fallback raw heap scan using string matching (for tables not in system catalog).
fn scan_raw_heap_for_string(heap_path: &std::path::Path, value: &str) -> Result<bool, String> {
    if !heap_path.exists() {
        return Ok(false);
    }

    let heap = match crate::backend::heap::HeapManager::open(heap_path.to_path_buf()) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("[Constraint] Failed to open heap for raw scan: {}", e);
            return Ok(false);
        }
    };

    for result in heap.scan() {
        let (_page_id, _slot_id, raw_bytes) = match result {
            Ok(triple) => triple,
            Err(_) => continue,
        };

        let raw_str = String::from_utf8_lossy(&raw_bytes);
        if raw_str.contains(value) {
            return Ok(true);
        }
    }

    Ok(false)
}
