//! CREATE INDEX — Build a B+ Tree index from a heap scan.
//!
//! Scans all tuples in a table's heap file, extracts the indexed column's value
//! from each tuple, and inserts it into a new B+ Tree index file at
//! `database/base/{db}/{table}.{index_name}.idx`.
//!
//! Also stores index metadata in a companion `.idx.meta` file so that DML
//! operations (INSERT, DELETE, UPDATE) can update the index automatically.
//!
//! # Multiple indexes per table
//!
//! Each index gets its own file pair:
//!   `database/base/{db}/{table}.{index_name}.idx`
//!   `database/base/{db}/{table}.{index_name}.idx.meta`
//!
//! The `update_index_on_*` functions query `sys_indexes` to discover all
//! indexes defined on a table and maintain each one individually.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::catalog::Catalog;
use crate::backend::heap::heap_manager::HeapManager;
use crate::backend::index::btree::BTree;
use crate::types::value::{DataValue, OrderedF32, OrderedF64};
use crate::types::DataType;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Construct the `.idx` file path for a named index.
fn index_file_path(db_name: &str, table_name: &str, index_name: &str) -> PathBuf {
    PathBuf::from(format!(
        "database/base/{}/{}.{}.idx",
        db_name, table_name, index_name
    ))
}

/// Construct the `.idx.meta` file path for a named index.
fn index_meta_file_path(db_name: &str, table_name: &str, index_name: &str) -> PathBuf {
    PathBuf::from(format!(
        "database/base/{}/{}.{}.idx.meta",
        db_name, table_name, index_name
    ))
}

/// Construct the legacy `.idx` file path (single-index fallback).
fn legacy_index_file_path(db_name: &str, table_name: &str) -> PathBuf {
    PathBuf::from(format!("database/base/{}/{}.idx", db_name, table_name))
}

/// Construct the legacy `.idx.meta` file path (single-index fallback).
fn legacy_index_meta_file_path(db_name: &str, table_name: &str) -> PathBuf {
    PathBuf::from(format!("database/base/{}/{}.idx.meta", db_name, table_name))
}

/// Metadata about an index, stored alongside the `.idx` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMeta {
    /// The name of the indexed column (for human reference).
    column_name: String,
    /// The ordinal position of the indexed column in the table schema.
    column_idx: usize,
    /// String representation of the key type (e.g. "INT", "VARCHAR(100)").
    key_type: String,
}

/// Load index metadata from an `.idx.meta` file.
fn load_index_meta(meta_path: &PathBuf) -> Result<IndexMeta, String> {
    let meta_json = std::fs::read_to_string(meta_path)
        .map_err(|e| format!("Failed to read index metadata: {}", e))?;
    serde_json::from_str(&meta_json)
        .map_err(|e| format!("Failed to parse index metadata: {}", e))
}

/// Load all indexes defined for a table from `sys_indexes`.
///
/// # Dependency
///
/// This function reads from the `sys_indexes` system table, which requires:
/// - `sys_databases` and `sys_tables` to have the database/table registered
/// - `sys_indexes` to be populated (i.e., `save_catalog()` has been called at
///   least once so the system tables contain this table's metadata).
///
/// During initial catalog bootstrap — before `save_catalog()` has persisted
/// the table's metadata — this function returns an empty list because the
/// system tables have not yet been populated.  Callers should fall back to
/// legacy `{table}.idx` file scanning when this function returns `Vec::new()`.
///
/// Returns `Vec<(index_name, column_name, is_unique)>`.
pub fn load_table_indexes(db_name: &str, table_name: &str) -> Result<Vec<(String, String, bool)>, String> {
    use crate::backend::system_table::SYS_INDEXES_SCHEMA;

    let (table_id, _) = match crate::backend::system_table::resolve_table_id(db_name, table_name) {
        Ok(ids) => ids,
        Err(_) => return Ok(Vec::new()),
    };

    let idx_path = PathBuf::from(format!("{}/indexes.dat", crate::layout::SYSTEM_DIR));
    if !idx_path.exists() {
        return Ok(Vec::new());
    }

    let heap = HeapManager::open(idx_path)
        .map_err(|e| format!("Failed to open sys_indexes: {}", e))?;

    let mut indexes = Vec::new();
    for result in heap.scan() {
        let (_page_id, _slot_id, raw_bytes) = result
            .map_err(|e| format!("Error scanning sys_indexes: {}", e))?;
        let decoded = crate::types::deserialize_nullable_row(SYS_INDEXES_SCHEMA, &raw_bytes)
            .map_err(|e| format!("Error deserializing sys_indexes: {}", e))?;

        // SYS_INDEXES_SCHEMA: index_id, table_id, name, is_unique, is_primary, columns
        if decoded.len() < 6 {
            continue;
        }

        let row_table_id = match &decoded[1] {
            Some(DataValue::Int(id)) => *id,
            _ => continue,
        };
        if row_table_id != table_id {
            continue;
        }

        let index_name = match &decoded[2] {
            Some(DataValue::Varchar(n)) => n.clone(),
            Some(DataValue::Char(n)) => n.clone(),
            _ => continue,
        };
        let column_name = match &decoded[5] {
            Some(DataValue::Varchar(c)) => c.clone(),
            Some(DataValue::Char(c)) => c.clone(),
            _ => continue,
        };
        let is_unique = match &decoded[3] {
            Some(DataValue::Bool(v)) => *v,
            _ => false,
        };

        indexes.push((index_name, column_name, is_unique));
    }

    Ok(indexes)
}

/// Check if a legacy single-index file exists for a table.
pub fn legacy_index_exists(db_name: &str, table_name: &str) -> bool {
    legacy_index_file_path(db_name, table_name).exists()
}

// ── Create Index ──────────────────────────────────────────────────────────────

/// Build a B+ Tree index on the specified column of a table.
///
/// Each index gets its own file pair:
///   `database/base/{db}/{table}.{index_name}.idx`
///   `database/base/{db}/{table}.{index_name}.idx.meta`
///
/// # Arguments
/// * `catalog` - The catalog (used to resolve the table schema).
/// * `db_name` - The current database name.
/// * `table_name` - The table to index.
/// * `index_name` - The name of the index (used in the file name).
/// * `column_name` - The column to index.
///
/// # Returns
/// * `Ok(tuple_count)` on success, with the number of indexed tuples.
/// * `Err(message)` on failure.
pub fn create_index(
    catalog: &Catalog,
    db_name: &str,
    table_name: &str,
    index_name: &str,
    column_name: &str,
) -> Result<usize, String> {
    log::info!(
        "[CreateIndex] Creating index '{}.{}.{}' on column '{}'",
        db_name, table_name, index_name, column_name
    );

    // 1. Resolve the table schema from the catalog
    let db = catalog.databases.get(db_name)
        .ok_or_else(|| format!("Database '{}' not found", db_name))?;
    let table = db.tables.get(table_name)
        .ok_or_else(|| format!("Table '{}' not found in database '{}'", table_name, db_name))?;

    // 2. Find the indexed column
    let indexed_col = table.columns.iter()
        .find(|c| c.name.eq_ignore_ascii_case(column_name))
        .ok_or_else(|| format!("Column '{}' not found in table '{}'", column_name, table_name))?;

    let indexed_col_idx = table.columns.iter()
        .position(|c| c.name.eq_ignore_ascii_case(column_name))
        .ok_or_else(|| format!("Column '{}' not found (index lookup)", column_name))?;

    let schema_types: Vec<DataType> = table.columns.iter().map(|c| c.data_type.clone()).collect();

    log::info!(
        "[CreateIndex] Indexed column '{}' at index {}, type={}",
        indexed_col.name, indexed_col_idx, indexed_col.data_type
    );

    // 3. Determine the actual index name (auto-generate if empty)
    let resolved_name = if index_name.is_empty() {
        format!("idx_{}_{}", table_name, column_name)
    } else {
        index_name.to_string()
    };

    // 4. Build paths — use index-name-based files
    let heap_path = PathBuf::from(format!("database/base/{}/{}.dat", db_name, table_name));
    let idx_path = index_file_path(db_name, table_name, &resolved_name);
    let meta_path = index_meta_file_path(db_name, table_name, &resolved_name);

    if !heap_path.exists() {
        return Err(format!("Heap file not found: {:?}", heap_path));
    }

    // 4. Open the heap file for scanning
    let heap_manager = HeapManager::open(heap_path)
        .map_err(|e| format!("Failed to open heap for table '{}': {}", table_name, e))?;

    // 5. Create the B+ Tree index file
    let mut btree = BTree::create(idx_path, indexed_col.data_type.clone())
        .map_err(|e| format!("Failed to create index file: {}", e))?;

    // 6. Scan all tuples and insert into the B+ Tree
    let mut inserted_count = 0usize;
    let scan_iter = heap_manager.scan();
    for result in scan_iter {
        let (page_id, slot_id, raw_bytes) = match result {
            Ok(triple) => triple,
            Err(e) => return Err(format!("Scan error: {}", e)),
        };

        // Deserialize the tuple
        let values = crate::types::deserialize_nullable_row(&schema_types, &raw_bytes)
            .map_err(|e| format!("Failed to deserialize tuple: {}", e))?;

        // Extract the indexed column value
        let key_value = match values.get(indexed_col_idx) {
            Some(Some(dv)) => dv,
            Some(None) => {
                // NULL key — skip (NULLs are not indexed in simple B+ Tree)
                log::trace!(
                    "[CreateIndex] Skipping NULL key at (page={}, slot={})",
                    page_id, slot_id
                );
                continue;
            }
            None => {
                return Err(format!(
                    "Tuple at (page={}, slot={}) has fewer columns than schema",
                    page_id, slot_id
                ));
            }
        };

        // Insert into the B+ Tree: key=column_value, value=(page_id, slot_id)
        if let Err(e) = btree.insert(key_value, page_id, slot_id) {
            return Err(format!(
                "Failed to insert into index at (page={}, slot={}): {}",
                page_id, slot_id, e
            ));
        }

        inserted_count += 1;
    }

    // 7. Sync the index to disk
    btree.sync()
        .map_err(|e| format!("Failed to sync index: {}", e))?;

    // 8. Write index metadata file for DML integration
    let meta = IndexMeta {
        column_name: indexed_col.name.clone(),
        column_idx: indexed_col_idx,
        key_type: format!("{}", indexed_col.data_type),
    };
    let meta_json = serde_json::to_string_pretty(&meta)
        .map_err(|e| format!("Failed to serialize index metadata: {}", e))?;
    std::fs::write(&meta_path, meta_json)
        .map_err(|e| format!("Failed to write index metadata: {}", e))?;

    // 9. Save index metadata to sys_indexes system table
    if let Err(e) = crate::backend::system_table::insert_index_metadata(
        db_name,
        table_name,
        &resolved_name,
        &indexed_col.name,
        false, // is_unique (not tracked from CREATE INDEX yet)
        false, // is_primary (not tracked from CREATE INDEX yet)
    ) {
        log::warn!(
            "[CreateIndex] Failed to save index metadata to sys_indexes: {}",
            e
        );
    }

    log::info!(
        "[CreateIndex] Index '{}' created with {} entries on {}.{}({})",
        resolved_name, inserted_count, db_name, table_name, column_name
    );

    Ok(inserted_count)
}

// ── DML Index Update Helpers ──────────────────────────────────────────────────

/// For a given table, discover all index files (named and legacy) and return
/// `Vec<(meta_path, idx_path, IndexMeta)>`.
fn discover_indexes_for_table(
    db_name: &str,
    table_name: &str,
) -> Vec<(PathBuf, PathBuf, IndexMeta)> {
    let mut discovered = Vec::new();

    // 1. Try loading all named indexes from sys_indexes
    if let Ok(indexes) = load_table_indexes(db_name, table_name) {
        for (idx_name, _col_name, _is_unique) in &indexes {
            let idx_path = index_file_path(db_name, table_name, idx_name);
            let meta_path = index_meta_file_path(db_name, table_name, idx_name);
            if idx_path.exists() && meta_path.exists() {
                if let Ok(meta) = load_index_meta(&meta_path) {
                    discovered.push((meta_path, idx_path, meta));
                } else {
                    log::warn!(
                        "[IndexDiscover] Failed to parse metadata for named index '{}'",
                        idx_name
                    );
                }
            }
        }
    }

    // 2. Fallback: check for legacy single-index file
    let legacy_idx = legacy_index_file_path(db_name, table_name);
    let legacy_meta = legacy_index_meta_file_path(db_name, table_name);
    if legacy_idx.exists() && legacy_meta.exists() {
        // Only add if not already covered by a named index (avoid duplicates)
        if !discovered.iter().any(|(_, p, _)| *p == legacy_idx) {
            if let Ok(meta) = load_index_meta(&legacy_meta) {
                log::info!(
                    "[IndexDiscover] Found legacy index for {}.{} at {:?}",
                    db_name, table_name, legacy_idx
                );
                discovered.push((legacy_meta, legacy_idx, meta));
            }
        }
    }

    discovered
}

/// After a tuple is inserted into the heap, update all existing B+ Tree indexes
/// for the table.
///
/// Discovers all indexes (named and legacy) via `sys_indexes` and file scanning,
/// then updates each one with the new tuple's key value.
///
/// Returns `Ok(())` on success or if no indexes exist (no-op).
pub fn update_index_on_insert(
    db_name: &str,
    table_name: &str,
    values: &[&str],
    page_id: u32,
    slot_id: u32,
) -> Result<(), String> {
    let indexes = discover_indexes_for_table(db_name, table_name);
    if indexes.is_empty() {
        return Ok(()); // No indexes to update
    }

    for (_meta_path, idx_path, meta) in &indexes {
        // Get the indexed column's string value
        let col_value = match values.get(meta.column_idx) {
            Some(v) => v,
            None => {
                log::warn!(
                    "[IndexInsert] Column index {} out of bounds for {}.{}, skipping this index",
                    meta.column_idx, db_name, table_name
                );
                continue;
            }
        };

        // Parse the key type string back to DataType
        let key_type: DataType = match meta.key_type.parse() {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "[IndexInsert] Failed to parse key type '{}': {}, skipping",
                    meta.key_type, e
                );
                continue;
            }
        };

        // Check if value is NULL
        if col_value.trim().eq_ignore_ascii_case("null") || col_value.trim().is_empty() {
            log::trace!(
                "[IndexInsert] Skipping NULL index key for {}.{}({})",
                db_name, table_name, meta.column_name
            );
            continue;
        }

        // Parse the string value into a DataValue for the BTree insert
        let key_value = match parse_string_to_value(&key_type, col_value) {
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "[IndexInsert] Failed to parse value for {}.{}('{}'): {}, skipping",
                    db_name, table_name, meta.column_name, e
                );
                continue;
            }
        };

        // Open the BTree and insert
        let mut btree = match BTree::open(idx_path.clone()) {
            Ok(b) => b,
            Err(e) => {
                log::warn!(
                    "[IndexInsert] Failed to open index {:?}: {}, skipping",
                    idx_path, e
                );
                continue;
            }
        };
        btree.set_key_type(key_type);

        if let Err(e) = btree.insert(&key_value, page_id, slot_id) {
            log::warn!(
                "[IndexInsert] Failed to insert into index {:?}: {}, skipping",
                idx_path, e
            );
            continue;
        }
        if let Err(e) = btree.sync() {
            log::warn!(
                "[IndexInsert] Failed to sync index {:?}: {}, skipping",
                idx_path, e
            );
        }

        log::trace!(
            "[IndexInsert] Inserted key {:?} → (page={}, slot={}) into index on {}.{}({})",
            key_value, page_id, slot_id, db_name, table_name, meta.column_name
        );
    }

    Ok(())
}

/// After a tuple is deleted from the heap, update all existing B+ Tree indexes
/// for the table.
///
/// Discovers all indexes (named and legacy) via `sys_indexes` and file scanning,
/// then removes the specific entry matching `(page_id, slot_id)` from each one.
///
/// Returns `Ok(true)` if at least one index was updated, `Ok(false)` if no
/// indexes exist, or `Err` on failure.
pub fn update_index_on_delete(
    db_name: &str,
    table_name: &str,
    columns: &[crate::catalog::types::Column],
    tuple_data: &[u8],
    page_id: u32,
    slot_id: u32,
) -> Result<bool, String> {
    let indexes = discover_indexes_for_table(db_name, table_name);
    if indexes.is_empty() {
        return Ok(false); // No indexes to update
    }

    // Pre-deserialize the tuple once for all indexes
    let schema_types: Vec<DataType> = columns.iter().map(|c| c.data_type.clone()).collect();
    let decoded = crate::types::deserialize_nullable_row(&schema_types, tuple_data)
        .map_err(|e| format!("Failed to deserialize tuple for index update: {}", e))?;

    let mut any_updated = false;

    for (_meta_path, idx_path, meta) in &indexes {
        let key_value = match decoded.get(meta.column_idx) {
            Some(Some(dv)) => dv.clone(),
            Some(None) => {
                // NULL key — not indexed
                log::trace!(
                    "[IndexDelete] Skipping NULL key for {}.{}({})",
                    db_name, table_name, meta.column_name
                );
                continue;
            }
            None => {
                log::warn!(
                    "[IndexDelete] Column index {} out of bounds for deserialized tuple",
                    meta.column_idx
                );
                continue;
            }
        };

        // Parse the key type string back to DataType
        let key_type: DataType = match meta.key_type.parse() {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "[IndexDelete] Failed to parse key type '{}': {}, skipping",
                    meta.key_type, e
                );
                continue;
            }
        };

        // Open the BTree and delete the specific entry (key, page_id, slot_id)
        let mut btree = match BTree::open(idx_path.clone()) {
            Ok(b) => b,
            Err(e) => {
                log::warn!(
                    "[IndexDelete] Failed to open index {:?}: {}, skipping",
                    idx_path, e
                );
                continue;
            }
        };
        btree.set_key_type(key_type);

        match btree.delete(&key_value, page_id, slot_id) {
            Ok(true) => {
                any_updated = true;
                log::trace!(
                    "[IndexDelete] Removed key {:?} at (page={}, slot={}) from index on {}.{}({})",
                    key_value, page_id, slot_id, db_name, table_name, meta.column_name
                );
            }
            Ok(false) => {
                log::warn!(
                    "[IndexDelete] Key {:?} at (page={}, slot={}) not found in index on {}.{}({})",
                    key_value, page_id, slot_id, db_name, table_name, meta.column_name
                );
            }
            Err(e) => {
                log::warn!(
                    "[IndexDelete] Failed to delete from index {:?}: {}, skipping",
                    idx_path, e
                );
                continue;
            }
        }

        if let Err(e) = btree.sync() {
            log::warn!(
                "[IndexDelete] Failed to sync index {:?}: {}, skipping",
                idx_path, e
            );
        }
    }

    Ok(any_updated)
}

/// After a tuple is updated, update all existing B+ Tree indexes for the table.
///
/// Discovers all indexes (named and legacy) via `sys_indexes` and file scanning,
/// then for each index:
///   - Deletes the old key matching `(old_page_id, old_slot_id)`
///   - Inserts the new key at `(new_page_id, new_slot_id)`
pub fn update_index_on_update(
    db_name: &str,
    table_name: &str,
    columns: &[crate::catalog::types::Column],
    old_tuple_data: &[u8],
    new_tuple_data: &[u8],
    old_page_id: u32,
    old_slot_id: u32,
    new_page_id: u32,
    new_slot_id: u32,
) -> Result<(), String> {
    let indexes = discover_indexes_for_table(db_name, table_name);
    if indexes.is_empty() {
        return Ok(()); // No indexes to update
    }

    // Pre-deserialize both tuples once
    let schema_types: Vec<DataType> = columns.iter().map(|c| c.data_type.clone()).collect();

    let old_decoded = crate::types::deserialize_nullable_row(&schema_types, old_tuple_data)
        .map_err(|e| format!("Failed to deserialize old tuple: {}", e))?;
    let new_decoded = crate::types::deserialize_nullable_row(&schema_types, new_tuple_data)
        .map_err(|e| format!("Failed to deserialize new tuple: {}", e))?;

    for (_meta_path, idx_path, meta) in &indexes {
        let old_key = match old_decoded.get(meta.column_idx) {
            Some(Some(dv)) => Some(dv.clone()),
            _ => None, // NULL — not indexed
        };
        let new_key = match new_decoded.get(meta.column_idx) {
            Some(Some(dv)) => Some(dv.clone()),
            _ => None, // NULL — not indexed
        };

        // If both keys are None, skip this index
        if old_key.is_none() && new_key.is_none() {
            log::trace!(
                "[IndexUpdate] Both old and new keys are NULL for {}.{}({}), skipping index.",
                db_name, table_name, meta.column_name
            );
            continue;
        }

        // Parse the key type string back to DataType
        let key_type: DataType = match meta.key_type.parse() {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "[IndexUpdate] Failed to parse key type '{}': {}, skipping",
                    meta.key_type, e
                );
                continue;
            }
        };

        let mut btree = match BTree::open(idx_path.clone()) {
            Ok(b) => b,
            Err(e) => {
                log::warn!(
                    "[IndexUpdate] Failed to open index {:?}: {}, skipping",
                    idx_path, e
                );
                continue;
            }
        };
        btree.set_key_type(key_type);

        // Delete old key (if it was indexed), matching by the OLD heap location
        if let Some(ref old) = old_key {
            let _ = btree.delete(old, old_page_id, old_slot_id)
                .map_err(|e| format!("Failed to delete old key from index: {}", e))?;
            log::trace!(
                "[IndexUpdate] Deleted old key {:?} at (page={}, slot={}) from index on {}.{}({})",
                old, old_page_id, old_slot_id, db_name, table_name, meta.column_name
            );
        }

        // Insert new key (if it's not NULL)
        if let Some(ref new) = new_key {
            btree.insert(new, new_page_id, new_slot_id)
                .map_err(|e| format!("Failed to insert new key into index: {}", e))?;
            log::trace!(
                "[IndexUpdate] Inserted new key {:?} → (page={}, slot={}) into index on {}.{}({})",
                new, new_page_id, new_slot_id, db_name, table_name, meta.column_name
            );
        }

        if let Err(e) = btree.sync() {
            log::warn!(
                "[IndexUpdate] Failed to sync index {:?}: {}",
                idx_path, e
            );
        }
    }

    Ok(())
}

/// Parse a raw string value into a `DataValue` for the given `DataType`.
/// This mirrors the DataValue construction logic in `DataValue::parse_and_encode`
/// but returns a `DataValue` instead of encoded bytes.
pub fn parse_string_to_value(ty: &DataType, input: &str) -> Result<DataValue, String> {
    let input = input.trim().trim_matches('"').trim_matches('\'');

    match ty {
        DataType::SmallInt => input
            .parse::<i16>()
            .map(DataValue::SmallInt)
            .map_err(|e| format!("Invalid SMALLINT '{}': {}", input, e)),
        DataType::Int => input
            .parse::<i32>()
            .map(DataValue::Int)
            .map_err(|e| format!("Invalid INT '{}': {}", input, e)),
        DataType::BigInt => input
            .parse::<i64>()
            .map(DataValue::BigInt)
            .map_err(|e| format!("Invalid BIGINT '{}': {}", input, e)),
        DataType::Real => input
            .parse::<f32>()
            .map(|v| DataValue::Real(OrderedF32(v)))
            .map_err(|e| format!("Invalid REAL '{}': {}", input, e)),
        DataType::DoublePrecision => input
            .parse::<f64>()
            .map(|v| DataValue::DoublePrecision(OrderedF64(v)))
            .map_err(|e| format!("Invalid DOUBLE '{}': {}", input, e)),
        DataType::Bool => match input.to_ascii_lowercase().as_str() {
            "true" | "t" | "1" => Ok(DataValue::Bool(true)),
            "false" | "f" | "0" => Ok(DataValue::Bool(false)),
            _ => Err(format!("Invalid BOOLEAN '{}': expected true/false", input)),
        },
        DataType::Char(_) | DataType::Character(_) => {
            Ok(DataValue::Char(input.to_string()))
        }
        DataType::Varchar(_) => {
            Ok(DataValue::Varchar(input.to_string()))
        }
        DataType::Date => {
            use chrono::NaiveDate;
            NaiveDate::parse_from_str(input, "%Y-%m-%d")
                .map(DataValue::Date)
                .map_err(|e| format!("Invalid DATE '{}': {}", input, e))
        }
        DataType::Time => {
            use chrono::NaiveTime;
            NaiveTime::parse_from_str(input, "%H:%M:%S%.f")
                .or_else(|_| NaiveTime::parse_from_str(input, "%H:%M:%S"))
                .map(DataValue::Time)
                .map_err(|e| format!("Invalid TIME '{}': {}", input, e))
        }
        DataType::Timestamp => {
            use chrono::NaiveDateTime;
            NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S"))
                .map(DataValue::Timestamp)
                .map_err(|e| format!("Invalid TIMESTAMP '{}': {}", input, e))
        }
        DataType::Bit(_) => {
            let bits = crate::types::bit_utils::normalize_bit_literal(input);
            Ok(DataValue::Bit(bits))
        }
        DataType::Numeric { .. } | DataType::Decimal { .. } => {
            Err(format!("Index maintenance for NUMERIC/DECIMAL types not yet supported"))
        }
    }
}
