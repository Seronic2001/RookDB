//! System Tables — Physical catalog stored as standard heap files.
//!
//! Step 8 of the SQL-99 roadmap replaces `database/global/catalog.json` with
//! proper slotted-page heap files under `database/system/`.  Each system table
//! is a normal `.dat` file with an `.fsm` fork, managed via `HeapManager`.
//!
//! ## System table listing
//!
//! | System table       | File                             | Contents                          |
//! |--------------------|----------------------------------|-----------------------------------|
//! | `sys_databases`    | `database/system/databases.dat`  | db_id, name                      |
//! | `sys_tables`       | `database/system/tables.dat`     | table_id, db_id, name, file_path |
//! | `sys_columns`      | `database/system/columns.dat`    | col_id, table_id, name, data_type, ordinal, nullable, has_default, default_value |
//! | `sys_constraints`  | `database/system/constraints.dat`| constraint_id, table_id, type, columns, ref_table, ref_columns |
//! | `sys_indexes`      | `database/system/indexes.dat`    | index_id, table_id, name, is_unique, is_primary, columns |
//!
//! ## Bootstrap flow
//!
//! 1. On first startup, `bootstrap_system_catalog()` checks if
//!    `database/system/` exists and whether `catalog.json` is present.
//! 2. If `catalog.json` exists but system tables are absent, it reads the
//!    JSON catalog, creates and populates all five heap files, then renames
//!    `catalog.json` → `catalog.json.migrated`.
//! 3. Subsequent startups go directly to the heap files.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::catalog::types::{Catalog, Column, Constraints, Database, Table};
use crate::types::DataType;
use rook_ast::logical::{ColumnInfo, ColumnSchema};

// ── System table file paths ───────────────────────────────────────────────────

const SYS_DIR: &str = crate::layout::SYSTEM_DIR;

fn sys_path(table: &str) -> PathBuf {
    PathBuf::from(format!("{}/{}.dat", SYS_DIR, table))
}

// ── Column schemas for each system table ──────────────────────────────────────
// These match the physical column order used for serialisation.

/// Schema for `sys_databases`: db_id:INT, name:VARCHAR(255)
const SYS_DATABASES_SCHEMA: &[DataType] = &[DataType::Int, DataType::Varchar(255)];

/// Schema for `sys_tables`: table_id:INT, db_id:INT, name:VARCHAR(255), file_path:VARCHAR(512)
pub const SYS_TABLES_SCHEMA: &[DataType] = &[
    DataType::Int,
    DataType::Int,
    DataType::Varchar(255),
    DataType::Varchar(512),
];

/// Schema for `sys_columns`: col_id:INT, table_id:INT, name:VARCHAR(255),
/// data_type:VARCHAR(100), ordinal:INT, nullable:BOOL, has_default:BOOL,
/// default_value:VARCHAR(255)
pub const SYS_COLUMNS_SCHEMA: &[DataType] = &[
    DataType::Int,
    DataType::Int,
    DataType::Varchar(255),
    DataType::Varchar(100),
    DataType::Int,
    DataType::Bool,
    DataType::Bool,
    DataType::Varchar(255),
];

/// Schema for `sys_constraints`: constraint_id:INT, table_id:INT,
/// constraint_type:VARCHAR(50), columns:TEXT, ref_table:VARCHAR(255),
/// ref_columns:TEXT
pub const SYS_CONSTRAINTS_SCHEMA: &[DataType] = &[
    DataType::Int,
    DataType::Int,
    DataType::Varchar(50),
    DataType::Varchar(1024),
    DataType::Varchar(255),
    DataType::Varchar(1024),
];

/// Schema for `sys_indexes`: index_id:INT, table_id:INT, name:VARCHAR(255),
/// is_unique:BOOL, is_primary:BOOL, columns:VARCHAR(255)
pub const SYS_INDEXES_SCHEMA: &[DataType] = &[
    DataType::Int,
    DataType::Int,
    DataType::Varchar(255),
    DataType::Bool,
    DataType::Bool,
    DataType::Varchar(255),
];

/// Schema for `sys_views`: view_id:INT, db_id:INT, name:VARCHAR(255),
/// query_json:TEXT
pub const SYS_VIEWS_SCHEMA: &[DataType] = &[
    DataType::Int,
    DataType::Int,
    DataType::Varchar(255),
    DataType::Varchar(8192),
];

// ── String helpers ────────────────────────────────────────────────────────────

/// Convert a `DataValue` to its string representation for storage in system
/// table VARCHAR columns.
fn value_to_string(dv: &crate::types::DataValue) -> String {
    match dv {
        crate::types::DataValue::SmallInt(v) => v.to_string(),
        crate::types::DataValue::Int(v) => v.to_string(),
        crate::types::DataValue::BigInt(v) => v.to_string(),
        crate::types::DataValue::Real(v) => v.0.to_string(),
        crate::types::DataValue::DoublePrecision(v) => v.0.to_string(),
        crate::types::DataValue::Bool(v) => v.to_string(),
        crate::types::DataValue::Char(v) | crate::types::DataValue::Varchar(v) => v.clone(),
        crate::types::DataValue::Date(_) => format!("{:?}", dv),
        crate::types::DataValue::Time(_) => format!("{:?}", dv),
        crate::types::DataValue::Timestamp(_) => format!("{:?}", dv),
        crate::types::DataValue::Numeric(_) => format!("{:?}", dv),
        crate::types::DataValue::Bit(_) => format!("{:?}", dv),
    }
}

/// Format a `DataType` to its canonical string representation.
fn datatype_to_string(dt: &DataType) -> String {
    format!("{}", dt)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Bootstrap the system catalog from `catalog.json` if needed.
///
/// Returns `true` if a migration happened, `false` if system tables already
/// existed (or if no JSON catalog was found).
pub fn bootstrap_system_catalog() -> bool {
    let sys_dir = std::path::Path::new(SYS_DIR);
    let catalog_json = std::path::Path::new(crate::layout::CATALOG_FILE);

    // If system directory already exists with at least one table file, skip.
    if sys_dir.exists() && sys_path("databases").exists() {
        return false;
    }

    // No JSON catalog to bootstrap from — just create empty system tables.
    if !catalog_json.exists() {
        log::info!("[SystemCatalog] No catalog.json found; creating empty system tables.");
        create_system_table_files();
        return false;
    }

    log::info!("[SystemCatalog] Migrating catalog.json → system tables ...");

    // Read the existing JSON catalog.
    let catalog = crate::catalog::load_catalog();

    // Create system directory and heap files.
    std::fs::create_dir_all(SYS_DIR).ok();
    create_system_table_files();

    // Populate system tables from the in-memory catalog.
    populate_system_tables(&catalog, Vec::new());

    // Rename catalog.json so we don't migrate again.
    let migrated_str = format!("{}.migrated", crate::layout::CATALOG_FILE);
    let _ = std::fs::rename(catalog_json, &migrated_str);

    log::info!("[SystemCatalog] Migration complete. catalog.json → system tables.");
    true
}

/// Load the full `Catalog` from system table heap files.
///
/// Scans each system table via direct disk reads and reconstructs the
/// in-memory `Catalog` hierarchy.
pub fn load_catalog_from_system() -> Catalog {
    let mut catalog = Catalog {
        databases: HashMap::new(),
    };

    let databases = match scan_system_table("databases", SYS_DATABASES_SCHEMA) {
        Ok(rows) => rows,
        Err(e) => {
            log::error!("[SystemCatalog] Failed to read sys_databases: {}", e);
            return catalog;
        }
    };

    let tables = match scan_system_table("tables", SYS_TABLES_SCHEMA) {
        Ok(rows) => rows,
        Err(e) => {
            log::error!("[SystemCatalog] Failed to read sys_tables: {}", e);
            return catalog;
        }
    };

    let columns = match scan_system_table("columns", SYS_COLUMNS_SCHEMA) {
        Ok(rows) => rows,
        Err(e) => {
            log::error!("[SystemCatalog] Failed to read sys_columns: {}", e);
            return catalog;
        }
    };

    let views_data = match scan_system_table("views", SYS_VIEWS_SCHEMA) {
        Ok(rows) => rows,
        Err(e) => {
            log::error!("[SystemCatalog] Failed to read sys_views: {}", e);
            return catalog;
        }
    };

    // Build the in-memory catalog
    for db_row in &databases {
        // db_row columns: db_id, name
        let db_name = match &db_row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(name))) => name.clone(),
            Some(Some(crate::types::DataValue::Char(name))) => name.clone(),
            _ => continue,
        };

        let db_id = match &db_row.get(0) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => continue,
        };

        let mut database = Database {
            tables: HashMap::new(),
            views: HashMap::new(),
        };

        // Find all tables belonging to this database
        for tbl_row in &tables {
            let tbl_db_id = match &tbl_row.get(1) {
                Some(Some(crate::types::DataValue::Int(id))) => *id,
                _ => continue,
            };

            if tbl_db_id != db_id {
                continue;
            }

            let tbl_name = match &tbl_row.get(2) {
                Some(Some(crate::types::DataValue::Varchar(name))) => name.clone(),
                Some(Some(crate::types::DataValue::Char(name))) => name.clone(),
                _ => continue,
            };

            let table_id = match &tbl_row.get(0) {
                Some(Some(crate::types::DataValue::Int(id))) => *id,
                _ => continue,
            };

            // Collect columns with their ordinal positions, then sort
            let mut col_with_ordinals: Vec<(i32, Column)> = Vec::new();
            for col_row in &columns {
                let col_table_id = match &col_row.get(1) {
                    Some(Some(crate::types::DataValue::Int(id))) => *id,
                    _ => continue,
                };
                if col_table_id != table_id {
                    continue;
                }
                let col_ordinal = match &col_row.get(4) {
                    Some(Some(crate::types::DataValue::Int(o))) => *o,
                    _ => continue,
                };
                let col_name = match &col_row.get(2) {
                    Some(Some(crate::types::DataValue::Varchar(n))) => n.clone(),
                    Some(Some(crate::types::DataValue::Char(n))) => n.clone(),
                    _ => continue,
                };
                let col_type_str = match &col_row.get(3) {
                    Some(Some(crate::types::DataValue::Varchar(s))) => s.clone(),
                    Some(Some(crate::types::DataValue::Char(s))) => s.clone(),
                    _ => continue,
                };
                let col_type = col_type_str.parse::<DataType>().unwrap_or(DataType::Varchar(255));
                let col_nullable = match &col_row.get(5) {
                    Some(Some(crate::types::DataValue::Bool(v))) => *v,
                    _ => true,
                };

                col_with_ordinals.push((
                    col_ordinal,
                    Column {
                        name: col_name,
                        data_type: col_type,
                        nullable: col_nullable,
                        constraints: Constraints::default(),
                    },
                ));
            }
            col_with_ordinals.sort_by_key(|(ord, _)| *ord);
            let table_cols: Vec<Column> = col_with_ordinals.into_iter().map(|(_, c)| c).collect();

            database
                .tables
                .insert(tbl_name, Table { columns: table_cols });
        }

        // Load views for this database
        for view_row in &views_data {
            let view_db_id = match &view_row.get(1) {
                Some(Some(crate::types::DataValue::Int(id))) => *id,
                _ => continue,
            };
            if view_db_id != db_id {
                continue;
            }
            let view_name = match &view_row.get(2) {
                Some(Some(crate::types::DataValue::Varchar(n))) => n,
                Some(Some(crate::types::DataValue::Char(n))) => n,
                _ => continue,
            };
            let view_query_json = match &view_row.get(3) {
                Some(Some(crate::types::DataValue::Varchar(q))) => q,
                Some(Some(crate::types::DataValue::Char(q))) => q,
                _ => continue,
            };
            database.views.insert(view_name.clone(), crate::catalog::types::ViewDef {
                query_json: view_query_json.clone(),
            });
        }

        catalog.databases.insert(db_name, database);
    }

    catalog
}

/// Persist the in-memory catalog to system table heap files.
///
/// Deletes and recreates all system table files from scratch to prevent
/// duplicate rows on subsequent saves.
pub fn save_catalog_to_system(catalog: &Catalog) -> std::io::Result<()> {
    // 1. Load existing FOREIGN KEY constraints from sys_constraints before deleting it.
    //
    // CRITICAL: FK rows store the child table's numeric `table_id`.  After system tables
    // are rebuilt below, `populate_system_tables()` assigns new table_ids by iterating
    // a HashMap — whose iteration order is NOT deterministic (SwissTable random seed).
    // To prevent FK rows from pointing to wrong tables, we resolve the numeric table_id
    // to the table NAME before deletion, then re-resolve to the new table_id after rebuild.
    let mut fk_rows = Vec::new();
    let constr_path = sys_path("constraints");
    if constr_path.exists() {
        // Also load sys_tables to resolve table_id → table_name
        let mut tbl_id_to_name: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
        let tbl_path = sys_path("tables");
        if tbl_path.exists() {
            if let Ok(heap) = crate::backend::heap::HeapManager::open(tbl_path) {
                for result in heap.scan() {
                    if let Ok((_, _, raw_bytes)) = result {
                        if let Ok(decoded) = crate::types::deserialize_nullable_row(SYS_TABLES_SCHEMA, &raw_bytes) {
                            if decoded.len() >= 3 {
                                if let Some(Some(crate::types::DataValue::Int(tid))) = decoded.get(0) {
                                    let name_opt = match decoded.get(2) {
                                        Some(Some(crate::types::DataValue::Varchar(name))) => Some(name.clone()),
                                        Some(Some(crate::types::DataValue::Char(name))) => Some(name.clone()),
                                        _ => None,
                                    };
                                    if let Some(name) = name_opt {
                                        tbl_id_to_name.insert(*tid, name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Ok(heap) = crate::backend::heap::HeapManager::open(constr_path.clone()) {
            for result in heap.scan() {
                if let Ok((_, _, raw_bytes)) = result {
                    if let Ok(decoded) = crate::types::deserialize_nullable_row(SYS_CONSTRAINTS_SCHEMA, &raw_bytes) {
                        if decoded.len() >= 6 {
                            let constr_type = match &decoded[2] {
                                Some(crate::types::DataValue::Varchar(s)) => s.as_str(),
                                Some(crate::types::DataValue::Char(s)) => s.as_str(),
                                _ => "",
                            };
                            if constr_type.to_uppercase().contains("FOREIGN KEY") {
                                // Resolve numeric table_id → table name for the CHILD table
                                let old_table_id = match &decoded[1] {
                                    Some(crate::types::DataValue::Int(id)) => *id,
                                    _ => 0,
                                };
                                let child_table_name = tbl_id_to_name.get(&old_table_id)
                                    .cloned()
                                    .unwrap_or_else(|| format!("<table_id={}>", old_table_id));

                                let constraint_type = decoded[2].as_ref().map(|dv| value_to_string(dv));
                                let columns = decoded[3].as_ref().map(|dv| value_to_string(dv));
                                let ref_table = decoded[4].as_ref().map(|dv| value_to_string(dv));
                                let ref_columns = decoded[5].as_ref().map(|dv| value_to_string(dv));

                                // Store CHILD TABLE NAME as the "table_id" field.
                                // After rebuilding, populate_system_tables will resolve
                                // this name back to a numeric table_id. We use a special
                                // "NAME:" prefix to distinguish name-based entries from
                                // legacy numeric entries loaded by older code paths.
                                fk_rows.push(vec![
                                    None,                                      // constraint_id (auto-assigned)
                                    Some(format!("TABLE_NAME:{}", child_table_name)),  // resolved by name
                                    constraint_type,
                                    columns,
                                    ref_table,
                                    ref_columns,
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Delete existing system table files to start fresh
    for name in &["databases", "tables", "columns", "constraints", "indexes", "views"] {
        let path = sys_path(name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let fsm_path_str = format!("{}.fsm", path.to_string_lossy());
        let fsm_path = std::path::Path::new(&fsm_path_str);
        if fsm_path.exists() {
            std::fs::remove_file(fsm_path)?;
        }
    }
    // Create fresh files and populate
    create_system_table_files();
    populate_system_tables(catalog, fk_rows);
    Ok(())
}

/// Delete all system table metadata (constraints, columns, indexes) for a specific table.
///
/// Scans `sys_constraints`, `sys_columns`, and `sys_indexes` for rows whose
/// `table_id` (column index 1) matches the resolved table_id for
/// `(db_name, table_name)`, and deletes them via `HeapManager::delete_tuple`.
///
/// Returns the total number of deleted rows, or an error if the table can't
/// be found in `sys_tables`.
pub fn delete_table_metadata(db_name: &str, table_name: &str) -> std::io::Result<usize> {
    // Resolve table_id from sys_tables
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_id = tbl_rows.iter().find_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id != db_id {
            return None;
        }
        let name = match row.get(2) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(table_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Table '{}.{}' not found in sys_tables", db_name, table_name))
    })?;

    let mut total = 0usize;
    total += delete_rows_by_table_id("constraints", SYS_CONSTRAINTS_SCHEMA, table_id)?;
    total += delete_rows_by_table_id("columns", SYS_COLUMNS_SCHEMA, table_id)?;
    total += delete_rows_by_table_id("indexes", SYS_INDEXES_SCHEMA, table_id)?;

    log::info!(
        "[SystemCatalog] Deleted {} metadata rows for table '{}.{}' (table_id={})",
        total, db_name, table_name, table_id
    );
    Ok(total)
}

/// Insert a single constraint row into `sys_constraints`.
///
/// Resolves the `table_id` by scanning `sys_databases` and `sys_tables`,
/// then inserts a row with the appropriate `constraint_id` (auto-incremented).
///
/// Returns `Ok(())` on success, or an error if the table isn't found.
pub fn insert_constraint_metadata(
    db_name: &str,
    table_name: &str,
    constraint_type: &str,      // e.g. "NOT NULL", "UNIQUE", "PRIMARY KEY", "CHECK", "FOREIGN KEY"
    columns: &str,               // e.g. "id" or "id,name"
    ref_table: Option<&str>,     // for FOREIGN KEY
    ref_columns: Option<&str>,   // for FOREIGN KEY
) -> std::io::Result<()> {
    // 1. Resolve db_name → db_id
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    // 2. Resolve (db_id, table_name) → table_id
    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_id = tbl_rows.iter().find_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id != db_id {
            return None;
        }
        let name = match row.get(2) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(table_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Table '{}.{}' not found in sys_tables", db_name, table_name))
    })?;

    // 3. Compute next constraint_id by scanning sys_constraints
    let constr_rows = scan_system_table("constraints", SYS_CONSTRAINTS_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let next_constr_id = constr_rows.iter().fold(1i32, |max_id, row| {
        match row.get(0) {
            Some(Some(crate::types::DataValue::Int(id))) => std::cmp::max(max_id, *id + 1),
            _ => max_id,
        }
    });

    // 4. Insert the new constraint row
    // SYS_CONSTRAINTS_SCHEMA: [constraint_id:INT, table_id:INT, constraint_type:VARCHAR(50),
    //                         columns:TEXT, ref_table:VARCHAR(255), ref_columns:TEXT]
    let constr_row = vec![
        Some(next_constr_id.to_string()),
        Some(table_id.to_string()),
        Some(constraint_type.to_string()),
        Some(columns.to_string()),
        Some(ref_table.unwrap_or("").to_string()),
        Some(ref_columns.unwrap_or("").to_string()),
    ];
    insert_system_rows("constraints", SYS_CONSTRAINTS_SCHEMA, &[constr_row])
}

/// Insert metadata for a newly created index into `sys_indexes`.
///
/// Resolves the `table_id` by scanning `sys_databases` and `sys_tables`,
/// then inserts a row with the appropriate `index_id` (auto-incremented).
///
/// Returns `Ok(())` on success, or an error if any system table can't be
/// read or the table isn't found.
pub fn insert_index_metadata(
    db_name: &str,
    table_name: &str,
    index_name: &str,
    column_name: &str,
    is_unique: bool,
    is_primary: bool,
) -> std::io::Result<()> {
    // 1. Resolve db_name → db_id from sys_databases
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    // 2. Resolve (db_id, table_name) → table_id from sys_tables
    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_id = tbl_rows.iter().find_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id != db_id {
            return None;
        }
        let name = match row.get(2) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(table_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Table '{}.{}' not found in sys_tables", db_name, table_name))
    })?;

    // 3. Compute next index_id by scanning sys_indexes
    let idx_rows = scan_system_table("indexes", SYS_INDEXES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let next_idx_id = idx_rows.iter().fold(1i32, |max_id, row| {
        match row.get(0) {
            Some(Some(crate::types::DataValue::Int(id))) => std::cmp::max(max_id, *id + 1),
            _ => max_id,
        }
    });

    // 4. Insert the new index row
    let idx_row = vec![
        Some(next_idx_id.to_string()),           // index_id
        Some(table_id.to_string()),              // table_id
        Some(index_name.to_string()),            // name
        Some(if is_unique { "true" } else { "false" }.to_string()),  // is_unique
        Some(if is_primary { "true" } else { "false" }.to_string()), // is_primary
        Some(column_name.to_string()),           // columns
    ];
    insert_system_rows("indexes", SYS_INDEXES_SCHEMA, &[idx_row])
}

/// Delete a specific index entry from sys_indexes by index name and table.
///
/// Scans sys_databases and sys_tables to resolve table_id, then scans
/// sys_indexes to find a matching row by name, and deletes it.
///
/// Returns the number of deleted rows (0 or 1).
pub fn delete_index_metadata(
    db_name: &str,
    table_name: &str,
    index_name: &str,
) -> std::io::Result<usize> {
    // Resolve db_name → db_id
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    // Resolve (db_id, table_name) → table_id
    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_id = tbl_rows.iter().find_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id != db_id {
            return None;
        }
        let name = match row.get(2) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(table_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Table '{}.{}' not found in sys_tables", db_name, table_name))
    })?;

    // Scan sys_indexes to find matching row
    let idx_path = sys_path("indexes");
    if !idx_path.exists() {
        return Ok(0);
    }

    let mut heap = crate::backend::heap::HeapManager::open(idx_path)?;
    let mut to_delete: Vec<(u32, u32)> = Vec::new();

    for result in heap.scan() {
        let (page_id, slot_id, raw_bytes) = result?;
        let decoded = crate::types::deserialize_nullable_row(SYS_INDEXES_SCHEMA, &raw_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // SYS_INDEXES_SCHEMA: index_id, table_id, name, is_unique, is_primary, columns
        if decoded.len() >= 3 {
            let row_table_id = match &decoded[1] {
                Some(crate::types::DataValue::Int(id)) => *id,
                _ => continue,
            };
            if row_table_id != table_id {
                continue;
            }
            let row_name = match &decoded[2] {
                Some(crate::types::DataValue::Varchar(n)) => n,
                Some(crate::types::DataValue::Char(n)) => n,
                _ => continue,
            };
            if row_name.eq_ignore_ascii_case(index_name) {
                to_delete.push((page_id, slot_id));
            }
        }
    }

    let count = to_delete.len();
    for (page_id, slot_id) in &to_delete {
        heap.delete_tuple(*page_id, *slot_id)?;
    }
    heap.flush()?;

    log::info!(
        "[SystemCatalog] Deleted {} index row(s) for '{}.{}[{})'",
        count, db_name, table_name, index_name
    );
    Ok(count)
}

/// Delete constraint rows from sys_constraints for a specific column name.
///
/// Scans sys_databases and sys_tables to resolve table_id, then scans
/// sys_constraints for rows whose `columns` field matches `column_name`
/// and whose `table_id` matches, and deletes them.
///
/// Used when dropping a column: removes NOT NULL, UNIQUE, CHECK, FOREIGN KEY
/// constraints that reference the dropped column.
pub fn delete_column_constraints(
    db_name: &str,
    table_name: &str,
    column_name: &str,
) -> std::io::Result<usize> {
    // Resolve db_name → db_id, (db_id, table_name) → table_id
    let (table_id, _db_id) = resolve_table_id(db_name, table_name)?;

    let constr_path = sys_path("constraints");
    if !constr_path.exists() {
        return Ok(0);
    }

    let mut heap = crate::backend::heap::HeapManager::open(constr_path)?;
    let mut to_delete: Vec<(u32, u32)> = Vec::new();

    for result in heap.scan() {
        let (page_id, slot_id, raw_bytes) = result?;
        let decoded = crate::types::deserialize_nullable_row(SYS_CONSTRAINTS_SCHEMA, &raw_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // SYS_CONSTRAINTS_SCHEMA: constraint_id, table_id, constraint_type, columns, ref_table, ref_columns
        if decoded.len() >= 4 {
            let row_table_id = match &decoded[1] {
                Some(crate::types::DataValue::Int(id)) => *id,
                _ => continue,
            };
            if row_table_id != table_id {
                continue;
            }
            let row_columns = match &decoded[3] {
                Some(crate::types::DataValue::Varchar(c)) => c,
                Some(crate::types::DataValue::Char(c)) => c,
                _ => continue,
            };
            // Check if the columns field references the dropped column
            // (could be exact match or comma-separated)
            let col_refs: Vec<&str> = row_columns.split(',').map(|s| s.trim()).collect();
            if col_refs.iter().any(|c| c.eq_ignore_ascii_case(column_name)) {
                to_delete.push((page_id, slot_id));
            }
        }
    }

    let count = to_delete.len();
    for (page_id, slot_id) in &to_delete {
        heap.delete_tuple(*page_id, *slot_id)?;
    }
    heap.flush()?;

    log::info!(
        "[SystemCatalog] Deleted {} constraint row(s) for column '{}.{}.{}'",
        count, db_name, table_name, column_name
    );
    Ok(count)
}

/// Update the column name in sys_constraints for a renamed column.
///
/// Scans sys_constraints for rows matching the table_id and whose `columns`
/// field contains the old column name, then updates the field to the new name.
/// For composite constraint entries (e.g. "id,name"), only the matching part
/// is replaced.
pub fn rename_column_in_constraints(
    db_name: &str,
    table_name: &str,
    old_name: &str,
    new_name: &str,
) -> std::io::Result<usize> {
    let (table_id, _db_id) = resolve_table_id(db_name, table_name)?;

    let constr_path = sys_path("constraints");
    if !constr_path.exists() {
        return Ok(0);
    }

    let mut heap = crate::backend::heap::HeapManager::open(constr_path.clone())?;
    let mut updates: Vec<(u32, u32, Vec<Option<String>>)> = Vec::new();

    for result in heap.scan() {
        let (page_id, slot_id, raw_bytes) = result?;
        let decoded = crate::types::deserialize_nullable_row(SYS_CONSTRAINTS_SCHEMA, &raw_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if decoded.len() < 4 {
            continue;
        }

        let row_table_id = match &decoded[1] {
            Some(crate::types::DataValue::Int(id)) => *id,
            _ => continue,
        };
        if row_table_id != table_id {
            continue;
        }

        let row_columns = match &decoded[3] {
            Some(crate::types::DataValue::Varchar(c)) => c.clone(),
            Some(crate::types::DataValue::Char(c)) => c.clone(),
            _ => continue,
        };

        // Check if columns field contains the old name
        let parts: Vec<&str> = row_columns.split(',').map(|s| s.trim()).collect();
        let has_old = parts.iter().any(|c| c.eq_ignore_ascii_case(old_name));
        if !has_old {
            continue;
        }

        // Replace old_name with new_name (case-insensitive match, case-preserving replace)
        let new_columns = parts.iter().map(|c| {
            if c.eq_ignore_ascii_case(old_name) { new_name.to_string() } else { c.to_string() }
        }).collect::<Vec<_>>().join(", ");

        // Reconstruct a row with the updated columns field
        let updated_row = vec![
            decoded.get(0).and_then(|v| v.as_ref().map(|dv| value_to_string(dv))),
            decoded.get(1).and_then(|v| v.as_ref().map(|dv| value_to_string(dv))),
            decoded.get(2).and_then(|v| v.as_ref().map(|dv| value_to_string(dv))),
            Some(new_columns),
            decoded.get(4).and_then(|v| v.as_ref().map(|dv| value_to_string(dv))),
            decoded.get(5).and_then(|v| v.as_ref().map(|dv| value_to_string(dv))),
        ];
        updates.push((page_id, slot_id, updated_row));
    }

    let count = updates.len();
    // Delete old rows and insert updated ones
    for (page_id, slot_id, _row) in &updates {
        heap.delete_tuple(*page_id, *slot_id)?;
    }
    for (_page_id, _slot_id, row) in &updates {
        let str_refs: Vec<Option<&str>> = row.iter().map(|opt| opt.as_deref()).collect();
        let tuple_bytes = crate::types::serialize_nullable_row(SYS_CONSTRAINTS_SCHEMA, &str_refs)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        heap.insert_tuple(&tuple_bytes)?;
    }
    heap.flush()?;

    log::info!(
        "[SystemCatalog] Renamed column in {} constraint row(s) for '{}.{}': '{}' → '{}'",
        count, db_name, table_name, old_name, new_name
    );
    Ok(count)
}

/// Internal helper to resolve (db_name, table_name) → (table_id, db_id).
pub fn resolve_table_id(db_name: &str, table_name: &str) -> std::io::Result<(i32, i32)> {
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_id = tbl_rows.iter().find_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id != db_id {
            return None;
        }
        let name = match row.get(2) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(table_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Table '{}.{}' not found in sys_tables", db_name, table_name))
    })?;

    Ok((table_id, db_id))
}

/// Check if the system tables exist and are accessible.
pub fn system_tables_exist() -> bool {
    let sys_dir = std::path::Path::new(SYS_DIR);
    sys_dir.exists()
        && sys_path("databases").exists()
        && sys_path("tables").exists()
        && sys_path("columns").exists()
        && sys_path("constraints").exists()
        && sys_path("indexes").exists()
        && sys_path("views").exists()
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Create empty system table heap files (with HeapManager).
fn create_system_table_files() {
    let _ = std::fs::create_dir_all(SYS_DIR);

    for name in &["databases", "tables", "columns", "constraints", "indexes", "views"] {
        let path = sys_path(name);
        if !path.exists() {
            match crate::backend::heap::HeapManager::create(path.clone()) {
                Ok(mut hm) => {
                    if let Err(e) = hm.flush() {
                        log::warn!("[SystemCatalog] Failed to flush {}: {}", name, e);
                    }
                }
                Err(e) => {
                    log::error!("[SystemCatalog] Failed to create system table {}: {}", name, e);
                }
            }
        }
    }
}

/// Public wrapper: ensure all system table files exist, creating any that are
/// missing. Called during startup to support incremental schema additions
/// (e.g. adding `sys_views` to an existing system catalog).
pub fn ensure_system_tables() {
    create_system_table_files();
}

/// Scan all rows from a system table and return decoded values.
fn scan_system_table(name: &str, schema: &[DataType]) -> Result<Vec<Vec<Option<crate::types::DataValue>>>, String> {
    let path = sys_path(name);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let heap = crate::backend::heap::HeapManager::open(path)
        .map_err(|e| format!("Failed to open {}: {}", name, e))?;

    let mut rows = Vec::new();
    for result in heap.scan() {
        let (_page_id, _slot_id, raw_bytes) = match result {
            Ok(triple) => triple,
            Err(e) => return Err(format!("Scan error on {}: {}", name, e)),
        };

        let decoded = crate::types::deserialize_nullable_row(schema, &raw_bytes)
            .map_err(|e| format!("Deserialize error on {}: {}", name, e))?;

        rows.push(decoded);
    }

    Ok(rows)
}

/// Insert multiple rows into a system table in a single batch.
///
/// Opens the HeapManager once, inserts all rows, and closes.
fn insert_system_rows(
    name: &str,
    schema: &[DataType],
    rows: &[Vec<Option<String>>],
) -> std::io::Result<()> {
    let path = sys_path(name);
    let mut heap = crate::backend::heap::HeapManager::open(path)?;

    for row in rows {
        // Convert Vec<Option<String>> into Vec<Option<&str>> for serialisation
        let str_refs: Vec<Option<&str>> = row.iter().map(|opt| opt.as_deref()).collect();
        let tuple_bytes = crate::types::serialize_nullable_row(schema, &str_refs)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        heap.insert_tuple(&tuple_bytes)?;
    }

    heap.flush()?;
    Ok(())
}

/// Populate all system tables from an in-memory Catalog.
///
/// Writes databases, tables, columns, constraints, and (where possible)
/// indexes to their respective heap files.
fn populate_system_tables(catalog: &Catalog, mut constr_rows: Vec<Vec<Option<String>>>) {
    // Sequence counters for auto-generated IDs
    let mut next_db_id = 1i32;
    let mut next_table_id = 1i32;
    let mut next_col_id = 1i32;

    // Build a table_name → table_id mapping BEFORE processing FK rows.
    // This is populated during the table iteration below, then used to
    // resolve "TABLE_NAME:xxx" entries in constr_rows.
    let mut table_name_to_new_id: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    // First pass: collect table names and assign table_ids (needed for FK
    // name resolution).  The iteration order matches the main loop below
    // because no HashMap mutations happen between passes.
    for (_db_name, database) in &catalog.databases {
        for (tbl_name, _table) in &database.tables {
            let table_id = next_table_id;
            table_name_to_new_id.insert(tbl_name.clone(), table_id);
            next_table_id += 1;
        }
    }
    // Reset counter — the real table iteration below will re-assign the same IDs
    next_table_id = 1i32;

    // Compute the next available constraint_id from existing rows
    let mut next_constr_id = constr_rows.iter().fold(1i32, |max_id, row| {
        if let Some(Some(id_str)) = row.first() {
            if let Ok(id) = id_str.parse::<i32>() {
                return std::cmp::max(max_id, id + 1);
            }
        }
        max_id
    });

    // Assign real constraint_ids to FK rows that were preserved with None
    // (they were loaded from the previous save cycle with constraint_id=None
    //  because the numeric ID was tied to the old table iteration order).
    for row in &mut constr_rows {
        if row.len() >= 1 && row[0].is_none() {
            row[0] = Some(next_constr_id.to_string());
            next_constr_id += 1;
        }
    }

    // Batch collections per system table
    let mut db_rows: Vec<Vec<Option<String>>> = Vec::new();
    let mut tbl_rows: Vec<Vec<Option<String>>> = Vec::new();
    let mut col_rows: Vec<Vec<Option<String>>> = Vec::new();

    for (db_name, database) in &catalog.databases {
        let db_id = next_db_id;
        next_db_id += 1;

        db_rows.push(vec![
            Some(db_id.to_string()),
            Some(db_name.clone()),
        ]);

        for (tbl_name, table) in &database.tables {
            let table_id = next_table_id;
            next_table_id += 1;

            let file_path = format!("database/base/{}/{}.dat", db_name, tbl_name);

            tbl_rows.push(vec![
                Some(table_id.to_string()),
                Some(db_id.to_string()),
                Some(tbl_name.clone()),
                Some(file_path),
            ]);

            for (ordinal, col) in table.columns.iter().enumerate() {
                let col_id = next_col_id;
                next_col_id += 1;

                let data_type_str = datatype_to_string(&col.data_type);
                let has_default = col.constraints.default.is_some();
                let default_value = col
                    .constraints
                    .default
                    .as_ref()
                    .map(|dv| value_to_string(dv))
                    .unwrap_or_default();
                let nullable_str = if col.nullable { "true" } else { "false" };
                let has_default_str = if has_default { "true" } else { "false" };

                col_rows.push(vec![
                    Some(col_id.to_string()),
                    Some(table_id.to_string()),
                    Some(col.name.clone()),
                    Some(data_type_str),
                    Some(ordinal.to_string()),
                    Some(nullable_str.to_string()),
                    Some(has_default_str.to_string()),
                    Some(default_value),
                ]);

                // ── Emit constraint rows from per-column constraints ──
                if col.constraints.not_null {
                    constr_rows.push(vec![
                        Some(next_constr_id.to_string()),
                        Some(table_id.to_string()),
                        Some("NOT NULL".to_string()),
                        Some(col.name.clone()),
                        None,
                        None,
                    ]);
                    next_constr_id += 1;
                }
                if col.constraints.unique {
                    constr_rows.push(vec![
                        Some(next_constr_id.to_string()),
                        Some(table_id.to_string()),
                        Some("UNIQUE".to_string()),
                        Some(col.name.clone()),
                        None,
                        None,
                    ]);
                    next_constr_id += 1;
                }
                if let Some(ref check_expr) = col.constraints.check {
                    constr_rows.push(vec![
                        Some(next_constr_id.to_string()),
                        Some(table_id.to_string()),
                        Some("CHECK".to_string()),
                        Some(format!("CHECK({})", check_expr)),
                        None,
                        None,
                    ]);
                    next_constr_id += 1;
                }
            }
        }
    }

    // ── Resolve "TABLE_NAME:xxx" FK entries ────────────────────────────
    // FK rows preserved from the previous save cycle use the child table's
    // NAME (prefixed with "TABLE_NAME:") instead of a numeric table_id.  We
    // resolve them to the newly-assigned numeric table_ids here.
    for row in &mut constr_rows {
        if row.len() < 6 {
            continue;
        }
        if let Some(Some(table_id_str)) = &row.get(1).cloned() {
            if let Some(table_name) = table_id_str.strip_prefix("TABLE_NAME:") {
                let resolved_id = table_name_to_new_id.get(table_name)
                    .cloned()
                    .unwrap_or(0);
                row[1] = if resolved_id != 0 {
                    Some(resolved_id.to_string())
                } else {
                    log::warn!(
                        "[SystemCatalog] Could not resolve table name '{}' for FK constraint, using table_id=0",
                        table_name
                    );
                    Some("0".to_string())
                };
            }
        }
    }

    // Batch-write each system table
    if let Err(e) = insert_system_rows("databases", SYS_DATABASES_SCHEMA, &db_rows) {
        log::error!("[SystemCatalog] Failed to write sys_databases: {}", e);
    }
    if let Err(e) = insert_system_rows("tables", SYS_TABLES_SCHEMA, &tbl_rows) {
        log::error!("[SystemCatalog] Failed to write sys_tables: {}", e);
    }
    if let Err(e) = insert_system_rows("columns", SYS_COLUMNS_SCHEMA, &col_rows) {
        log::error!("[SystemCatalog] Failed to write sys_columns: {}", e);
    }
    if let Err(e) = insert_system_rows("constraints", SYS_CONSTRAINTS_SCHEMA, &constr_rows) {
        log::error!("[SystemCatalog] Failed to write sys_constraints: {}", e);
    }

    // ── Populate sys_indexes by scanning .idx.meta files on disk ────────
    // Build a table_name → table_id mapping for index rows.
    let mut table_name_to_id: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    for row in &tbl_rows {
        // tbl_rows: [table_id, db_id, name, file_path]
        if let (Some(table_id_str), Some(table_name)) = (&row[0], &row[2]) {
            if let Ok(tid) = table_id_str.parse::<i32>() {
                table_name_to_id.insert(table_name.clone(), tid);
            }
        }
    }
    // ── Populate sys_views ──────────────────────────────────────────────
    let mut view_rows: Vec<Vec<Option<String>>> = Vec::new();
    let mut next_view_id = 1i32;
    for (db_name, database) in &catalog.databases {
        let db_id = db_rows.iter().find_map(|row| {
            match (&row[0], &row[1]) {
                (Some(id_str), Some(name)) if name == db_name => id_str.parse::<i32>().ok(),
                _ => None,
            }
        }).unwrap_or(1);

        for (view_name, view_def) in &database.views {
            view_rows.push(vec![
                Some(next_view_id.to_string()), // view_id
                Some(db_id.to_string()),        // db_id
                Some(view_name.clone()),        // name
                Some(view_def.query_json.clone()),   // query_json
            ]);
            next_view_id += 1;
        }
    }
    if let Err(e) = insert_system_rows("views", SYS_VIEWS_SCHEMA, &view_rows) {
        log::error!("[SystemCatalog] Failed to write sys_views: {}", e);
    }

    populate_indexes_from_meta(&table_name_to_id);

    let view_count: usize = catalog.databases.values().map(|d| d.views.len()).sum();
    log::info!(
        "[SystemCatalog] Populated system tables: {} databases, {} tables, {} constraints, {} views",
        catalog.databases.len(),
        {
            let mut count = 0;
            for db in catalog.databases.values() {
                count += db.tables.len();
            }
            count
        },
        next_constr_id - 1,
        view_count
    );
}

/// Scan `database/base/**/*.idx.meta` files and populate `sys_indexes`.
///
/// `table_name_to_id` maps table names to their assigned table IDs (from
/// `sys_tables`) so the index rows can reference the correct table.
fn populate_indexes_from_meta(table_name_to_id: &std::collections::HashMap<String, i32>) {
    let base_dir = std::path::Path::new(crate::layout::DATABASE_DIR);
    if !base_dir.exists() {
        return;
    }

    let mut idx_rows: Vec<Vec<Option<String>>> = Vec::new();
    let mut next_idx_id = 1i32;

    // Walk database/base/*/ looking for .idx.meta files
    let walker = match std::fs::read_dir(base_dir) {
        Ok(r) => r,
        Err(_) => return,
    };

    for entry in walker.flatten() {
        let entry_path = entry.path();
        if !entry_path.is_dir() {
            continue;
        }

        // Look for *.idx.meta files in this database directory
        let meta_files = match std::fs::read_dir(&entry_path) {
            Ok(r) => r,
            Err(_) => continue,
        };

        for meta_entry in meta_files.flatten() {
            let meta_path = meta_entry.path();
            let fname = meta_path.to_string_lossy();
            if !fname.ends_with(".idx.meta") {
                continue;
            }

            // Extract table name and index name from the .idx.meta filename.
            //
            // Named format:  {table}.{index_name}.idx.meta
            //   e.g. "users.idx_users_name.idx.meta" → stem = "users.idx_users_name.idx"
            //   strip ".idx" → "users.idx_users_name" → split on first '.' → table="users", index="idx_users_name"
            //
            // Legacy format: {table}.idx.meta
            //   e.g. "users.idx.meta" → stem = "users.idx" → strip ".idx" → "users"
            //   no '.' in the result → legacy, index_name = "idx_{table}_{column}"
            let filename = match meta_path.file_stem() {
                Some(s) => s.to_string_lossy().to_string(),
                None => continue,
            };
            let stripped = filename.strip_suffix(".idx").unwrap_or(&filename).to_string();

            // Determine if this is a named or legacy format by checking for extra dots
            let (table_name, index_name) = if stripped.contains('.') {
                // Named format: "table.index_name" → split at first dot
                if let Some(dot_pos) = stripped.find('.') {
                    let tbl = stripped[..dot_pos].to_string();
                    let idx = stripped[dot_pos + 1..].to_string();
                    (tbl, idx)
                } else {
                    (stripped.clone(), format!("idx_{}", stripped))
                }
            } else {
                // Legacy format: just "table" → index_name = "idx_{table}_{column}"
                // We'll read the meta to get the column name later
                (stripped.clone(), String::new())
            };

            // Resolve table_id from the mapping
            let table_id = match table_name_to_id.get(&table_name) {
                Some(id) => *id,
                None => {
                    log::warn!("[SystemCatalog] Index meta references unknown table '{}', skipping", table_name);
                    continue;
                }
            };

            // Read and parse the metadata JSON
            let meta_content = match std::fs::read_to_string(&meta_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Parse IndexMeta from the metadata file
            #[derive(serde::Deserialize)]
            #[allow(dead_code)]
            struct IndexMeta {
                column_name: String,
                column_idx: usize,
                key_type: String,
            }

            let meta: IndexMeta = match serde_json::from_str(&meta_content) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Compute the index name: use parsed name if available, otherwise generate from column
            let idx_name = if !index_name.is_empty() {
                index_name
            } else {
                format!("idx_{}_{}", table_name, meta.column_name)
            };

            // SYS_INDEXES_SCHEMA: [index_id:INT, table_id:INT, name:VARCHAR(255),
            //                     is_unique:BOOL, is_primary:BOOL, columns:VARCHAR(255)]
            idx_rows.push(vec![
                Some(next_idx_id.to_string()),           // index_id
                Some(table_id.to_string()),              // table_id
                Some(idx_name),                          // name
                Some("true".to_string()),               // is_unique (assumed true for legacy compat)
                Some("false".to_string()),              // is_primary
                Some(meta.column_name.clone()),          // columns (indexed column)
            ]);
            next_idx_id += 1;
        }
    }

    if !idx_rows.is_empty() {
        if let Err(e) = insert_system_rows("indexes", SYS_INDEXES_SCHEMA, &idx_rows) {
            log::error!("[SystemCatalog] Failed to write sys_indexes: {}", e);
        }
    }
}

/// Return the physical `DataType` schema for a system table by name.
///
/// Used by the physical planner's `plan_table_scan` to correctly deserialise
/// system table heap file tuples using the actual on-disk column types rather
/// than the VARCHAR-heavy INFORMATION_SCHEMA view schema.
pub fn system_table_schema(name: &str) -> &'static [DataType] {
    match name {
        "databases" => SYS_DATABASES_SCHEMA,
        "tables" => SYS_TABLES_SCHEMA,
        "columns" => SYS_COLUMNS_SCHEMA,
        "constraints" => SYS_CONSTRAINTS_SCHEMA,
        "indexes" => SYS_INDEXES_SCHEMA,
        "views" => SYS_VIEWS_SCHEMA,
        // Fallback: treat all columns as VARCHAR(255)
        _ => &[DataType::Varchar(255)],
    }
}

/// Return a `ColumnSchema` for the given INFORMATION_SCHEMA view name.
///
/// Maps standard `information_schema.xxx` view names to the corresponding
/// system table column definitions so the logical planner can provide proper
/// schema metadata.
///
/// **Important**: The number of columns returned by this function MUST match
/// the number of physical columns in the corresponding system table
/// (`SYS_DATABASES_SCHEMA`, `SYS_TABLES_SCHEMA`, etc.) because the physical
/// SeqScanOperator uses this schema to label deserialised tuples. Column
/// names follow the SQL standard INFORMATION_SCHEMA convention for each view.
pub fn info_schema_column_schema(view_name: &str) -> ColumnSchema {
    let cols = match view_name.to_ascii_lowercase().as_str() {
        // information_schema.tables → sys_tables (4 cols)
        "tables" => vec![
            ("table_catalog", "VARCHAR(255)"),
            ("table_schema", "VARCHAR(255)"),
            ("table_name", "VARCHAR(255)"),
            ("table_type", "VARCHAR(50)"),
        ],
        // information_schema.schemata → sys_databases (2 cols)
        "schemata" => vec![
            ("catalog_name", "VARCHAR(255)"),
            ("schema_name", "VARCHAR(255)"),
        ],
        // information_schema.columns → sys_columns (8 cols)
        "columns" => vec![
            ("table_catalog", "VARCHAR(255)"),
            ("table_schema", "VARCHAR(255)"),
            ("table_name", "VARCHAR(255)"),
            ("column_name", "VARCHAR(255)"),
            ("ordinal_position", "INT"),
            ("data_type", "VARCHAR(100)"),
            ("is_nullable", "VARCHAR(3)"),
            ("column_default", "VARCHAR(255)"),
        ],
        // information_schema.table_constraints → sys_constraints (6 cols)
        "table_constraints" => vec![
            ("constraint_catalog", "VARCHAR(255)"),
            ("constraint_schema", "VARCHAR(255)"),
            ("constraint_name", "VARCHAR(255)"),
            ("table_name", "VARCHAR(255)"),
            ("constraint_type", "VARCHAR(50)"),
            ("constraint_columns", "VARCHAR(1024)"),
        ],
        // information_schema.statistics / information_schema.indexes → sys_indexes (6 cols)
        "statistics" | "indexes" => vec![
            ("table_catalog", "VARCHAR(255)"),
            ("table_schema", "VARCHAR(255)"),
            ("table_name", "VARCHAR(255)"),
            ("index_name", "VARCHAR(255)"),
            ("unique", "VARCHAR(3)"),
            ("primary", "VARCHAR(3)"),
        ],
        // information_schema.key_column_usage → sys_columns (8 cols)
        // Provides a standard view of which columns participate in key constraints.
        // Currently backed by sys_columns for broad coverage; constraint metadata
        // will be enriched as the catalog gains native constraint tracking.
        "key_column_usage" => vec![
            ("table_catalog", "VARCHAR(255)"),
            ("table_schema", "VARCHAR(255)"),
            ("table_name", "VARCHAR(255)"),
            ("column_name", "VARCHAR(255)"),
            ("ordinal_position", "INT"),
            ("constraint_catalog", "VARCHAR(255)"),
            ("constraint_schema", "VARCHAR(255)"),
            ("constraint_name", "VARCHAR(255)"),
        ],
        // information_schema.views → sys_views (4 cols)
        "views" => vec![
            ("table_catalog", "VARCHAR(255)"),
            ("table_schema", "VARCHAR(255)"),
            ("view_name", "VARCHAR(255)"),
            ("view_definition", "VARCHAR(8192)"),
        ],
        _other => {
            // Fallback: generic description
            vec![("name", "VARCHAR(255)")]
        }
    };
    ColumnSchema {
        columns: cols.into_iter().map(|(name, dt)| ColumnInfo {
            name: name.to_string(),
            data_type: dt.to_string(),
            nullable: true }).collect(),
    }
}

/// Delete all system table metadata for an entire database.
///
/// Deletes the database entry from `sys_databases`, all its tables from
/// `sys_tables`, and all associated columns, constraints, and indexes.
///
/// Returns the total number of deleted rows, or an error if the database
/// can't be found in `sys_databases`.
pub fn delete_database_metadata(db_name: &str) -> std::io::Result<usize> {
    // 1. Resolve db_name → db_id
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    // 2. Find all tables belonging to this database
    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let table_ids: Vec<i32> = tbl_rows.iter().filter_map(|row| {
        let tbl_db_id = match row.get(1) {
            Some(Some(crate::types::DataValue::Int(id))) => *id,
            _ => return None,
        };
        if tbl_db_id == db_id {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).collect();

    let mut total = 0usize;

    // 3. Delete metadata for each table (columns, constraints, indexes)
    for table_id in &table_ids {
        total += delete_rows_by_table_id("constraints", SYS_CONSTRAINTS_SCHEMA, *table_id)?;
        total += delete_rows_by_table_id("columns", SYS_COLUMNS_SCHEMA, *table_id)?;
        total += delete_rows_by_table_id("indexes", SYS_INDEXES_SCHEMA, *table_id)?;
    }

    // 4. Delete table entries from sys_tables (db_id is at column index 1)
    total += delete_rows_by_column("tables", SYS_TABLES_SCHEMA, 1, db_id)?;

    // 5. Delete view entries from sys_views (db_id is at column index 1)
    total += delete_rows_by_column("views", SYS_VIEWS_SCHEMA, 1, db_id)?;

    // 6. Delete the database entry from sys_databases (db_id is at column index 0)
    total += delete_rows_by_column("databases", SYS_DATABASES_SCHEMA, 0, db_id)?;

    log::info!(
        "[SystemCatalog] Deleted {} metadata rows for database '{}' (db_id={})",
        total, db_name, db_id
    );
    Ok(total)
}

/// Internal helper: scan a system table and delete all rows where the
/// column at `column_idx` matches `target_id`.
///
/// Used by `delete_database_metadata` to delete from `sys_databases` (col 0)
/// and `sys_tables` (col 1), and by `delete_table_metadata` via the
/// more specific `delete_rows_by_table_id` wrapper.
fn delete_rows_by_column(name: &str, schema: &[DataType], column_idx: usize, target_id: i32) -> std::io::Result<usize> {
    let path = sys_path(name);
    if !path.exists() {
        return Ok(0);
    }

    let mut heap = crate::backend::heap::HeapManager::open(path)?;
    let mut to_delete: Vec<(u32, u32)> = Vec::new();

    for result in heap.scan() {
        let (page_id, slot_id, raw_bytes) = result?;
        let decoded = crate::types::deserialize_nullable_row(schema, &raw_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if let Some(Some(crate::types::DataValue::Int(tid))) = decoded.get(column_idx) {
            if *tid == target_id {
                to_delete.push((page_id, slot_id));
            }
        }
    }

    let count = to_delete.len();
    for (page_id, slot_id) in &to_delete {
        heap.delete_tuple(*page_id, *slot_id)?;
    }
    heap.flush()?;

    log::trace!(
        "[SystemCatalog] delete_rows_by_column({}, col={}): removed {} rows for id={}",
        name, column_idx, count, target_id
    );
    Ok(count)
}

/// Internal helper: scan a system table and delete all rows where the
/// `table_id` column (index 1) matches `target_table_id`.
///
/// Uses `HeapManager::delete_tuple` to mark each matching slot as deleted.
fn delete_rows_by_table_id(name: &str, schema: &[DataType], target_table_id: i32) -> std::io::Result<usize> {
    delete_rows_by_column(name, schema, 1, target_table_id)
}

/// Scan `sys_constraints` and delete all FOREIGN KEY rows referencing `parent_table_name`.
pub fn delete_referencing_foreign_keys(
    db_name: &str,
    parent_table_name: &str,
) -> std::io::Result<usize> {
    // Resolve db_id
    let db_rows = scan_system_table("databases", SYS_DATABASES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let db_id = db_rows.iter().find_map(|row| {
        let name = match row.get(1) {
            Some(Some(crate::types::DataValue::Varchar(n))) => n,
            Some(Some(crate::types::DataValue::Char(n))) => n,
            _ => return None,
        };
        if name.eq_ignore_ascii_case(db_name) {
            if let Some(Some(crate::types::DataValue::Int(id))) = row.get(0) {
                return Some(*id);
            }
        }
        None
    }).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound,
            format!("Database '{}' not found in sys_databases", db_name))
    })?;

    // Load sys_tables for the db to create a table_id → db_id mapping
    let tbl_rows = scan_system_table("tables", SYS_TABLES_SCHEMA)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut table_id_to_db_id = std::collections::HashMap::new();
    for row in &tbl_rows {
        if let (Some(Some(crate::types::DataValue::Int(tid))), Some(Some(crate::types::DataValue::Int(t_db_id)))) = (row.get(0), row.get(1)) {
            table_id_to_db_id.insert(*tid, *t_db_id);
        }
    }

    let constr_path = sys_path("constraints");
    if !constr_path.exists() {
        return Ok(0);
    }

    let mut heap = crate::backend::heap::HeapManager::open(constr_path)?;
    let mut to_delete: Vec<(u32, u32)> = Vec::new();

    for result in heap.scan() {
        let (page_id, slot_id, raw_bytes) = result?;
        let decoded = crate::types::deserialize_nullable_row(SYS_CONSTRAINTS_SCHEMA, &raw_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        if decoded.len() >= 6 {
            let row_table_id = match &decoded[1] {
                Some(crate::types::DataValue::Int(id)) => *id,
                _ => continue,
            };
            // Ensure this constraint belongs to a table in the target database
            if table_id_to_db_id.get(&row_table_id) != Some(&db_id) {
                continue;
            }
            let constraint_type = match &decoded[2] {
                Some(crate::types::DataValue::Varchar(t)) => t,
                Some(crate::types::DataValue::Char(t)) => t,
                _ => continue,
            };
            if !constraint_type.eq_ignore_ascii_case("FOREIGN KEY") {
                continue;
            }
            let ref_table = match &decoded[4] {
                Some(crate::types::DataValue::Varchar(t)) => t,
                Some(crate::types::DataValue::Char(t)) => t,
                _ => continue,
            };
            if ref_table.eq_ignore_ascii_case(parent_table_name) {
                to_delete.push((page_id, slot_id));
            }
        }
    }

    let count = to_delete.len();
    for (page_id, slot_id) in &to_delete {
        heap.delete_tuple(*page_id, *slot_id)?;
    }
    heap.flush()?;
    Ok(count)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::types::*;

    #[allow(dead_code)]
    fn make_test_catalog() -> Catalog {
        let mut catalog = Catalog {
            databases: HashMap::new(),
        };

        let users = Table {
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Int,
                    nullable: false,
                    constraints: Constraints::default(),
                },
                Column {
                    name: "name".to_string(),
                    data_type: DataType::Varchar(100),
                    nullable: true,
                    constraints: Constraints::default(),
                },
            ],
        };

        let mut tables = HashMap::new();
        tables.insert("users".to_string(), users);

        catalog
            .databases
            .insert("test_db".to_string(), Database { tables, views: HashMap::new() });

        catalog
    }

    #[test]
    fn test_schema_constants_are_valid() {
        // Just verify schemas compile and have expected column counts
        assert_eq!(SYS_DATABASES_SCHEMA.len(), 2);
        assert_eq!(SYS_TABLES_SCHEMA.len(), 4);
        assert_eq!(SYS_COLUMNS_SCHEMA.len(), 8);
        assert_eq!(SYS_CONSTRAINTS_SCHEMA.len(), 6);
        assert_eq!(SYS_INDEXES_SCHEMA.len(), 6);
        assert_eq!(SYS_VIEWS_SCHEMA.len(), 4);
    }

    #[test]
    fn test_schema_serialization_roundtrip() {
        // Verify that the system table schemas produce valid serialization
        let schema = vec![DataType::Int, DataType::Varchar(255)];
        let result = crate::types::serialize_nullable_row(
            &schema,
            &[Some("42"), Some("hello")],
        );
        assert!(result.is_ok());

        let bytes = result.unwrap();
        let decoded = crate::types::deserialize_nullable_row(&schema, &bytes)
            .expect("Roundtrip deserialization failed");
        assert_eq!(decoded.len(), 2);

        // Verify values roundtrip
        match &decoded[0] {
            Some(crate::types::DataValue::Int(v)) => assert_eq!(*v, 42),
            _ => panic!("Expected Int(42)"),
        }
        match &decoded[1] {
            Some(crate::types::DataValue::Varchar(v)) => assert_eq!(v, "hello"),
            _ => panic!("Expected Varchar(hello)"),
        }
    }
}
