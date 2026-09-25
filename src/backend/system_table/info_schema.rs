//! Split of `system_table` — see `mod.rs` for the module overview.

use rook_ast::logical::{ColumnInfo, ColumnSchema};

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
        columns: cols
            .into_iter()
            .map(|(name, dt)| ColumnInfo {
                name: name.to_string(),
                data_type: dt.to_string(),
                nullable: true,
            })
            .collect(),
    }
}
