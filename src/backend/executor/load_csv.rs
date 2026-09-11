use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use crate::backend::heap::HeapManager;
use crate::catalog::types::Catalog;
use crate::types::DataValue;
use crate::types::validation::validate_value;
use crate::types::row::serialize_nullable_row;

/// Load CSV file with full validation and error handling using HeapManager.
///
/// Before loading any data:
/// 1. Validates that all column data types are supported
/// 2. Checks that CSV file is readable
/// 3. Performs row-by-row validation and type checking
/// 4. Uses HeapManager for FSM-aware insertion
///
/// Returns count of successfully inserted rows on success.
pub fn load_csv(
    catalog: &Catalog,
    db_name: &str,
    table_name: &str,
    csv_path: &str,
) -> io::Result<u32> {
    log::info!(" Starting CSV load operation");
    log::info!(
        " Database: '{}', Table: '{}', CSV: '{}'",
        db_name,
        table_name,
        csv_path
    );

    // --- 1. Fetch table schema from catalog ---
    let db = catalog.databases.get(db_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Database '{}' not found", db_name),
        )
    })?;

    let table = db.tables.get(table_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Table '{}' not found", table_name),
        )
    })?;

    let columns = &table.columns;
    if columns.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Table has no columns",
        ));
    }

    log::info!(" Found table with {} columns", columns.len());

    // --- 2. VALIDATE ALL DATA TYPES BEFORE LOADING ---
    log::info!(" Validating schema data types...");
    for (idx, col) in columns.iter().enumerate() {
        log::info!("   Column {}: '{}' → {}", idx + 1, col.name, col.data_type);
        log::info!("   Supported data type: {:?}", col.data_type);
    }
    log::info!(" All data types validated successfully");

    // --- 3. Open and read the CSV file ---
    log::info!(" Opening CSV file: '{}'", csv_path);
    let csv_file = match File::open(csv_path) {
        Ok(f) => {
            log::info!(" CSV file opened successfully");
            f
        }
        Err(e) => {
            log::info!(" Failed to open CSV file: {}", e);
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to open CSV file '{}': {}", csv_path, e),
            ));
        }
    };

    let reader = BufReader::new(csv_file);
    let mut lines = reader.lines();

    // Skip header line
    log::info!(" Reading CSV header...");
    if let Some(Ok(header)) = lines.next() {
        log::info!(" Header: {}", header);
    }

    // --- 3a. Open HeapManager for FSM-aware insertion ---
    let table_path = PathBuf::from(format!("database/base/{}/{}.dat", db_name, table_name));
    let mut heap_manager = match HeapManager::open(table_path.clone()) {
        Ok(hm) => {
            log::info!(" Opened HeapManager (FSM will be created/updated)");
            hm
        }
        Err(e) => {
            log::info!(" CRITICAL: Failed to open HeapManager: {}", e);
            return Err(e);
        }
    };

    // --- 4. Iterate through rows with detailed validation ---
    let mut inserted = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;

    for (line_num, line) in lines.enumerate() {
        let line_idx = line_num + 2; // +2 because we skipped header and start from line 2

        let row = match line {
            Ok(r) => r,
            Err(e) => {
                log::info!(" Line {}: Error reading line: {}", line_idx, e);
                failed += 1;
                continue;
            }
        };

        if row.trim().is_empty() {
            log::debug!("Line {}: Skipping empty row", line_idx);
            skipped += 1;
            continue;
        }

        let values: Vec<&str> = row.split(',').map(|v| v.trim()).collect();

        if values.len() != columns.len() {
            log::warn!(
                "[CSV LOADER] Line {}: Expected {} columns, found {}. Skipping row.",
                line_idx,
                columns.len(),
                values.len()
            );
            skipped += 1;
            continue;
        }

        // --- 5. Validate each value before serialization ---
        let mut validation_passed = true;
        for (col_idx, (val, col)) in values.iter().zip(columns.iter()).enumerate() {
            let data_type = &col.data_type;
            if let Err(validation_err) = validate_value(data_type, val) {
                log::warn!(
                    "[CSV LOADER] Line {}, Column {} ('{}'): {} Value: '{}'",
                    line_idx,
                    col_idx + 1,
                    col.name,
                    validation_err,
                    val
                );
                validation_passed = false;
                break;
            }
        }

        if !validation_passed {
            log::error!("Line {}: Validation failed. Skipping row.", line_idx);
            failed += 1;
            continue;
        }

        // --- 6. Serialize row based on schema ---
        let mut row_ok = true;

        for (val, col) in values.iter().zip(columns.iter()) {
            match DataValue::parse_and_encode(&col.data_type, val) {
                Ok(_) => {}
                Err(e) => {
                    println!("Skipping row {}: column '{}' — {}", line_idx, col.name, e);
                    row_ok = false;
                    break;
                }
            }
        }

        if !row_ok {
            continue;
        }

        // Build datatype list
        let data_types: Vec<_> = columns
            .iter()
            .map(|c| c.data_type.clone())
            .collect();

        // Build nullable value list
        let nullable_values: Vec<Option<&str>> =
            values.iter().map(|v| Some(*v)).collect();

        // Serialize using tuple layout serializer
        let tuple_bytes = match serialize_nullable_row(&data_types, &nullable_values) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!(
                    "Line {}: Failed to serialize row: {}",
                    line_idx,
                    e
                );
                failed += 1;
                continue;
            }
        };

        // --- 7. Insert tuple using HeapManager (FSM-aware) ---
        match heap_manager.insert_tuple(&tuple_bytes) {
            Ok((page_id, slot_id)) => {
                // Update any existing B+ Tree index
                if let Err(e) = crate::backend::executor::create_index::update_index_on_insert(
                    db_name, table_name, &values, page_id, slot_id,
                ) {
                    log::warn!("Failed to update index for row {}: {}", line_idx, e);
                }

                inserted += 1;
                if inserted.is_multiple_of(100) {
                    log::info!("Inserted {} rows so far...", inserted);
                }
            }
            Err(e) => {
                log::error!("Line {}: Failed to insert row: {}", line_idx, e);
                failed += 1;
            }
        }
    }

    log::info!("═══════════════════════════════════");
    log::info!(" CSV Load Summary:");
    log::info!(" Successfully inserted: {}", inserted);
    log::info!(" Skipped (formatting): {}", skipped);
    log::info!(" Failed (validation/insert): {}", failed);
    log::info!(" Total rows processed: {}", inserted + skipped + failed);
    log::info!(" ═══════════════════════════════════\n");

    if inserted == 0 && (skipped > 0 || failed > 0) {
        log::warn!(
            "WARNING: No rows were inserted. Please check your CSV file format and data types."
        );
    }

    Ok(inserted)
}

/// Insert a single tuple manually using HeapManager (FSM-aware)
///
/// Hot-path notes: the HeapManager for the table is process-cached
/// (`backend::cache::with_heap`) so repeated calls do not re-open files, and
/// index maintenance goes through the cached B+ Tree registry with batched
/// fsyncs. Callers that need data on disk should hit a checkpoint
/// (`backend::cache::checkpoint`) — heap scans do this automatically.
pub fn insert_single_tuple(
    catalog: &Catalog,
    db_name: &str,
    table_name: &str,
    values: &[&str],
) -> io::Result<bool> {
    log::info!(" Starting single tuple insertion");

    let db = catalog.databases.get(db_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Database '{}' not found", db_name),
        )
    })?;

    let table = db.tables.get(table_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Table '{}' not found", table_name),
        )
    })?;

    let columns = &table.columns;

    if values.len() != columns.len() {
        log::info!(" Expected {} values, got {}", columns.len(), values.len());
        return Ok(false);
    }

    // Validate all values
    for (val, col) in values.iter().zip(columns.iter()) {
        let data_type = col.data_type.clone();
        let trimmed = val.trim();
        if col.nullable && (trimmed.eq_ignore_ascii_case("null") || trimmed.is_empty()) {
            continue;
        }

        if let Err(e) = validate_value(&data_type, val) {
            log::info!("Column '{}': {}", col.name, e);
            return Ok(false);
        }
    }

    // Constraint validation: NOT NULL, UNIQUE, FK, CHECK
    if let Err(e) = crate::backend::constraint::validate_row_insert(
        catalog, db_name, table_name, values,
    ) {
        log::info!("Constraint violation: {}", e);
        return Ok(false);
    }

    // Serialize tuple
    let mut row_ok = true;

    for (val, col) in values.iter().zip(columns.iter()) {
        let data_type = &col.data_type;
        let trimmed = val.trim();
        if col.nullable && (trimmed.eq_ignore_ascii_case("null") || trimmed.is_empty()) {
            continue;
        }

        match DataValue::parse_and_encode(data_type, val) {
            Ok(_) => {}
            Err(e) => {
                log::info!(" Failed to serialize column '{}': {}", col.name, e);
                row_ok = false;
                break;
            }
        }
    }

    if !row_ok {
        return Ok(false);
    }

    // Build datatype list
    let data_types: Vec<_> = columns
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // Build nullable value list
    let nullable_values: Vec<Option<&str>> =
        values.iter().zip(columns.iter()).map(|(v, col)| {
            let trimmed = v.trim();
            if col.nullable && (trimmed.eq_ignore_ascii_case("null") || trimmed.is_empty()) {
                None
            } else {
                Some(*v)
            }
        }).collect();

    // Serialize using tuple layout serializer
    let tuple_bytes = match serialize_nullable_row(&data_types, &nullable_values) {
        Ok(bytes) => bytes,
        Err(e) => {
            log::info!(" Failed to serialize tuple: {}", e);
            return Ok(false);
        }
    };

    // Open HeapManager and insert tuple — via the process-cached manager
    // (no per-row file open / pool flush churn).
    let table_path = PathBuf::from(format!("database/base/{}/{}.dat", db_name, table_name));

    let inserted =
        crate::backend::cache::with_heap(&table_path, |hm| hm.insert_tuple(&tuple_bytes));

    match inserted {
        Ok((page_id, slot_id)) => {
            log::info!(
                " Successfully inserted at (page={}, slot={})",
                page_id,
                slot_id
            );

            // Update any existing B+ Tree index (cached handles + batched sync)
            if let Err(e) = crate::backend::executor::create_index::update_index_on_insert(
                db_name, table_name, values, page_id, slot_id,
            ) {
                log::warn!(" Failed to update index: {}", e);
            }

            Ok(true)
        }
        Err(e) => {
            log::info!(" Failed to insert tuple: {}", e);
            Ok(false)
        }
    }
}