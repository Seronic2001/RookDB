use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use storage_manager::backend::disk::read_header_page;
use storage_manager::catalog::Column;
use storage_manager::catalog::load_catalog;
use storage_manager::executor::parse_set_clause;
use storage_manager::executor::{compaction_table, insert_single_tuple, load_csv};
use storage_manager::types::deserialize_nullable_row;

/// Gracefully load CSV file with comprehensive validation and error handling
pub fn load_csv_cmd(current_db: &Option<String>) -> io::Result<()> {
    log::info!("Starting CSV load operation");

    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim().to_string();

    if table.is_empty() {
        println!("Table name cannot be empty");
        return Ok(());
    }

    let mut csv_path = String::new();
    print!("Enter CSV path: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut csv_path)?;
    let csv_path = csv_path.trim();

    // --- 1. VALIDATE CSV PATH FIRST ---
    log::info!("Verifying CSV path: '{}'", csv_path);

    if csv_path.is_empty() {
        println!("CSV path cannot be empty");
        return Ok(());
    }

    let csv_file_path = Path::new(csv_path);
    if !csv_file_path.exists() {
        println!("CSV file not found at: '{}'", csv_path);
        println!("Please check the file path and try again.");
        println!("Make sure the file exists and the path is correct.");
        return Ok(());
    }

    if !csv_file_path.is_file() {
        println!("Path is not a file: '{}'", csv_path);
        println!("Please provide a path to a file, not a directory.");
        return Ok(());
    }

    log::info!("CSV file verified successfully: '{}'", csv_path);

    // Load catalog and insert data using the improved load_csv function
    let catalog = load_catalog();

    log::info!("Starting data insertion...\n");

    // Use the improved load_csv function with validation (HeapManager handles FSM)
    match load_csv(&catalog, &db, &table, csv_path) {
        Ok(inserted_count) => {
            if inserted_count == 0 {
                log::warn!("No data was inserted from the CSV file.");
                println!("   Please check:");
                println!("   1. CSV file is not empty (excluding header)");
                println!("   2. Data types match the table schema");
                println!("   3. Each row has the correct number of columns");
            } else {
                log::info!("Successfully inserted {} rows from CSV", inserted_count);
                println!("\n FSM fork file has been created/updated");
            }
        }
        Err(e) => {
            log::error!("Error during CSV loading: {}", e);
            println!("\nThis usually means:");
            if e.kind() == io::ErrorKind::NotFound {
                println!("  - The CSV file path is incorrect");
                println!("  - The file no longer exists");
            } else if e.kind() == io::ErrorKind::PermissionDenied {
                println!("  - Permission denied accessing the file");
                println!("  - Try running with appropriate permissions");
            } else if e.kind() == io::ErrorKind::InvalidData {
                println!("  - Data validation failed");
                println!("  - Check your CSV format and data types");
            } else {
                println!("  - An I/O error occurred: {}", e);
            }
            println!("\nPlease fix the issue and try again.");
        }
    }

    Ok(())
}

/// Insert a single tuple manually
pub fn insert_tuple_cmd(current_db: &Option<String>) -> io::Result<()> {
    log::info!("Starting single tuple insertion");

    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim();

    if table.is_empty() {
        println!("Table name cannot be empty");
        return Ok(());
    }

    // Load catalog to get schema
    let catalog = load_catalog();

    let db_obj = match catalog.databases.get(&db) {
        Some(d) => d,
        None => {
            println!("Database '{}' not found", db);
            return Ok(());
        }
    };

    let table_schema = match db_obj.tables.get(table) {
        Some(t) => t,
        None => {
            println!("Table '{}' not found in database '{}'", table, db);
            return Ok(());
        }
    };

    // Display schema
    log::trace!("Table schema:");
    for (idx, col) in table_schema.columns.iter().enumerate() {
        println!("  {}: {} (type: {})", idx + 1, col.name, col.data_type);
    }

    // Collect values
    println!("Enter values for each column:");
    let mut values = Vec::new();

    for col in &table_schema.columns {
        print!("  {} [{}]: ", col.name, col.data_type);
        io::stdout().flush()?;

        let mut value = String::new();
        io::stdin().read_line(&mut value)?;
        values.push(value.trim().to_string());
    }

    // Convert to string references
    let value_refs: Vec<&str> = values.iter().map(|v| v.as_str()).collect();

    // Insert tuple using HSM-aware insert (with FSM)
    log::info!("Inserting tuple...");
    match insert_single_tuple(&catalog, &db, table, &value_refs) {
        Ok(success) => {
            if success {
                log::info!("Tuple inserted successfully!");
                log::info!("FSM fork file updated");
            } else {
                log::error!("Failed to insert tuple. Please check your data types and values.");
            }
        }
        Err(e) => {
            log::error!("Error inserting tuple: {}", e);
        }
    }

    Ok(())
}

pub fn show_tuples_cmd(current_db: &Option<String>) -> io::Result<()> {
    log::debug!("Starting tuple display");

    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim();

    if table.is_empty() {
        println!("Table name cannot be empty");
        return Ok(());
    }

    // Optional WHERE clause — empty means show every row.
    println!();
    println!("Optional WHERE clause (=, !=, <, <=, >, >=, AND/OR, parentheses).");
    println!("Leave empty to show all rows.");
    print!("WHERE clause: ");
    io::stdout().flush()?;
    let mut where_input = String::new();
    io::stdin().read_line(&mut where_input)?;

    // Selection runs on the Volcano engine via a wildcard scan.
    let catalog = load_catalog();
    let selection =
        storage_manager::backend::executor::row_select::parse_where_text(&where_input)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let pointers = storage_manager::backend::executor::row_select::select_matching_pointers(
        &catalog, &db, table, selection,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // Fetch and display each matching row through the heap manager.
    let heap_path = format!("database/base/{}/{}.dat", db, table);
    let mut heap = storage_manager::heap::HeapManager::open(std::path::PathBuf::from(
        &heap_path,
    ))
    .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;

    let columns: Vec<Column> = catalog
        .databases
        .get(&db)
        .and_then(|d| d.tables.get(table))
        .map(|t| t.columns.clone())
        .unwrap_or_default();
    let schema_types: Vec<storage_manager::types::DataType> =
        columns.iter().map(|c| c.data_type.clone()).collect();

    println!("\n=== Matching tuples in '{}.{}' ===", db, table);
    for (i, (page_id, slot_id)) in pointers.iter().enumerate() {
        print!("Tuple {}: ", i + 1);
        match heap.get_tuple(*page_id, *slot_id) {
            Ok(raw) => match deserialize_nullable_row(&schema_types, &raw) {
                Ok(values) => {
                    for (col, val_opt) in columns.iter().zip(values.iter()) {
                        match val_opt {
                            Some(val) => print!("{}={} ", col.name, val),
                            None => print!("{}=NULL ", col.name),
                        }
                    }
                }
                Err(e) => print!("<decode-error: {}> ", e),
            },
            Err(e) => print!("<read-error at ({},{}): {}> ", page_id, slot_id, e),
        }
        println!();
    }
    println!("\n{} row(s) matched.\n", pointers.len());
    Ok(())
}

/// Interactive DELETE command.
///
/// Accepts a single WHERE clause string with full AND / OR / parentheses support.
///
/// Examples:
///   (leave empty)                                       → DELETE ALL rows
///   price > 10                                          → simple condition
///   dept = HR AND salary < 50000                        → AND
///   dept = HR OR dept = Sales                           → OR
///   (dept = HR AND salary < 50000) OR dept = Sales      → mixed
///   (c1 = 1 AND c2 = 2) AND (c3 = 3 OR c4 = 4)        → nested (auto-expanded to DNF)
pub fn delete_tuples_cmd(current_db: &Option<String>) -> io::Result<()> {
    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first.");
            return Ok(());
        }
    };

    // -- table name --
    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim().to_string();

    let catalog = load_catalog();
    if !catalog
        .databases
        .get(&db)
        .map(|d| d.tables.contains_key(table.as_str()))
        .unwrap_or(false)
    {
        println!("Table '{}' does not exist in '{}'.", table, db);
        return Ok(());
    }

    // -- WHERE clause (parsed with the real SQL grammar) --
    println!();
    println!("Supported operators : =  !=  <  <=  >  >=  IN  BETWEEN  LIKE");
    println!("Logical connectors  : AND  OR  NOT (parentheses supported)");
    println!("Leave empty         : delete ALL rows");
    println!();
    print!("WHERE clause: ");
    io::stdout().flush()?;
    let mut where_input = String::new();
    io::stdin().read_line(&mut where_input)?;

    let selection =
        storage_manager::backend::executor::row_select::parse_where_text(&where_input)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    if selection.is_none() {
        println!(
            "No WHERE clause \u{2013} this will delete ALL rows in '{}'.",
            table
        );
        print!("Are you sure? (yes/no): ");
        io::stdout().flush()?;
        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        if !confirm.trim().eq_ignore_ascii_case("yes") {
            println!("Aborted.");
            return Ok(());
        }
    }

    // -- RETURNING --
    print!("\nPrint deleted rows? (y/n): ");
    io::stdout().flush()?;
    let mut ret_input = String::new();
    io::stdin().read_line(&mut ret_input)?;
    let returning = ret_input.trim().eq_ignore_ascii_case("y");

    // -- select matching rows on the Volcano engine, then mutate --
    let pointers = storage_manager::backend::executor::row_select::select_matching_pointers(
        &catalog, &db, &table, selection,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    if returning && !pointers.is_empty() {
        // Snapshot the rows before deletion so they can be printed.
        let mut heap = storage_manager::heap::HeapManager::open(std::path::PathBuf::from(
            format!("database/base/{}/{}.dat", db, table),
        ))
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;
        let columns: Vec<Column> = catalog.databases[&db].tables[&table].columns.clone();
        let schema_types: Vec<storage_manager::types::DataType> =
            columns.iter().map(|c| c.data_type.clone()).collect();
        println!("\n=== Deleted rows ===");
        for &(page_id, slot_id) in &pointers {
            if let Ok(raw) = heap.get_tuple(page_id, slot_id) {
                if let Ok(values) =
                    storage_manager::types::deserialize_nullable_row(&schema_types, &raw)
                {
                    let cells: Vec<String> = columns
                        .iter()
                        .zip(values.iter())
                        .map(|(c, v)| match v {
                            Some(val) => format!("{}={}", c.name, val),
                            None => format!("{}=NULL", c.name),
                        })
                        .collect();
                    println!("  {}", cells.join("  |  "));
                }
            }
        }
        println!("===================");
    }

    match storage_manager::executor::delete_by_pointers(&catalog, &db, &table, &pointers) {
        Ok(result) => {
            println!("\nDeleted {} row(s).", result.deleted_count);
        }
        Err(e) => println!("Delete failed: {}", e),
    }

    Ok(())
}

/// Interactive UPDATE command.
///
/// Prompts for SET assignments and an optional WHERE clause.
///
/// Examples:
///   SET  : age = 25
///   SET  : age = age + 1
///   SET  : salary = salary * 1.10 , dept = Engineering
///   WHERE: id > 5 AND dept = HR
pub fn update_tuples_cmd(current_db: &Option<String>) -> io::Result<()> {
    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first.");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim().to_string();

    let catalog = load_catalog();
    if !catalog
        .databases
        .get(&db)
        .map(|d| d.tables.contains_key(table.as_str()))
        .unwrap_or(false)
    {
        println!("Table '{}' does not exist in '{}'.", table, db);
        return Ok(());
    }

    // -- SET clause --
    println!();
    println!("SET clause examples:");
    println!("  age = 25");
    println!("  age = age + 1");
    println!("  salary = salary * 1.10 , dept = Engineering");
    println!();
    print!("SET: ");
    io::stdout().flush()?;
    let mut set_input = String::new();
    io::stdin().read_line(&mut set_input)?;
    let set_input = set_input.trim();

    let assignments = match parse_set_clause(set_input) {
        Some(a) if !a.is_empty() => a,
        _ => {
            println!("Could not parse SET clause. Aborted.");
            return Ok(());
        }
    };

    // -- WHERE clause --
    println!();
    println!("WHERE clause (leave empty to update ALL rows):");
    print!("WHERE: ");
    io::stdout().flush()?;
    let mut where_input = String::new();
    io::stdin().read_line(&mut where_input)?;

    let selection =
        storage_manager::backend::executor::row_select::parse_where_text(&where_input)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

    if selection.is_none() {
        println!(
            "No WHERE clause \u{2013} this will update ALL rows in '{}'.",
            table
        );
        print!("Are you sure? (yes/no): ");
        io::stdout().flush()?;
        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        if !confirm.trim().eq_ignore_ascii_case("yes") {
            println!("Aborted.");
            return Ok(());
        }
    }

    // -- select matching rows on the Volcano engine, then mutate --
    let pointers = storage_manager::backend::executor::row_select::select_matching_pointers(
        &catalog, &db, &table, selection,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // -- RETURNING --
    print!("\nPrint updated rows? (y/n): ");
    io::stdout().flush()?;
    let mut ret_input = String::new();
    io::stdin().read_line(&mut ret_input)?;
    let returning = ret_input.trim().eq_ignore_ascii_case("y");

    match storage_manager::executor::update_by_pointers(
        &catalog,
        &db,
        &table,
        &pointers,
        &assignments,
    ) {
        Ok(result) => {
            println!("\nUpdated {} row(s).", result.updated_count);

            if returning && !pointers.is_empty() {
                // Show post-update contents of exactly the touched rows.
                let mut heap =
                    storage_manager::heap::HeapManager::open(std::path::PathBuf::from(format!(
                        "database/base/{}/{}.dat",
                        db, table
                    )))
                    .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e))?;
                let columns: Vec<Column> =
                    catalog.databases[&db].tables[&table].columns.clone();
                let schema_types: Vec<storage_manager::types::DataType> =
                    columns.iter().map(|c| c.data_type.clone()).collect();
                println!("\n=== Updated rows (after) ===");
                for &(page_id, slot_id) in &pointers {
                    if let Ok(raw) = heap.get_tuple(page_id, slot_id) {
                        if let Ok(values) = storage_manager::types::deserialize_nullable_row(
                            &schema_types,
                            &raw,
                        ) {
                            let cells: Vec<String> = columns
                                .iter()
                                .zip(values.iter())
                                .map(|(c, v)| match v {
                                    Some(val) => format!("{}={}", c.name, val),
                                    None => format!("{}=NULL", c.name),
                                })
                                .collect();
                            println!("  {}", cells.join("  |  "));
                        }
                    }
                }
                println!("========================");
            }
        }
        Err(e) => println!("Update failed: {}", e),
    }

    Ok(())
}

pub fn compact_table_cmd(current_db: &Option<String>) -> io::Result<()> {
    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first.");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim().to_string();

    let path = format!("database/base/{}/{}.dat", db, table);
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(f) => f,
        Err(e) => {
            println!("Could not open table '{}': {}", table, e);
            return Ok(());
        }
    };

    // file was opened just to validate the table exists; compaction_table opens it internally
    drop(file);
    let pages_compacted = compaction_table(&db, &table)?;
    println!(
        "\nCompaction complete. {} page(s) had dead tuples removed.",
        pages_compacted
    );

    Ok(())
}

/// Check heap health and display FSM statistics for a table.
pub fn check_heap_cmd(current_db: &Option<String>) -> io::Result<()> {
    let db = match current_db {
        Some(db) => db.clone(),
        None => {
            println!("No database selected. Please select a database first");
            return Ok(());
        }
    };

    let mut table = String::new();
    print!("Enter table name: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut table)?;
    let table = table.trim();

    let heap_path = PathBuf::from(format!("database/base/{}/{}.dat", db, table));

    if !heap_path.exists() {
        log::warn!("Heap file not found: {:?}", heap_path);
        println!("Table may not exist. Try creating the table first.");
        return Ok(());
    }

    println!("\n╔════════════════════════════════════════╗");
    println!("║         HEAP DIAGNOSTICS               ║");
    println!("╚════════════════════════════════════════╝");

    println!("\nHeap Info: {}.{}", db, table);
    // println!("════════════════════════════════════════");

    // Try to read header
    match OpenOptions::new().read(true).write(true).open(&heap_path) {
        Ok(mut file) => match read_header_page(&mut file) {
            Ok(header) => {
                println!("Total Heap Pages:  {}", header.page_count);
                println!("FSM Fork Pages:    {}", header.fsm_page_count);
                println!("Total Tuples:      {}", header.total_tuples);
                println!(
                    "Last Vacuum:       {}",
                    if header.last_vacuum == 0 {
                        "Never".to_string()
                    } else {
                        format!(
                            "{}s ago",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs()
                                - header.last_vacuum as u64
                        )
                    }
                );

                println!("\nHeap is healthy and accessible");
                storage_manager::backend::instrumentation::StatsSnapshot::capture().print_table();
            }
            Err(e) => {
                log::warn!("Could not read header: {}", e);
                println!("   Heap file may need FSM rebuild");
            }
        },
        Err(e) => {
            log::error!("Error opening heap file: {}", e);
        }
    }

    // Check FSM fork file
    let fsm_path = PathBuf::from(format!("{}.fsm", heap_path.to_string_lossy()));
    if fsm_path.exists() {
        match std::fs::metadata(&fsm_path) {
            Ok(meta) => {
                let fsm_pages = meta.len() / 8192;
                println!("\nFSM Fork File:");
                println!("  Path: {:?}", fsm_path);
                println!("  Size: {} bytes ({} pages)", meta.len(), fsm_pages);
            }
            Err(e) => {
                println!("\nFSM Fork file exists but cannot stat: {}", e);
            }
        }
    } else {
        println!("\n FSM Fork file not yet created (will be created on first insert)");
    }

    println!("════════════════════════════════════════\n");

    Ok(())
}
