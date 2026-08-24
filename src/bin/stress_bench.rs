//! Stress benchmark harness for RookDB.
//!
//! Build & run (from repo root):
//!   cargo build --release --bin stress_bench
//!   ./target/release/stress_bench raw    100000
//!   ./target/release/stress_bench raw    1000000
//!   ./target/release/stress_bench raw    10000000
//!   ./target/release/stress_bench engine 100000        # plain, no index
//!   ./target/release/stress_bench engine 1000000 idx   # unique index on id
//!   ./target/release/stress_bench sql    100000        # full SQL text pipeline
//!   ./target/release/stress_bench join   100000        # hash join a x b
//!   ./target/release/stress_bench mutate 1000000      # update/delete/vacuum
//!
//! Emits machine-parseable lines starting with `[RESULT]`.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use storage_manager::catalog::{
    create_database, create_table, init_catalog, load_catalog, Catalog, Column,
};
use storage_manager::heap::HeapManager;
use storage_manager::types::DataType;

// ── deterministic RNG (LCG) ──────────────────────────────────────────────────
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn result(name: &str, n: u64, ms: u128, extra: &str) {
    let rps = if ms > 0 { n as u128 * 1000 / ms } else { 0 };
    println!(
        "[RESULT] phase={} n={} elapsed_ms={} rows_per_sec={} {}",
        name, n, ms, rps, extra
    );
    let _ = std::io::stdout().flush();
}

// ── isolated scratch workspace ───────────────────────────────────────────────
struct Ws(PathBuf);
impl Ws {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rook_bench_p{}_{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("base")).expect("ws create");
        std::env::set_current_dir(&dir).expect("chdir");
        init_catalog();
        Ws(dir)
    }
}
impl Drop for Ws {
    fn drop(&mut self) {
        if std::env::set_current_dir(self.0.parent().unwrap()).is_ok() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn col(name: &str, ty: DataType) -> Column {
    Column {
        name: name.to_string(),
        data_type: ty,
        nullable: true,
        constraints: Default::default(),
    }
}

fn make_db_with_staff(db: &str, with_index: bool) -> Catalog {
    let mut c = load_catalog();
    create_database(&mut c, db);
    let mut c = load_catalog();
    create_table(
        &mut c,
        db,
        "staff",
        vec![
            col("id", DataType::Int),
            col("name", DataType::Varchar(24)),
            col("salary", DataType::Int),
        ],
    );
    let _ = storage_manager::catalog::save_catalog(&c);
    if with_index {
        let c = load_catalog();
        storage_manager::executor::create_index(
            &c, db, "staff", "idx_id", &["id".to_string()],
        )
        .expect("create index");
    }
    load_catalog()
}

fn staff_row(i: u64) -> Vec<String> {
    vec![
        i.to_string(),
        format!("user_{:06}", (i % 1_000_000) as usize),
        (i % 100_000).to_string(),
    ]
}

// ── RAW storage tier (no typing / validation / planner) ──────────────────────
const RAW_LEN: usize = 32;

fn raw_pack(i: u64) -> Vec<u8> {
    let mut b = vec![0u8; RAW_LEN];
    b[0..4].copy_from_slice(&(i as u32).to_le_bytes());
    let name = format!("u{:0>10}", (i % 1_000_000_000) as usize); // "u" + 10 = 11 B
    b[4..15].copy_from_slice(name.as_bytes());
    b[16..24].copy_from_slice(&i.to_le_bytes());
    b[24..32].copy_from_slice(&i.wrapping_mul(0x9E37_79B9).to_le_bytes());
    b
}

fn bench_raw(n: u64, tag: &str) {
    let _ws = Ws::new(tag);
    let path = PathBuf::from("database/base/raw.dat");
    std::fs::create_dir_all("database/base").ok();
    let _ = std::fs::remove_file(&path);
    let mut hm = HeapManager::create(path.clone()).expect("raw create");

    let t = Instant::now();
    for i in 0..n {
        if let Err(e) = hm.insert_tuple(&raw_pack(i)) {
            eprintln!("[ERROR] raw insert {}: {}", i, e);
            return;
        }
        if i % 2_000_000 == 0 && i > 0 {
            eprintln!("  ...raw insert {}/{}", i, n);
        }
    }
    hm.flush().unwrap();
    let ms = t.elapsed().as_millis();
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    result("raw_heap_insert", n, ms, &format!("file_bytes={}", size));

    // sequential scan (live rows only, streaming)
    // (HeapManager::scan checkpoints executor caches automatically.)
    let t = Instant::now();
    let mut cnt = 0u64;
    for r in hm.scan() {
        if r.is_ok() {
            cnt += 1;
        }
    }
    result("raw_seq_scan", cnt, t.elapsed().as_millis(), "");

    // point gets: 10k random samples via stored locations
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let samples = 10_000u64.min(n);
    // rebuild a location index by scanning once (id order == slot order here)
    let mut locs: Vec<(u32, u32)> = Vec::with_capacity(n as usize);
    for r in hm.scan() {
        if let Ok((p, s, _)) = r {
            locs.push((p, s));
        }
    }
    let t = Instant::now();
    for _ in 0..samples {
        let (p, s) = locs[rng.below(n) as usize];
        if hm.get_tuple(p, s).is_err() {
            eprintln!("[ERROR] raw point get failed");
            return;
        }
    }
    result("raw_point_get", samples, t.elapsed().as_millis(), "queries");
}

// ── ENGINE tier (typed validation + serialization, optional index upkeep) ────
fn bench_engine(n: u64, tag: &str, with_index: bool) {
    let ws_tag = format!("{}_idx{}", tag, with_index as u8);
    let _ws = Ws::new(&ws_tag);
    let db = "bench";
    let catalog = make_db_with_staff(db, with_index);

    let t = Instant::now();
    for i in 0..n {
        let vals = staff_row(i);
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        match storage_manager::insert_single_tuple(&catalog, db, "staff", &refs) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("[ERROR] typed insert {} rejected", i);
                return;
            }
            Err(e) => {
                eprintln!("[ERROR] typed insert {}: {}", i, e);
                return;
            }
        }
        if i % 500_000 == 0 && i > 0 {
            eprintln!("  ...typed insert {}/{} (indexed={})", i, n, with_index);
        }
    }
    result(
        if with_index { "typed_insert_indexed" } else { "typed_insert_plain" },
        n,
        t.elapsed().as_millis(),
        "",
    );

    // Durability checkpoint before the read phases.
    storage_manager::backend::cache::checkpoint();

    // COUNT(*) through planner + Volcano aggregate
    let t = Instant::now();
    match count_via_sql(&catalog, db, "SELECT COUNT(*) FROM staff") {
        Ok(c) => result("volcano_count_star", c, t.elapsed().as_millis(), ""),
        Err(e) => eprintln!("[ERROR] count: {}", e),
    }

    // indexed point SELECTs through full SQL text (parse+plan+IndexScan)
    if with_index {
        let mut rng = Lcg(0xDEAD_BEEF);
        let samples = 1_000u64.min(n);
        let t = Instant::now();
        for _ in 0..samples {
            let id = rng.below(n);
            let sql = format!("SELECT id, name FROM staff WHERE id = {}", id);
            if let Err(e) = count_via_sql(&catalog, db, &sql) {
                eprintln!("[ERROR] point select: {}", e);
                return;
            }
        }
        result("sql_point_select", samples, t.elapsed().as_millis(), "queries");
    }
}

fn count_via_sql(catalog: &Catalog, db: &str, sql: &str) -> Result<u64, String> {
    use storage_manager::types::DataValue;
    let plan = rook_parser::parse_sql(sql)?;
    let logical = storage_manager::planner::plan_query(&plan, catalog, db)
        .map_err(|e| e.to_string())?;
    let tuples =
        storage_manager::backend::executor::physical::engine::execute_plan_collect(
            &logical, catalog, db,
        )
        .map_err(|e| e.to_string())?;
    Ok(match tuples.first().and_then(|t| t.values.first()) {
        Some(Some(DataValue::Int(v))) => *v as u64,
        Some(Some(DataValue::BigInt(v))) => *v as u64,
        Some(Some(other)) => return Err(format!("unexpected count type {:?}", other)),
        _ => 0,
    })
}

// ── FULL SQL TEXT pipeline inserts (parser included) ─────────────────────────
fn bench_pipeline(n: u64, tag: &str) {
    let _ws = Ws::new(tag);
    let db = "bench";
    let catalog = make_db_with_staff(db, false);

    let t = Instant::now();
    for i in 0..n {
        let r = staff_row(i);
        let sql = format!("INSERT INTO staff VALUES ({}, '{}', {})", r[0], r[1], r[2]);
        match route_insert(&catalog, db, &sql) {
            Ok(1) => {}
            other => {
                eprintln!("[ERROR] pipeline insert {} -> {:?}", i, other);
                return;
            }
        }
        if i % 50_000 == 0 && i > 0 {
            eprintln!("  ...pipeline insert {}/{}", i, n);
        }
    }
    result("pipeline_insert_sqltext", n, t.elapsed().as_millis(), "");
}

fn route_insert(catalog: &Catalog, db: &str, sql: &str) -> Result<usize, String> {
    use rook_ast::QueryPlan;
    let plan = rook_parser::parse_sql(sql)?;
    match plan {
        QueryPlan::Insert(ref ip) if ip.source_select.is_none() => {
            let logical = storage_manager::planner::plan_query(&plan, catalog, db)
                .map_err(|e| e.to_string())?;
            let tuples =
                storage_manager::backend::executor::physical::engine::execute_plan_collect(
                    &logical, catalog, db,
                )
                .map_err(|e| e.to_string())?;
            Ok(tuples.len())
        }
        _ => Err("not a simple VALUES insert".into()),
    }
}

// ── HASH JOIN ────────────────────────────────────────────────────────────────
fn bench_join(m: u64, tag: &str) {
    let _ws = Ws::new(tag);
    let db = "bench";
    make_db_with_staff(db, false);
    // Mutate ONE catalog handle: create_table persists internally, and the
    // same in-memory catalog is reused for every insert below (a per-row
    // load_catalog() used to re-read + re-parse the catalog each time).
    let mut catalog = load_catalog();
    create_table(
        &mut catalog,
        db,
        "orders",
        vec![
            col("id", DataType::Int),
            col("staff_id", DataType::Int),
            col("amount", DataType::Int),
        ],
    );

    // populate staff (m rows) and orders (2 rows per staff member)
    let t = Instant::now();
    for i in 0..m {
        let vals = vec![i.to_string(), format!("user_{:06}", (i % 1_000_000) as usize), (i % 100_000).to_string()];
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        if !storage_manager::insert_single_tuple(&catalog, db, "staff", &refs).unwrap_or(false) {
            eprintln!("[ERROR] staff insert {} rejected", i);
            return;
        }
    }
    for i in 0..(m * 2) {
        let vals = vec![i.to_string(), (i % m).to_string(), (i % 500).to_string()];
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        if !storage_manager::insert_single_tuple(&catalog, db, "orders", &refs)
            .unwrap_or(false)
        {
            eprintln!("[ERROR] orders insert {} rejected", i);
            return;
        }
    }
    result("join_setup_inserts", m * 3, t.elapsed().as_millis(), "");

    // Make everything durable before the read query.
    storage_manager::backend::cache::checkpoint();

    let sql = "SELECT COUNT(*) FROM staff JOIN orders ON staff.id = orders.staff_id";
    match count_via_sql(&catalog, db, sql) {
        Ok(c) => println!(
            "[RESULT] phase=hash_join_count matched={} side_a={} side_b={}",
            c, m, m * 2
        ),
        Err(e) => eprintln!("[ERROR] hash join: {}", e),
    }
}

// ── MUTATION tier: range UPDATE, range DELETE, then VACUUM ───────────────────
fn bench_mutate(n: u64, tag: &str) {
    let _ws = Ws::new(tag);
    let db = "bench";
    let catalog = make_db_with_staff(db, true); // indexed table

    // populate n rows via the typed path
    for i in 0..n {
        let vals = staff_row(i);
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        if !storage_manager::insert_single_tuple(&catalog, db, "staff", &refs)
            .unwrap_or(false)
        {
            eprintln!("[ERROR] setup insert {} rejected", i);
            return;
        }
        if i % 500_000 == 0 && i > 0 {
            eprintln!("  ...mutate setup {}/{}", i, n);
        }
    }

    // Durability checkpoint before mutation phases.
    storage_manager::backend::cache::checkpoint();

    use storage_manager::backend::executor::row_select::{
        parse_where_text, select_matching_pointers,
    };
    use storage_manager::executor::{delete_by_pointers, parse_set_clause, update_by_pointers};

    // range UPDATE over half the keyspace
    eprintln!("  [mutate] setup done, parsing where...");
    let sel = parse_where_text(&format!("id >= 0 AND id < {}", n / 2)).unwrap();
    eprintln!("  [mutate] selecting pointers...");
    let t = Instant::now();
    let ptrs = select_matching_pointers(&catalog, db, "staff", sel).expect("upd select");
    eprintln!("  [mutate] {} pointers in {} ms", ptrs.len(), t.elapsed().as_millis());
    let assignments = parse_set_clause("salary = salary + 1").expect("set clause");
    let tu = Instant::now();
    let upd = update_by_pointers(
        &catalog,
        db,
        "staff",
        &ptrs,
        &assignments,
    )
    .expect("range update");
    result("update_range_pointers", upd.updated_count as u64, t.elapsed().as_millis(), "");
    eprintln!("  [mutate] update took {} ms total", tu.elapsed().as_millis());

    // range DELETE of the same span, then VACUUM
    eprintln!("  [mutate] starting delete selection...");
    let sel = parse_where_text(&format!("id >= 0 AND id < {}", n / 2)).unwrap();
    let t = Instant::now();
    let ptrs = select_matching_pointers(&catalog, db, "staff", sel).expect("del select");
    eprintln!("  [mutate] delete pointers: {} in {} ms", ptrs.len(), t.elapsed().as_millis());
    let del = delete_by_pointers(&catalog, db, "staff", &ptrs).expect("range delete");
    result("delete_range_pointers", del.deleted_count as u64, t.elapsed().as_millis(), "");

    let catalog = load_catalog();
    let t = Instant::now();
    match storage_manager::backend::executor::vacuum::vacuum_table(&catalog, db, "staff") {
        Ok(st) => {
            let ms = t.elapsed().as_millis();
            println!(
                "[RESULT] phase=vacuum pages_compacted={} dead_reclaimed={} indexes_rebuilt={} elapsed_ms={}",
                st.pages_compacted, st.dead_tuples_before, st.indexes_rebuilt, ms
            );
            let _ = std::io::stdout().flush();
        }
        Err(e) => eprintln!("[ERROR] vacuum: {}", e),
    }
}

// ── main dispatch ────────────────────────────────────────────────────────────
fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: stress_bench <raw|engine|sql|join|mutate> <n> [idx]\n\
             engine phases accept optional third arg 'idx' to maintain an index"
        );
        std::process::exit(2);
    }
    let phase = args[1].clone();
    let n: u64 = args[2].parse().expect("n must be a number");

    match phase.as_str() {
        "raw" => bench_raw(n, &format!("raw_{}", n)),
        "engine" => {
            let idx = args.get(3).map(|s| s == "idx").unwrap_or(false);
            bench_engine(n, &format!("eng_{}_{}", n, idx), idx);
        }
        "sql" => bench_pipeline(n, &format!("sql_{}", n)),
        "join" => bench_join(n, &format!("join_{}", n)),
        "mutate" => bench_mutate(n, &format!("mut_{}", n)),
        other => {
            eprintln!("unknown phase: {}", other);
            std::process::exit(2);
        }
    }
}
