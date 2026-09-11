//! Query Plan Cache & Prepared Statement Support.
//!
//! Avoids redundant AST parsing, semantic analysis, and physical plan generation
//! for repeated query forms (such as bulk INSERTs and repetitive SELECTs).

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

use rook_ast::logical::LogicalPlan;
use rook_ast::QueryPlan;
use crate::catalog::Catalog;
use crate::backend::executor::insert_single_tuple;

const DEFAULT_CACHE_CAPACITY: usize = 512;

/// Normalize a SQL statement by replacing literals (strings, numbers) with `$1, $2, ...`
/// and extracting parameter values in order.
pub fn normalize_sql(sql: &str) -> (String, Vec<String>) {
    let mut normalized = String::with_capacity(sql.len());
    let mut params = Vec::new();
    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut param_idx = 1;

    while i < n {
        let ch = chars[i];

        // Single-quoted string literal: '...'
        if ch == '\'' {
            i += 1;
            let mut s = String::new();
            while i < n {
                if chars[i] == '\'' {
                    if i + 1 < n && chars[i + 1] == '\'' {
                        s.push('\'');
                        i += 2;
                    } else {
                        i += 1;
                        break;
                    }
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            params.push(s);
            normalized.push('$');
            normalized.push_str(&param_idx.to_string());
            param_idx += 1;
            continue;
        }

        // Numeric literal:
        let is_digit = ch.is_ascii_digit();
        let is_negative_number = ch == '-'
            && i + 1 < n
            && chars[i + 1].is_ascii_digit()
            && (i == 0 || (!chars[i - 1].is_alphanumeric() && chars[i - 1] != '_'));

        let prev_is_ident = if i > 0 {
            chars[i - 1].is_alphanumeric() || chars[i - 1] == '_'
        } else {
            false
        };

        if (is_digit || is_negative_number) && !prev_is_ident {
            let mut num_str = String::new();
            if is_negative_number {
                num_str.push('-');
                i += 1;
            }
            while i < n && (chars[i].is_ascii_digit()
                || chars[i] == '.'
                || chars[i] == 'e'
                || chars[i] == 'E'
                || ((chars[i] == '+' || chars[i] == '-')
                    && i > 0
                    && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
            {
                num_str.push(chars[i]);
                i += 1;
            }
            params.push(num_str);
            normalized.push('$');
            normalized.push_str(&param_idx.to_string());
            param_idx += 1;
            continue;
        }

        // Collapse multiple whitespaces
        if ch.is_whitespace() {
            if !normalized.ends_with(' ') && !normalized.is_empty() {
                normalized.push(' ');
            }
            i += 1;
            continue;
        }

        // Strip trailing semicolon
        if ch == ';' && i == n - 1 {
            i += 1;
            continue;
        }

        normalized.push(ch);
        i += 1;
    }

    (normalized.trim().to_string(), params)
}

/// Cached fast-path template for an INSERT statement.
#[derive(Clone, Debug)]
pub struct CachedInsert {
    pub table: String,
    pub columns: Vec<String>,
    pub arity: usize,
}

impl CachedInsert {
    /// Execute the insert using the cached template and extracted parameter values.
    pub fn execute(&self, catalog: &Catalog, db_name: &str, values: &[&str]) -> Result<usize, String> {
        let db = catalog
            .databases
            .get(db_name)
            .ok_or_else(|| format!("Database '{}' not found", db_name))?;
        let table = db
            .tables
            .get(&self.table)
            .ok_or_else(|| format!("Table '{}' not found", self.table))?;

        let final_values: Vec<String>;
        let final_refs: Vec<&str>;

        if self.columns.is_empty() {
            if values.len() != table.columns.len() {
                return Err(format!(
                    "Expected {} values for table '{}', got {}",
                    table.columns.len(),
                    self.table,
                    values.len()
                ));
            }
            final_refs = values.to_vec();
        } else {
            final_values = vec![String::from("NULL"); table.columns.len()];
            let mut buf = final_values;
            for (i, col_name) in self.columns.iter().enumerate() {
                if let Some(pos) = table
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(col_name))
                {
                    if i < values.len() {
                        buf[pos] = values[i].to_string();
                    }
                }
            }
            final_refs = buf.iter().map(|s| s.as_str()).collect();
            return match insert_single_tuple(catalog, db_name, &self.table, &final_refs) {
                Ok(true) => Ok(1),
                Ok(false) => Err("Constraint violation or invalid data".to_string()),
                Err(e) => Err(format!("Insert failed: {}", e)),
            };
        }

        match insert_single_tuple(catalog, db_name, &self.table, &final_refs) {
            Ok(true) => Ok(1),
            Ok(false) => Err("Constraint violation or invalid data".to_string()),
            Err(e) => Err(format!("Insert failed: {}", e)),
        }
    }
}

/// A cached entry in the plan cache.
#[derive(Clone, Debug)]
pub enum PlanCacheEntry {
    Insert(CachedInsert),
    Plan(LogicalPlan),
}

/// Process-wide Plan Cache.
pub struct PlanCache {
    entries: HashMap<(String, String), PlanCacheEntry>,
    order: VecDeque<(String, String)>,
    capacity: usize,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn get(&self, db_name: &str, normalized_sql: &str) -> Option<PlanCacheEntry> {
        let key = (db_name.to_string(), normalized_sql.to_string());
        self.entries.get(&key).cloned()
    }

    pub fn insert(&mut self, db_name: &str, normalized_sql: &str, entry: PlanCacheEntry) {
        let key = (db_name.to_string(), normalized_sql.to_string());
        if self.entries.contains_key(&key) {
            self.entries.insert(key, entry);
            return;
        }

        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, entry);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

fn global_plan_cache() -> &'static Mutex<PlanCache> {
    static CACHE: OnceLock<Mutex<PlanCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(PlanCache::new(DEFAULT_CACHE_CAPACITY)))
}

/// Invalidate all entries in the plan cache (called when DDL runs).
pub fn invalidate_plan_cache() {
    let mut cache = global_plan_cache().lock().unwrap_or_else(|p| p.into_inner());
    cache.clear();
}

/// Look up an entry in the plan cache.
pub fn get_cached_plan(db_name: &str, normalized_sql: &str) -> Option<PlanCacheEntry> {
    let cache = global_plan_cache().lock().unwrap_or_else(|p| p.into_inner());
    cache.get(db_name, normalized_sql)
}

/// Store an entry in the plan cache.
pub fn store_cached_plan(db_name: &str, normalized_sql: &str, entry: PlanCacheEntry) {
    let mut cache = global_plan_cache().lock().unwrap_or_else(|p| p.into_inner());
    cache.insert(db_name, normalized_sql, entry);
}

/// Register a SQL parser function (e.g. from rook-parser).
type SqlParserFn = Box<dyn Fn(&str) -> Result<QueryPlan, String> + Send + Sync>;
static SQL_PARSER: OnceLock<SqlParserFn> = OnceLock::new();

pub fn register_sql_parser(parser: impl Fn(&str) -> Result<QueryPlan, String> + Send + Sync + 'static) {
    let _ = SQL_PARSER.set(Box::new(parser));
}

fn parse_sql(sql: &str) -> Result<QueryPlan, String> {
    if let Some(parser) = SQL_PARSER.get() {
        parser(sql)
    } else {
        Err("SQL parser not registered (call register_sql_parser first)".to_string())
    }
}

/// Parse a normalized INSERT statement template without external parser dependencies.
pub fn parse_insert_template(normalized_sql: &str, arity: usize) -> Option<CachedInsert> {
    let trimmed = normalized_sql.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("insert into ") {
        return None;
    }
    let rest = trimmed["insert into ".len()..].trim_start();
    let values_idx = rest.to_ascii_lowercase().find("values")?;
    let header = rest[..values_idx].trim();
    let (table, cols) = if let Some(paren_idx) = header.find('(') {
        let table = header[..paren_idx].trim().to_string();
        let close_idx = header.rfind(')')?;
        let col_str = &header[paren_idx + 1..close_idx];
        let cols: Vec<String> = col_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        (table, cols)
    } else {
        (header.to_string(), Vec::new())
    };

    if table.is_empty() {
        return None;
    }

    Some(CachedInsert {
        table,
        columns: cols,
        arity,
    })
}

/// Execute an insert SQL query through the plan cache if possible.
/// Returns `Ok(Some(count))` if routed through the plan cache, or `Ok(None)` if not an insert.
pub fn execute_cached_insert(
    catalog: &Catalog,
    db_name: &str,
    sql: &str,
) -> Result<Option<usize>, String> {
    let (normalized, params) = normalize_sql(sql);
    let key_matched = normalized.starts_with("INSERT INTO") || normalized.starts_with("insert into");

    if !key_matched {
        return Ok(None);
    }

    if let Some(entry) = get_cached_plan(db_name, &normalized) {
        if let PlanCacheEntry::Insert(cached_insert) = entry {
            let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
            return cached_insert
                .execute(catalog, db_name, &param_refs)
                .map(Some);
        }
    }

    // Cache miss: construct template
    if let Some(cached_insert) = parse_insert_template(&normalized, params.len()) {
        let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
        let res = cached_insert.execute(catalog, db_name, &param_refs)?;
        store_cached_plan(db_name, &normalized, PlanCacheEntry::Insert(cached_insert));
        return Ok(Some(res));
    }

    // If fast template failed (e.g. complex expression), fall back to SQL parser if registered
    if let Ok(plan) = parse_sql(sql) {
        if let QueryPlan::Insert(ref ip) = plan {
            if ip.source_select.is_none() {
                let cached_insert = CachedInsert {
                    table: ip.table.clone(),
                    columns: ip.columns.clone(),
                    arity: params.len(),
                };
                let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
                let res = cached_insert.execute(catalog, db_name, &param_refs)?;
                store_cached_plan(db_name, &normalized, PlanCacheEntry::Insert(cached_insert));
                return Ok(Some(res));
            }
        }
    }

    Ok(None)
}

/// Prepared Statement.
#[derive(Clone, Debug)]
pub struct PreparedStatement {
    pub sql_template: String,
    pub db_name: String,
    pub param_count: usize,
    pub entry: PlanCacheEntry,
}

impl PreparedStatement {
    /// Prepare a statement from a SQL string.
    pub fn prepare(catalog: &Catalog, db_name: &str, sql: &str) -> Result<Self, String> {
        let (normalized, default_params) = normalize_sql(sql);
        if let Some(entry) = get_cached_plan(db_name, &normalized) {
            return Ok(Self {
                sql_template: normalized,
                db_name: db_name.to_string(),
                param_count: default_params.len(),
                entry,
            });
        }

        if let Some(cached_insert) = parse_insert_template(&normalized, default_params.len()) {
            let entry = PlanCacheEntry::Insert(cached_insert);
            store_cached_plan(db_name, &normalized, entry.clone());
            return Ok(Self {
                sql_template: normalized,
                db_name: db_name.to_string(),
                param_count: default_params.len(),
                entry,
            });
        }

        let plan = parse_sql(sql)?;
        let entry = match plan {
            QueryPlan::Insert(ref ip) if ip.source_select.is_none() => {
                PlanCacheEntry::Insert(CachedInsert {
                    table: ip.table.clone(),
                    columns: ip.columns.clone(),
                    arity: default_params.len(),
                })
            }
            _ => {
                let logical = crate::backend::planner::plan_query(&plan, catalog, db_name)
                    .map_err(|e| e.to_string())?;
                PlanCacheEntry::Plan(logical)
            }
        };

        store_cached_plan(db_name, &normalized, entry.clone());
        Ok(Self {
            sql_template: normalized,
            db_name: db_name.to_string(),
            param_count: default_params.len(),
            entry,
        })
    }

    /// Execute the prepared statement with given parameters.
    pub fn execute(&self, catalog: &Catalog, params: &[&str]) -> Result<usize, String> {
        match &self.entry {
            PlanCacheEntry::Insert(ins) => ins.execute(catalog, &self.db_name, params),
            PlanCacheEntry::Plan(plan) => {
                let tuples = crate::backend::executor::physical::engine::execute_plan_collect(
                    plan,
                    catalog,
                    &self.db_name,
                )
                .map_err(|e| e.to_string())?;
                Ok(tuples.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_sql_insert() {
        let sql = "INSERT INTO staff VALUES (1, 'user_000001', 100);";
        let (norm, params) = normalize_sql(sql);
        assert_eq!(norm, "INSERT INTO staff VALUES ($1, $2, $3)");
        assert_eq!(params, vec!["1", "user_000001", "100"]);
    }

    #[test]
    fn test_normalize_sql_select() {
        let sql = "SELECT id, name FROM staff WHERE id = 42";
        let (norm, params) = normalize_sql(sql);
        assert_eq!(norm, "SELECT id, name FROM staff WHERE id = $1");
        assert_eq!(params, vec!["42"]);
    }

    #[test]
    fn test_normalize_sql_count_star() {
        let sql = "SELECT COUNT(*) FROM staff";
        let (norm, params) = normalize_sql(sql);
        assert_eq!(norm, "SELECT COUNT(*) FROM staff");
        assert!(params.is_empty());
    }

    #[test]
    fn test_normalize_sql_strings_with_quotes() {
        let sql = "INSERT INTO users VALUES ('O''Reilly', -10)";
        let (norm, params) = normalize_sql(sql);
        assert_eq!(norm, "INSERT INTO users VALUES ($1, $2)");
        assert_eq!(params, vec!["O'Reilly", "-10"]);
    }
}
