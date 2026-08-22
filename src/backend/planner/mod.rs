//! Logical Planner — Translates `rook_ast::QueryPlan` into a `LogicalPlan` tree.
//!
//! The planner performs two main passes:
//!
//! 1. **Semantic analysis**: resolve column names, expand `SELECT *`, validate
//!    table/column existence, apply type coercions.
//! 2. **Plan building**: construct the logical operator tree matching the query's
//!    intent (filter, project, sort, aggregate, join, etc.).

pub mod semantic;
pub mod optimizer;
pub mod helpers;
#[cfg(test)]
pub mod tests;

use rook_ast::logical::*;
use rook_ast::*;

use crate::catalog::Catalog;
use crate::statistics::collect_table_statistics;

use self::helpers::*;

/// Errors produced during planning / semantic analysis.
#[derive(Debug, Clone)]
pub struct PlanError {
    pub message: String,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PlanError {}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load table statistics for all tables in the given database.
///
/// Collects tuple counts and page usage from each table's heap file.
/// Tables whose heap files don't exist yet (e.g. freshly created with no
/// data) are silently skipped. Returns an empty map if no stats can be
/// collected.
fn load_table_statistics(db_name: &str, catalog: &Catalog) -> std::collections::HashMap<String, crate::statistics::TableStatistics> {
    let db = match catalog.databases.get(db_name) {
        Some(d) => d,
        None => return std::collections::HashMap::new(),
    };

    let mut stats = std::collections::HashMap::new();
    for table_name in db.tables.keys() {
        if let Ok(table_stats) = collect_table_statistics(db_name, table_name) {
            stats.insert(table_name.clone(), table_stats);
        }
    }
    stats
}

/// Plan an INSERT INTO ... SELECT into a LogicalPlan.
fn plan_insert(
    insert: &InsertPlan,
    catalog: &Catalog,
    db_name: &str,
) -> Result<LogicalPlan, PlanError> {
    let source_select = insert.source_select.as_ref().ok_or_else(|| PlanError {
        message: "plan_insert requires a source SELECT (INSERT INTO ... SELECT)".to_string(),
    })?;

    let child_plan = plan_select(source_select, catalog, db_name)?;

    Ok(LogicalPlan::Insert(LogicalInsert {
        table: insert.table.clone(),
        columns: insert.columns.clone(),
        child: Box::new(child_plan),
    }))
}

/// Plan a `QueryPlan` into a `LogicalPlan` using the given catalog.
pub fn plan_query(query: &QueryPlan, catalog: &Catalog, db_name: &str) -> Result<LogicalPlan, PlanError> {
    match query {
        QueryPlan::Select(select) => {
            let plan = plan_select(select, catalog, db_name)?;
            let table_stats = load_table_statistics(db_name, catalog);
            let optimizer = if table_stats.is_empty() {
                optimizer::Optimizer::new()
            } else {
                optimizer::Optimizer::with_statistics(table_stats)
            };
            Ok(optimizer.optimize(plan))
        }
        QueryPlan::SetOperation(set_op) => {
            let left = plan_select(&set_op.left, catalog, db_name)?;
            let right = plan_select(&set_op.right, catalog, db_name)?;
            let plan = LogicalPlan::SetOp(LogicalSetOp {
                left: Box::new(left),
                right: Box::new(right),
                op: match set_op.op.to_ascii_uppercase().as_str() {
                    "INTERSECT" => logical::SetOpType::Intersect,
                    "EXCEPT" => logical::SetOpType::Except,
                    _ => logical::SetOpType::Union,
                },
                all: set_op.all,
            });
            let table_stats = load_table_statistics(db_name, catalog);
            let optimizer = if table_stats.is_empty() {
                optimizer::Optimizer::new()
            } else {
                optimizer::Optimizer::with_statistics(table_stats)
            };
            Ok(optimizer.optimize(plan))
        }
        QueryPlan::Insert(ins) => {
            if ins.source_select.is_some() {
                plan_insert(ins, catalog, db_name)
            } else {
                Err(PlanError {
                    message: "INSERT ... VALUES must be handled by the CLI executor, not the logical planner".to_string(),
                })
            }
        }
        QueryPlan::CreateTable(_) | QueryPlan::DropTable(_) | QueryPlan::DropDatabase(_)
        | QueryPlan::AlterTable(_) | QueryPlan::CreateView(_) | QueryPlan::DropView(_)
        | QueryPlan::CreateIndex(_) | QueryPlan::CreateDatabase(_)
        | QueryPlan::CreateTableAsSelect(_) | QueryPlan::ShowTables | QueryPlan::ShowDatabases
        | QueryPlan::UseDatabase(_) => Err(PlanError {
            message: format!("DDL/DQL statement '{}' must be handled by the executor, not the logical planner", query.statement_type()),
        }),
        _ => Err(PlanError {
            message: format!("Planning not yet supported for {}", query.statement_type()),
        }),
    }
}

// ── SELECT planning ───────────────────────────────────────────────────────────

/// Plan a SELECT query into a LogicalPlan.
fn plan_select(
    select: &SelectPlan,
    catalog: &Catalog,
    db_name: &str,
) -> Result<LogicalPlan, PlanError> {
    plan_select_with_ctes(select, catalog, db_name, None)
}

/// Like `plan_select`, but accepts an optional pre-populated CTE registry.
fn plan_select_with_ctes(
    select: &SelectPlan,
    catalog: &Catalog,
    db_name: &str,
    existing_ctes: Option<&std::collections::HashMap<String, (LogicalPlan, ColumnSchema)>>,
) -> Result<LogicalPlan, PlanError> {
    // 0. Resolve CTEs and build a registry
    let db = catalog
        .databases
        .get(db_name)
        .ok_or_else(|| PlanError {
            message: format!("Database '{}' not found", db_name),
        })?;

    let mut cte_registry: std::collections::HashMap<String, (LogicalPlan, ColumnSchema)> =
        std::collections::HashMap::new();

    if let Some(existing) = existing_ctes {
        for (key, val) in existing {
            cte_registry.entry(key.clone()).or_insert_with(|| val.clone());
        }
    }

    for cte_def in &select.ctes {
        let inner_plan = plan_select(&cte_def.query, catalog, db_name)?;
        let schema = derive_schema(&inner_plan);
        cte_registry.insert(cte_def.name.to_ascii_lowercase(), (inner_plan, schema));
    }

    // 1. Resolve table references and build the base scan
    let mut current_plan = if select.from.is_empty() {
        LogicalPlan::TableScan(LogicalTableScan {
            table: "__singlerow__".to_string(),
            alias: None,
            schema: ColumnSchema::empty(),
            system_table_name: Some("__singlerow__".to_string()),
        })
    } else {
        let first_table = &select.from[0];
        resolve_from_item(first_table, db, catalog, db_name, &cte_registry)?
    };

    // 2. Handle JOINs (if any)
    for join in &select.joins {
        let right_plan = resolve_from_item(&join.relation, db, catalog, db_name, &cte_registry)?;
        current_plan = LogicalPlan::Join(LogicalJoin {
            left: Box::new(current_plan),
            right: Box::new(right_plan),
            join_type: join.join_type,
            condition: join.condition.clone(),
        });
    }

    // 3. Apply WHERE filter
    if let Some(ref pred) = select.selection {
        current_plan = LogicalPlan::Filter(LogicalFilter {
            predicate: pred.clone(),
            child: Box::new(current_plan),
        });
    }

    // 4. Apply GROUP BY / HAVING
    let has_aggregates = !select.group_by.is_empty() || select.having.is_some()
        || select.projections.iter().any(|p| contains_aggregate(p));

    if has_aggregates {
        let aggregates = extract_aggregates(&select.projections);
        current_plan = LogicalPlan::Aggregate(LogicalAggregate {
            group_by: select.group_by.clone(),
            aggregates,
            having: select.having.clone(),
            child: Box::new(current_plan),
        });
    }

    // 5. Apply projection
    let projections = expand_projections(&select.projections, &current_plan)?;
    current_plan = LogicalPlan::Project(LogicalProject {
        expressions: projections,
        child: Box::new(current_plan),
    });

    // 6. Apply DISTINCT
    if select.distinct {
        current_plan = LogicalPlan::Distinct(LogicalDistinct {
            child: Box::new(current_plan),
        });
    }

    // 7. Apply ORDER BY
    if !select.order_by.is_empty() {
        let limit_hint = select.limit.as_ref().map(|l| l.limit);
        current_plan = LogicalPlan::Sort(LogicalSort {
            order_by: select.order_by.clone(),
            child: Box::new(current_plan),
            limit: limit_hint,
        });
    }

    // 8. Apply LIMIT / OFFSET
    if let Some(ref limit_clause) = select.limit {
        current_plan = LogicalPlan::Limit(LogicalLimit {
            limit: limit_clause.limit,
            offset: limit_clause.offset.unwrap_or(0),
            child: Box::new(current_plan),
        });
    }

    // 9. Wrap in LogicalCte / LogicalRecursiveCte nodes
    for cte_def in select.ctes.iter().rev() {
        let cte_key = cte_def.name.to_ascii_lowercase();

        if cte_def.recursive_term.is_some() {
            let (non_rec_logical, non_rec_schema) = cte_registry
                .remove(&cte_key)
                .expect("CTE must be in registry at wrap time");

            let dummy_plan = LogicalPlan::TableScan(LogicalTableScan {
                table: "__recursive_cte_dummy__".to_string(),
                alias: None,
                schema: non_rec_schema.clone(),
                system_table_name: None,
            });
            cte_registry.insert(cte_key.clone(), (dummy_plan, non_rec_schema.clone()));

            let rec_select_plan = cte_def.recursive_term.as_ref().unwrap();

            let adjusted_rec_plan: rook_ast::SelectPlan = if rec_select_plan.from.is_empty() {
                let mut adjusted = (**rec_select_plan).clone();
                adjusted.from = vec![rook_ast::TableRef {
                    name: format!("__cte__:{}", cte_def.name),
                    alias: None,
                }];
                adjusted
            } else {
                (**rec_select_plan).clone()
            };

            let rec_logical = plan_select_with_ctes(
                &adjusted_rec_plan, catalog, db_name, Some(&cte_registry),
            )?;

            cte_registry.remove(&cte_key);

            current_plan = LogicalPlan::RecursiveCte(LogicalRecursiveCte {
                name: cte_def.name.clone(),
                non_recursive: Box::new(non_rec_logical),
                recursive: Box::new(rec_logical),
                union_all: cte_def.union_all,
                schema: non_rec_schema,
                outer: Box::new(current_plan),
            });
        } else {
            let (inner_logical, _schema) = cte_registry
                .remove(&cte_key)
                .expect("CTE must be in registry at wrap time");
            current_plan = LogicalPlan::Cte(LogicalCte {
                name: cte_def.name.clone(),
                inner: Box::new(inner_logical),
                outer: Box::new(current_plan),
            });
        }
    }

    Ok(current_plan)
}
