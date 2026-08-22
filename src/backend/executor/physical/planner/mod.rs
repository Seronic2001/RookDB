//! Physical Planner — converts a `LogicalPlan` into a tree of operators.
//!
//! The planner walks the logical plan tree top-down and builds the matching
//! physical operator tree, using the catalog to resolve table schemas and
//! heap file paths for scans.
//!
//! This stage implements the core relational operators:
//! scans, filters, projections, sorting, limiting and deduplication.
//! JOINs, aggregates, set operations, subqueries/CTEs and INSERT pipelines
//! are planned by later stages and report themselves as unsupported here.

use std::path::PathBuf;

use rook_ast::logical::*;

use super::tuple::{ColumnInfo, Tuple};
use super::expr::{expr_from_ast, predicate_from_ast, Expr, Predicate};
use super::operators::{
    CteScanOperator, DistinctOperator, FilterOperator, LimitOperator, PhysicalOperator,
    ProjectionOperator, SeqScanOperator, SingleRowOperator, SortOperator,
};
use crate::backend::catalog::types::Catalog;
use crate::backend::heap::heap_manager::HeapManager;
use crate::types::datatype::DataType;
use self::helpers::infer_expr_type_from_ast;

pub mod helpers;

/// Converts a `LogicalPlan` into an executable `PhysicalOperator` tree.
pub struct PhysicalPlanner {
    /// The catalog for schema lookups.
    pub(crate) catalog: Catalog,
    /// Current database name.
    pub(crate) db_name: String,
}

impl PhysicalPlanner {
    pub fn new(catalog: Catalog, db_name: String) -> Self {
        Self { catalog, db_name }
    }

    /// Plan a logical query plan into a physical operator tree.
    ///
    /// Returns the root of the operator tree.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<Box<dyn PhysicalOperator>, String> {
        // CTEs are materialised on demand once the advanced operators land;
        // until then the registry stays empty.
        let mut cte_registry: std::collections::HashMap<String, Vec<Tuple>> =
            std::collections::HashMap::new();
        self.plan_internal(logical_plan, &mut cte_registry)
    }

    pub(crate) fn plan_internal(
        &self,
        plan: &LogicalPlan,
        cte_registry: &mut std::collections::HashMap<String, Vec<Tuple>>,
    ) -> Result<Box<dyn PhysicalOperator>, String> {
        match plan {
            LogicalPlan::TableScan(ts) => self.plan_table_scan(ts),

            LogicalPlan::Filter(f) => {
                let child = self.plan_internal(&f.child, cte_registry)?;
                let predicate = self.make_predicate(&f.predicate, child.schema())?;
                Ok(Box::new(FilterOperator::new(child, predicate)))
            }

            LogicalPlan::Project(p) => {
                let child = self.plan_internal(&p.child, cte_registry)?;
                let child_schema = child.schema();
                let column_names: Vec<String> =
                    child_schema.iter().map(|c| c.name.clone()).collect();
                let child_types: Vec<_> =
                    child_schema.iter().map(|c| c.data_type.clone()).collect();

                if p.expressions.is_empty() {
                    // No explicit expressions — treat as SELECT *
                    return Ok(Box::new(ProjectionOperator::star(child)));
                }

                let mut projections = Vec::new();
                for ne in &p.expressions {
                    let (e, data_type) =
                        self.plan_projection_expr(&ne.expr, &column_names, &child_types)?;
                    projections.push((e, ne.name.clone(), data_type));
                }

                Ok(Box::new(ProjectionOperator::new(child, projections)))
            }

            LogicalPlan::Distinct(d) => {
                let child = self.plan_internal(&d.child, cte_registry)?;
                Ok(Box::new(DistinctOperator::new(child)))
            }

            LogicalPlan::Sort(s) => {
                let child = self.plan_internal(&s.child, cte_registry)?;
                // Clone the schema immediately to avoid borrowing `child`
                // (which will be moved later).
                let child_schema: Vec<ColumnInfo> = child.schema().to_vec();
                let column_names: Vec<String> =
                    child_schema.iter().map(|c| c.name.clone()).collect();
                let child_types: Vec<_> =
                    child_schema.iter().map(|c| c.data_type.clone()).collect();

                // Build sort keys. Simple column references resolve directly;
                // complex expressions (`ORDER BY a + b`) are projected as a
                // hidden extra column, sorted by its index, then stripped.
                let mut sort_keys: Vec<(usize, bool)> = Vec::new();
                let mut extra_projections: Vec<(Expr, String, DataType)> = Vec::new();
                let mut has_complex = false;

                for ob in &s.order_by {
                    match &ob.expr {
                        rook_ast::ExprNode::Column(name) => {
                            let idx = column_names.iter().position(|n| n == name)
                                .ok_or_else(|| format!("Sort column '{}' not found", name))?;
                            sort_keys.push((idx, !ob.ascending));
                        }
                        rook_ast::ExprNode::Compound(parts) => {
                            let name = parts.last().cloned().unwrap_or_default();
                            let idx = column_names.iter().position(|n| n == &name)
                                .ok_or_else(|| format!("Sort column '{}' not found", name))?;
                            sort_keys.push((idx, !ob.ascending));
                        }
                        complex_expr => {
                            let (phys_expr, data_type) = self.plan_projection_expr(
                                complex_expr, &column_names, &child_types,
                            )?;
                            let col_name = format!("__sort_col_{}", extra_projections.len());
                            let sort_idx = column_names.len() + extra_projections.len();
                            sort_keys.push((sort_idx, !ob.ascending));
                            extra_projections.push((phys_expr, col_name.clone(), data_type));
                            has_complex = true;
                        }
                    }
                }

                if has_complex {
                    // Project all existing columns plus the hidden sort keys.
                    let mut all_projections: Vec<(Expr, String, DataType)> = child_schema
                        .iter()
                        .map(|ci| {
                            let expr = Expr::Column { table: None, column: ci.name.clone() };
                            (expr, ci.name.clone(), ci.data_type.clone())
                        })
                        .collect();
                    all_projections.extend(extra_projections);
                    let child_with_extra =
                        Box::new(ProjectionOperator::new(child, all_projections));

                    let sort_op = SortOperator::new(child_with_extra, sort_keys);

                    // Strip the hidden columns after sorting.
                    let result_schema: Vec<String> =
                        child_schema.iter().map(|c| c.name.clone()).collect();
                    let indices: Vec<usize> = (0..column_names.len()).collect();

                    if let Some(limit) = s.limit {
                        let limited = Box::new(LimitOperator::new(
                            Box::new(sort_op), limit as usize, 0,
                        ));
                        Ok(Box::new(ProjectionOperator::from_indices(
                            limited, &indices, &result_schema,
                        )?))
                    } else {
                        Ok(Box::new(ProjectionOperator::from_indices(
                            Box::new(sort_op), &indices, &result_schema,
                        )?))
                    }
                } else {
                    // Simple case: every sort key is a plain column reference.
                    let sort_op = SortOperator::new(child, sort_keys);

                    if let Some(limit) = s.limit {
                        // Push the limit into the sort pipeline as a top-k hint.
                        Ok(Box::new(LimitOperator::new(
                            Box::new(sort_op), limit as usize, 0,
                        )))
                    } else {
                        Ok(Box::new(sort_op))
                    }
                }
            }

            LogicalPlan::Limit(l) => {
                let child = self.plan_internal(&l.child, cte_registry)?;
                Ok(Box::new(LimitOperator::new(
                    child,
                    l.limit as usize,
                    l.offset as usize,
                )))
            }

            LogicalPlan::CteScan(cs) => {
                let tuples = cte_registry
                    .get(&cs.name.to_ascii_lowercase())
                    .cloned()
                    .ok_or_else(|| format!(
                        "CTE '{}' is not supported yet (advanced-operator stage)",
                        cs.name
                    ))?;
                let schema: Vec<ColumnInfo> = cs.schema.columns.iter().map(|c| {
                    ColumnInfo {
                        name: c.name.clone(),
                        data_type: DataType::Varchar(u16::MAX),
                        table: Some(cs.name.clone()),
                    }
                }).collect();
                Ok(Box::new(CteScanOperator::new(tuples, schema)))
            }

            // ── Planned by later stages ─────────────────────────────────────
            LogicalPlan::Aggregate(_) => unsupported_later("GROUP BY / aggregates"),
            LogicalPlan::Join(_) => unsupported_later("JOINs"),
            LogicalPlan::SetOp(_) => unsupported_later("UNION / INTERSECT / EXCEPT"),
            LogicalPlan::Subquery(_) => unsupported_later("subqueries in FROM"),
            LogicalPlan::RecursiveCte(_) => unsupported_later("recursive CTEs"),
            LogicalPlan::Cte(_) => unsupported_later("common table expressions"),
            LogicalPlan::Insert(_) => unsupported_later("INSERT INTO ... SELECT"),
        }
    }
}

fn unsupported_later(feature: &str) -> Result<Box<dyn PhysicalOperator>, String> {
    Err(format!(
        "{} are recognised by the planner but not executable yet \
         (planned for the advanced-operators stage).",
        feature
    ))
}

// ── Table scan planning ───────────────────────────────────────────────────────

impl PhysicalPlanner {
    /// Create a scan operator for the given `LogicalTableScan`.
    ///
    /// System-table scans only support the virtual `__singlerow__` source at
    /// this stage; real system tables arrive with the metadata stage.
    pub(crate) fn plan_table_scan(
        &self,
        ts: &LogicalTableScan,
    ) -> Result<Box<dyn PhysicalOperator>, String> {
        // ── Virtual single-row table (SELECT without FROM) ────────────────
        if let Some(sys_name) = &ts.system_table_name {
            if sys_name == "__singlerow__" {
                return Ok(Box::new(SingleRowOperator::new()));
            }
            return Err(format!(
                "System table '{}' requires the metadata stage (system tables).",
                sys_name
            ));
        }

        // ── Regular user table path ───────────────────────────────────────
        let table_schema = self.resolve_table_schema(&ts.table)?;
        let table_name = ts.table.clone();
        let column_info: Vec<ColumnInfo> = table_schema
            .iter()
            .map(|c| ColumnInfo {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                table: Some(table_name.clone()),
            })
            .collect();

        let heap_path = PathBuf::from(format!(
            "database/base/{}/{}.dat",
            self.db_name, ts.table
        ));

        if !heap_path.exists() {
            return Err(format!("Heap file not found: {:?}", heap_path));
        }

        let heap_manager = HeapManager::open(heap_path)
            .map_err(|e| format!("Failed to open heap for table '{}': {}", ts.table, e))?;

        Ok(Box::new(SeqScanOperator::new(heap_manager, column_info)))
    }
}

// ── Schema helpers ────────────────────────────────────────────────────────────

impl PhysicalPlanner {
    /// Resolve the table's schema from the catalog.
    pub(crate) fn resolve_table_schema(
        &self,
        table_name: &str,
    ) -> Result<Vec<crate::backend::catalog::types::Column>, String> {
        let db = self.catalog.databases.get(&self.db_name)
            .ok_or_else(|| format!("Database '{}' not found", self.db_name))?;
        let table = db.tables.get(table_name)
            .ok_or_else(|| format!("Table '{}' not found in database '{}'", table_name, self.db_name))?;
        Ok(table.columns.clone())
    }

    /// Build a physical predicate from the AST predicate node, resolving
    /// column names against the child operator's schema.
    pub(crate) fn make_predicate(
        &self,
        pred_node: &rook_ast::PredicateNode,
        schema: &[ColumnInfo],
    ) -> Result<Predicate, String> {
        let column_names: Vec<String> = schema.iter().map(|c| c.name.clone()).collect();
        predicate_from_ast(pred_node, &column_names)
    }
}

// ── Projection expression planning ────────────────────────────────────────────

impl PhysicalPlanner {
    /// Plan a single projection expression into a physical expression plus
    /// its inferred output type.
    pub(crate) fn plan_projection_expr(
        &self,
        expr: &rook_ast::ExprNode,
        column_names: &[String],
        child_types: &[DataType],
    ) -> Result<(Expr, DataType), String> {
        let e = expr_from_ast(expr, column_names)?;
        let data_type = infer_expr_type_from_ast(expr, child_types, column_names)?;
        Ok((e, data_type))
    }
}
