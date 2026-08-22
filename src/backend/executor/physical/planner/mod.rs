//! Physical Planner — Converts a `LogicalPlan` into a tree of `PhysicalOperator` nodes.
//!
//! The planner walks the logical plan tree bottom-up and builds the corresponding
//! physical operator tree. It uses the catalog to resolve table schemas and file
//! paths needed for operators like `SeqScan`.

use std::path::PathBuf;

use rook_ast::logical::*;

use super::tuple::ColumnInfo;
use super::expr::{Predicate, Expr, expr_from_ast};
use crate::types::datatype::DataType;
use super::operators::{
    PhysicalOperator,
    SeqScanOperator,
    CteScanOperator,
    SingleRowOperator,
    FilterOperator,
    ProjectionOperator,
    LimitOperator,
    DistinctOperator,
    SortOperator,
    SetOpOperator,
    SetOpType as PhysicalSetOpType,
    SubqueryExecOperator,
    SubqueryType as PhysicalSubqueryType,
    InsertOperator,
};

use crate::backend::catalog::types::Catalog;
use crate::backend::heap::heap_manager::HeapManager;
use self::helpers::infer_expr_type_from_ast;
use crate::types::DataValue;

pub mod helpers;
pub mod subqueries;
pub mod joins;

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
        // Start with an empty CTE registry — CTEs are materialised on demand
        // as the planner descends into LogicalCte nodes.
        let mut cte_registry: std::collections::HashMap<String, Vec<super::tuple::Tuple>> =
            std::collections::HashMap::new();
        self.plan_internal(logical_plan, &mut cte_registry)
    }

    pub(crate) fn plan_internal(
        &self,
        plan: &LogicalPlan,
        cte_registry: &mut std::collections::HashMap<String, Vec<super::tuple::Tuple>>,
    ) -> Result<Box<dyn PhysicalOperator>, String> {
        match plan {
            LogicalPlan::TableScan(ts) => self.plan_table_scan(ts),

            LogicalPlan::Filter(f) => {
                // NOTE: index-accelerated scans arrive with the B+Tree stage;
                // filters always run as SeqScan + FilterOperator here.
                let child = self.plan_internal(&f.child, cte_registry)?;
                let predicate = self.make_predicate(&f.predicate, child.schema())?;
                Ok(Box::new(FilterOperator::new(child, predicate)))
            }

            LogicalPlan::Project(p) => {
                let child = self.plan_internal(&p.child, cte_registry)?;
                let child_schema = child.schema();
                let column_names: Vec<String> = child_schema.iter().map(|c| c.name.clone()).collect();
                let child_types: Vec<_> = child_schema.iter().map(|c| c.data_type.clone()).collect();

                // Convert project expressions to physical expressions
                if p.expressions.is_empty() {
                    // No explicit expressions — treat as SELECT *
                    return Ok(Box::new(ProjectionOperator::star(child)));
                }

                let mut projections = Vec::new();
                for ne in &p.expressions {
                    let (e, data_type) = self.plan_projection_expr(&ne.expr, &column_names, &child_types)?;
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
                // Clone the schema immediately to avoid borrowing `child` (which will be moved later)
                let child_schema: Vec<ColumnInfo> = child.schema().to_vec();
                let column_names: Vec<String> = child_schema.iter().map(|c| c.name.clone()).collect();
                let child_types: Vec<_> = child_schema.iter().map(|c| c.data_type.clone()).collect();

                // ── Build sort keys, handling complex expressions ──────────────
                //
                // For simple column references, we look up the column index directly.
                // For complex expressions (e.g. `ORDER BY a+b`), we:
                //   1. Plan the expression as an extra physical expression
                //   2. Project it as an additional hidden column
                //   3. Sort by the hidden column's index
                //   4. Strip the hidden column after sorting
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
                            // Complex sort expression: project as a hidden column
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

                // ── Build the operator chain ──────────────────────────────────

                // Option 1: Project all existing columns + extra sort columns
                if has_complex {
                    let mut all_projections: Vec<(Expr, String, DataType)> = child_schema
                        .iter()
                        .map(|ci| {
                            let expr = Expr::Column { table: None, column: ci.name.clone() };
                            (expr, ci.name.clone(), ci.data_type.clone())
                        })
                        .collect();
                    all_projections.extend(extra_projections);
                    let child_with_extra = Box::new(ProjectionOperator::new(child, all_projections));

                    let sort_op = SortOperator::new(child_with_extra, sort_keys);

                    // Strip the extra sort columns
                    let result_schema: Vec<String> = child_schema.iter().map(|c| c.name.clone()).collect();
                    let indices: Vec<usize> = (0..column_names.len()).collect();

                    if let Some(limit) = s.limit {
                        log::info!("[Volcano] Sort has limit={}, wrapping in LimitOperator", limit);
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
                    // Simple case: all sort keys are column references
                    let sort_op = SortOperator::new(child, sort_keys);

                    if let Some(limit) = s.limit {
                        log::info!("[Volcano] Sort has limit={}, wrapping in LimitOperator", limit);
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

            LogicalPlan::Aggregate(a) => {
                self.plan_aggregate_with_ctes(a, cte_registry)
            }

            LogicalPlan::Join(j) => {
                self.plan_join_with_ctes(j, cte_registry)
            }

            LogicalPlan::SetOp(s) => {
                let left = self.plan_internal(&s.left, cte_registry)?;
                let right = self.plan_internal(&s.right, cte_registry)?;

                let op_type = match s.op {
                    rook_ast::logical::SetOpType::Union => PhysicalSetOpType::Union,
                    rook_ast::logical::SetOpType::Intersect => PhysicalSetOpType::Intersect,
                    rook_ast::logical::SetOpType::Except => PhysicalSetOpType::Except,
                };

                log::info!(
                    "[Volcano] Planning {:?} (all={})",
                    op_type,
                    s.all
                );

                Ok(Box::new(SetOpOperator::new(left, right, op_type, s.all)))
            }

            LogicalPlan::Subquery(sq) => {
                // For FROM-clause subqueries (derived tables), plan the inner query
                // and optionally wrap in SubqueryExecOperator in table (passthrough) mode.
                let inner = self.plan_internal(&sq.subquery, cte_registry)?;

                // If the subquery has an alias, it's a derived table in FROM clause.
                // Just return the inner plan directly — it acts as a scan of the subquery result.
                if sq.alias.is_some() {
                    return Ok(inner);
                }

                // No alias means this is likely a scalar subquery used in an expression.
                // Wrap it in SubqueryExecOperator in Scalar mode.
                log::info!("[Volcano] Wrapping subquery in Scalar SubqueryExecOperator");
                Ok(Box::new(SubqueryExecOperator::new(
                    inner,
                    PhysicalSubqueryType::Scalar,
                    None,
                )))
            }

            // ── Recursive CTE node ──────────────────────────────────────────

            LogicalPlan::RecursiveCte(rc) => {
                self.plan_recursive_cte(rc, cte_registry)
            }

            // ── CTE nodes ────────────────────────────────────────────────────

            LogicalPlan::Cte(c) => {
                // 1. Materialise the CTE inner plan into an in-memory Vec<Tuple>.
                //
                // We fully execute the inner plan now (before planning the outer
                // query) so that any CteScan leaves in the outer query can look
                // up the cached tuples in O(1) without re-executing the CTE body.
                let mut inner_op = self.plan_internal(&c.inner, cte_registry)?;
                let mut tuples = Vec::new();
                while let Some(tuple) = inner_op.next()? {
                    tuples.push(tuple);
                }
                log::info!(
                    "[Volcano] CTE '{}' materialised {} tuples",
                    c.name,
                    tuples.len()
                );

                // 2. Register the materialised tuples so CteScan leaves can find them.
                cte_registry.insert(c.name.to_ascii_lowercase(), tuples);

                // 3. Plan the outer query (which may contain CteScan leaves that
                //    reference the CTE we just registered).
                self.plan_internal(&c.outer, cte_registry)
            }

            LogicalPlan::Insert(inp) => {
                let child = self.plan_internal(&inp.child, cte_registry)?;
                log::info!(
                    "[Volcano] Planning Insert into '{}' with child operator '{}'",
                    inp.table,
                    child.name()
                );
                Ok(Box::new(InsertOperator::new(
                    child,
                    inp.table.clone(),
                    self.db_name.clone(),
                    self.catalog.clone(),
                )))
            }

            LogicalPlan::CteScan(cs) => {
                // Retrieve the pre-materialised tuples from the CTE registry.
                // For recursive CTEs, `plan_recursive_cte` updates the registry before
                // each iteration, so the operator always sees the latest working table.
                let tuples = cte_registry
                    .get(&cs.name.to_ascii_lowercase())
                    .cloned()
                    .ok_or_else(|| format!("CTE '{}' not found in registry during physical planning", cs.name))?;

                // Build the schema for CteScanOperator from the CteScan node.
                // The CTE name serves as the table qualifier so that table-qualified
                // references (e.g. `my_cte.col`) and NATURAL JOIN disambiguation work
                // correctly when a CTE is on the right side of a join.
                let cte_table_name = cs.name.clone();
                let schema: Vec<super::tuple::ColumnInfo> = cs.schema.columns.iter().map(|c| {
                    super::tuple::ColumnInfo {
                        name: c.name.clone(),
                        data_type: crate::types::datatype::DataType::Varchar(u16::MAX),
                        table: Some(cte_table_name.clone()),
                    }
                }).collect();

                log::info!(
                    "[Volcano] CteScan '{}': {} tuples available",
                    cs.name,
                    tuples.len()
                );

                Ok(Box::new(CteScanOperator::new(tuples, schema)))
            }
        }
    }
}

// ── Table scan planning ───────────────────────────────────────────────────────

impl PhysicalPlanner {
    /// Create a table scan operator for the given LogicalTableScan.
    ///
    /// System-table scans only support the virtual `__singlerow__` source at
    /// this stage; real system tables arrive with the metadata stage. User
    /// tables are scanned sequentially (index-driven scans arrive with the
    /// B+Tree stage).
    pub(crate) fn plan_table_scan(&self, ts: &LogicalTableScan) -> Result<Box<dyn PhysicalOperator>, String> {
        // ── System table path ─────────────────────────────────────────────
        if let Some(sys_name) = &ts.system_table_name {
            // Virtual single-row table: no heap file needed
            if sys_name == "__singlerow__" {
                return Ok(Box::new(SingleRowOperator::new()));
            }
            // Real system tables (INFORMATION_SCHEMA backing stores) are
            // introduced by the metadata stage.
            return Err(format!(
                "System table '{}' requires the metadata stage (system tables).",
                sys_name
            ));
        }


        let table_schema = self.resolve_table_schema(&ts.table)?;
        let table_name = ts.table.clone();
        let column_info: Vec<ColumnInfo> = table_schema.iter().map(|c| {
            ColumnInfo {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                table: Some(table_name.clone()),
            }
        }).collect();

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
    /// Return the column-to-physical-index mapping for a given info_schema view.
    ///
    /// Many INFORMATION_SCHEMA views expose column names that do NOT positionally
    /// align with the physical system table columns (e.g. "columns" view's
    /// `column_name` maps to physical index 2, not 3).  This function returns
    /// the correct `(view_col_idx -> phys_col_idx)` mapping so the SeqScan
    /// operator can reorder deserialised values to match view column order.
    fn info_schema_column_mapping(sys_name: &str) -> Vec<usize> {
        match sys_name.to_ascii_lowercase().as_str() {
            // sys_columns: [col_id(0), table_id(1), name(2), data_type(3),
            //               ordinal(4), nullable(5), has_default(6), default_value(7)]
            // COLUMNS view: [table_catalog, table_schema, table_name, column_name,
            //                ordinal_position, data_type, is_nullable, column_default]
            "columns" => vec![0, 1, 2, 2, 4, 3, 5, 7],

            // sys_constraints: [constraint_id(0), table_id(1), type(2),
            //                   columns(3), ref_table(4), ref_columns(5)]
            // TABLE_CONSTRAINTS view: [constraint_catalog, constraint_schema,
            //                          constraint_name, table_name,
            //                          constraint_type, constraint_columns]
            "constraints" | "table_constraints" => vec![0, 1, 2, 3, 2, 3],

            // sys_indexes: [index_id(0), table_id(1), name(2), is_unique(3),
            //               is_primary(4), columns(5)]
            // INDEXES/STATISTICS view: [table_catalog, table_schema, table_name,
            //                          index_name, unique, primary]
            "indexes" | "statistics" => vec![0, 1, 2, 2, 3, 4],

            // Default: empty = identity mapping (positions align).
            // Caller will generate identity based on sys_schema length.
            _ => Vec::new(),
        }
    }

    /// Map INFORMATION_SCHEMA view column names to the correct physical column types.
    ///
    /// Returns `(Vec<ColumnInfo>, Vec<usize>)` where:
    /// - `ColumnInfo` contains view column names with physical DataTypes
    /// - `Vec<usize>` maps each view column position to its physical column index
    ///
    /// The `SeqScanOperator` uses the physical DataTypes for correct deserialisation
    /// and applies the mapping to reorder values into view column order.
    pub(crate) fn map_info_schema_columns(
        &self,
        sys_name: &str,
        info_schema: &rook_ast::logical::ColumnSchema,
        sys_schema: &[DataType],
    ) -> (Vec<ColumnInfo>, Vec<usize>) {
        let mut mapping = Self::info_schema_column_mapping(sys_name);

        // If no explicit mapping (empty vec), use identity mapping: each view
        // column maps to the same physical position.  This is correct for views
        // like "tables", "schemata", "views" where columns align positionally.
        if mapping.is_empty() {
            mapping = (0..sys_schema.len()).collect();
        }

        let column_info: Vec<ColumnInfo> = info_schema.columns.iter().zip(mapping.iter())
            .map(|(c, &phys_idx)| {
                let dt = sys_schema.get(phys_idx)
                    .cloned()
                    .unwrap_or(DataType::Varchar(255));
                ColumnInfo {
                    name: c.name.clone(),
                    data_type: dt,
                    table: None,
                }
            })
            .collect();

        (column_info, mapping)
    }

    /// Resolve the table's schema from the catalog.
    pub(crate) fn resolve_table_schema(&self, table_name: &str) -> Result<Vec<crate::backend::catalog::types::Column>, String> {
        let db = self.catalog.databases.get(&self.db_name)
            .ok_or_else(|| format!("Database '{}' not found", self.db_name))?;
        let table = db.tables.get(table_name)
            .ok_or_else(|| format!("Table '{}' not found in database '{}'", table_name, self.db_name))?;
        Ok(table.columns.clone())
    }

    /// Build a predicate from the AST predicate node, resolving column names
    /// against the child operator's schema. Handles subqueries (EXISTS, IN)
    /// by materializing them during physical plan construction.
    pub(crate) fn make_predicate(
        &self,
        pred_node: &rook_ast::PredicateNode,
        schema: &[ColumnInfo],
    ) -> Result<Predicate, String> {
        let column_names: Vec<String> = schema.iter().map(|c| c.name.clone()).collect();
        self.build_predicate_with_subqueries(pred_node, &column_names, schema)
    }
}

// ── Projection expression planning ────────────────────────────────────────────

impl PhysicalPlanner {
    /// Plan a single projection expression, handling `ScalarSubquery` nodes
    /// directly without going through the ConstantValue round-trip.
    pub(crate) fn plan_projection_expr(
        &self,
        expr: &rook_ast::ExprNode,
        column_names: &[String],
        child_types: &[DataType],
    ) -> Result<(Expr, DataType), String> {
        match expr {
            // Top-level scalar subquery: materialize and create Expr::Constant directly
            rook_ast::ExprNode::ScalarSubquery(info) => {
                let (value, data_type) = self.materialize_scalar_subquery(&info.select)?;
                let phys_expr = match value {
                    Some(dv) => Expr::Constant(dv),
                    None => Expr::Null,
                };
                Ok((phys_expr, data_type))
            }
            // Nested subqueries inside Binary or Cast need AST replacement first
            rook_ast::ExprNode::Binary { left, op, right } => {
                let left = self.materialize_nested_subqueries(left);
                let right = self.materialize_nested_subqueries(right);
                // Now convert the materialized AST (no more subqueries inside)
                let left_expr = expr_from_ast(&left?, column_names)?;
                let right_expr = expr_from_ast(&right?, column_names)?;
                let phys_op = match op {
                    rook_ast::ArithOp::Add => Expr::Add(Box::new(left_expr), Box::new(right_expr)),
                    rook_ast::ArithOp::Sub => Expr::Sub(Box::new(left_expr), Box::new(right_expr)),
                    rook_ast::ArithOp::Mul => Expr::Mul(Box::new(left_expr), Box::new(right_expr)),
                    rook_ast::ArithOp::Div => Expr::Div(Box::new(left_expr), Box::new(right_expr)),
                };
                let data_type = infer_expr_type_from_ast(expr, child_types, column_names)?;
                Ok((phys_op, data_type))
            }
            rook_ast::ExprNode::Cast { expr: inner, data_type } => {
                let inner = self.materialize_nested_subqueries(inner)?;
                let inner_expr = expr_from_ast(&inner, column_names)?;
                let target_dt: DataType = data_type.parse()
                    .map_err(|e: String| format!("Invalid CAST target type '{}': {}", data_type, e))?;
                Ok((Expr::Cast(Box::new(inner_expr), target_dt.clone()), target_dt))
            }
            // All other expression types: use the standard converter
            _ => {
                let e = expr_from_ast(expr, column_names)?;
                let data_type = infer_expr_type_from_ast(expr, child_types, column_names)?;
                Ok((e, data_type))
            }
        }
    }
}
