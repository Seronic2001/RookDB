//! Subquery planning for the PhysicalPlanner.
//!
//! Handles correlated EXISTS/IN subqueries, scalar subquery materialization,
//! and predicate building with subquery support.

use std::cell::RefCell;
use std::rc::Rc;


use super::super::tuple::ColumnInfo;
use super::super::expr::{Predicate, Expr, ComparisonOp, expr_from_ast, predicate_from_ast};
use super::super::engine::execute_plan_collect;
use crate::types::datatype::DataType;
use crate::types::DataValue;
use super::super::operators::{
    PhysicalOperator,
    FilterOperator,
};
use super::PhysicalPlanner;

impl PhysicalPlanner {
    /// Recursively convert a `PredicateNode` to a physical `Predicate`,
    /// materializing any subqueries (EXISTS, IN subquery) at plan time.
    pub(crate) fn build_predicate_with_subqueries(
        &self,
        node: &rook_ast::PredicateNode,
        column_names: &[String],
        schema: &[ColumnInfo],
    ) -> Result<Predicate, String> {
        match node {
            rook_ast::PredicateNode::BinaryOp { left, op, right } => {
                let l = self.build_predicate_with_subqueries(left, column_names, schema)?;
                let r = self.build_predicate_with_subqueries(right, column_names, schema)?;
                match op {
                    rook_ast::BinaryOp::And => Ok(Predicate::and(l, r)),
                    rook_ast::BinaryOp::Or => Ok(Predicate::or(l, r)),
                }
            }
            rook_ast::PredicateNode::Not(inner) => {
                let inner = self.build_predicate_with_subqueries(inner, column_names, schema)?;
                Ok(Predicate::not(inner))
            }
            rook_ast::PredicateNode::Exists(subquery_info) => {
                // Check if this EXISTS is correlated (references outer columns)
                let inner_table_names: Vec<String> = subquery_info.select.from
                    .iter().map(|t| t.name.clone())
                    .collect();
                if self.is_predicate_correlated(
                    subquery_info.select.selection.as_ref(),
                    &inner_table_names,
                ) {
                    log::info!("[Planner] Building correlated EXISTS subquery");
                    self.build_correlated_exists(subquery_info, schema)
                } else {
                    log::info!("[Planner] Materializing non-correlated EXISTS subquery");
                    let exists = self.materialize_exists_subquery(subquery_info.select.as_ref())?;
                    Ok(Predicate::ExistsResult(exists))
                }
            }
            rook_ast::PredicateNode::InSubquery {
                expr,
                subquery,
                negated,
            } => {
                let inner_table_names: Vec<String> = subquery.select.from
                    .iter().map(|t| t.name.clone())
                    .collect();
                if self.is_predicate_correlated(
                    subquery.select.selection.as_ref(),
                    &inner_table_names,
                ) {
                    log::info!("[Planner] Building correlated IN subquery");
                    let lhs_expr = expr_from_ast(expr, column_names)?;
                    self.build_correlated_in_subquery(
                        subquery, lhs_expr, *negated, schema,
                    )
                } else {
                    log::info!("[Planner] Materializing non-correlated IN subquery");
                    let lhs_expr = expr_from_ast(expr, column_names)?;
                    let values = self.materialize_in_subquery(subquery.select.as_ref())?;
                    Ok(Predicate::InSubqueryResult(lhs_expr, values, *negated))
                }
            }
            // IS DISTINCT FROM and IS Boolean — handle directly
            rook_ast::PredicateNode::IsDistinctFrom { left, right } => {
                let l = expr_from_ast(left, column_names)?;
                let r = expr_from_ast(right, column_names)?;
                Ok(Predicate::IsDistinctFrom(l, r))
            }
            rook_ast::PredicateNode::IsBoolean {
                expr,
                test,
                negated,
            } => {
                let e = expr_from_ast(expr, column_names)?;
                let bt = match test {
                    rook_ast::BooleanTest::True => super::super::expr::BooleanTest::True,
                    rook_ast::BooleanTest::False => super::super::expr::BooleanTest::False,
                    rook_ast::BooleanTest::Unknown => super::super::expr::BooleanTest::Unknown,
                };
                Ok(Predicate::IsBoolean {
                    expr: e,
                    test: bt,
                    negated: *negated,
                })
            }
            // All standard predicate nodes delegate to predicate_from_ast
            _ => {
                // Use the standard converter
                predicate_from_ast(node, column_names)
            }
        }
    }

    /// Plan and execute an EXISTS subquery, returning whether any rows exist.
    pub(crate) fn materialize_exists_subquery(
        &self,
        select: &rook_ast::SelectPlan,
    ) -> Result<bool, String> {
        // Build QueryPlan around the SelectPlan
        let query_plan = rook_ast::QueryPlan::Select(select.clone());

        // Plan the inner query into a LogicalPlan
        let logical_plan = crate::planner::plan_query(&query_plan, &self.catalog, &self.db_name)
            .map_err(|e| format!("Failed to plan EXISTS subquery: {}", e.message))?;

        // Execute the plan and check if any tuples are produced
        let tuples = execute_plan_collect(&logical_plan, &self.catalog, &self.db_name).map_err(|e| e.to_string())?;
        Ok(!tuples.is_empty())
    }

    /// Plan and execute an IN subquery, returning the list of values from the
    /// first column of the subquery result.
    pub(crate) fn materialize_in_subquery(
        &self,
        select: &rook_ast::SelectPlan,
    ) -> Result<Vec<Option<DataValue>>, String> {
        let query_plan = rook_ast::QueryPlan::Select(select.clone());

        let logical_plan = crate::planner::plan_query(&query_plan, &self.catalog, &self.db_name)
            .map_err(|e| format!("Failed to plan IN subquery: {}", e.message))?;

        // Execute and collect tuples
        let tuples = execute_plan_collect(&logical_plan, &self.catalog, &self.db_name).map_err(|e| e.to_string())?;

        // Extract the first column value from each tuple
        let values: Vec<Option<DataValue>> = tuples
            .into_iter()
            .map(|t| t.values.into_iter().next().unwrap_or(None))
            .collect();

        log::info!(
            "[Planner] IN subquery materialized {} values",
            values.len()
        );

        Ok(values)
    }

    /// Materialize a single scalar subquery and return (value, data_type).
    pub(crate) fn materialize_scalar_subquery(
        &self,
        select: &Box<rook_ast::SelectPlan>,
    ) -> Result<(Option<DataValue>, DataType), String> {
        log::info!("[Planner] Materializing scalar subquery");
        let query_plan = rook_ast::QueryPlan::Select((**select).clone());

        let logical_plan = crate::planner::plan_query(&query_plan, &self.catalog, &self.db_name)
            .map_err(|e| format!("Failed to plan scalar subquery: {}", e.message))?;

        let tuples = execute_plan_collect(&logical_plan, &self.catalog, &self.db_name).map_err(|e| e.to_string())?;

        match tuples.len() {
            0 => Ok((None, DataType::Int)), // NULL → default to Int
            1 => {
                let mut tuple = tuples.into_iter().next().unwrap();
                let value = tuple.values.drain(..).next().flatten();
                // Infer the data type from the column info
                let data_type = tuple.column_info.first()
                    .map(|ci| ci.data_type.clone())
                    .unwrap_or(DataType::Int);
                Ok((value, data_type))
            }
            n => Err(format!(
                "Scalar subquery returned more than one row (got {})",
                n
            )),
        }
    }

    /// Recursively walk an `ExprNode` tree and replace any `ScalarSubquery`
    /// nodes with `Constant` nodes. Used for nested subqueries inside
    /// `Binary`, `Cast`, etc.
    pub(crate) fn materialize_nested_subqueries(
        &self,
        expr: &rook_ast::ExprNode,
    ) -> Result<rook_ast::ExprNode, String> {
        match expr {
            rook_ast::ExprNode::ScalarSubquery(info) => {
                let (value, _) = self.materialize_scalar_subquery(&info.select)?;
                let constant = match value {
                    Some(dv) => super::helpers::data_value_to_constant_value(&dv),
                    None => rook_ast::ConstantValue::Null,
                };
                Ok(rook_ast::ExprNode::Constant(constant))
            }
            rook_ast::ExprNode::Binary { left, op, right } => {
                let left = self.materialize_nested_subqueries(left)?;
                let right = self.materialize_nested_subqueries(right)?;
                Ok(rook_ast::ExprNode::Binary {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                })
            }
            rook_ast::ExprNode::Cast { expr: inner, data_type } => {
                let inner = self.materialize_nested_subqueries(inner)?;
                Ok(rook_ast::ExprNode::Cast {
                    expr: Box::new(inner),
                    data_type: data_type.clone(),
                })
            }
            // Leaf nodes: Column, Compound, Constant — no subqueries inside
            _ => Ok(expr.clone()),
        }
    }
}

// ── Correlated subquery detection and building ────────────────────────────────

impl PhysicalPlanner {
    /// Check if a predicate (from an inner query's SELECT) references columns
    /// belonging to an **outer** table (i.e., columns that are NOT in any of
    /// the inner FROM tables). If so, the subquery is correlated.
    pub(crate) fn is_predicate_correlated(
        &self,
        pred_node: Option<&rook_ast::PredicateNode>,
        inner_table_names: &[String],
    ) -> bool {
        let Some(node) = pred_node else {
            return false;
        };
        self.pred_contains_outer_ref(node, inner_table_names)
    }

    /// Recursively walk a predicate tree looking for any column reference that
    /// does not belong to the inner query's tables.
    fn pred_contains_outer_ref(
        &self,
        node: &rook_ast::PredicateNode,
        inner_tables: &[String],
    ) -> bool {
        match node {
            rook_ast::PredicateNode::BinaryOp { left, right, .. } => {
                self.pred_contains_outer_ref(left, inner_tables)
                    || self.pred_contains_outer_ref(right, inner_tables)
            }
            rook_ast::PredicateNode::Not(inner) => {
                self.pred_contains_outer_ref(inner, inner_tables)
            }
            rook_ast::PredicateNode::Compare { left, right, .. } => {
                self.expr_is_outer_ref(left, inner_tables)
                    || self.expr_is_outer_ref(right, inner_tables)
            }
            rook_ast::PredicateNode::IsNull(expr)
            | rook_ast::PredicateNode::IsNotNull(expr) => {
                self.expr_is_outer_ref(expr, inner_tables)
            }
            rook_ast::PredicateNode::Between { expr, low, high } => {
                self.expr_is_outer_ref(expr, inner_tables)
                    || self.expr_is_outer_ref(low, inner_tables)
                    || self.expr_is_outer_ref(high, inner_tables)
            }
            rook_ast::PredicateNode::InList { expr, list } => {
                self.expr_is_outer_ref(expr, inner_tables)
                    || list.iter().any(|e| self.expr_is_outer_ref(e, inner_tables))
            }
            rook_ast::PredicateNode::Like { expr, .. } => {
                self.expr_is_outer_ref(expr, inner_tables)
            }
            rook_ast::PredicateNode::IsDistinctFrom { left, right } => {
                self.expr_is_outer_ref(left, inner_tables)
                    || self.expr_is_outer_ref(right, inner_tables)
            }
            rook_ast::PredicateNode::IsBoolean { expr, .. } => {
                self.expr_is_outer_ref(expr, inner_tables)
            }
            // EXISTS and IN-subquery inside a predicate are not expected here
            rook_ast::PredicateNode::Exists(_)
            | rook_ast::PredicateNode::InSubquery { .. } => false,
        }
    }

    /// Check whether an expression node references an outer table column.
    ///
    /// A `Compound([table, col])` is an outer ref if `table` does not match any
    /// of the inner table names. A simple `Column(name)` is assumed to be an
    /// inner ref unless proven otherwise (conservative heuristic).
    fn expr_is_outer_ref(&self, expr: &rook_ast::ExprNode, inner_tables: &[String]) -> bool {
        match expr {
            rook_ast::ExprNode::Compound(parts) => {
                if let Some(table_part) = parts.first() {
                    !inner_tables.iter().any(|t| t.eq_ignore_ascii_case(table_part))
                } else {
                    false
                }
            }
            // Simple Column refs: if the column name exists in an inner table
            // schema, it's an inner ref. Otherwise conservatively assume it's
            // an outer ref — this handles cases like `x = outer.col` where `x`
            // is an unqualified inner column and `outer.col` is compound.
            rook_ast::ExprNode::Column(name) => {
                // Check if it's a column in any inner table
                !self.column_exists_in_any_inner_table(name, inner_tables)
            }
            rook_ast::ExprNode::Constant(_) | rook_ast::ExprNode::ScalarSubquery(_) => false,
            rook_ast::ExprNode::Function { .. } => false,
            rook_ast::ExprNode::Binary { left, right, .. } => {
                self.expr_is_outer_ref(left, inner_tables)
                    || self.expr_is_outer_ref(right, inner_tables)
            }
            rook_ast::ExprNode::Cast { expr: inner, .. } => {
                self.expr_is_outer_ref(inner, inner_tables)
            }
            rook_ast::ExprNode::Case { when_then_pairs, else_result } => {
                for (when, then) in when_then_pairs {
                    if self.expr_is_outer_ref(when, inner_tables)
                        || self.expr_is_outer_ref(then, inner_tables)
                    {
                        return true;
                    }
                }
                if let Some(else_node) = else_result {
                    self.expr_is_outer_ref(else_node, inner_tables)
                } else {
                    false
                }
            }
        }
    }

    /// Check whether a bare column name exists in any of the inner tables'
    /// catalog schemas.
    fn column_exists_in_any_inner_table(
        &self,
        col_name: &str,
        inner_tables: &[String],
    ) -> bool {
        let db = match self.catalog.databases.get(&self.db_name) {
            Some(db) => db,
            None => return false,
        };
        for table_name in inner_tables {
            if let Some(table) = db.tables.get(table_name) {
                if table.columns.iter().any(|c| c.name.eq_ignore_ascii_case(col_name)) {
                    return true;
                }
            }
        }
        false
    }

    /// Build a correlated EXISTS subquery with per-row execution.
    ///
    /// Supports multi-column correlations: extracts ALL correlation predicates
    /// from the inner WHERE clause (connected by AND) and creates a filter with
    /// multiple CorrelatedParam bindings.
    fn build_correlated_exists(
        &self,
        subquery_info: &rook_ast::SubqueryInfo,
        outer_schema: &[ColumnInfo],
    ) -> Result<Predicate, String> {
        let select = &subquery_info.select;

        // ── 1. Determine inner table (or its alias) and column info ──
        let inner_tref = select
            .from
            .first()
            .ok_or_else(|| "EXISTS subquery has no FROM table".to_string())?;
        // Use alias if present, since predicates reference the alias not the raw table name
        let inner_table_alias = inner_tref.alias.as_deref().unwrap_or(&inner_tref.name).to_string();
        let inner_table_name = inner_tref.name.clone();

        let inner_table_schema = self.resolve_table_schema(&inner_table_name)?;
        let inner_col_names: Vec<String> =
            inner_table_schema.iter().map(|c| c.name.clone()).collect();

        // ── 2. Extract ALL correlation pairs from the inner predicate ──
        // Use the table alias (or raw name if no alias) for column matching
        // since predicates reference the alias in compound identifiers.
        let correlation_pairs = if let Some(ref selection) = select.selection {
            self.extract_all_correlations(
                selection,
                &inner_table_alias,
                &inner_col_names,
            )?
        } else {
            return Err("Correlated EXISTS subquery has no WHERE clause".to_string());
        };

        if correlation_pairs.is_empty() {
            return Err("No correlation found in EXISTS subquery".to_string());
        }

        // ── 3. Build the inner plan directly as a SeqScan ──────────────
        let inner_table_scan = rook_ast::logical::LogicalTableScan {
            table: inner_table_name.clone(),
            alias: None,
            schema: rook_ast::logical::ColumnSchema {
                columns: inner_col_names.iter().map(|name| {
                    rook_ast::logical::ColumnInfo {
                        name: name.clone(),
                        data_type: "UNKNOWN".to_string(),
                        nullable: true }
                }).collect(),
            },
            system_table_name: None,
        };
        let inner_base = self.plan_table_scan(&inner_table_scan)?;

        // ── 4. Build filter predicates with CorrelatedParams for EACH pair ──
        let mut params = Vec::new();
        let mut outer_indices = Vec::new();
        let mut filter_predicate: Option<Predicate> = None;

        for (inner_col_name, outer_col_name, comp_op) in &correlation_pairs {
            // Resolve inner column index
            let _inner_col_idx = inner_col_names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(inner_col_name))
                .ok_or_else(|| {
                    format!(
                        "Inner column '{}' not found in table '{}'",
                        inner_col_name, inner_table_name
                    )
                })?;

            // Resolve outer column index
            let outer_col_idx = outer_schema
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(outer_col_name))
                .ok_or_else(|| {
                    format!(
                        "Outer column '{}' not found in outer schema (available: {:?})",
                        outer_col_name,
                        outer_schema.iter().map(|c| &c.name).collect::<Vec<_>>()
                    )
                })?;

            let param = Rc::new(RefCell::new(None));
            let phys_comparison = match comp_op {
                rook_ast::ComparisonOp::Eq => ComparisonOp::Equals,
                rook_ast::ComparisonOp::Ne => ComparisonOp::NotEquals,
                rook_ast::ComparisonOp::Lt => ComparisonOp::LessThan,
                rook_ast::ComparisonOp::Le => ComparisonOp::LessOrEqual,
                rook_ast::ComparisonOp::Gt => ComparisonOp::GreaterThan,
                rook_ast::ComparisonOp::Ge => ComparisonOp::GreaterOrEqual,
            };

            let col_pred = Predicate::Compare(
                Expr::Column { table: None, column: inner_col_name.clone() },
                phys_comparison,
                Expr::CorrelatedParam(param.clone()),
            );

            filter_predicate = Some(match filter_predicate {
                None => col_pred,
                Some(acc) => Predicate::and(acc, col_pred),
            });

            params.push(param);
            outer_indices.push(outer_col_idx);

            log::info!(
                "[Planner] Correlated EXISTS: inner col '{}' (idx={}) = outer.col '{}' (idx={})",
                inner_col_name, _inner_col_idx, outer_col_name, outer_col_idx
            );
        }

        let filter_op: Box<dyn PhysicalOperator> = Box::new(FilterOperator::new(
            inner_base,
            filter_predicate.unwrap(),
        ));
        let inner_plan_rc = Rc::new(RefCell::new(filter_op));

        Ok(Predicate::CorrelatedExists {
            inner_plan: inner_plan_rc,
            params,
            outer_col_indices: outer_indices,
        })
    }

    /// Build a correlated IN (subquery) with per-row execution.
    ///
    /// Supports multi-column correlations: extracts ALL correlation predicates
    /// from the inner WHERE clause and creates a filter with multiple
    /// CorrelatedParam bindings.
    fn build_correlated_in_subquery(
        &self,
        subquery: &rook_ast::SubqueryInfo,
        lhs_expr: Expr,
        negated: bool,
        outer_schema: &[ColumnInfo],
    ) -> Result<Predicate, String> {
        let select = &subquery.select;

        // ── 1. Determine inner table (or its alias) and column info ──
        let inner_tref = select
            .from
            .first()
            .ok_or_else(|| "IN subquery has no FROM table".to_string())?;
        // Use alias if present, since predicates reference the alias not the raw table name
        let inner_table_alias = inner_tref.alias.as_deref().unwrap_or(&inner_tref.name).to_string();
        let inner_table_name = inner_tref.name.clone();

        let inner_table_schema = self.resolve_table_schema(&inner_table_name)?;
        let inner_col_names: Vec<String> =
            inner_table_schema.iter().map(|c| c.name.clone()).collect();

        // ── 2. Extract ALL correlation pairs ───────────────────────────
        // Use the table alias (or raw name if no alias) since predicates
        // reference the alias in compound identifiers.
        let correlation_pairs = if let Some(ref selection) = select.selection {
            self.extract_all_correlations(
                selection,
                &inner_table_alias,
                &inner_col_names,
            )?
        } else {
            return Err("Correlated IN subquery has no WHERE clause".to_string());
        };

        if correlation_pairs.is_empty() {
            return Err("No correlation found in IN subquery".to_string());
        }

        // ── 3. Build the inner plan (no WHERE clause, just SeqScan) ────
        let mut inner_select = (**select).clone();
        inner_select.selection = None;

        let query_plan = rook_ast::QueryPlan::Select(inner_select);
        let logical_plan =
            crate::planner::plan_query(&query_plan, &self.catalog, &self.db_name)
                .map_err(|e| format!("Failed to plan correlated IN subquery: {}", e.message))?;

        let inner_planner = PhysicalPlanner::new(self.catalog.clone(), self.db_name.clone());
        let inner_base = inner_planner.plan(&logical_plan)?;

        // ── 4. Build filter predicates with CorrelatedParams for EACH pair ──
        let mut params = Vec::new();
        let mut outer_indices = Vec::new();
        let mut filter_predicate: Option<Predicate> = None;

        for (inner_col_name, outer_col_name, comp_op) in &correlation_pairs {
            let _inner_col_idx = inner_col_names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(inner_col_name))
                .ok_or_else(|| {
                    format!(
                        "Inner column '{}' not found in table '{}'",
                        inner_col_name, inner_table_name
                    )
                })?;

            let outer_col_idx = outer_schema
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(outer_col_name))
                .ok_or_else(|| {
                    format!(
                        "Outer column '{}' not found in outer schema",
                        outer_col_name
                    )
                })?;

            let param = Rc::new(RefCell::new(None));
            let phys_comparison = match comp_op {
                rook_ast::ComparisonOp::Eq => ComparisonOp::Equals,
                rook_ast::ComparisonOp::Ne => ComparisonOp::NotEquals,
                rook_ast::ComparisonOp::Lt => ComparisonOp::LessThan,
                rook_ast::ComparisonOp::Le => ComparisonOp::LessOrEqual,
                rook_ast::ComparisonOp::Gt => ComparisonOp::GreaterThan,
                rook_ast::ComparisonOp::Ge => ComparisonOp::GreaterOrEqual,
            };

            let col_pred = Predicate::Compare(
                Expr::Column { table: None, column: inner_col_name.clone() },
                phys_comparison,
                Expr::CorrelatedParam(param.clone()),
            );

            filter_predicate = Some(match filter_predicate {
                None => col_pred,
                Some(acc) => Predicate::and(acc, col_pred),
            });

            params.push(param);
            outer_indices.push(outer_col_idx);

            log::info!(
                "[Planner] Correlated IN subquery: inner col '{}' (idx={}) = outer.col '{}' (idx={})",
                inner_col_name, _inner_col_idx, outer_col_name, outer_col_idx
            );
        }

        let filter_op: Box<dyn PhysicalOperator> = Box::new(FilterOperator::new(
            inner_base,
            filter_predicate.unwrap(),
        ));
        let inner_plan_rc = Rc::new(RefCell::new(filter_op));

        Ok(Predicate::CorrelatedInSubquery {
            inner_plan: inner_plan_rc,
            params,
            outer_col_indices: outer_indices,
            lhs_expr,
            negated,
        })
    }

    /// Extract all (inner_column_name, outer_column_name, comparison_op) pairs
    /// from a predicate that compares inner-table columns to outer-table columns.
    ///
    /// Walks the predicate tree, collecting ALL Compare nodes that reference
    /// both an inner and an outer column. For AND-connected comparisons, all
    /// pairs are returned. For OR-connected comparisons, returns the first found
    /// (simpler single-column EXISTS fallback).
    fn extract_all_correlations(
        &self,
        node: &rook_ast::PredicateNode,
        inner_table_name: &str,
        inner_col_names: &[String],
    ) -> Result<Vec<(String, String, rook_ast::ComparisonOp)>, String> {
        let mut pairs = Vec::new();
        self.collect_correlations(node, inner_table_name, inner_col_names, &mut pairs)?;
        if pairs.is_empty() {
            return Err(
                "No correlation found in subquery predicate".to_string()
            );
        }
        Ok(pairs)
    }

    /// Recursively collect correlation pairs from a predicate tree.
    fn collect_correlations(
        &self,
        node: &rook_ast::PredicateNode,
        inner_table_name: &str,
        inner_col_names: &[String],
        pairs: &mut Vec<(String, String, rook_ast::ComparisonOp)>,
    ) -> Result<(), String> {
        match node {
            rook_ast::PredicateNode::Compare { left, op, right } => {
                if let Ok((inner_name, outer_name)) =
                    self.resolve_inner_outer_columns(left, right, inner_table_name, inner_col_names)
                {
                    pairs.push((inner_name, outer_name, *op));
                }
                // If not a valid correlation, just skip it (not an error)
                Ok(())
            }
            rook_ast::PredicateNode::BinaryOp {
                left, op, right
            } => {
                match op {
                    rook_ast::BinaryOp::And => {
                        // AND: collect correlations from both sides
                        self.collect_correlations(left, inner_table_name, inner_col_names, pairs)?;
                        self.collect_correlations(right, inner_table_name, inner_col_names, pairs)?;
                        Ok(())
                    }
                    rook_ast::BinaryOp::Or => {
                        // OR: complex — just try to extract single from either side
                        if let Ok(mut single_pairs) = self.extract_all_correlations(
                            left,
                            inner_table_name,
                            inner_col_names,
                        ) {
                            if !single_pairs.is_empty() {
                                pairs.append(&mut single_pairs);
                                return Ok(());
                            }
                        }
                        if let Ok(mut single_pairs) = self.extract_all_correlations(
                            right,
                            inner_table_name,
                            inner_col_names,
                        ) {
                            pairs.append(&mut single_pairs);
                        }
                        Ok(())
                    }
                }
            }
            _ => Ok(()), // Skip non-comparison predicates
        }
    }

    /// Legacy wrapper: extract a single (inner_column_name, outer_column_name, comparison_op)
    /// from the first valid correlation found.
    #[allow(dead_code)]
    fn extract_correlation_from_predicate(
        &self,
        node: &rook_ast::PredicateNode,
        inner_table_name: &str,
        inner_col_names: &[String],
    ) -> Result<(String, String, rook_ast::ComparisonOp), String> {
        let pairs = self.extract_all_correlations(node, inner_table_name, inner_col_names)?;
        if pairs.is_empty() {
            return Err(
                "Correlated subquery predicate must contain a column comparison"
                    .to_string(),
            );
        }
        Ok(pairs[0].clone())
    }

    /// Figure out which of `left` / `right` is the inner column and which is
    /// the outer column reference.
    fn resolve_inner_outer_columns(
        &self,
        left: &rook_ast::ExprNode,
        right: &rook_ast::ExprNode,
        inner_table_name: &str,
        inner_col_names: &[String],
    ) -> Result<(String, String), String> {
        // Check (left = inner, right = outer)
        if let Some(inner_name) = self.extract_column_name_if_inner(left, inner_table_name, inner_col_names)
        {
            if let Some(outer_name) =
                self.extract_column_name_if_outer(right, inner_table_name, inner_col_names)
            {
                return Ok((inner_name, outer_name));
            }
        }

        // Check (left = outer, right = inner)
        if let Some(inner_name) =
            self.extract_column_name_if_inner(right, inner_table_name, inner_col_names)
        {
            if let Some(outer_name) =
                self.extract_column_name_if_outer(left, inner_table_name, inner_col_names)
            {
                return Ok((inner_name, outer_name));
            }
        }

        Err(
            "Could not resolve inner/outer columns in correlated subquery predicate"
                .to_string(),
        )
    }

    /// If `expr` references an inner table column, return the column name.
    fn extract_column_name_if_inner<'a>(
        &self,
        expr: &'a rook_ast::ExprNode,
        inner_table_name: &str,
        inner_col_names: &[String],
    ) -> Option<String> {
        match expr {
            rook_ast::ExprNode::Compound(parts) if parts.len() >= 2 => {
                if parts[0].eq_ignore_ascii_case(inner_table_name) {
                    return Some(parts[1].clone());
                }
                None
            }
            rook_ast::ExprNode::Column(name) => {
                // Unqualified column — check if it exists in inner table
                if inner_col_names
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(name))
                {
                    Some(name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// If `expr` references an outer table column, return the column name
    /// (the last component of a Compound, or the Column name itself).
    fn extract_column_name_if_outer<'a>(
        &self,
        expr: &'a rook_ast::ExprNode,
        inner_table_name: &str,
        inner_col_names: &[String],
    ) -> Option<String> {
        match expr {
            rook_ast::ExprNode::Compound(parts) if parts.len() >= 2 => {
                // Outer reference: table prefix does NOT match inner table
                if !parts[0].eq_ignore_ascii_case(inner_table_name) {
                    Some(parts[parts.len() - 1].clone())
                } else {
                    None
                }
            }
            rook_ast::ExprNode::Column(name) => {
                // Unqualified column — outer if it's NOT in the inner table
                if !inner_col_names
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(name))
                {
                    Some(name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
