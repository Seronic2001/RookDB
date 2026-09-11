//! PhysicalOperator trait and all concrete operator implementations.
//!
//! Each operator implements the Volcano pull-based `next()` protocol and
//! wraps zero or more child operators. Operators form a tree where the root
//! produces the final result set.
//!
//! This module is split into per-category files for maintainability:
//! - `trait_.rs`: PhysicalOperator trait definition
//! - `scans.rs`: SeqScanOperator, IndexScanOperator, IndexScanMode
//! - `joins.rs`: NestedLoopJoinOperator, HashJoinOperator, JoinType
//! - `filter_project.rs`: Filter, Projection, Limit, Distinct, SingleRow,
//!   Null and CteScan operators
//! - `sort.rs`: SortOperator
//! - `aggregate.rs`: AggregateOperator with GROUP BY / HAVING support
//! - `set_op.rs`: `UNION [ALL] / INTERSECT [ALL] / EXCEPT [ALL]`
//! - `subquery.rs`: scalar and EXISTS subquery execution
//! - `insert.rs`: INSERT INTO ... SELECT pipeline operator
//! - `utils.rs`: normalise_value_for_key
//! - `tests.rs`: unit tests (mock operator + operator behaviour)

mod trait_;
mod scans;
mod joins;
mod filter_project;
mod sort;
mod aggregate;
mod set_op;
mod subquery;
mod insert;
mod utils;

#[cfg(test)]
mod tests;

// Re-export everything from submodules
pub use trait_::{PhysicalOperator, DEFAULT_BATCH_SIZE};
pub use scans::{SeqScanOperator, IndexScanOperator, IndexScanMode};
pub use joins::{NestedLoopJoinOperator, HashJoinOperator, JoinType};
pub use insert::{InsertOperator, ValuesOperator};

pub use filter_project::{
    SingleRowOperator, NullOperator, CteScanOperator,
    FilterOperator, ProjectionOperator, LimitOperator, DistinctOperator,
};
pub use sort::SortOperator;
pub use aggregate::{
    AggregateOperator, AggregateFunction, AggregateInfo, PerGroupState,
    infer_aggregate_output_type,
};
pub use set_op::{SetOpOperator, SetOpType};
pub use subquery::{SubqueryExecOperator, SubqueryType};
pub use utils::normalise_value_for_key;
