//! Volcano-style physical execution engine.
//!
//! Converts a `LogicalPlan` into a tree of `PhysicalOperator` nodes and
//! drives them in a pull-based (Volcano) model. Each operator implements
//! `next() -> Option<Result<Tuple>>` and the root of the tree produces
//! the final result set as an iterator of `Tuple` values.
//!
//! # Architecture
//!
//! ```text
//!            ┌──────────────┐
//!            │  Display /   │
//!            │  CLI output  │
//!            └──────┬───────┘
//!                   │ next()
//!            ┌──────▼───────┐
//!            │  Sort /      │
//!            │  Distinct /  │
//!            │  Limit /     │
//!            │  Project     │
//!            └──────┬───────┘
//!                   │ next()
//!            ┌──────▼───────┐
//!            │   Filter     │
//!            └──────┬───────┘
//!                   │ next()
//!            ┌──────▼───────┐
//!            │   SeqScan    │
//!            │  (heap file) │
//!            └──────────────┘
//! ```

pub mod tuple;
pub mod expr;
pub mod operators;
pub mod planner;
pub mod engine;
pub mod external_sort;

pub use tuple::{ColumnInfo, Tuple, display_tuples, format_tuple,
                serialize_tuple_to_bytes, deserialize_tuple_from_bytes};
pub use expr::{Expr, Predicate, ComparisonOp, evaluate_predicate};
pub use operators::{PhysicalOperator, SeqScanOperator, FilterOperator, ProjectionOperator,
                    LimitOperator, DistinctOperator, SortOperator, NullOperator, SingleRowOperator,
                    CteScanOperator};
pub use external_sort::{ExternalSortOperator, ExternalSortConfig};
pub use planner::PhysicalPlanner;
pub use engine::{execute_plan, execute_plan_collect};
