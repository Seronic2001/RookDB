//! PhysicalOperator trait and the core operator implementations.
//!
//! Each operator implements the Volcano pull-based `next()` protocol and
//! wraps zero or more child operators. Operators form a tree where the root
//! produces the final result set.
//!
//! Current operators:
//! - `trait_.rs`: PhysicalOperator trait definition
//! - `scans.rs`: SeqScanOperator (heap scan)
//! - `filter_project.rs`: Filter, Projection, Limit, Distinct, SingleRow,
//!   Null and CteScan operators
//! - `sort.rs`: SortOperator
//! - `utils.rs`: normalise_value_for_key
//! - `tests.rs`: unit tests (mock operator + operator behaviour)
//!
//! JOIN / aggregate / set-operation / subquery operators arrive with the
//! advanced-operators stage.

mod trait_;
mod scans;
mod filter_project;
mod sort;
mod utils;

#[cfg(test)]
mod tests;

pub use trait_::PhysicalOperator;
pub use scans::SeqScanOperator;

pub use filter_project::{
    SingleRowOperator, NullOperator, CteScanOperator,
    FilterOperator, ProjectionOperator, LimitOperator, DistinctOperator,
};
pub use sort::SortOperator;
pub use utils::normalise_value_for_key;
