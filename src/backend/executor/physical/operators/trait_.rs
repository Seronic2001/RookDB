use crate::backend::error::{RookError, RookResult};
use super::super::tuple::{Tuple, ColumnInfo};

/// A Volcano-style physical operator that produces tuples lazily.
///
/// Call `next()` repeatedly to get the result stream. Returns `Ok(None)` when
/// the stream is exhausted.
pub trait PhysicalOperator {
    /// Produce the next tuple, if any.
    fn next(&mut self) -> RookResult<Option<Tuple>>;

    /// Return the schema (column metadata) this operator produces.
    fn schema(&self) -> &[ColumnInfo];

    /// Reset the operator so it can be iterated again (if supported).
    fn reset(&mut self) -> RookResult<()> {
        Err(RookError::Internal("reset() not supported".to_string()))
    }

    /// Estimate how many tuples this operator will produce (0 = unknown).
    fn estimate_cardinality(&self) -> usize { 0 }

    /// A human-readable name for the operator (for debug printing).
    fn name(&self) -> &'static str;
}
