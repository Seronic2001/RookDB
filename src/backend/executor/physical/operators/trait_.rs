use crate::backend::error::{RookError, RookResult};
use super::super::tuple::{Tuple, ColumnInfo};

/// Default batch size for vectorized execution.
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// A Volcano-style physical operator that produces tuples lazily.
///
/// Call `next()` repeatedly to get the result stream. Returns `Ok(None)` when
/// the stream is exhausted. Operators may also implement `next_batch` to stream
/// chunks of tuples with reduced virtual-dispatch overhead.
pub trait PhysicalOperator {
    /// Produce the next tuple, if any.
    fn next(&mut self) -> RookResult<Option<Tuple>>;

    /// Pull the next batch of tuples into `batch`.
    ///
    /// Clears `batch` before populating. Returns the number of tuples pulled.
    /// Default implementation calls `next()` repeatedly up to `DEFAULT_BATCH_SIZE`.
    fn next_batch(&mut self, batch: &mut Vec<Tuple>) -> RookResult<usize> {
        batch.clear();
        while batch.len() < DEFAULT_BATCH_SIZE {
            match self.next()? {
                Some(tuple) => batch.push(tuple),
                None => break,
            }
        }
        Ok(batch.len())
    }

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

    /// Physical ordering of output tuples: list of (column_index, is_descending).
    /// Returns `None` if output is unordered or ordering is unknown.
    fn ordering(&self) -> Option<Vec<(usize, bool)>> {
        None
    }
}

