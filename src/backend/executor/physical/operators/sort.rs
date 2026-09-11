use std::cmp::Ordering;

use crate::backend::error::RookResult;
use super::super::tuple::{Tuple, ColumnInfo};
use super::super::external_sort::{ExternalSortOperator, ExternalSortConfig};
use super::trait_::PhysicalOperator;
use super::filter_project::NullOperator;

use crate::types::comparison::compare_nullable;

// ── Sort Operator ─────────────────────────────────────────────────────────────

/// Sorts all tuples from the child by the specified columns.
///
/// Uses in-memory sort for small datasets and transparently switches to
/// external merge sort (spilling to temp files) for larger datasets.
pub struct SortOperator {
    child: Box<dyn PhysicalOperator>,
    /// Sort columns: (column_index, descending).
    sort_keys: Vec<(usize, bool)>,
    /// Whether to use external merge sort.
    use_external: bool,
    /// External sort operator (lazily initialised).
    external: Option<ExternalSortOperator>,
    /// Whether the child has been fully consumed (in-memory mode only).
    loaded: bool,
    /// Sorted buffer of tuples (in-memory mode).
    buffer: Vec<Tuple>,
    /// Current position in buffer.
    pos: usize,
    /// Threshold for switching to external sort (0 = always in-memory).
    external_threshold: usize,
    /// Cached schema (survives child replacement during external init).
    cached_schema: Vec<ColumnInfo>,
}

impl SortOperator {
    /// Create a new in-memory-only sort operator.
    pub fn new(child: Box<dyn PhysicalOperator>, sort_keys: Vec<(usize, bool)>) -> Self {
        let cached_schema = child.schema().to_vec();
        Self {
            child,
            sort_keys,
            use_external: false,
            external: None,
            loaded: false,
            buffer: Vec::new(),
            pos: 0,
            external_threshold: 0,
            cached_schema,
        }
    }

    /// Create an adaptive sort operator that switches to external merge sort
    /// when the estimated cardinality exceeds `threshold`.
    pub fn new_adaptive(
        child: Box<dyn PhysicalOperator>,
        sort_keys: Vec<(usize, bool)>,
        threshold: usize,
    ) -> Self {
        let cached_schema = child.schema().to_vec();
        let estimated = child.estimate_cardinality();
        let use_external = threshold > 0 && estimated > threshold;
        log::info!(
            "[Sort] Adaptive mode: estimated={}, threshold={}, use_external={}",
            estimated,
            threshold,
            use_external
        );
        Self {
            child,
            sort_keys,
            use_external,
            external: None,
            loaded: false,
            buffer: Vec::new(),
            pos: 0,
            external_threshold: threshold,
            cached_schema,
        }
    }

    /// Get or initialise the external sort operator.
    fn init_external(&mut self) -> RookResult<&mut ExternalSortOperator> {
        if self.external.is_none() {
            log::info!("[Sort] Switching to external merge sort");
            let config = ExternalSortConfig {
                max_tuples_per_run: self.external_threshold.max(100),
                ..Default::default()
            };
            let child = std::mem::replace(&mut self.child, Box::new(NullOperator::new(Vec::new())));
            self.external = Some(ExternalSortOperator::new(
                child,
                self.sort_keys.clone(),
                config,
            ));
        }
        Ok(self.external.as_mut().unwrap())
    }

    /// Ensure the child has been consumed and sorted in memory (or flagged for external).
    fn load_if_needed(&mut self) -> RookResult<()> {
        if self.loaded {
            return Ok(());
        }

        // Consume all tuples from child in batches
        let mut child_batch = Vec::with_capacity(super::trait_::DEFAULT_BATCH_SIZE);
        while self.child.next_batch(&mut child_batch)? > 0 {
            self.buffer.append(&mut child_batch);
        }

        // Check if we should have used external sort
        if self.external_threshold > 0 && self.buffer.len() > self.external_threshold {
            log::info!(
                "[Sort] In-memory sort exceeded threshold ({} > {}), switching to external",
                self.buffer.len(),
                self.external_threshold
            );
            self.use_external = true;
            return Ok(());
        }

        // Sort in place
        self.buffer.sort_by(|a, b| {
            for &(key_idx, descending) in &self.sort_keys {
                let av = a.values.get(key_idx).and_then(|v| v.as_ref());
                let bv = b.values.get(key_idx).and_then(|v| v.as_ref());
                let ordering = match (av, bv) {
                    (Some(a_val), Some(b_val)) => {
                        compare_nullable(Some(a_val), Some(b_val))
                            .unwrap_or(None)
                            .unwrap_or(Ordering::Equal)
                    }
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                };
                let ordering = if descending { ordering.reverse() } else { ordering };
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            Ordering::Equal
        });

        self.loaded = true;
        Ok(())
    }
}

impl PhysicalOperator for SortOperator {
    fn next(&mut self) -> RookResult<Option<Tuple>> {
        if self.use_external {
            let external = self.init_external()?;
            return external.next();
        }

        // ── In-memory path ──
        if !self.loaded {
            self.load_if_needed()?;
            if self.use_external {
                let external = self.init_external()?;
                return external.next();
            }
        }

        if self.pos < self.buffer.len() {
            let tuple = self.buffer[self.pos].clone();
            self.pos += 1;
            Ok(Some(tuple))
        } else {
            Ok(None)
        }
    }

    fn next_batch(&mut self, batch: &mut Vec<Tuple>) -> RookResult<usize> {
        if self.use_external {
            let external = self.init_external()?;
            return external.next_batch(batch);
        }

        if !self.loaded {
            self.load_if_needed()?;
            if self.use_external {
                let external = self.init_external()?;
                return external.next_batch(batch);
            }
        }

        let available = self.buffer.len().saturating_sub(self.pos);
        if available == 0 {
            return Ok(0);
        }
        let take = available.min(super::trait_::DEFAULT_BATCH_SIZE);
        batch.reserve(take);
        for i in 0..take {
            batch.push(self.buffer[self.pos + i].clone());
        }
        self.pos += take;
        Ok(take)
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.cached_schema
    }

    fn reset(&mut self) -> RookResult<()> {
        if self.use_external
            && let Some(ref mut ext) = self.external {
                return ext.reset();
            }
        self.child.reset()?;
        self.loaded = false;
        self.buffer.clear();
        self.pos = 0;
        Ok(())
    }

    fn estimate_cardinality(&self) -> usize {
        self.child.estimate_cardinality()
    }

    fn name(&self) -> &'static str {
        "Sort"
    }
}
