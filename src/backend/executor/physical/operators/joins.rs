use std::collections::HashMap;

use super::super::tuple::{Tuple, ColumnInfo};
use super::super::expr::{Expr, Predicate, evaluate_predicate};
use super::trait_::PhysicalOperator;
use super::utils::normalise_value_for_key;

use crate::types::value::DataValue;

// ── Join Type ─────────────────────────────────────────────────────────────────

/// Join type matching SQL semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

// ── NestedLoopJoin Operator ───────────────────────────────────────────────────

/// A nested-loop join operator that works for any join predicate.
///
/// For each tuple from the left (outer) child, scans all tuples from the
/// right (inner) child and evaluates the join predicate on each pair.
/// Tuples that match the predicate are concatenated and yielded.
///
/// Supports: INNER, LEFT, RIGHT, FULL, CROSS join types.
/// For CROSS joins (no predicate), all pairs match.
pub struct NestedLoopJoinOperator {
    left: Box<dyn PhysicalOperator>,
    right: Box<dyn PhysicalOperator>,
    /// The join predicate (None for CROSS JOIN).
    predicate: Option<Predicate>,
    /// The join type.
    join_type: JoinType,
    /// Output schema (left columns + right columns).
    output_schema: Vec<ColumnInfo>,

    // ── Runtime state ──
    /// All tuples from the left child (materialised on first pass).
    left_tuples: Vec<Tuple>,
    /// All tuples from the right child (materialised on first pass).
    right_tuples: Vec<Tuple>,
    /// Which left tuples have found at least one match (for LEFT/FULL OUTER).
    left_matched: Vec<bool>,
    /// Which right tuples have found at least one match (for RIGHT/FULL OUTER).
    right_matched: Vec<bool>,
    /// Output buffer of result tuples.
    output_buffer: Vec<Tuple>,
    /// Current read position in output_buffer.
    output_pos: usize,
    /// Whether children have been consumed.
    consumed: bool,
}

impl NestedLoopJoinOperator {
    /// Create a new nested-loop join operator.
    ///
    /// `left` and `right` are the child operators.
    /// `predicate` is the join condition (None for CROSS JOIN).
    /// `join_type` controls outer join null-extension behaviour.
    pub fn new(
        left: Box<dyn PhysicalOperator>,
        right: Box<dyn PhysicalOperator>,
        predicate: Option<Predicate>,
        join_type: JoinType,
    ) -> Self {
        // Build output schema: left columns + right columns.
        // ColumnInfo.table is populated with actual table names by the scan operators,
        // so table-qualified references like `t1.id` vs `t2.id` resolve correctly
        // without sentinel markers. The planner also uses the original column info
        // when building the combined schema for predicate resolution.
        let mut output_schema = left.schema().to_vec();
        output_schema.extend(right.schema().iter().cloned());

        Self {
            left,
            right,
            predicate,
            join_type,
            output_schema,
            left_tuples: Vec::new(),
            right_tuples: Vec::new(),
            left_matched: Vec::new(),
            right_matched: Vec::new(),
            output_buffer: Vec::new(),
            output_pos: 0,
            consumed: false,
        }
    }

    /// Materialise all tuples from both children, compute the join, and buffer results.
    fn materialise(&mut self) -> Result<(), String> {
        // Consume both children
        self.left_tuples.clear();
        self.right_tuples.clear();
        while let Some(t) = self.left.next()? {
            self.left_tuples.push(t);
        }
        while let Some(t) = self.right.next()? {
            self.right_tuples.push(t);
        }

        let num_left = self.left_tuples.len();
        let num_right = self.right_tuples.len();

        // Initialise match tracking for outer joins
        self.left_matched = vec![false; num_left];
        self.right_matched = vec![false; num_right];

        // Helper: create a joined tuple with the operator's output_schema
        // (which preserves actual table names from child operators).
        let join_tuples = |left: &Tuple, right: &Tuple| -> Tuple {
            let mut t = left.concatenate(right);
            t.column_info = self.output_schema.clone();
            t
        };

        // Compute the join
        if self.predicate.is_none() || self.join_type == JoinType::Cross {
            // CROSS JOIN (no predicate) or predicate is None
            for l_idx in 0..num_left {
                for r_idx in 0..num_right {
                    let joined = join_tuples(&self.left_tuples[l_idx], &self.right_tuples[r_idx]);
                    self.output_buffer.push(joined);
                    self.left_matched[l_idx] = true;
                    self.right_matched[r_idx] = true;
                }
            }
        } else {
            let pred = self.predicate.as_ref().unwrap();
            for l_idx in 0..num_left {
                for r_idx in 0..num_right {
                    let joined = join_tuples(&self.left_tuples[l_idx], &self.right_tuples[r_idx]);
                    match evaluate_predicate(pred, &joined)? {
                        Some(true) => {
                            self.output_buffer.push(joined);
                            self.left_matched[l_idx] = true;
                            self.right_matched[r_idx] = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Handle outer join NULL-extensions
        let left_cols = self.left.schema().to_vec();
        let right_cols = self.right.schema().to_vec();

        match self.join_type {
            JoinType::Left | JoinType::Full => {
                // Emit NULL-extended rows for unmatched left tuples
                for (l_idx, &matched) in self.left_matched.iter().enumerate() {
                    if !matched {
                        let null_values: Vec<Option<DataValue>> =
                            right_cols.iter().map(|_| None).collect();
                        let null_right = Tuple::new(null_values, right_cols.clone());
                        self.output_buffer
                            .push(join_tuples(&self.left_tuples[l_idx], &null_right));
                    }
                }
            }
            _ => {}
        }

        match self.join_type {
            JoinType::Right | JoinType::Full => {
                // Emit NULL-extended rows for unmatched right tuples
                for (r_idx, &matched) in self.right_matched.iter().enumerate() {
                    if !matched {
                        let null_values: Vec<Option<DataValue>> =
                            left_cols.iter().map(|_| None).collect();
                        let null_left = Tuple::new(null_values, left_cols.clone());
                        self.output_buffer
                            .push(join_tuples(&null_left, &self.right_tuples[r_idx]));
                    }
                }
            }
            _ => {}
        }

        self.consumed = true;
        Ok(())
    }
}

impl PhysicalOperator for NestedLoopJoinOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if !self.consumed {
            self.materialise()?;
        }

        if self.output_pos < self.output_buffer.len() {
            let tuple = self.output_buffer[self.output_pos].clone();
            self.output_pos += 1;
            Ok(Some(tuple))
        } else {
            Ok(None)
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.output_schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.left.reset()?;
        self.right.reset()?;
        self.left_tuples.clear();
        self.right_tuples.clear();
        self.left_matched.clear();
        self.right_matched.clear();
        self.output_buffer.clear();
        self.output_pos = 0;
        self.consumed = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }
}

// ── HashJoin Operator ─────────────────────────────────────────────────────────

/// A hash join operator for equality joins.
///
/// Builds a hash table on the left (build) side, then probes with the
/// right (probe) side. Only supports INNER joins with equality conditions.
///
/// For non-equality conditions or outer joins, use NestedLoopJoinOperator.
pub struct HashJoinOperator {
    build: Box<dyn PhysicalOperator>,
    probe: Box<dyn PhysicalOperator>,
    /// Expressions to hash on the build side.
    build_keys: Vec<Expr>,
    /// Expressions to hash on the probe side.
    probe_keys: Vec<Expr>,
    /// Any remaining predicate (non-equality conditions that must also be satisfied).
    remaining_predicate: Option<Predicate>,
    /// Output schema.
    output_schema: Vec<ColumnInfo>,

    // ── Runtime state ──
    /// Hash table: key string → list of build tuples.
    hash_table: HashMap<String, Vec<Tuple>>,
    /// All probe tuples (for replay during reset).
    probe_tuples: Vec<Tuple>,
    /// Current position in probe_tuples.
    probe_pos: usize,
    /// Current batch of matching build tuples for the current probe tuple.
    current_matches: Vec<Tuple>,
    /// Current position within current_matches.
    match_pos: usize,
    /// Whether build phase is complete.
    build_done: bool,
    /// Whether all output has been consumed.
    exhausted: bool,
}

impl HashJoinOperator {
    /// Create a new hash join operator.
    ///
    /// `build` and `probe` are the child operators.
    /// `build_keys` are the expressions evaluated on the build side to form the hash key.
    /// `probe_keys` are the expressions evaluated on the probe side to form the hash key.
    /// They must have the same length and corresponding entries must be type-compatible.
    /// `remaining_predicate` is any non-equality condition that must also be satisfied.
    pub fn new(
        build: Box<dyn PhysicalOperator>,
        probe: Box<dyn PhysicalOperator>,
        build_keys: Vec<Expr>,
        probe_keys: Vec<Expr>,
        remaining_predicate: Option<Predicate>,
    ) -> Self {
        let mut output_schema = build.schema().to_vec();
        output_schema.extend(probe.schema().iter().cloned());

        Self {
            build,
            probe,
            build_keys,
            probe_keys,
            remaining_predicate,
            output_schema,
            hash_table: HashMap::new(),
            probe_tuples: Vec::new(),
            probe_pos: 0,
            current_matches: Vec::new(),
            match_pos: 0,
            build_done: false,
            exhausted: false,
        }
    }

    /// Compute the hash key string from a tuple using a set of key expressions.
    /// Returns `None` if any key is NULL (meaning this tuple cannot match in an
    /// INNER hash join, since SQL NULL != NULL).
    fn make_hash_key(keys: &[Expr], tuple: &Tuple) -> Result<Option<String>, String> {
        if keys.is_empty() {
            return Ok(Some("__global__".to_string()));
        }
        let mut parts = Vec::with_capacity(keys.len());
        for expr in keys {
            let val = expr.evaluate(tuple)?;
            match val {
                Some(dv) => parts.push(normalise_value_for_key(&dv)),
                None => return Ok(None),  // NULL key → can never match (NULL != NULL)
            }
        }
        Ok(Some(parts.join("|")))
    }

    /// Build the hash table from the build side.
    /// Tuples with NULL join keys are skipped (they can never match in SQL).
    fn build_hash_table(&mut self) -> Result<(), String> {
        while let Some(tuple) = self.build.next()? {
            if let Some(key) = Self::make_hash_key(&self.build_keys, &tuple)? {
                self.hash_table.entry(key).or_insert_with(Vec::new).push(tuple);
            }
            // NULL-keyed tuples are skipped — they can never match (NULL != NULL)
        }
        self.build_done = true;
        Ok(())
    }

    /// Consume all probe tuples (needed for reset support).
    fn load_probe(&mut self) -> Result<(), String> {
        while let Some(tuple) = self.probe.next()? {
            self.probe_tuples.push(tuple);
        }
        Ok(())
    }
}

impl PhysicalOperator for HashJoinOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        // Build phase
        if !self.build_done {
            self.build_hash_table()?;
            self.load_probe()?;
        }

        if self.exhausted {
            return Ok(None);
        }

        // Try to yield from current batch of matches
        if self.match_pos < self.current_matches.len() {
            let build_tuple = &self.current_matches[self.match_pos];
            self.match_pos += 1;
            let mut joined = build_tuple.concatenate(&self.probe_tuples[self.probe_pos - 1]);
            // Explicitly set column_info to match output_schema for consistency
            joined.column_info = self.output_schema.clone();
            return Ok(Some(joined));
        }

        // Find the next probe tuple that has matches
        while self.probe_pos < self.probe_tuples.len() {
            let probe_tuple = &self.probe_tuples[self.probe_pos];
            self.probe_pos += 1;

            let key = match Self::make_hash_key(&self.probe_keys, probe_tuple)? {
                Some(k) => k,
                None => continue,  // NULL probe key → skip (NULL != NULL)
            };

            if let Some(build_matches) = self.hash_table.get(&key) {
                // Check remaining predicate for each match
                self.current_matches.clear();
                for build_tuple in build_matches {
                    if let Some(ref remaining) = self.remaining_predicate {
                        let joined = build_tuple.concatenate(probe_tuple);
                        match evaluate_predicate(remaining, &joined)? {
                            Some(true) => self.current_matches.push(build_tuple.clone()),
                            _ => {}
                        }
                    } else {
                        self.current_matches.push(build_tuple.clone());
                    }
                }

                if !self.current_matches.is_empty() {
                    self.match_pos = 1;
                    let build_tuple = &self.current_matches[0];
                    let mut joined = build_tuple.concatenate(probe_tuple);
                    joined.column_info = self.output_schema.clone();
                    return Ok(Some(joined));
                }
            }
            // No matches — skip this probe tuple (INNER JOIN semantics)
        }

        self.exhausted = true;
        Ok(None)
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.output_schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.build.reset()?;
        self.probe.reset()?;
        self.hash_table.clear();
        self.probe_tuples.clear();
        self.probe_pos = 0;
        self.current_matches.clear();
        self.match_pos = 0;
        self.build_done = false;
        self.exhausted = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "HashJoin"
    }
}
