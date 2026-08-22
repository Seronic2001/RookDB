use std::collections::{HashMap, HashSet};

use super::super::tuple::{Tuple, ColumnInfo};
use super::trait_::PhysicalOperator;
use super::utils::normalise_value_for_key;


// ── SetOp Type ────────────────────────────────────────────────────────────────

/// Set operation type matching SQL semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
}

// ── SetOp Operator ────────────────────────────────────────────────────────────

/// A set-operation operator (UNION, INTERSECT, EXCEPT).
///
/// Implements standard SQL set operations between two child operators that
/// produce tuples with identical schemas.
pub struct SetOpOperator {
    left: Box<dyn PhysicalOperator>,
    right: Box<dyn PhysicalOperator>,
    op_type: SetOpType,
    all: bool,
    /// Output schema (from left child; must match right child).
    output_schema: Vec<ColumnInfo>,

    // ── Runtime state ──
    left_tuples: Vec<Tuple>,
    right_tuples: Vec<Tuple>,
    output_buffer: Vec<Tuple>,
    output_pos: usize,
    consumed: bool,
}

impl SetOpOperator {
    /// Create a new set operation operator.
    pub fn new(
        left: Box<dyn PhysicalOperator>,
        right: Box<dyn PhysicalOperator>,
        op_type: SetOpType,
        all: bool,
    ) -> Self {
        let output_schema = left.schema().to_vec();
        Self {
            left,
            right,
            op_type,
            all,
            output_schema,
            left_tuples: Vec::new(),
            right_tuples: Vec::new(),
            output_buffer: Vec::new(),
            output_pos: 0,
            consumed: false,
        }
    }

    /// Serialise a tuple to a string key for hashing/comparison.
    fn tuple_key(tuple: &Tuple) -> String {
        tuple.values.iter().map(|v| match v {
            Some(dv) => normalise_value_for_key(dv),
            None => "\x00N\x00".to_string(),
        }).collect::<Vec<_>>().join("|")
    }

    /// Materialise both children and compute the set operation result.
    fn materialise(&mut self) -> Result<(), String> {
        self.left_tuples.clear();
        self.right_tuples.clear();
        while let Some(t) = self.left.next()? {
            self.left_tuples.push(t);
        }
        while let Some(t) = self.right.next()? {
            self.right_tuples.push(t);
        }

        match (self.op_type, self.all) {
            (SetOpType::Union, true) => {
                self.output_buffer = self.left_tuples.drain(..).collect();
                self.output_buffer.extend(self.right_tuples.drain(..));
            }

            (SetOpType::Union, false) => {
                let mut seen = HashSet::new();
                for tuple in self.left_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if seen.insert(key) {
                        self.output_buffer.push(tuple);
                    }
                }
                for tuple in self.right_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if seen.insert(key) {
                        self.output_buffer.push(tuple);
                    }
                }
            }

            (SetOpType::Intersect, true) => {
                let mut left_counts: HashMap<String, (usize, Tuple)> = HashMap::new();
                let mut order: Vec<String> = Vec::new();
                for tuple in self.left_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    let entry = left_counts.entry(key.clone()).or_insert((0, tuple));
                    entry.0 += 1;
                    if entry.0 == 1 { order.push(key); }
                }

                let mut results: HashMap<String, Vec<Tuple>> = HashMap::new();
                for tuple in self.right_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if let Some((count, _)) = left_counts.get_mut(&key) {
                        if *count > 0 {
                            *count -= 1;
                            results.entry(key.clone()).or_insert_with(Vec::new).push(tuple);
                        }
                    }
                }

                for key in order {
                    if let Some(tuples) = results.remove(&key) {
                        self.output_buffer.extend(tuples);
                    }
                }
            }

            (SetOpType::Intersect, false) => {
                let left_set: HashSet<String> =
                    self.left_tuples.iter().map(Self::tuple_key).collect();
                let mut seen = HashSet::new();
                for tuple in self.right_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if left_set.contains(&key) && seen.insert(key) {
                        self.output_buffer.push(tuple);
                    }
                }
            }

            (SetOpType::Except, true) => {
                let mut left_counts: HashMap<String, (usize, Vec<Tuple>)> = HashMap::new();
                let mut order: Vec<String> = Vec::new();
                for tuple in self.left_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    let entry = left_counts.entry(key.clone()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(tuple);
                    if entry.0 == 1 { order.push(key); }
                }

                for tuple in self.right_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if let Some((count, _)) = left_counts.get_mut(&key) {
                        if *count > 0 { *count -= 1; }
                    }
                }

                for key in order {
                    if let Some((count, tuples)) = left_counts.remove(&key) {
                        for tuple in tuples.into_iter().take(count) {
                            self.output_buffer.push(tuple);
                        }
                    }
                }
            }

            (SetOpType::Except, false) => {
                let right_set: HashSet<String> =
                    self.right_tuples.iter().map(Self::tuple_key).collect();
                let mut seen = HashSet::new();
                for tuple in self.left_tuples.drain(..) {
                    let key = Self::tuple_key(&tuple);
                    if !right_set.contains(&key) && seen.insert(key) {
                        self.output_buffer.push(tuple);
                    }
                }
            }
        }

        self.consumed = true;
        Ok(())
    }
}

impl PhysicalOperator for SetOpOperator {
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
        self.output_buffer.clear();
        self.output_pos = 0;
        self.consumed = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        match (self.op_type, self.all) {
            (SetOpType::Union, true) => "UnionAll",
            (SetOpType::Union, false) => "Union",
            (SetOpType::Intersect, true) => "IntersectAll",
            (SetOpType::Intersect, false) => "Intersect",
            (SetOpType::Except, true) => "ExceptAll",
            (SetOpType::Except, false) => "Except",
        }
    }
}
