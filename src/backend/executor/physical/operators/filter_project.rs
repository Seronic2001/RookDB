use std::collections::HashSet;

use super::super::tuple::{Tuple, ColumnInfo};
use super::super::expr::{Expr, Predicate, evaluate_predicate};
use super::trait_::PhysicalOperator;

use crate::types::datatype::DataType;

// ── SingleRowOperator ────────────────────────────────────────────────────────

/// An operator that produces exactly one empty tuple (zero-column row).
pub struct SingleRowOperator {
    schema: Vec<ColumnInfo>,
    emitted: bool,
}

impl SingleRowOperator {
    pub fn new() -> Self {
        Self {
            schema: Vec::new(),
            emitted: false,
        }
    }
}

impl PhysicalOperator for SingleRowOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(Some(Tuple::new(Vec::new())))
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.emitted = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SingleRow"
    }
}

// ── Null Source ───────────────────────────────────────────────────────────────

/// An operator that produces no tuples — used as a placeholder.
pub struct NullOperator {
    schema: Vec<ColumnInfo>,
}

impl NullOperator {
    pub fn new(schema: Vec<ColumnInfo>) -> Self {
        Self { schema }
    }
}

impl PhysicalOperator for NullOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        Ok(None)
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn name(&self) -> &'static str {
        "Null"
    }
}

// ── CteScan Operator ──────────────────────────────────────────────────────────

/// Scans over a pre-materialised CTE result set stored in memory.
pub struct CteScanOperator {
    /// All tuples produced by the CTE, in evaluation order.
    tuples: Vec<Tuple>,
    /// Current read position.
    pos: usize,
    /// Schema forwarded from the CTE's output.
    schema: Vec<ColumnInfo>,
}

impl CteScanOperator {
    /// Create a new CTE scan operator.
    pub fn new(tuples: Vec<Tuple>, schema: Vec<ColumnInfo>) -> Self {
        Self { tuples, pos: 0, schema }
    }
}

impl PhysicalOperator for CteScanOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.pos >= self.tuples.len() {
            return Ok(None);
        }
        let tuple = self.tuples[self.pos].clone();
        self.pos += 1;
        Ok(Some(tuple))
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.pos = 0;
        Ok(())
    }

    fn estimate_cardinality(&self) -> usize {
        self.tuples.len()
    }

    fn name(&self) -> &'static str {
        "CteScan"
    }
}

// ── Filter Operator ───────────────────────────────────────────────────────────

/// Filters tuples from a child operator by evaluating a predicate.
pub struct FilterOperator {
    child: Box<dyn PhysicalOperator>,
    predicate: Predicate,
}

impl FilterOperator {
    pub fn new(child: Box<dyn PhysicalOperator>, predicate: Predicate) -> Self {
        Self { child, predicate }
    }
}

impl PhysicalOperator for FilterOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        loop {
            match self.child.next()? {
                Some(tuple) => {
                    match evaluate_predicate(&self.predicate, &tuple, self.child.schema())? {
                        Some(true) => return Ok(Some(tuple)),
                        _ => continue,
                    }
                }
                None => return Ok(None),
            }
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        self.child.schema()
    }

    fn reset(&mut self) -> Result<(), String> {
        self.child.reset()
    }

    fn estimate_cardinality(&self) -> usize {
        self.child.estimate_cardinality()
    }

    fn name(&self) -> &'static str {
        "Filter"
    }
}

// ── Projection Operator ───────────────────────────────────────────────────────

/// Evaluates expressions to produce projected output tuples.
pub struct ProjectionOperator {
    child: Box<dyn PhysicalOperator>,
    projections: Vec<(Expr, String, DataType)>,
    output_schema: Vec<ColumnInfo>,
}

impl ProjectionOperator {
    pub fn new(
        child: Box<dyn PhysicalOperator>,
        projections: Vec<(Expr, String, DataType)>,
    ) -> Self {
        let output_schema: Vec<ColumnInfo> = projections.iter()
            .map(|(_, name, dt)| ColumnInfo {
                name: name.clone(),
                data_type: dt.clone(),
                table: None,
            })
            .collect();
        Self { child, projections, output_schema }
    }

    pub fn new_with_mapping(
        child: Box<dyn PhysicalOperator>,
        indices: &[usize],
        names: &[String],
    ) -> Result<Self, String> {
        let child_schema = child.schema();
        let mut projections = Vec::with_capacity(indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            let name = names.get(i).cloned().unwrap_or_else(|| {
                child_schema.get(idx).map(|c| c.name.clone()).unwrap_or_else(|| format!("col{}", idx))
            });
            let dt = child_schema.get(idx).map(|c| c.data_type.clone())
                .ok_or_else(|| format!("Column index {} out of bounds", idx))?;
            projections.push((Expr::Column { table: None, column: name.clone() }, name, dt));
        }
        Ok(Self::new(child, projections))
    }

    pub fn from_indices(
        child: Box<dyn PhysicalOperator>,
        indices: &[usize],
        names: &[String],
    ) -> Result<Self, String> {
        Self::new_with_mapping(child, indices, names)
    }

    pub fn star(child: Box<dyn PhysicalOperator>) -> Self {
        let child_schema = child.schema().to_vec();
        let projections: Vec<(Expr, String, DataType)> = child_schema.iter().enumerate()
            .map(|(_i, ci)| (Expr::Column { table: None, column: ci.name.clone() }, ci.name.clone(), ci.data_type.clone()))
            .collect();
        let output_schema = child_schema;
        Self { child, projections, output_schema }
    }
}

impl PhysicalOperator for ProjectionOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        match self.child.next()? {
            Some(child_tuple) => {
                let mut values = Vec::with_capacity(self.projections.len());
                for (expr, _, _) in &self.projections {
                    let val = expr.evaluate(&child_tuple, self.child.schema())?;
                    values.push(val);
                }
                // Each output row derives from exactly one input row: carry
                // its heap location through so pointer-based UPDATE/DELETE
                // can still address the row (a synthetic source — aggregate
                // output — stays location-less).
                Ok(Some(Tuple::new(values)
                    .with_location_from(&child_tuple)))
            }
            None => Ok(None),
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.output_schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.child.reset()
    }

    fn name(&self) -> &'static str {
        "Project"
    }
}

// ── Limit Operator ────────────────────────────────────────────────────────────

/// Limits the number of tuples produced (SQL `LIMIT`).
pub struct LimitOperator {
    child: Box<dyn PhysicalOperator>,
    limit: usize,
    offset: usize,
    emitted: usize,
    skipped: usize,
}

impl LimitOperator {
    pub fn new(child: Box<dyn PhysicalOperator>, limit: usize, offset: usize) -> Self {
        Self { child, limit, offset, emitted: 0, skipped: 0 }
    }
}

impl PhysicalOperator for LimitOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        while self.skipped < self.offset {
            match self.child.next()? {
                Some(_) => self.skipped += 1,
                None => return Ok(None),
            }
        }

        if self.emitted >= self.limit {
            return Ok(None);
        }

        match self.child.next()? {
            Some(tuple) => {
                self.emitted += 1;
                Ok(Some(tuple))
            }
            None => Ok(None),
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        self.child.schema()
    }

    fn reset(&mut self) -> Result<(), String> {
        self.child.reset()?;
        self.emitted = 0;
        self.skipped = 0;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Limit"
    }
}

// ── Distinct Operator ─────────────────────────────────────────────────────────

/// Removes duplicate tuples (SQL `DISTINCT`).
pub struct DistinctOperator {
    child: Box<dyn PhysicalOperator>,
    seen: HashSet<String>,
    loaded: bool,
    buffer: Vec<Tuple>,
    pos: usize,
}

impl DistinctOperator {
    pub fn new(child: Box<dyn PhysicalOperator>) -> Self {
        Self {
            child,
            seen: HashSet::new(),
            loaded: false,
            buffer: Vec::new(),
            pos: 0,
        }
    }
}

impl PhysicalOperator for DistinctOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if !self.loaded {
            while let Some(tuple) = self.child.next()? {
                let key = tuple.values.iter().map(|v| match v {
                    Some(dv) => format!("{:?}", dv),
                    None => "__NULL__".to_string(),
                }).collect::<Vec<_>>().join("|");
                if self.seen.insert(key) {
                    self.buffer.push(tuple);
                }
            }
            self.loaded = true;
        }

        if self.pos < self.buffer.len() {
            let tuple = self.buffer[self.pos].clone();
            self.pos += 1;
            Ok(Some(tuple))
        } else {
            Ok(None)
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        self.child.schema()
    }

    fn reset(&mut self) -> Result<(), String> {
        self.child.reset()?;
        self.seen.clear();
        self.buffer.clear();
        self.pos = 0;
        self.loaded = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Distinct"
    }
}
