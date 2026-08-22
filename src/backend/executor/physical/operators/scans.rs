
use crate::backend::heap::heap_manager::{HeapManager, HeapScanIterator};

use super::super::tuple::{Tuple, ColumnInfo};
use super::trait_::PhysicalOperator;

use crate::types::value::DataValue;
use crate::types::datatype::DataType;

// ── SeqScan Operator ──────────────────────────────────────────────────────────

/// Scans all tuples from a heap file by iterating through all data pages.
///
/// Deserialises raw tuple bytes into `Tuple` values using the table's schema.
pub struct SeqScanOperator {
    /// The HeapManager provides file access.
    heap_manager: HeapManager,
    /// The scan iterator (reads pages sequentially).
    scan_iter: HeapScanIterator,
    /// Column metadata for the scanned table (view column names + physical DataTypes).
    column_info: Vec<ColumnInfo>,
    /// Data types in PHYSICAL column order (used for correct deserialisation).
    schema_types: Vec<DataType>,
    /// Optional mapping from view column position to physical column position.
    /// When non-empty, deserialised values are reordered from physical order
    /// to view column order before being yielded.
    column_mapping: Vec<usize>,
    /// Whether the scan has been fully consumed.
    exhausted: bool,
}

impl SeqScanOperator {
    /// Create a new sequential scan operator (identity mapping).
    ///
    /// `heap_manager` must be already opened/created for the target table.
    /// `column_info` should contain the table's column metadata (from the catalog).
    /// The DataTypes in `column_info` are used both for deserialisation and labeling.
    pub fn new(
        heap_manager: HeapManager,
        column_info: Vec<ColumnInfo>,
    ) -> Self {
        let schema_types: Vec<DataType> = column_info.iter().map(|c| c.data_type.clone()).collect();
        let identity: Vec<usize> = (0..column_info.len()).collect();
        let scan_iter = heap_manager.scan();
        Self {
            heap_manager,
            scan_iter,
            column_info,
            schema_types,
            column_mapping: identity,
            exhausted: false,
        }
    }

    /// Create a new sequential scan operator with explicit column mapping.
    ///
    /// `schema_types` — DataTypes in PHYSICAL column order (for deserialisation).
    /// `column_info` — Column metadata with VIEW column names.
    /// `column_mapping` — Maps each view position (0..column_info.len()) to the
    ///    corresponding physical column index in `schema_types`.
    ///
    /// After deserialisation using `schema_types`, values are reordered according
    /// to `column_mapping` so that each view column gets the correct data value.
    pub fn new_with_mapping(
        heap_manager: HeapManager,
        schema_types: Vec<DataType>,
        column_info: Vec<ColumnInfo>,
        column_mapping: Vec<usize>,
    ) -> Self {
        let scan_iter = heap_manager.scan();
        Self {
            heap_manager,
            scan_iter,
            column_info,
            schema_types,
            column_mapping,
            exhausted: false,
        }
    }
}

impl PhysicalOperator for SeqScanOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.exhausted {
            return Ok(None);
        }

        match self.scan_iter.next() {
            Some(Ok((page_id, slot_id, raw_bytes))) => {
                // Deserialise the raw tuple bytes using PHYSICAL column schema
                let phys_values = crate::types::deserialize_nullable_row(&self.schema_types, &raw_bytes)
                    .map_err(|e| format!("Failed to deserialise tuple: {}", e))?;

                // Reorder values from physical order to view column order
                let values: Vec<Option<DataValue>> = if self.column_mapping.len() == phys_values.len()
                    && self.column_mapping.iter().enumerate().all(|(i, &idx)| idx == i)
                {
                    // Identity mapping — skip reordering for efficiency
                    phys_values
                } else {
                    self.column_mapping.iter()
                        .map(|&phys_idx| phys_values.get(phys_idx).cloned().unwrap_or(None))
                        .collect()
                };

                Ok(Some(Tuple::new_with_location(values, self.column_info.clone(), page_id, slot_id)))
            }
            Some(Err(e)) => Err(format!("Scan error: {}", e)),
            None => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.column_info
    }

    fn reset(&mut self) -> Result<(), String> {
        self.scan_iter = self.heap_manager.scan();
        self.exhausted = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SeqScan"
    }
}
