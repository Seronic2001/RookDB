use crate::types::datatype::DataType;
use crate::types::value::DataValue;
use crate::types::{serialize_nullable_typed_row, deserialize_nullable_row};

/// Metadata describing one column in a query result set.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: DataType,
    /// Optional table name for qualified column references (e.g. `t1.id`).
    /// Used by join operators to disambiguate columns with the same name
    /// from different tables. Currently informational; lookups use `name` only.
    pub table: Option<String>,
}

/// A single row in the result set.
///
/// `values[i]` corresponds to `column_info[i]` — both use the same logical
/// ordering (the projection order for the query, or the table's schema order
/// for a full `SELECT *`).
///
/// The optional `page_id` / `slot_id` fields record the heap location of the
/// source row.  These are populated by scan operators and propagated through
/// filter/evaluation operators, allowing DML mutations (UPDATE/DELETE) to
/// identify which physical row to modify without a separate heap scan.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuple {
    pub values: Vec<Option<DataValue>>,
    /// Heap page identifier where this tuple originated (if known).
    pub page_id: Option<u32>,
    /// Heap slot identifier within `page_id` (if known).
    pub slot_id: Option<u32>,
}

impl Tuple {
    /// Create a new tuple from deserialized values.
    pub fn new(values: Vec<Option<DataValue>>) -> Self {
        Self { values, page_id: None, slot_id: None }
    }

    /// Create a new tuple with heap location metadata.
    pub fn new_with_location(
        values: Vec<Option<DataValue>>,
        page_id: u32,
        slot_id: u32,
    ) -> Self {
        Self { values, page_id: Some(page_id), slot_id: Some(slot_id) }
    }

    /// Set heap location metadata.
    pub fn with_location(mut self, page_id: u32, slot_id: u32) -> Self {
        self.page_id = Some(page_id);
        self.slot_id = Some(slot_id);
        self
    }

    /// Attach the source tuple's heap location (if any) to this derived
    /// tuple.
    ///
    /// Operators that build NEW tuples from one input row per output row
    /// (e.g. Projection) must carry the row's `(page_id, slot_id)` through,
    /// otherwise pointer-based UPDATE/DELETE cannot address the row. A
    /// synthetic source (aggregate output, single-row stub) simply stays
    /// location-less.
    pub fn with_location_from(mut self, src: &Tuple) -> Self {
        self.page_id = src.page_id;
        self.slot_id = src.slot_id;
        self
    }

    /// How many columns this tuple has.
    pub fn arity(&self) -> usize {
        self.values.len()
    }

    /// Return a reference to the value at `index`, if within bounds.
    pub fn get(&self, index: usize) -> Option<&Option<DataValue>> {
        self.values.get(index)
    }

    /// Concatenate two tuples into one (used by join operators).
    /// The resulting tuple has `self`'s values followed by `other`'s values.
    /// Location metadata is cleared (joined tuples don't have a single source).
    pub fn concatenate(&self, other: &Tuple) -> Self {
        let mut combined_values = self.values.clone();
        combined_values.extend(other.values.iter().cloned());
        Self {
            values: combined_values,
            page_id: None,
            slot_id: None,
        }
    }

    /// Project a subset of columns from this tuple.
    pub fn project(&self, indices: &[usize]) -> Self {
        let mut projected_values = Vec::with_capacity(indices.len());
        for &i in indices {
            projected_values.push(self.values.get(i).cloned().unwrap_or(None));
        }
        Self {
            values: projected_values,
            page_id: self.page_id,
            slot_id: self.slot_id,
        }
    }
}

// ── Display / Formatting ──────────────────────────────────────────────────────

/// Format a single tuple as a human-readable string.
pub fn format_tuple(tuple: &Tuple, schema: &[ColumnInfo], _col_width: usize) -> String {
    let mut parts = Vec::new();
    for (i, val_opt) in tuple.values.iter().enumerate() {
        let name = schema.get(i).map(|c| c.name.as_str()).unwrap_or("?");
        match val_opt {
            Some(val) => parts.push(format!("{}={}", name, val)),
            None => parts.push(format!("{}=NULL", name)),
        }
    }
    if parts.is_empty() {
        "(empty row)".to_string()
    } else {
        parts.join(" | ")
    }
}

/// Display a set of tuples in a formatted table with borders.
///
/// Returns the number of tuples displayed.
pub fn display_tuples(tuples: &[Tuple], schema: &[ColumnInfo]) -> usize {
    if tuples.is_empty() {
        println!("(0 rows)");
        return 0;
    }

    let col_count = schema.len();

    // Compute column widths
    let col_width = 22usize;

    // Build borders
    let mut top_border = String::from("┌");
    let mut mid_border = String::from("├");
    let mut bot_border = String::from("└");

    for idx in 0..col_count {
        let line = "─".repeat(col_width + 2);
        if idx < col_count - 1 {
            top_border.push_str(&format!("{}┬", line));
            mid_border.push_str(&format!("{}┼", line));
            bot_border.push_str(&format!("{}┴", line));
        } else {
            top_border.push_str(&format!("{}┐", line));
            mid_border.push_str(&format!("{}┤", line));
            bot_border.push_str(&format!("{}┘", line));
        }
    }

    println!("{}", top_border);
    print!("│");
    for col in schema.iter() {
        let display = format!("{}: {}", col.name, col.data_type);
        let truncated = if display.len() > col_width {
            format!("{}…", &display[..col_width - 1])
        } else {
            display
        };
        print!(" {:<width$} │", truncated, width = col_width);
    }
    println!();
    println!("{}", mid_border);

    for (row_idx, tuple) in tuples.iter().enumerate() {
        print!("│ {:>3} │", row_idx + 1);
        for val_opt in &tuple.values {
            let display = match val_opt {
                Some(val) => format!("{}", val),
                None => "NULL".to_string(),
            };
            if display.len() > col_width {
                print!(" {}… │", &display[..col_width - 1]);
            } else {
                print!(" {:<width$} │", display, width = col_width);
            }
        }
        println!();
    }

    println!("{}", bot_border);
    println!("({} rows)\n", tuples.len());
    tuples.len()
}

// ── Tuple serialization for external sort / temp files ───────────────────────

/// Serialize a Tuple to binary bytes using the existing row serialization.
///
/// The returned bytes can be written to a temp file and later deserialized
/// back into a Tuple using `deserialize_tuple_from_bytes` with the same schema.
pub fn serialize_tuple_to_bytes(tuple: &Tuple, schema: &[ColumnInfo]) -> Result<Vec<u8>, String> {
    let schema_types: Vec<DataType> = schema
        .iter()
        .map(|c| c.data_type.clone())
        .collect();
    serialize_nullable_typed_row(&schema_types, &tuple.values)
}

/// Deserialize a Tuple from binary bytes produced by `serialize_tuple_to_bytes`.
///
/// The `schema_types` and `column_info` must match those of the original tuple.
pub fn deserialize_tuple_from_bytes(
    bytes: &[u8],
    column_info: &[ColumnInfo],
) -> Result<Tuple, String> {
    let schema_types: Vec<DataType> = column_info.iter().map(|c| c.data_type.clone()).collect();
    let values = deserialize_nullable_row(&schema_types, bytes)?;
    Ok(Tuple::new(values))
}
