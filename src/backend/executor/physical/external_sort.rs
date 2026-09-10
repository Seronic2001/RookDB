//! External Merge Sort — sorts datasets larger than available memory.
//!
//! # Algorithm (two-phase)
//!
//! **Phase 1 (Run Generation):** Read tuples from the child operator in
//! fixed-size chunks (`max_tuples_per_run`). Sort each chunk entirely in
//! memory and write it to a temporary file as a "sorted run".
//!
//! **Phase 2 (Merge):** Open all sorted-run files simultaneously and perform
//! a k-way merge using a min-heap (`std::collections::BinaryHeap`). The
//! smallest tuple among all runs is yielded first, maintaining global sort
//! order.
//!
//! # Temp file format
//!
//! Each sorted run is stored in a temporary file with this binary layout:
//!
//! ```text
//! [4 bytes]  — tuple count (u32 LE)
//! [N tuples] — each prefixed with a 4-byte length (u32 LE) followed by
//!              the serialized row bytes from serialize_tuple_to_bytes()
//! ```
//!
//! # Transparency
//!
//! The `ExternalSortOperator` implements `PhysicalOperator` and is used by
//! `SortOperator` when the estimated cardinality exceeds the configured
//! threshold. Downstream operators see no difference.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use super::operators::PhysicalOperator;
use super::tuple::{Tuple, ColumnInfo, serialize_tuple_to_bytes, deserialize_tuple_from_bytes};
use crate::types::comparison::compare_nullable;

// ── Global temp-file counter ──────────────────────────────────────────────────

/// Process-wide monotone counter so every `TempFileManager` instance gets a
/// globally unique `instance_id`, preventing collisions when multiple
/// `ExternalSortOperator`s run concurrently in the same process (e.g. tests).
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── Configuration ────────────────────────────────────────────────────────────

/// Tuning parameters for the external merge sort.
#[derive(Debug, Clone)]
pub struct ExternalSortConfig {
    /// Maximum number of tuples per sorted run (written as one temp file).
    /// Larger values reduce the number of runs (and thus the merge depth)
    /// but consume more memory per run.
    pub max_tuples_per_run: usize,

    /// Directory for temporary sort files.
    /// Files are named `rook_sort_<pid>_<instance_id>_<seq>.tmp` and cleaned up on drop.
    pub temp_dir: PathBuf,
}

impl Default for ExternalSortConfig {
    fn default() -> Self {
        Self {
            // ~80 KB per run for simple (4 col) int tuples at ~20 bytes each
            max_tuples_per_run: 10_000,
            temp_dir: PathBuf::from("/tmp/rookdb_sort"),
        }
    }
}

impl ExternalSortConfig {
    /// Create a config with a custom memory budget (approximate).
    ///
    /// `memory_budget_bytes` is used to derive `max_tuples_per_run` assuming
    /// an average tuple size of 64 bytes.
    pub fn with_memory_budget(memory_budget_bytes: usize, temp_dir: PathBuf) -> Self {
        let avg_tuple_size = 64usize; // conservative average
        let max_tuples_per_run = (memory_budget_bytes / avg_tuple_size).max(100);
        Self {
            max_tuples_per_run,
            temp_dir,
        }
    }
}

// ── Temp file management ─────────────────────────────────────────────────────

/// Manages temporary files created during external sort.
///
/// All files are deleted when this struct is dropped.
///
/// File names include a globally unique `instance_id` drawn from
/// `TEMP_FILE_COUNTER`, ensuring no two `TempFileManager` instances in the
/// same process can produce colliding paths — even when tests run in parallel.
struct TempFileManager {
    files: Vec<PathBuf>,
    /// Monotone counter local to this manager instance.
    seq: u64,
    /// Globally unique ID assigned at construction time.
    instance_id: u64,
    temp_dir: PathBuf,
}

impl TempFileManager {
    fn new(temp_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&temp_dir);
        let instance_id = TEMP_FILE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        Self {
            files: Vec::new(),
            seq: 0,
            instance_id,
            temp_dir,
        }
    }

    /// Create a new unique temp file path and track it for cleanup.
    ///
    /// The path is `<temp_dir>/rook_sort_<pid>_<instance_id>_<seq>.tmp`.
    /// `instance_id` is globally unique per process, so parallel operator
    /// instances cannot collide.
    fn create_file(&mut self) -> std::io::Result<PathBuf> {
        let seq = self.seq;
        self.seq += 1;
        let path = self.temp_dir.join(format!(
            "rook_sort_{}_{}_{}.tmp",
            std::process::id(),
            self.instance_id,
            seq,
        ));
        self.files.push(path.clone());

        // Create (or truncate) the file
        File::create(&path)?;

        Ok(path)
    }
}

impl Drop for TempFileManager {
    fn drop(&mut self) {
        for path in &self.files {
            let _ = fs::remove_file(path);
        }
    }
}

// ── Helper: compare two tuples by sort keys ─────────────────────────────────

/// Compare two tuples using the same logic as `SortOperator::sort_by` closure.
///
/// Returns `Ordering::Less` if `a` should sort before `b` according to the
/// sort key specifications.
fn compare_tuples_by_keys(
    a: &Tuple,
    b: &Tuple,
    sort_keys: &[(usize, bool)],
) -> Ordering {
    for &(key_idx, descending) in sort_keys {
        let av = a.values.get(key_idx).and_then(|v| v.as_ref());
        let bv = b.values.get(key_idx).and_then(|v| v.as_ref());
        let ordering = match (av, bv) {
            (Some(a_val), Some(b_val)) => compare_nullable(Some(a_val), Some(b_val))
                .unwrap_or(None)
                .unwrap_or(Ordering::Equal),
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,  // NULLs first
            (Some(_), None) => Ordering::Greater,
        };
        let ordering = if descending { ordering.reverse() } else { ordering };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

// ── Sorted run file I/O ───────────────────────────────────────────────────────

/// Writes a batch of sorted tuples to a temp file as a single sorted run.
#[allow(dead_code)]
struct SortedRunWriter {
    file: File,
    path: PathBuf,
    tuple_count: u32,
    schema: Vec<ColumnInfo>,
}

impl SortedRunWriter {
    /// Create a new writer for a sorted run.
    fn new(path: PathBuf, schema: Vec<ColumnInfo>) -> std::io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Reserve space for tuple count (will be written on close)
        file.write_all(&0u32.to_le_bytes())?;

        Ok(Self {
            file,
            path,
            tuple_count: 0,
            schema,
        })
    }

    /// Append one tuple to the sorted run.
    fn write_tuple(&mut self, tuple: &Tuple) -> Result<(), String> {
        let bytes = serialize_tuple_to_bytes(tuple, &self.schema)?;
        let len = bytes.len() as u32;

        self.file
            .write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write tuple length to sort temp file: {}", e))?;
        self.file
            .write_all(&bytes)
            .map_err(|e| format!("Failed to write tuple data to sort temp file: {}", e))?;

        self.tuple_count += 1;
        Ok(())
    }

    /// Finalize the run by writing the tuple count at the start of the file.
    fn finalize(&mut self) -> Result<(), String> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to seek in sort temp file: {}", e))?;
        self.file
            .write_all(&self.tuple_count.to_le_bytes())
            .map_err(|e| format!("Failed to write tuple count to sort temp file: {}", e))?;
        self.file
            .flush()
            .map_err(|e| format!("Failed to flush sort temp file: {}", e))?;
        Ok(())
    }

    /// Number of tuples written to this run.
    #[allow(dead_code)]
    fn len(&self) -> u32 {
        self.tuple_count
    }
}

/// Reads and yields tuples from a single sorted run file.
struct SortedRunReader {
    file: File,
    /// Total tuples in this run.
    tuple_count: u32,
    /// Tuples read so far.
    tuples_read: u32,
    /// Schema for deserialization.
    schema: Vec<ColumnInfo>,
    /// Whether the file has been exhausted.
    exhausted: bool,
    /// Run index (for stable ordering during merge).
    #[allow(dead_code)]
    run_index: usize,
}

impl SortedRunReader {
    /// Open a sorted run file for reading.
    fn open(path: &Path, schema: Vec<ColumnInfo>, run_index: usize) -> Result<Self, String> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| format!("Failed to open sort temp file: {}", e))?;

        // Read tuple count
        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)
            .map_err(|e| format!("Failed to read sort temp file header: {}", e))?;
        let tuple_count = u32::from_le_bytes(count_buf);
        log::info!(
            "[ExternalSort] Opened run {} with {} tuples",
            run_index,
            tuple_count
        );

        Ok(Self {
            file,
            tuple_count,
            tuples_read: 0,
            schema,
            exhausted: false,
            run_index,
        })
    }

    /// Read and return the next tuple, advancing the reader.
    fn next_tuple(&mut self) -> Result<Option<Tuple>, String> {
        if self.exhausted || self.tuples_read >= self.tuple_count {
            self.exhausted = true;
            return Ok(None);
        }

        // Read length prefix
        let mut len_buf = [0u8; 4];
        if self.file.read_exact(&mut len_buf).is_err() {
            self.exhausted = true;
            return Ok(None);
        }
        let data_len = u32::from_le_bytes(len_buf) as usize;

        // Read tuple data
        let mut data_buf = vec![0u8; data_len];
        self.file
            .read_exact(&mut data_buf)
            .map_err(|e| format!("Failed to read tuple from sort temp file: {}", e))?;

        self.tuples_read += 1;
        if self.tuples_read >= self.tuple_count {
            self.exhausted = true;
        }

        let tuple = deserialize_tuple_from_bytes(&data_buf, &self.schema)?;
        Ok(Some(tuple))
    }
}

// ── Merge heap entry ──────────────────────────────────────────────────────────

/// An entry in the merge heap representing the smallest tuple from one run.
///
/// `Ord` is implemented to make `BinaryHeap` behave as a min-heap over the
/// sort-key order, with run_index as tiebreaker for stability.
struct HeapEntry {
    tuple: Tuple,
    /// Index of the run this tuple came from.
    run_index: usize,
    sort_keys: Vec<(usize, bool)>,
}

impl HeapEntry {
    fn new(tuple: Tuple, run_index: usize, sort_keys: Vec<(usize, bool)>) -> Self {
        Self {
            tuple,
            run_index,
            sort_keys,
        }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.run_index == other.run_index
            && compare_tuples_by_keys(&self.tuple, &other.tuple, &self.sort_keys)
                == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we reverse to get ascending (min-heap)
        // behavior for the sort key.
        let key_ordering =
            compare_tuples_by_keys(&self.tuple, &other.tuple, &self.sort_keys);
        // Reverse: smaller tuple should be popped FIRST from max-heap
        key_ordering
            .reverse()
            .then_with(|| other.run_index.cmp(&self.run_index))
    }
}

// ── ExternalSortOperator ─────────────────────────────────────────────────────

/// A `PhysicalOperator` that performs an external merge sort on its child's
/// output, spilling intermediate sorted runs to temporary files when the
/// dataset exceeds the in-memory threshold.
///
/// # Usage
///
/// This operator is created by `SortOperator::new_external()` or by the
/// `PhysicalPlanner` when the estimated cardinality exceeds the threshold.
pub struct ExternalSortOperator {
    child: Box<dyn PhysicalOperator>,
    sort_keys: Vec<(usize, bool)>,
    config: ExternalSortConfig,

    // ── Runtime state (Phase 1: run generation) ──
    temp_files: TempFileManager,
    sorted_runs: Vec<PathBuf>,
    run_schema: Vec<ColumnInfo>,
    phase1_done: bool,

    // ── Runtime state (Phase 2: merge) ──
    readers: Vec<SortedRunReader>,
    heap: BinaryHeap<HeapEntry>,
    merge_exhausted: bool,
}

impl ExternalSortOperator {
    /// Create a new external sort operator.
    pub fn new(
        child: Box<dyn PhysicalOperator>,
        sort_keys: Vec<(usize, bool)>,
        config: ExternalSortConfig,
    ) -> Self {
        let run_schema = child.schema().to_vec();
        let temp_dir = config.temp_dir.clone();
        Self {
            child,
            sort_keys,
            config,
            temp_files: TempFileManager::new(temp_dir),
            sorted_runs: Vec::new(),
            run_schema,
            phase1_done: false,
            readers: Vec::new(),
            heap: BinaryHeap::new(),
            merge_exhausted: false,
        }
    }

    /// Phase 1: Consume all tuples from child, sort in chunks, write sorted runs.
    fn generate_runs(&mut self) -> Result<(), String> {
        log::info!("[ExternalSort] Phase 1: Generating sorted runs");

        let mut buffer: Vec<Tuple> = Vec::with_capacity(self.config.max_tuples_per_run);

        while let Some(tuple) = self.child.next()? {
            buffer.push(tuple);

            if buffer.len() >= self.config.max_tuples_per_run {
                self.flush_run(&mut buffer)?;
            }
        }

        // Flush remaining tuples
        if !buffer.is_empty() {
            self.flush_run(&mut buffer)?;
        }

        log::info!(
            "[ExternalSort] Phase 1 complete: {} sorted runs generated",
            self.sorted_runs.len()
        );

        self.phase1_done = true;
        Ok(())
    }

    /// Sort the buffer and write it as a sorted run to a temp file.
    fn flush_run(&mut self, buffer: &mut Vec<Tuple>) -> Result<(), String> {
        // Sort the batch
        let sort_keys = self.sort_keys.clone();
        buffer.sort_by(|a, b| compare_tuples_by_keys(a, b, &sort_keys));

        // Write to temp file
        let path = self
            .temp_files
            .create_file()
            .map_err(|e| format!("Failed to create sort temp file: {}", e))?;

        let mut writer = SortedRunWriter::new(path.clone(), self.run_schema.clone())
            .map_err(|e| format!("Failed to create run writer: {}", e))?;

        for tuple in buffer.drain(..) {
            writer
                .write_tuple(&tuple)
                .map_err(|e| format!("Failed to write tuple to run: {}", e))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize run: {}", e))?;

        self.sorted_runs.push(path);
        Ok(())
    }

    /// Phase 2: Open all sorted runs and initialize the merge heap.
    fn init_merge(&mut self) -> Result<(), String> {
        log::info!(
            "[ExternalSort] Phase 2: Merging {} sorted runs",
            self.sorted_runs.len()
        );

        if self.sorted_runs.is_empty() {
            self.merge_exhausted = true;
            return Ok(());
        }

        let sort_keys = self.sort_keys.clone();
        let schema = self.run_schema.clone();

        for (idx, path) in self.sorted_runs.iter().enumerate() {
            let mut reader =
                SortedRunReader::open(path, schema.clone(), idx)?;
            if let Some(tuple) = reader.next_tuple()? {
                self.heap.push(HeapEntry::new(tuple, idx, sort_keys.clone()));
            }
            self.readers.push(reader);
        }

        log::info!(
            "[ExternalSort] Merge heap initialized with {} entries",
            self.heap.len()
        );

        Ok(())
    }

    /// Yield the next tuple from the merge phase.
    fn merge_next(&mut self) -> Result<Option<Tuple>, String> {
        if self.merge_exhausted || self.heap.is_empty() {
            return Ok(None);
        }

        // Pop the smallest tuple from the heap
        let entry = self.heap.pop().unwrap();

        // Read the next tuple from the same run and push it to the heap
        if let Some(next_tuple) = self.readers[entry.run_index].next_tuple()? {
            let sort_keys = self.sort_keys.clone();
            self.heap
                .push(HeapEntry::new(next_tuple, entry.run_index, sort_keys));
        }

        Ok(Some(entry.tuple))
    }
}

impl PhysicalOperator for ExternalSortOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        // Phase 1: generate sorted runs
        if !self.phase1_done {
            self.generate_runs()?;
            self.init_merge()?;
        }

        // Phase 2: merge
        self.merge_next()
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.run_schema
    }

    fn reset(&mut self) -> Result<(), String> {
        // Re-initialize state
        self.sorted_runs.clear();
        self.readers.clear();
        self.heap.clear();
        self.phase1_done = false;
        self.merge_exhausted = false;
        self.child.reset()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ExternalSort"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::executor::physical::tuple::Tuple;
    use crate::backend::executor::physical::operators::PhysicalOperator;
    use crate::types::datatype::DataType;
    use crate::types::value::DataValue;

    /// A mock operator that yields a fixed set of tuples (local to test module).
    struct MockOperator {
        tuples: Vec<Tuple>,
        pos: usize,
        schema: Vec<ColumnInfo>,
    }

    impl MockOperator {
        fn new(tuples: Vec<Tuple>, schema: Vec<ColumnInfo>) -> Self {
            Self { tuples, pos: 0, schema }
        }
    }

    impl PhysicalOperator for MockOperator {
        fn next(&mut self) -> Result<Option<Tuple>, String> {
            if self.pos < self.tuples.len() {
                let t = self.tuples[self.pos].clone();
                self.pos += 1;
                Ok(Some(t))
            } else {
                Ok(None)
            }
        }

        fn schema(&self) -> &[ColumnInfo] {
            &self.schema
        }

        fn reset(&mut self) -> Result<(), String> {
            self.pos = 0;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    /// Helper to create a test schema and tuples.
    fn int_schema() -> Vec<ColumnInfo> {
        vec![
            ColumnInfo {
                name: "a".into(),
                data_type: DataType::Int, table: None },
            ColumnInfo {
                name: "b".into(),
                data_type: DataType::Int, table: None },
        ]
    }

    fn int_tuples(data: Vec<(Option<i32>, Option<i32>)>) -> (Vec<Tuple>, Vec<ColumnInfo>) {
        let schema = int_schema();
        let tuples = data
            .into_iter()
            .map(|(a, b)| {
                Tuple::new(
                    vec![a.map(DataValue::Int), b.map(DataValue::Int)],
                )
            })
            .collect();
        (tuples, schema)
    }

    #[test]
    fn test_compare_tuples_by_keys_asc() {
        let a = Tuple::new(
            vec![Some(DataValue::Int(1)), Some(DataValue::Int(10))],
        );
        let b = Tuple::new(
            vec![Some(DataValue::Int(2)), Some(DataValue::Int(20))],
        );

        let keys = vec![(0, false)]; // sort by col 0 ascending
        assert_eq!(
            compare_tuples_by_keys(&a, &b, &keys),
            Ordering::Less
        );
        assert_eq!(
            compare_tuples_by_keys(&b, &a, &keys),
            Ordering::Greater
        );
        assert_eq!(compare_tuples_by_keys(&a, &a, &keys), Ordering::Equal);
    }

    #[test]
    fn test_compare_tuples_by_keys_desc() {
        let a = Tuple::new(
            vec![Some(DataValue::Int(1)), Some(DataValue::Int(10))],
        );
        let b = Tuple::new(
            vec![Some(DataValue::Int(2)), Some(DataValue::Int(20))],
        );

        let keys = vec![(0, true)]; // sort by col 0 descending
        assert_eq!(
            compare_tuples_by_keys(&a, &b, &keys),
            Ordering::Greater // 1 > 2 in descending order
        );
    }

    #[test]
    fn test_compare_tuples_by_keys_nulls_first() {
        let a = Tuple::new(
            vec![None, Some(DataValue::Int(10))],
        );
        let b = Tuple::new(
            vec![Some(DataValue::Int(5)), Some(DataValue::Int(20))],
        );

        let keys = vec![(0, false)]; // NULLs first
        assert_eq!(compare_tuples_by_keys(&a, &b, &keys), Ordering::Less);
    }

    #[test]
    fn test_external_sort_small_dataset() {
        // Small dataset that fits in a single run
        let (tuples, schema) = int_tuples(vec![
            (Some(3), Some(30)),
            (Some(1), Some(10)),
            (Some(2), Some(20)),
        ]);

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 100, // large enough for all tuples
            ..Default::default()
        };

        let mut sorter = ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);

        let t1 = sorter.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(1)));
        let t2 = sorter.next().unwrap().unwrap();
        assert_eq!(t2.values[0], Some(DataValue::Int(2)));
        let t3 = sorter.next().unwrap().unwrap();
        assert_eq!(t3.values[0], Some(DataValue::Int(3)));
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_multiple_runs() {
        // Force multiple runs by setting max_tuples_per_run to 2
        let (tuples, schema) = int_tuples(vec![
            (Some(10), Some(100)),
            (Some(3), Some(30)),
            (Some(7), Some(70)),
            (Some(1), Some(10)),
            (Some(5), Some(50)),
        ]);

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 2, // forces 3 runs
            ..Default::default()
        };

        let mut sorter = ExternalSortOperator::new(
            Box::new(child),
            vec![(0, false)], // sort by col 0 ascending
            config,
        );

        let results: Vec<Tuple> = std::iter::from_fn(|| sorter.next().transpose())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].values[0], Some(DataValue::Int(1)));
        assert_eq!(results[1].values[0], Some(DataValue::Int(3)));
        assert_eq!(results[2].values[0], Some(DataValue::Int(5)));
        assert_eq!(results[3].values[0], Some(DataValue::Int(7)));
        assert_eq!(results[4].values[0], Some(DataValue::Int(10)));
    }

    #[test]
    fn test_external_sort_descending() {
        let (tuples, schema) = int_tuples(vec![
            (Some(1), Some(10)),
            (Some(3), Some(30)),
            (Some(2), Some(20)),
        ]);

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 2,
            ..Default::default()
        };

        let mut sorter = ExternalSortOperator::new(
            Box::new(child),
            vec![(0, true)], // sort by col 0 descending
            config,
        );

        let t1 = sorter.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(3)));
        let t2 = sorter.next().unwrap().unwrap();
        assert_eq!(t2.values[0], Some(DataValue::Int(2)));
        let t3 = sorter.next().unwrap().unwrap();
        assert_eq!(t3.values[0], Some(DataValue::Int(1)));
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_empty_child() {
        let (tuples, schema) = int_tuples(vec![]);
        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig::default();
        let mut sorter =
            ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_single_tuple() {
        let (tuples, schema) = int_tuples(vec![(Some(42), Some(100))]);
        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig::default();
        let mut sorter =
            ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);
        let t = sorter.next().unwrap().unwrap();
        assert_eq!(t.values[0], Some(DataValue::Int(42)));
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_multi_key() {
        let schema = vec![
            ColumnInfo {
                name: "x".into(),
                data_type: DataType::Int, table: None },
            ColumnInfo {
                name: "y".into(),
                data_type: DataType::Int, table: None },
        ];
        let tuples = vec![
            Tuple::new(
                vec![Some(DataValue::Int(1)), Some(DataValue::Int(20))],
            ),
            Tuple::new(
                vec![Some(DataValue::Int(1)), Some(DataValue::Int(10))],
            ),
            Tuple::new(
                vec![Some(DataValue::Int(2)), Some(DataValue::Int(5))],
            ),
        ];

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 2,
            ..Default::default()
        };
        let mut sorter = ExternalSortOperator::new(
            Box::new(child),
            vec![(0, false), (1, false)], // sort by x ASC, then y ASC
            config,
        );

        let t1 = sorter.next().unwrap().unwrap();
        assert_eq!(t1.values, vec![Some(DataValue::Int(1)), Some(DataValue::Int(10))]);
        let t2 = sorter.next().unwrap().unwrap();
        assert_eq!(t2.values, vec![Some(DataValue::Int(1)), Some(DataValue::Int(20))]);
        let t3 = sorter.next().unwrap().unwrap();
        assert_eq!(t3.values, vec![Some(DataValue::Int(2)), Some(DataValue::Int(5))]);
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_reset() {
        let (tuples, schema) = int_tuples(vec![(Some(2), Some(20)), (Some(1), Some(10))]);
        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 1, // force separate runs
            ..Default::default()
        };
        let mut sorter =
            ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);

        // First pass
        let t1 = sorter.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(1)));
        let t2 = sorter.next().unwrap().unwrap();
        assert_eq!(t2.values[0], Some(DataValue::Int(2)));
        assert!(sorter.next().unwrap().is_none());

        // Reset and re-read
        sorter.reset().unwrap();
        let r1 = sorter.next().unwrap().unwrap();
        assert_eq!(r1.values[0], Some(DataValue::Int(1)));
        let r2 = sorter.next().unwrap().unwrap();
        assert_eq!(r2.values[0], Some(DataValue::Int(2)));
        assert!(sorter.next().unwrap().is_none());
    }

    #[test]
    fn test_external_sort_duplicates() {
        let (tuples, schema) = int_tuples(vec![
            (Some(2), Some(20)),
            (Some(2), Some(20)), // duplicate
            (Some(1), Some(10)),
            (Some(1), Some(10)), // duplicate
        ]);

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 2,
            ..Default::default()
        };
        let mut sorter =
            ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);

        let results: Vec<Tuple> = std::iter::from_fn(|| sorter.next().transpose())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].values[0], Some(DataValue::Int(1)));
        assert_eq!(results[1].values[0], Some(DataValue::Int(1)));
        assert_eq!(results[2].values[0], Some(DataValue::Int(2)));
        assert_eq!(results[3].values[0], Some(DataValue::Int(2)));
    }

    #[test]
    fn test_external_sort_with_nulls() {
        let (tuples, schema) = int_tuples(vec![
            (None, Some(30)),
            (Some(2), Some(20)),
            (None, Some(10)),
            (Some(1), Some(5)),
        ]);

        let child = MockOperator::new(tuples, schema);
        let config = ExternalSortConfig {
            max_tuples_per_run: 2,
            ..Default::default()
        };
        let mut sorter =
            ExternalSortOperator::new(Box::new(child), vec![(0, false)], config);

        let results: Vec<Tuple> = std::iter::from_fn(|| sorter.next().transpose())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // NULLs should come first, then sorted values
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].values[0], None); // NULL
        assert_eq!(results[1].values[0], None); // NULL
        assert_eq!(results[2].values[0], Some(DataValue::Int(1)));
        assert_eq!(results[3].values[0], Some(DataValue::Int(2)));
    }
}
