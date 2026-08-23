//! B+ Tree — Page-based balanced tree for disk-resident key-value storage.
//!
//! # Design
//!
//! - 8 KB pages stored in a separate `.idx` file per index.
//! - Nodes are either **Internal** (page_type=0) or **Leaf** (page_type=1).
//! - Leaves are linked into a doubly-linked list for efficient range scans.
//! - Keys are variable-length: `[key_len: u16][encoded_key_bytes]`.
//! - Values are heap tuple identifiers `(page_id: u32, slot_id: u32)`.
//! - Page-level write latching (via `PageWriteLock`) protects splits.
//!
//! # Page Format
//!
//! **Header (shared):**
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 4 | page_type: 0=internal, 1=leaf |
//! | 4 | 4 | num_keys |
//!
//! **Internal node (after header):**
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 8 | 4×(n+1) | child_0 … child_n |
//! | 8+4×(n+1) | * | key_0_len(2) + key_0_bytes  …  key_n-1_len(2) + key_n-1_bytes |
//!
//! **Leaf node (after header):**
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 8 | 4 | next_leaf_page_id |
//! | 12 | 4 | prev_leaf_page_id |
//! | 16 | 8×n | (page_id, slot_id) per entry |
//! | 16+8×n | * | key_0_len(2) + key_bytes …  key_n-1_len(2) + key_bytes |

use std::cmp::Ordering;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::backend::page::page_lock::PageWriteLock;
use crate::backend::table::table_file::file_identity_from_file;
use crate::types::value::DataValue;
use crate::types::Comparable;
use crate::types::DataType;

// ─── Constants ───────────────────────────────────────────────────────────────

/// B+ Tree page size (same as heap pages: 8 KB).
pub const BTREE_PAGE_SIZE: usize = 8192;

/// Minimum number of keys in a node (half-full, for non-root splits).
/// Allow at least a few entries even for large keys.
pub const BTREE_MIN_KEYS: usize = 2;

/// Page type markers.
const PAGE_TYPE_INTERNAL: u32 = 0;
const PAGE_TYPE_LEAF: u32 = 1;

/// Size of the shared page header.
const PAGE_HEADER: usize = 8; // page_type (4) + num_keys (4)
/// Leaf-specific header beyond the shared header.
const LEAF_HEADER: usize = 16; // shared(8) + next_leaf(4) + prev_leaf(4)
/// Size of a value tuple (page_id + slot_id).
const VALUE_SIZE: usize = 8; // page_id(4) + slot_id(4)
/// Size of a child pointer.
const CHILD_SIZE: usize = 4; // u32
/// Size of the key length prefix.
const KEY_LEN_SIZE: usize = 2; // u16

// ─── Encoded Key Helpers ────────────────────────────────────────────────────

/// Sidecar file storing the current root page id (`<idx>.root`, u32 LE).
///
/// The index file itself has no header page, so without this every
/// `BTree::open` assumed page 0 — the ORIGINAL leftmost leaf — and all
/// searches degenerated to next_leaf chain walks.
fn root_sidecar_path(idx_path: &Path) -> PathBuf {
    let mut s = idx_path.as_os_str().to_owned();
    s.push(".root");
    PathBuf::from(s)
}

fn write_root_sidecar(idx_path: &Path, root_page_id: u32) -> io::Result<()> {
    std::fs::write(root_sidecar_path(idx_path), root_page_id.to_le_bytes())
}

fn read_root_sidecar(idx_path: &Path) -> io::Result<u32> {
    let bytes = std::fs::read(root_sidecar_path(idx_path))?;
    if bytes.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short root sidecar"));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Encode a DataValue to the on-disk key format: `[u16 length][bytes]`.
fn encode_key(key: &DataValue) -> Vec<u8> {
    let bytes = key.to_bytes();
    let mut out = Vec::with_capacity(KEY_LEN_SIZE + bytes.len());
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(&bytes);
    out
}

/// Decode a key from on-disk format. Returns `(key, bytes_consumed)`.
fn decode_key(data: &[u8], ty: &DataType) -> io::Result<(DataValue, usize)> {
    if data.len() < KEY_LEN_SIZE {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key length"));
    }
    let len = u16::from_le_bytes([data[0], data[1]]) as usize;
    if data.len() < KEY_LEN_SIZE + len {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key data"));
    }
    let key_bytes = &data[KEY_LEN_SIZE..KEY_LEN_SIZE + len];
    let value = DataValue::from_bytes(ty, key_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((value, KEY_LEN_SIZE + len))
}

/// Encode a composite key as one framed entry:
/// `[u16 total_len][seg1_len][seg1][seg2_len][seg2]…`
///
/// The outer envelope lets leaf serialisation/deserialisation treat the
/// whole multi-segment key as a single opaque `[len][bytes]` entry, exactly
/// like single-column keys (ANALYSIS.md Tier 2 #8).
pub(crate) fn encode_key_multi(values: &[DataValue]) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    for v in values {
        body.extend_from_slice(&encode_key(v));
    }
    let total = body.len();
    if total > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Composite key too large: {} bytes", total),
        ));
    }
    let mut out = Vec::with_capacity(KEY_LEN_SIZE + total);
    out.extend_from_slice(&(total as u16).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Calculate the byte size of an encoded key.
#[allow(dead_code)]
fn encoded_key_size(encoded: &[u8]) -> usize {
    if encoded.len() < KEY_LEN_SIZE {
        return 0;
    }
    let len = u16::from_le_bytes([encoded[0], encoded[1]]) as usize;
    KEY_LEN_SIZE + len
}

/// Get the total byte size of n keys stored sequentially.
fn total_keys_size(keys: &[Vec<u8>]) -> usize {
    keys.iter().map(|k| k.len()).sum()
}

// ─── BTreeNode ───────────────────────────────────────────────────────────────

/// In-memory representation of a B+ Tree node.
#[derive(Debug, Clone)]
pub enum BTreeNode {
    /// An internal (branch) node: contains keys and child page pointers.
    Internal {
        num_keys: u32,
        /// Encoded keys in sorted order, each as `[u16 len][key_bytes]`.
        keys: Vec<Vec<u8>>,
        /// Child page IDs: length = num_keys + 1.
        children: Vec<u32>,
    },
    /// A leaf node: contains keys and heap tuple identifiers.
    Leaf {
        num_keys: u32,
        /// Encoded keys in sorted order.
        keys: Vec<Vec<u8>>,
        /// Heap tuple identifiers: (page_id, slot_id) per entry.
        values: Vec<(u32, u32)>,
        /// Next leaf in the linked list (0 = none).
        next_leaf: u32,
        /// Previous leaf in the linked list (0 = none).
        prev_leaf: u32,
    },
}

impl BTreeNode {
    /// Create a new empty leaf node.
    fn new_leaf() -> Self {
        BTreeNode::Leaf {
            num_keys: 0,
            keys: Vec::new(),
            values: Vec::new(),
            next_leaf: 0,
            prev_leaf: 0,
        }
    }

    /// Create a new empty internal node.
    fn new_internal() -> Self {
        BTreeNode::Internal {
            num_keys: 0,
            keys: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Serialize this node into a byte buffer for writing to disk.
    fn serialize(&self) -> Vec<u8> {
        let mut buf = vec![0u8; BTREE_PAGE_SIZE];
        match self {
            BTreeNode::Internal { num_keys, keys, children } => {
                // Header
                buf[0..4].copy_from_slice(&PAGE_TYPE_INTERNAL.to_le_bytes());
                buf[4..8].copy_from_slice(&num_keys.to_le_bytes());

                // Children (starts at offset 8)
                let children_start = PAGE_HEADER;
                for (i, &child) in children.iter().enumerate() {
                    let off = children_start + i * CHILD_SIZE;
                    if off + CHILD_SIZE <= BTREE_PAGE_SIZE {
                        buf[off..off + CHILD_SIZE].copy_from_slice(&child.to_le_bytes());
                    }
                }

                // Keys (after children)
                let mut key_off = children_start + children.len() * CHILD_SIZE;
                for key in keys {
                    if key_off + key.len() <= BTREE_PAGE_SIZE {
                        buf[key_off..key_off + key.len()].copy_from_slice(key);
                        key_off += key.len();
                    }
                }
            }
            BTreeNode::Leaf { num_keys, keys, values, next_leaf, prev_leaf } => {
                // Header
                buf[0..4].copy_from_slice(&PAGE_TYPE_LEAF.to_le_bytes());
                buf[4..8].copy_from_slice(&num_keys.to_le_bytes());
                buf[8..12].copy_from_slice(&next_leaf.to_le_bytes());
                buf[12..16].copy_from_slice(&prev_leaf.to_le_bytes());

                // Values (starts at offset 16)
                let values_start = LEAF_HEADER;
                for (i, &(page_id, slot_id)) in values.iter().enumerate() {
                    let off = values_start + i * VALUE_SIZE;
                    if off + VALUE_SIZE <= BTREE_PAGE_SIZE {
                        buf[off..off + 4].copy_from_slice(&page_id.to_le_bytes());
                        buf[off + 4..off + 8].copy_from_slice(&slot_id.to_le_bytes());
                    }
                }

                // Keys (after values)
                let mut key_off = values_start + values.len() * VALUE_SIZE;
                for key in keys {
                    if key_off + key.len() <= BTREE_PAGE_SIZE {
                        buf[key_off..key_off + key.len()].copy_from_slice(key);
                        key_off += key.len();
                    }
                }
            }
        }
        buf
    }

    /// Deserialize a node from a byte buffer.
    fn deserialize(buf: &[u8]) -> io::Result<Self> {
        if buf.len() < PAGE_HEADER {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Page too short for header"));
        }

        let page_type = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let num_keys = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let nk = num_keys as usize;

        match page_type {
            PAGE_TYPE_INTERNAL => {
                let children_start = PAGE_HEADER;
                let num_children = nk + 1;

                // Read children
                let mut children = Vec::with_capacity(num_children);
                for i in 0..num_children {
                    let off = children_start + i * CHILD_SIZE;
                    if off + CHILD_SIZE > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated children"));
                    }
                    let child = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                    children.push(child);
                }

                // Read keys
                let mut key_off = children_start + num_children * CHILD_SIZE;
                let mut keys = Vec::with_capacity(nk);
                for _ in 0..nk {
                    if key_off + KEY_LEN_SIZE > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key length"));
                    }
                    let key_len = u16::from_le_bytes([buf[key_off], buf[key_off + 1]]) as usize;
                    let total_entry = KEY_LEN_SIZE + key_len;
                    if key_off + total_entry > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key data"));
                    }
                    keys.push(buf[key_off..key_off + total_entry].to_vec());
                    key_off += total_entry;
                }

                Ok(BTreeNode::Internal { num_keys, keys, children })
            }
            PAGE_TYPE_LEAF => {
                if buf.len() < LEAF_HEADER {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Page too short for leaf header"));
                }
                let next_leaf = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
                let prev_leaf = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

                // Read values
                let values_start = LEAF_HEADER;
                let mut values = Vec::with_capacity(nk);
                for i in 0..nk {
                    let off = values_start + i * VALUE_SIZE;
                    if off + VALUE_SIZE > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated values"));
                    }
                    let page_id = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                    let slot_id = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
                    values.push((page_id, slot_id));
                }

                // Read keys
                let mut key_off = values_start + nk * VALUE_SIZE;
                let mut keys = Vec::with_capacity(nk);
                for _ in 0..nk {
                    if key_off + KEY_LEN_SIZE > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key length"));
                    }
                    let key_len = u16::from_le_bytes([buf[key_off], buf[key_off + 1]]) as usize;
                    let total_entry = KEY_LEN_SIZE + key_len;
                    if key_off + total_entry > buf.len() {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated key data"));
                    }
                    keys.push(buf[key_off..key_off + total_entry].to_vec());
                    key_off += total_entry;
                }

                Ok(BTreeNode::Leaf { num_keys, keys, values, next_leaf, prev_leaf })
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown page type: {}", page_type))),
        }
    }

    /// Check whether this node can accept another entry of the given key size.
    fn has_space_for(&self, encoded_key_size: usize) -> bool {
        match self {
            BTreeNode::Internal { num_keys, keys, children: _ } => {
                let nk = *num_keys as usize;
                let used = PAGE_HEADER
                    + (nk + 1) * CHILD_SIZE  // children
                    + total_keys_size(keys);  // keys
                let needed = CHILD_SIZE + encoded_key_size; // one more child + one key
                used + needed <= BTREE_PAGE_SIZE
            }
            BTreeNode::Leaf { num_keys, keys, values: _, .. } => {
                let nk = *num_keys as usize;
                let used = LEAF_HEADER
                    + nk * VALUE_SIZE  // values
                    + total_keys_size(keys);  // keys
                let needed = VALUE_SIZE + encoded_key_size; // one more value + one key
                used + needed <= BTREE_PAGE_SIZE
            }
        }
    }
}

// ─── BTree ───────────────────────────────────────────────────────────────────

/// A page-based B+ Tree index stored in a `.idx` file.
///
/// Provides point lookups, range scans, and ordered insertion.
pub struct BTree {
    /// Path to the `.idx` file.
    #[allow(dead_code)]
    file_path: PathBuf,
    /// Open file handle.
    file: File,
    /// The data type(s) of the indexed column(s). A single entry is a plain
    /// single-column index; multiple entries form a composite key where each
    /// value contributes one `[len][bytes]` segment in order.
    key_types: Vec<DataType>,
    /// Cached root page ID.
    root_page_id: u32,
    /// Total number of pages in the index file.
    total_pages: u32,
}

impl BTree {
    /// Create a new B+ Tree index file.
    pub fn create(file_path: PathBuf, key_type: DataType) -> io::Result<Self> {
        Self::create_composite(file_path, vec![key_type])
    }

    /// Create a new composite-key B+ Tree index file.
    pub fn create_composite(file_path: PathBuf, key_types: Vec<DataType>) -> io::Result<Self> {
        log::info!("[BTree::create] Creating new index at {:?} ({} col)", file_path, key_types.len());

        if key_types.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Composite index requires at least one key column",
            ));
        }

        // Remove existing file if present
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)?;

        // Create the root node (empty leaf)
        let root = BTreeNode::new_leaf();
        let buf = root.serialize();
        file.write_all(&buf)?;
        file.flush()?;
        file.sync_all()?;

        // Persist the root pointer sidecar so reopen lands on the real root.
        write_root_sidecar(&file_path, 0)?;

        log::info!("[BTree::create] Created with 1 page, root=0");
        Ok(Self {
            file_path,
            file,
            key_types,
            root_page_id: 0,
            total_pages: 1,
        })
    }

    /// Open an existing B+ Tree index file.
    pub fn open(file_path: PathBuf) -> io::Result<Self> {
        log::info!("[BTree::open] Opening index at {:?}", file_path);

        if !file_path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "Index file not found"));
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file_path)?;

        let file_size = file.metadata()?.len();
        let total_pages = (file_size as usize / BTREE_PAGE_SIZE) as u32;

        // The key_type is not stored in the file yet — callers must provide it
        // In a full implementation, type info would be stored in a header page.
        // For now, use a placeholder (will be set explicitly).
        let key_types = vec![DataType::Int];

        // Restore the persisted root pointer. Without it, every reopened
        // index descended into the original leftmost leaf and searched via
        // the next_leaf chain — O(n) lookups on any real tree.
        let root_page_id = read_root_sidecar(&file_path).unwrap_or(0);

        log::info!(
            "[BTree::open] Opened with {} pages, root={}",
            total_pages, root_page_id
        );
        Ok(Self {
            file_path,
            file,
            key_types,
            root_page_id,
            total_pages,
        })
    }

    /// Set the key type (required after opening if the placeholder was used).
    pub fn set_key_type(&mut self, ty: DataType) {
        self.key_types = vec![ty];
    }

    /// Set the composite key types (required after opening).
    pub fn set_key_types(&mut self, types: Vec<DataType>) {
        assert!(!types.is_empty(), "key_types must not be empty");
        self.key_types = types;
    }

    /// Return the single-column key type (first segment for composites).
    pub fn key_type(&self) -> &DataType {
        &self.key_types[0]
    }

    /// Debug: root page id.
    pub fn root_page_id(&self) -> u32 {
        self.root_page_id
    }

    /// Debug: decoded separator keys stored in the root node.
    pub fn debug_root_keys(&mut self) -> Vec<String> {
        let node = self.read_node(self.root_page_id).expect("read root");
        match node {
            BTreeNode::Internal { keys, children, .. } => {
                let _ = children;
                keys.iter()
                    .map(|k| {
                        let (v, _) = decode_key(k, &self.key_types[0])
                            .map_err(|e| format!("decode err: {}", e))
                            .unwrap_or((DataValue::Int(-1), 0));
                        format!("{:?}", v)
                    })
                    .collect()
            }
            BTreeNode::Leaf { keys, .. } => vec![format!("<leaf with {} keys>", keys.len())],
        }
    }

    /// Return all key segment types.
    pub fn key_types(&self) -> &[DataType] {
        &self.key_types
    }

    // ─── Composite Key Comparison Helpers ──────────────────────────────────

    /// Encode lookup/insert keys in the format this tree stores:
    /// bare `[len][bytes]` for single-column trees (legacy on-disk
    /// compatibility), or the framed composite envelope otherwise.
    fn encode_key_for(&self, keys: &[DataValue]) -> io::Result<Vec<u8>> {
        if self.key_types.len() == 1 {
            if keys.len() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "single-column index takes exactly one key value",
                ));
            }
            Ok(encode_key(&keys[0]))
        } else {
            encode_key_multi(keys)
        }
    }


    /// Compare two encoded keys segment-by-segment using `key_types`.
    ///
    /// Encoded keys are opaque `[len][bytes]` concatenations; decoding each
    /// segment with its declared type preserves value ordering (plain byte
    /// comparison would not — little-endian integers sort wrong as bytes).
    ///
    /// Single-column trees store bare segments (legacy format); composite
    /// trees wrap the whole key in an envelope (see `encode_key_multi`).
    fn cmp_encoded_keys(&self, a: &[u8], b: &[u8]) -> io::Result<Ordering> {
        use std::cmp::Ordering as O;
        if self.key_types.len() == 1 {
            let (va, _) = decode_key(a, &self.key_types[0])?;
            let (vb, _) = decode_key(b, &self.key_types[0])?;
            return va
                .compare(&vb)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
        }
        // Composite: strip the outer envelope from both sides, then walk
        // inner segments in lockstep.
        if a.len() < KEY_LEN_SIZE || b.len() < KEY_LEN_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated composite key envelope"));
        }
        let mut ia = KEY_LEN_SIZE;
        let mut ib = KEY_LEN_SIZE;
        let mut seg = 0usize;
        loop {
            match (ia >= a.len(), ib >= b.len()) {
                (true, true) => return Ok(O::Equal),
                (true, false) => return Ok(O::Less),
                (false, true) => return Ok(O::Greater),
                (false, false) => {}
            }
            // Segments align positionally: type for the current segment index.
            let ty = &self.key_types[seg % self.key_types.len()];
            let (va, ca) = decode_key(&a[ia..], ty)?;
            let (vb, cb) = decode_key(&b[ib..], ty)?;
            let c = va
                .compare(&vb)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if c != O::Equal {
                return Ok(c);
            }
            ia += ca;
            ib += cb;
            seg += 1;
        }
    }

    /// Compare an encoded key against a slice of typed values (one per key
    /// segment). Used by range bounds and lookups supplied as values.
    fn cmp_encoded_vs_values(&self, encoded: &[u8], values: &[DataValue]) -> io::Result<Ordering> {
        use std::cmp::Ordering as O;
        debug_assert_eq!(values.len(), self.key_types.len(), "value/key arity mismatch");
        if self.key_types.len() == 1 {
            let (ev, _) = decode_key(encoded, &self.key_types[0])?;
            return ev
                .compare(&values[0])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
        }
        if encoded.len() < KEY_LEN_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated composite key envelope"));
        }
        let mut off = KEY_LEN_SIZE;
        for (i, v) in values.iter().enumerate() {
            if off >= encoded.len() {
                return Ok(O::Less);
            }
            let (ev, consumed) = decode_key(&encoded[off..], &self.key_types[i])?;
            let c = ev
                .compare(v)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if c != O::Equal {
                return Ok(c);
            }
            off += consumed;
        }
        Ok(O::Equal)
    }

    /// Scan all entries in the B+ Tree by traversing the leaf linked list.
    ///
    /// Returns every `(page_id, slot_id)` pair stored in the index,
    /// in key order. This implements a full index scan.
    pub fn scan_all(&mut self) -> io::Result<Vec<(u32, u32)>> {
        let mut results = Vec::new();

        // Find the leftmost leaf by always following child[0]
        let first_leaf = self.find_leftmost_leaf(self.root_page_id)?;

        // Scan all leaves via the doubly-linked list
        let mut current = first_leaf;
        loop {
            let node = self.read_node(current)?;
            match node {
                BTreeNode::Leaf { values, next_leaf, .. } => {
                    results.extend(values);
                    if next_leaf == 0 {
                        break;
                    }
                    current = next_leaf;
                }
                _ => unreachable!("Expected leaf node in scan_all"),
            }
        }

        Ok(results)
    }

    /// Navigate to the leftmost leaf by always following child[0] from the root.
    fn find_leftmost_leaf(&mut self, page_id: u32) -> io::Result<u32> {
        let node = self.read_node(page_id)?;
        match node {
            BTreeNode::Internal { children, .. } => {
                self.find_leftmost_leaf(children[0])
            }
            BTreeNode::Leaf { .. } => Ok(page_id),
        }
    }

    /// Return the total number of pages in the index file.
    pub fn total_pages(&self) -> u32 {
        self.total_pages
    }

    // ─── Page I/O ──────────────────────────────────────────────────────────

    /// Read a node page from disk.
    fn read_node(&mut self, page_id: u32) -> io::Result<BTreeNode> {
        let offset = page_id as u64 * BTREE_PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; BTREE_PAGE_SIZE];
        self.file.read_exact(&mut buf)?;
        BTreeNode::deserialize(&buf)
    }

    /// Write a node page to disk.
    fn write_node(&mut self, page_id: u32, node: &BTreeNode) -> io::Result<()> {
        let offset = page_id as u64 * BTREE_PAGE_SIZE as u64;
        let buf = node.serialize();
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&buf)?;
        self.file.flush()?;
        Ok(())

    }

    /// Allocate a new page (append to the file).
    fn allocate_page(&mut self, node: &BTreeNode) -> io::Result<u32> {
        let page_id = self.total_pages;
        let buf = node.serialize();
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&buf)?;
        self.file.flush()?;
        self.total_pages += 1;
        Ok(page_id)
    }

    /// Sync all changes to durable storage.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    // ─── Public Search API ─────────────────────────────────────────────────

    /// Search for a key in the B+ Tree.
    ///
    /// Returns `Ok(Some((page_id, slot_id)))` if the key was found, or `Ok(None)` if not.
    pub fn search(&mut self, key: &DataValue) -> io::Result<Option<(u32, u32)>> {
        self.search_keys(std::slice::from_ref(key))
    }

    /// Search for a (possibly composite) key in the B+ Tree.
    pub fn search_keys(&mut self, keys: &[DataValue]) -> io::Result<Option<(u32, u32)>> {
        let encoded = self.encode_key_for(keys)?;
        self.search_in_tree(self.root_page_id, &encoded)
    }

    /// Range search: find all entries with keys in `[low, high]`.
    ///
    /// Returns a vector of heap tuple identifiers.
    pub fn search_range(&mut self, low: &DataValue, high: &DataValue) -> io::Result<Vec<(u32, u32)>> {
        self.search_range_keys(std::slice::from_ref(low), std::slice::from_ref(high))
    }

    /// Range search over composite keys: find entries with key segments
    /// lexicographically within `[low_parts, high_parts]`.
    pub fn search_range_keys(&mut self, low: &[DataValue], high: &[DataValue]) -> io::Result<Vec<(u32, u32)>> {
        let low_encoded = self.encode_key_for(low)?;
        let mut results = Vec::new();

        // Navigate to the leaf containing `low`
        let leaf_id = self.find_leaf(self.root_page_id, &low_encoded)?;

        // Scan the leaf and subsequent leaves
        let mut current_leaf = leaf_id;
        loop {
            let node = self.read_node(current_leaf)?;
            match node {
                BTreeNode::Leaf { ref keys, ref values, next_leaf, .. } => {
                    for (i, key) in keys.iter().enumerate() {
                        let cmp_low = self.cmp_encoded_vs_values(key, low)?;
                        let cmp_high = self.cmp_encoded_vs_values(key, high)?;
                        if cmp_low != Ordering::Less && cmp_high != Ordering::Greater {
                            results.push(values[i]);
                        } else if cmp_high == Ordering::Greater {
                            // Keys are sorted — we've passed the range
                            return Ok(results);
                        }
                    }
                    if next_leaf == 0 {
                        break;
                    }
                    current_leaf = next_leaf;
                }
                _ => unreachable!("Expected leaf node in range scan"),
            }
        }

        Ok(results)
    }

    /// Check whether a key exists in the tree.
    pub fn contains(&mut self, key: &DataValue) -> io::Result<bool> {
        self.search(key).map(|opt| opt.is_some())
    }

    /// Check whether a (possibly composite) key exists in the tree.
    pub fn contains_keys(&mut self, keys: &[DataValue]) -> io::Result<bool> {
        self.search_keys(keys).map(|opt| opt.is_some())
    }

    // ─── Public Search All API ────────────────────────────────────────────

    /// Search for all entries matching a key in the B+ Tree.
    ///
    /// Returns all `(page_id, slot_id)` pairs whose key equals the given key.
    /// For indexes without duplicates, this returns zero or one entry.
    /// For non-unique indexes, this returns all matching entries.
    pub fn search_all(&mut self, key: &DataValue) -> io::Result<Vec<(u32, u32)>> {
        self.search_all_keys(std::slice::from_ref(key))
    }

    /// Search for all entries matching a (possibly composite) key.
    pub fn search_all_keys(&mut self, keys: &[DataValue]) -> io::Result<Vec<(u32, u32)>> {
        let encoded = self.encode_key_for(keys)?;
        self.search_all_in_tree(self.root_page_id, &encoded)
    }

    /// Internal search_all: return all entries matching encoded_key.
    fn search_all_in_tree(&mut self, page_id: u32, encoded_key: &[u8]) -> io::Result<Vec<(u32, u32)>> {
        let node = self.read_node(page_id)?;
        match node {
            BTreeNode::Internal { keys, children, .. } => {
                let mut child_idx = children.len() - 1;
                for (i, key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                    if cmp == Ordering::Greater {
                        child_idx = i;
                        break;
                    }
                }
                self.search_all_in_tree(children[child_idx], encoded_key)
            }
            BTreeNode::Leaf { keys, values, .. } => {
                let mut results = Vec::new();
                for (i, key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                    if cmp == Ordering::Equal {
                        results.push(values[i]);
                    } else if cmp == Ordering::Greater {
                        // Keys are sorted — we've passed all matching entries
                        break;
                    }
                }
                Ok(results)
            }
        }
    }

    // ─── Public Delete API ─────────────────────────────────────────────────

    /// Delete a key from the B+ Tree.
    ///
    /// Searches for an entry matching BOTH the key and the heap location
    /// `(page_id, slot_id)`, and removes it if found. Matching by value is
    /// essential for non-unique indexes where the same key can appear in
    /// multiple entries.
    ///
    /// Does NOT rebalance or merge underfull nodes — the tree remains
    /// searchable but may have reduced space utilization.
    ///
    /// Returns `true` if the entry was found and deleted, `false` if not found.
    pub fn delete(&mut self, key: &DataValue, page_id: u32, slot_id: u32) -> io::Result<bool> {
        self.delete_keys(std::slice::from_ref(key), page_id, slot_id)
    }

    /// Delete a (possibly composite) key from the B+ Tree.
    pub fn delete_keys(&mut self, keys: &[DataValue], page_id: u32, slot_id: u32) -> io::Result<bool> {
        let encoded = self.encode_key_for(keys)?;
        self.delete_from_tree(self.root_page_id, &encoded, page_id, slot_id)
    }

    /// Internal delete: search and remove a specific (key, page_id, slot_id) entry.
    fn delete_from_tree(&mut self, page_id: u32, encoded_key: &[u8], value_page_id: u32, value_slot_id: u32) -> io::Result<bool> {
        let node = self.read_node(page_id)?;

        if matches!(node, BTreeNode::Leaf { .. }) {
            // Leaf node — find and remove the specific entry
            let mut node = node;
            match &mut node {
                BTreeNode::Leaf { num_keys, keys, values, .. } => {
                    for i in 0..keys.len() {
                        let cmp = self.cmp_encoded_keys(&keys[i], encoded_key)?;
                        if cmp == Ordering::Equal && values[i] == (value_page_id, value_slot_id) {
                            // Exact match — remove this specific entry
                            keys.remove(i);
                            values.remove(i);
                            *num_keys -= 1;
                            self.write_node(page_id, &node)?;
                            return Ok(true);
                        }
                        if cmp == Ordering::Greater {
                            // Key not in this leaf (past sorted position)
                            break;
                        }
                    }
                    Ok(false) // Key not found
                }
                _ => unreachable!(),
            }
        } else {
            // Internal node — find the child to descend into
            let (keys, children) = match &node {
                BTreeNode::Internal { keys, children, .. } => (keys.clone(), children.clone()),
                _ => unreachable!(),
            };

            let mut child_idx = children.len() - 1;
            for (i, key) in keys.iter().enumerate() {
                let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                if cmp == Ordering::Greater {
                    child_idx = i;
                    break;
                }
            }

            self.delete_from_tree(children[child_idx], encoded_key, value_page_id, value_slot_id)
        }
    }

    // ─── Public Insert API ─────────────────────────────────────────────────

    /// Insert a key-value pair into the B+ Tree.
    ///
    /// The value is a heap tuple identifier `(page_id, slot_id)`.
    /// If the key already exists, the value is overwritten.
    pub fn insert(&mut self, key: &DataValue, page_id: u32, slot_id: u32) -> io::Result<()> {
        self.insert_keys(std::slice::from_ref(key), page_id, slot_id)
    }

    /// Insert a (possibly composite) key-value pair into the B+ Tree.
    pub fn insert_keys(&mut self, keys: &[DataValue], page_id: u32, slot_id: u32) -> io::Result<()> {
        debug_assert_eq!(keys.len(), self.key_types.len(), "key arity must match index arity");
        let encoded = self.encode_key_for(keys)?;

        // Acquire page write lock for the root
        let file_id = file_identity_from_file(&self.file)?;
        let _lock = PageWriteLock::acquire(file_id, self.root_page_id);

        let result = self.insert_internal(self.root_page_id, &encoded, page_id, slot_id)?;

        // Check if the root was split
        if let Some((new_key, new_child)) = result {
            // Create a new internal node as the new root
            let old_root = self.root_page_id;
            let mut new_root_node = BTreeNode::new_internal();
            match &mut new_root_node {
                BTreeNode::Internal { num_keys, keys, children } => {
                    *num_keys = 1;
                    keys.push(new_key);
                    children.push(old_root);
                    children.push(new_child);
                }
                _ => unreachable!(),
            }

            let new_root_id = self.allocate_page(&new_root_node)?;
            self.root_page_id = new_root_id;
            write_root_sidecar(&self.file_path, new_root_id)?;
        }

        Ok(())
    }

    // ─── Internal Tree Operations ──────────────────────────────────────────

    /// Recursive search within the tree starting from the given page.
    fn search_in_tree(&mut self, page_id: u32, encoded_key: &[u8]) -> io::Result<Option<(u32, u32)>> {
        let node = self.read_node(page_id)?;
        match node {
            BTreeNode::Internal { keys, children, .. } => {
                // keys[i] = min key in children[i+1]
                // If search_key < keys[i]: go to children[i]
                // If search_key >= keys[i]: continue (will go to children[i+1] ultimately)
                let mut child_idx = children.len() - 1;
                for (i, key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                    if cmp == Ordering::Greater {
                        // key > target, so target < key → belongs in child i
                        child_idx = i;
                        break;
                    }
                }
                self.search_in_tree(children[child_idx], encoded_key)
            }
            BTreeNode::Leaf { keys, values, .. } => {
                for (i, key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                    if cmp == Ordering::Equal {
                        return Ok(Some(values[i]));
                    }
                    if cmp == Ordering::Greater {
                        break;
                    }
                }
                Ok(None)
            }
        }
    }

    /// Find the leaf node that should contain (or would contain) `encoded_key`.
    fn find_leaf(&mut self, page_id: u32, encoded_key: &[u8]) -> io::Result<u32> {
        let node = self.read_node(page_id)?;
        match node {
            BTreeNode::Internal { keys, children, .. } => {
                let mut child_idx = children.len() - 1;
                for (i, key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                    if cmp == Ordering::Greater {
                        child_idx = i;
                        break;
                    }
                }
                self.find_leaf(children[child_idx], encoded_key)
            }
            BTreeNode::Leaf { .. } => Ok(page_id),
        }
    }

    /// Convert the first key segment from on-disk encoded bytes to a DataValue.
    #[allow(dead_code)]
    fn key_from_encoded(&self, encoded: &[u8]) -> DataValue {
        let (key, _) = decode_key(encoded, &self.key_types[0]).unwrap_or_else(|_| {
            (DataValue::Int(0), 0)
        });
        key
    }



    /// Internal insert: returns `Ok(None)` if no split, or `Ok(Some((middle_key, new_child_id)))` if split.
    fn insert_internal(
        &mut self,
        page_id: u32,
        encoded_key: &[u8],
        value_page_id: u32,
        value_slot_id: u32,
    ) -> io::Result<Option<(Vec<u8>, u32)>> {
        let node = self.read_node(page_id)?;
        let is_leaf = matches!(node, BTreeNode::Leaf { .. });

        if is_leaf {
            let has_space = node.has_space_for(encoded_key.len());
            if has_space {
                self.insert_into_leaf(page_id, encoded_key, value_page_id, value_slot_id)?;
                Ok(None)
            } else {
                let (new_key, new_leaf_id) = self.split_leaf(page_id, encoded_key, value_page_id, value_slot_id)?;
                Ok(Some((new_key, new_leaf_id)))
            }
        } else {
            // Internal node — find child to descend into
            let (keys, children) = match &node {
                BTreeNode::Internal { keys, children, .. } => (keys.clone(), children.clone()),
                _ => unreachable!(),
            };

            let mut child_idx = children.len() - 1;
            for (i, key) in keys.iter().enumerate() {
                let cmp = self.cmp_encoded_keys(key, encoded_key)?;
                if cmp == Ordering::Greater {
                    child_idx = i;
                    break;
                }
            }

            let child_id = children[child_idx];

            // Acquire write lock for the child before modifying
            let file_id = file_identity_from_file(&self.file)?;
            let _lock = PageWriteLock::acquire(file_id, child_id);

            // Recursively insert
            let result = self.insert_internal(child_id, encoded_key, value_page_id, value_slot_id)?;

            // If child was split, we need to insert the promoted key into this node
            if let Some((promoted_key, new_child_id)) = result {
                // Re-read node to check space with the actual promoted key size
                // (the promoted key may have a different size than encoded_key for varlen types)
                let current_node = self.read_node(page_id)?;
                if current_node.has_space_for(promoted_key.len()) {
                    self.insert_into_internal(page_id, child_idx, &promoted_key, new_child_id)?;
                    Ok(None)
                } else {
                    let (new_key, new_sibling_id) = self.split_internal(page_id, child_idx, &promoted_key, new_child_id)?;
                    Ok(Some((new_key, new_sibling_id)))
                }
            } else {
                Ok(None)
            }
        }
    }

    /// Insert a key-value pair into a leaf node (assumes space is available).
    ///
    /// Supports duplicate keys: if the key already exists, the new entry is
    /// inserted alongside existing entries with the same key, preserving
    /// the sorted order and allowing multiple values per key.
    fn insert_into_leaf(
        &mut self,
        page_id: u32,
        encoded_key: &[u8],
        value_page_id: u32,
        value_slot_id: u32,
    ) -> io::Result<()> {
        let mut node = self.read_node(page_id)?;

        match &mut node {
            BTreeNode::Leaf { num_keys, keys, values, .. } => {
                let nk = *num_keys as usize;
                let target_encoded = encoded_key.to_vec();

                // Find the insertion position.
                // For duplicate keys, insert AFTER all entries with the same key
                // so that the sort order is: key < == == < <
                let mut insert_pos = nk;
                for (i, existing_key) in keys.iter().enumerate() {
                    let cmp = self.cmp_encoded_keys(existing_key, encoded_key)?;
                    if cmp == Ordering::Greater {
                        insert_pos = i;
                        break;
                    }
                    // If cmp == Equal, continue scanning to find the insertion
                    // point after all entries with the same key.
                }

                // Insert new key and value
                keys.insert(insert_pos, target_encoded);
                values.insert(insert_pos, (value_page_id, value_slot_id));
                *num_keys += 1;
            }
            _ => unreachable!("insert_into_leaf called on non-leaf"),
        }

        self.write_node(page_id, &node)?;
        Ok(())
    }

    /// Insert a key and new child pointer into an internal node (assumes space is available).
    fn insert_into_internal(
        &mut self,
        page_id: u32,
        child_idx: usize,
        promoted_key: &[u8],
        new_child_id: u32,
    ) -> io::Result<()> {
        let mut node = self.read_node(page_id)?;

        match &mut node {
            BTreeNode::Internal { num_keys, keys, children, .. } => {
                // The promoted key goes at position child_idx in the key array.
                // The new child goes at position child_idx + 1 in the children array.
                keys.insert(child_idx, promoted_key.to_vec());
                children.insert(child_idx + 1, new_child_id);
                *num_keys += 1;
            }
            _ => unreachable!("insert_into_internal called on non-leaf"),
        }

        self.write_node(page_id, &node)?;
        Ok(())
    }

    /// Split a leaf node that is full. Distributes keys between the old node and a new sibling.
    /// Returns the middle key and the new sibling's page ID.
    fn split_leaf(
        &mut self,
        page_id: u32,
        encoded_key: &[u8],
        value_page_id: u32,
        value_slot_id: u32,
    ) -> io::Result<(Vec<u8>, u32)> {
        let node = self.read_node(page_id)?;

        let (old_keys, old_values, old_next_leaf, old_prev_leaf) = match &node {
            BTreeNode::Leaf { keys, values, next_leaf, prev_leaf, .. } => {
                (keys.clone(), values.clone(), *next_leaf, *prev_leaf)
            }
            _ => unreachable!(),
        };

        // Build combined entries
        let mut all_keys = old_keys;
        let mut all_values = old_values;

        let target_encoded = encoded_key.to_vec();
        let mut insert_pos = all_keys.len();
        for (i, key) in all_keys.iter().enumerate() {
            let cmp = self.cmp_encoded_keys(key, encoded_key)?;
            if cmp == Ordering::Greater {
                insert_pos = i;
                break;
            }
        }
        all_keys.insert(insert_pos, target_encoded);
        all_values.insert(insert_pos, (value_page_id, value_slot_id));

        // Split point: keep first half in old node, move second half to new node
        let split_point = all_keys.len() / 2;

        // First key in the new node (promoted to parent)
        let promoted_key = all_keys[split_point].clone();

        // Left half stays in the original node
        let left_keys = all_keys[..split_point].to_vec();
        let left_values = all_values[..split_point].to_vec();

        // Right half goes to a new node
        let right_keys = all_keys[split_point..].to_vec();
        let right_values = all_values[split_point..].to_vec();

        // Update the old node (now the left sibling)
        let left_node = BTreeNode::Leaf {
            num_keys: left_keys.len() as u32,
            keys: left_keys,
            values: left_values,
            next_leaf: 0, // Will be set after allocating the new node
            prev_leaf: old_prev_leaf,
        };

        // Allocate the right sibling
        let right_node = BTreeNode::Leaf {
            num_keys: right_keys.len() as u32,
            keys: right_keys,
            values: right_values,
            next_leaf: old_next_leaf,
            prev_leaf: 0, // Set after left node is written
        };

        // Write the new sibling first so we know its page ID
        let new_leaf_id = self.allocate_page(&right_node)?;

        // Update left node's next_leaf to point to the new sibling
        let left_node = match left_node {
            BTreeNode::Leaf { num_keys, keys, values, next_leaf: _, prev_leaf } => {
                BTreeNode::Leaf {
                    num_keys,
                    keys,
                    values,
                    next_leaf: new_leaf_id,
                    prev_leaf,
                }
            }
            _ => unreachable!(),
        };
        self.write_node(page_id, &left_node)?;

        // Update the right sibling's prev_leaf
        let right_node = match right_node {
            BTreeNode::Leaf { num_keys, keys, values, next_leaf, prev_leaf: _ } => {
                BTreeNode::Leaf {
                    num_keys,
                    keys,
                    values,
                    next_leaf,
                    prev_leaf: page_id,
                }
            }
            _ => unreachable!(),
        };
        self.write_node(new_leaf_id, &right_node)?;

        // If there was a next leaf, update its prev_leaf to point to the new sibling
        if old_next_leaf != 0 && old_next_leaf < self.total_pages {
            if let Ok(mut next_node) = self.read_node(old_next_leaf) {
                match &mut next_node {
                    BTreeNode::Leaf { prev_leaf, .. } => {
                        *prev_leaf = new_leaf_id;
                    }
                    _ => {}
                }
                self.write_node(old_next_leaf, &next_node)?;
            }
        }

        Ok((promoted_key, new_leaf_id))
    }

    /// Split an internal node. Similar to split_leaf but handles children pointers.
    fn split_internal(
        &mut self,
        page_id: u32,
        child_idx: usize,
        promoted_key: &[u8],
        new_child_id: u32,
    ) -> io::Result<(Vec<u8>, u32)> {
        let node = self.read_node(page_id)?;

        let (old_keys, old_children) = match &node {
            BTreeNode::Internal { keys, children, .. } => {
                (keys.clone(), children.clone())
            }
            _ => unreachable!(),
        };

        // Build combined entries
        let mut all_keys = old_keys;
        let mut all_children = old_children;

        all_keys.insert(child_idx, promoted_key.to_vec());
        all_children.insert(child_idx + 1, new_child_id);

        // Split point
        let split_point = all_keys.len() / 2;

        // The middle key is promoted to the parent
        let promoted_to_parent = all_keys[split_point].clone();

        // Left half (becomes the existing node)
        let left_keys = all_keys[..split_point].to_vec();
        let left_children = all_children[..=split_point].to_vec();

        // Right half (new sibling node)
        let right_keys = all_keys[split_point + 1..].to_vec();
        let right_children = all_children[split_point + 1..].to_vec();

        // Update the old node
        let left_node = BTreeNode::Internal {
            num_keys: left_keys.len() as u32,
            keys: left_keys,
            children: left_children,
        };
        self.write_node(page_id, &left_node)?;

        // Create the new sibling
        let right_node = BTreeNode::Internal {
            num_keys: right_keys.len() as u32,
            keys: right_keys,
            children: right_children,
        };
        let new_sibling_id = self.allocate_page(&right_node)?;

        Ok((promoted_to_parent, new_sibling_id))
    }

    /// Print the tree structure for debugging.
    #[allow(dead_code)]
    pub fn print_tree(&mut self) -> io::Result<()> {
        println!("=== B+ Tree ===");
        println!("Root page: {}", self.root_page_id);
        println!("Total pages: {}", self.total_pages);
        self.print_node(self.root_page_id, 0)?;
        println!("==============");
        Ok(())
    }

    fn print_node(&mut self, page_id: u32, depth: usize) -> io::Result<()> {
        let indent = "  ".repeat(depth);
        let node = self.read_node(page_id)?;
        match &node {
            BTreeNode::Internal { num_keys, keys, children } => {
                println!("{}[Internal] page={}, num_keys={}, children={:?}",
                    indent, page_id, num_keys, children);
                for (i, key) in keys.iter().enumerate() {
                    let (val, _) = decode_key(key, &self.key_types[0]).unwrap_or((DataValue::Int(0), 0));
                    println!("{}  key[{}] = {}", indent, i, val);
                }
                // Recursively print children
                for &child in children {
                    self.print_node(child, depth + 1)?;
                }
            }
            BTreeNode::Leaf { num_keys, keys, values, next_leaf, prev_leaf } => {
                println!("{}[Leaf] page={}, num_keys={}, next={}, prev={}",
                    indent, page_id, num_keys, next_leaf, prev_leaf);
                for (i, (key, val)) in keys.iter().zip(values.iter()).enumerate() {
                    let (dv, _) = decode_key(key, &self.key_types[0]).unwrap_or((DataValue::Int(0), 0));
                    println!("{}  entry[{}]: key={}, tid=({},{})", indent, i, dv, val.0, val.1);
                }
            }
        }
        Ok(())
    }
}

impl Drop for BTree {
    fn drop(&mut self) {
        if let Err(e) = self.sync() {
            log::error!("[BTree::drop] Error syncing on drop: {}", e);
        }
    }
}

// ─── Helper function to decode a key from a leaf entry ───────────────────────

/// Decode a DataValue from an encoded key byte slice.
pub fn decode_btree_key(encoded: &[u8], ty: &DataType) -> Result<(DataValue, usize), String> {
    decode_key(encoded, ty).map_err(|e| e.to_string())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_index_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("rookdb_btree_test_{}.idx", nanos))
    }

    fn setup_btree() -> (PathBuf, BTree) {
        let path = unique_index_path();
        let btree = BTree::create(path.clone(), DataType::Int).unwrap();
        (path, btree)
    }

    fn cleanup(path: &PathBuf) {
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_create_and_open() {
        let path = unique_index_path();
        let mut btree = BTree::create(path.clone(), DataType::Int).unwrap();
        assert_eq!(btree.total_pages(), 1);
        assert_eq!(btree.root_page_id, 0);

        // Search on empty tree
        let result = btree.search(&DataValue::Int(42)).unwrap();
        assert_eq!(result, None);

        cleanup(&path);
    }

    #[test]
    fn test_insert_and_search_single() {
        let (path, mut btree) = setup_btree();

        btree.insert(&DataValue::Int(42), 1, 5).unwrap();
        let result = btree.search(&DataValue::Int(42)).unwrap();
        assert_eq!(result, Some((1, 5)));

        // Non-existent key
        let result = btree.search(&DataValue::Int(999)).unwrap();
        assert_eq!(result, None);

        cleanup(&path);
    }

    #[test]
    fn test_insert_multiple_and_search() {
        let (path, mut btree) = setup_btree();

        btree.insert(&DataValue::Int(10), 1, 0).unwrap();
        btree.insert(&DataValue::Int(20), 2, 1).unwrap();
        btree.insert(&DataValue::Int(30), 3, 2).unwrap();

        assert_eq!(btree.search(&DataValue::Int(10)).unwrap(), Some((1, 0)));
        assert_eq!(btree.search(&DataValue::Int(20)).unwrap(), Some((2, 1)));
        assert_eq!(btree.search(&DataValue::Int(30)).unwrap(), Some((3, 2)));

        cleanup(&path);
    }

    #[test]
    fn test_insert_duplicate_keys() {
        let (path, mut btree) = setup_btree();

        // Insert two entries with the same key but different values
        btree.insert(&DataValue::Int(42), 1, 5).unwrap();
        btree.insert(&DataValue::Int(42), 2, 10).unwrap();

        // search() returns the FIRST entry with this key
        let result = btree.search(&DataValue::Int(42)).unwrap();
        assert_eq!(result, Some((1, 5))); // First entry preserved

        // search_all() returns ALL entries with this key
        let all = btree.search_all(&DataValue::Int(42)).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], (1, 5));
        assert_eq!(all[1], (2, 10));

        cleanup(&path);
    }

    #[test]
    fn test_delete_specific_duplicate_entry() {
        let (path, mut btree) = setup_btree();

        // Insert three entries: two with key=42, one with key=99
        btree.insert(&DataValue::Int(42), 1, 5).unwrap();
        btree.insert(&DataValue::Int(99), 3, 0).unwrap();
        btree.insert(&DataValue::Int(42), 2, 10).unwrap();

        // Verify three total entries
        let all_42 = btree.search_all(&DataValue::Int(42)).unwrap();
        assert_eq!(all_42.len(), 2);

        // Delete one specific entry (42, page=1, slot=5)
        let deleted = btree.delete(&DataValue::Int(42), 1, 5).unwrap();
        assert!(deleted);

        // Verify only the other duplicate remains
        let all_42 = btree.search_all(&DataValue::Int(42)).unwrap();
        assert_eq!(all_42.len(), 1);
        assert_eq!(all_42[0], (2, 10));

        // Key 99 should be untouched
        assert_eq!(btree.search(&DataValue::Int(99)).unwrap(), Some((3, 0)));

        cleanup(&path);
    }

    #[test]
    fn test_delete_nonexistent_entry() {
        let (path, mut btree) = setup_btree();
        btree.insert(&DataValue::Int(42), 1, 5).unwrap();

        // Wrong page_id should not match
        let deleted = btree.delete(&DataValue::Int(42), 999, 0).unwrap();
        assert!(!deleted);

        // Original entry should be untouched
        assert_eq!(btree.search(&DataValue::Int(42)).unwrap(), Some((1, 5)));

        cleanup(&path);
    }

    #[test]
    fn test_contains() {
        let (path, mut btree) = setup_btree();

        btree.insert(&DataValue::Int(100), 1, 0).unwrap();
        assert!(btree.contains(&DataValue::Int(100)).unwrap());
        assert!(!btree.contains(&DataValue::Int(200)).unwrap());

        cleanup(&path);
    }

    #[test]
    fn test_range_search() {
        let (path, mut btree) = setup_btree();

        for i in 0..20 {
            btree.insert(&DataValue::Int(i * 10), i as u32, 0).unwrap();
        }

        // Search range [50, 150]
        let results = btree.search_range(&DataValue::Int(50), &DataValue::Int(150)).unwrap();
        // Expect: 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        // Our data: 0, 10, 20, ..., 190
        // In range: 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        assert_eq!(results.len(), 11);
        assert_eq!(results[0], (5, 0)); // key=50
        assert_eq!(results[10], (15, 0)); // key=150

        cleanup(&path);
    }

    #[test]
    fn test_many_inserts_trigger_split() {
        let (path, mut btree) = setup_btree();

        // Insert enough keys to trigger splits
        // With 8KB pages and 4-byte INT keys (2+4 bytes each), we can fit ~800 keys per leaf
        // Insert 2000 keys to force multiple splits and a multi-level tree
        for i in 0..2000u32 {
            btree.insert(&DataValue::Int(i as i32), i, 0).unwrap();
        }

        // Verify all keys are searchable
        for i in 0..2000u32 {
            let result = btree.search(&DataValue::Int(i as i32)).unwrap();
            assert!(result.is_some(), "Key {} should exist", i);
            assert_eq!(result.unwrap().0, i);
        }

        // Verify tree structure (should have multiple levels now)
        assert!(btree.total_pages > 1, "Tree should have multiple pages after 2000 inserts");

        cleanup(&path);
    }

    #[test]
    fn test_insert_out_of_order() {
        let (path, mut btree) = setup_btree();

        // Insert keys in reverse order
        for i in (0..100u32).rev() {
            btree.insert(&DataValue::Int(i as i32), i, 0).unwrap();
        }

        // Verify all keys are present
        for i in 0..100u32 {
            let result = btree.search(&DataValue::Int(i as i32)).unwrap();
            assert!(result.is_some(), "Key {} should exist", i);
            assert_eq!(result.unwrap().0, i);
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistence() {
        let path = unique_index_path();
        let key_type = DataType::Int;

        {
            let mut btree = BTree::create(path.clone(), key_type.clone()).unwrap();
            for i in 0u32..50 {
                btree.insert(&DataValue::Int(i as i32), i, 0).unwrap();
            }
            // Explicitly sync
            btree.sync().unwrap();
        }

        // Re-open the file
        {
            let mut btree = BTree::open(path.clone()).unwrap();
            btree.set_key_type(key_type.clone());

            // Verify keys survived
            for i in 0u32..50 {
                let result = btree.search(&DataValue::Int(i as i32)).unwrap();
                assert!(result.is_some(), "Key {} should survive reopen", i);
                assert_eq!(result.unwrap().0, i);
            }
        }

        cleanup(&path);
    }

    #[test]
    fn test_string_keys() {
        let path = unique_index_path();
        let key_type = DataType::Varchar(100);

        let mut btree = BTree::create(path.clone(), key_type).unwrap();

        btree.insert(&DataValue::Varchar("apple".to_string()), 1, 0).unwrap();
        btree.insert(&DataValue::Varchar("banana".to_string()), 2, 1).unwrap();
        btree.insert(&DataValue::Varchar("cherry".to_string()), 3, 2).unwrap();

        let result = btree.search(&DataValue::Varchar("banana".to_string())).unwrap();
        assert_eq!(result, Some((2, 1)));

        let result = btree.search(&DataValue::Varchar("grape".to_string())).unwrap();
        assert_eq!(result, None);

        cleanup(&path);
    }

    #[test]
    fn test_range_search_strings() {
        let path = unique_index_path();
        let key_type = DataType::Varchar(100);

        let mut btree = BTree::create(path.clone(), key_type).unwrap();

        let fruits = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"];
        for (i, fruit) in fruits.iter().enumerate() {
            btree.insert(&DataValue::Varchar(fruit.to_string()), i as u32, 0).unwrap();
        }

        let results = btree.search_range(
            &DataValue::Varchar("banana".to_string()),
            &DataValue::Varchar("fig".to_string()),
        ).unwrap();

        // Range [banana, fig] inclusive → 5 items
        assert_eq!(results.len(), 5); // banana, cherry, date, elderberry, fig
        assert_eq!(results[0], (1, 0)); // banana
        assert_eq!(results[4], (5, 0)); // fig

        cleanup(&path);
    }

    #[test]
    fn test_large_string_keys() {
        let path = unique_index_path();
        let key_type = DataType::Varchar(1000);

        let mut btree = BTree::create(path.clone(), key_type).unwrap();

        // Insert keys that are long enough to cause splits with fewer entries
        for i in 0..100u32 {
            let s = format!("key_{}_with_padding_{}_done", i, "x".repeat(i as usize % 50));
            btree.insert(&DataValue::Varchar(s.clone()), i, 0).unwrap();
        }

        for i in 0..100u32 {
            let s = format!("key_{}_with_padding_{}_done", i, "x".repeat(i as usize % 50));
            let result = btree.search(&DataValue::Varchar(s)).unwrap();
            assert!(result.is_some(), "Key {} should exist", i);
        }

        cleanup(&path);
    }

    #[test]
    fn test_scan_all_empty() {
        let (path, mut btree) = setup_btree();
        let results = btree.scan_all().unwrap();
        assert!(results.is_empty());
        cleanup(&path);
    }

    #[test]
    fn test_scan_all_single() {
        let (path, mut btree) = setup_btree();
        btree.insert(&DataValue::Int(42), 1, 5).unwrap();
        let results = btree.scan_all().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (1, 5));
        cleanup(&path);
    }

    #[test]
    fn test_scan_all_multiple() {
        let (path, mut btree) = setup_btree();
        let mut expected = Vec::new();
        for i in 0u32..100 {
            btree.insert(&DataValue::Int(i as i32), i, i * 10).unwrap();
            expected.push((i, i * 10));
        }
        let results = btree.scan_all().unwrap();
        assert_eq!(results.len(), 100);
        // Results should be in key order
        for (i, &(page_id, slot_id)) in results.iter().enumerate() {
            assert_eq!(page_id, i as u32, "page_id mismatch at index {}", i);
            assert_eq!(slot_id, (i as u32) * 10, "slot_id mismatch at index {}", i);
        }
        cleanup(&path);
    }
}
