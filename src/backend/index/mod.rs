//! B+ Tree Indexing Module
//!
//! Provides a page-based B+ Tree for efficient key-based lookup and range scans.
//! Each index is stored in a separate `<table>.idx` file with 8 KB pages.
//!
//! # Page Layout
//!
//! ## Internal Node (page_type = 0)
//! ```text
//! [page_type: u32(4)] [num_keys: u32(4)]
//! [child_0: u32(4)] [child_1: u32(4)] ... [child_n: u32(4)]
//! [key_0_len: u16(2)] [key_0_bytes...]
//! [key_1_len: u16(2)] [key_1_bytes...]
//! ...
//! [key_n-1_len: u16(2)] [key_n-1_bytes...]
//! ```
//!
//! ## Leaf Node (page_type = 1)
//! ```text
//! [page_type: u32(4)] [num_keys: u32(4)] [next_leaf: u32(4)] [prev_leaf: u32(4)]
//! [val_0_page_id: u32(4)] [val_0_slot_id: u32(4)]
//! [val_1_page_id: u32(4)] [val_1_slot_id: u32(4)]
//! ...
//! [val_n-1_page_id: u32(4)] [val_n-1_slot_id: u32(4)]
//! [key_0_len: u16(2)] [key_0_bytes...]
//! [key_1_len: u16(2)] [key_1_bytes...]
//! ...
//! [key_n-1_len: u16(2)] [key_n-1_bytes...]
//! ```

pub mod btree;
pub use btree::BTree;
