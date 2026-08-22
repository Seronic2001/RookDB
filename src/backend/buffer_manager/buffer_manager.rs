use std::io;

use super::buffer_pool::BufferPool;

/// BufferManager — High-level buffer manager wrapping the CLOCK BufferPool.
///
/// Provides backward-compatible API for existing callers while leveraging the
/// new CLOCK-sweep eviction pool internally.
///
/// # Default Pool Capacity
///
/// The default pool size is 128 frames (1 MB). This can be adjusted with
/// `with_pool_capacity()`.
pub struct BufferManager {
    /// The underlying CLOCK buffer pool.
    pub pool: BufferPool,
}

impl BufferManager {
    /// Create a new buffer manager with default capacity (128 frames = 1 MB).
    pub fn new() -> Self {
        Self::with_pool_capacity(128)
    }

    /// Create a new buffer manager with a specified pool capacity.
    pub fn with_pool_capacity(capacity: usize) -> Self {
        log::info!(
            "BufferManager initialized with CLOCK buffer pool ({} frames = {} KB)",
            capacity,
            capacity * 8
        );
        Self {
            pool: BufferPool::new(capacity),
        }
    }

    /// Allocate ONE new data page via the buffer pool.
    ///
    /// Returns the page_id of the newly allocated page.
    pub fn allocate_page(&mut self) -> io::Result<u32> {
        let (page_id, frame_id) = self.pool.new_page()?;
        self.pool.unpin(frame_id, false);
        Ok(page_id)
    }

    /// Loads table from disk into buffer (opens an existing table).
    /// This opens the file and makes it available through the buffer pool.
    pub fn load_table_from_disk(&mut self, db_name: &str, table_name: &str) -> io::Result<()> {
        let table_path = format!("database/base/{}/{}.dat", db_name, table_name);
        let path = std::path::Path::new(&table_path);

        if !path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Table file not found: {}", table_path),
            ));
        }

        log::info!(
            "BufferManager: loading table '{}.{}' into CLOCK buffer pool",
            db_name,
            table_name
        );

        self.pool.open(path)?;

        let total_pages = self.pool.total_pages();
        log::info!(
            "BufferManager: loaded {} pages into buffer pool ({} frame capacity)",
            total_pages,
            self.pool.capacity()
        );

        Ok(())
    }
}
