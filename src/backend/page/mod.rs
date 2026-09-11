// Page size in bytes (8 KB)
pub const PAGE_SIZE: usize = 8192;

// Page header size: stores lower & upper pointers
pub const PAGE_HEADER_SIZE: u32 = 8;

// Size of one item slot
pub const ITEM_ID_SIZE: u32 = 8;

// Slot flag bit: tuple is soft-deleted
pub const SLOT_FLAG_DELETED: u16 = 0b0000_0000_0000_0001;

// Page size in bytes (8 KB)
pub mod page_lock;

// Represents a single database page
pub struct Page {
    // Raw page bytes
    pub data: Vec<u8>,
}

impl Default for Page {
    fn default() -> Self {
        Self::new()
    }
}

impl Page {
    // Create an empty page
    pub fn new() -> Self {
        Self {
            data: vec![0; PAGE_SIZE],
        }
    }
}

// Initialize page header pointers
pub fn init_page(page: &mut Page) {
    // Lower starts after header
    page.data[0..4].copy_from_slice(&PAGE_HEADER_SIZE.to_le_bytes());

    // Upper starts at page end
    page.data[4..8].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
}

// Return available free space in the page
pub fn page_free_space(page: &Page) -> std::io::Result<u32> {
    let lower = u32::from_le_bytes(page.data[0..4].try_into().unwrap());
    let upper = u32::from_le_bytes(page.data[4..8].try_into().unwrap());

    if lower < PAGE_HEADER_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid page lower pointer: {} < {}", lower, PAGE_HEADER_SIZE),
        ));
    }

    if lower > PAGE_SIZE as u32 || upper > PAGE_SIZE as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Page pointers out of bounds: lower={}, upper={}, page_size={}",
                lower, upper, PAGE_SIZE
            ),
        ));
    }

    if upper < lower {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Corrupted page pointers: upper ({}) < lower ({})",
                upper, lower
            ),
        ));
    }

    Ok(upper - lower)
}

/// Get the number of tuples currently stored in a page.
/// This is calculated as (lower - PAGE_HEADER_SIZE) / ITEM_ID_SIZE.
/// 
/// # Errors
/// Returns error if reading the page header fails.
pub fn get_tuple_count(page: &Page) -> std::io::Result<u32> {
    let lower = u32::from_le_bytes(page.data[0..4].try_into().unwrap());
    let upper = u32::from_le_bytes(page.data[4..8].try_into().unwrap());
    
    if lower < PAGE_HEADER_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid page lower pointer: {} < {}", lower, PAGE_HEADER_SIZE),
        ));
    }

    if lower > PAGE_SIZE as u32 || upper > PAGE_SIZE as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Page pointers out of bounds: lower={}, upper={}, page_size={}",
                lower, upper, PAGE_SIZE
            ),
        ));
    }

    if upper < lower {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Corrupted page pointers: upper ({}) < lower ({})",
                upper, lower
            ),
        ));
    }

    if !(lower - PAGE_HEADER_SIZE).is_multiple_of(ITEM_ID_SIZE) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid slot array alignment: lower={}, header={}, item_id_size={}",
                lower, PAGE_HEADER_SIZE, ITEM_ID_SIZE
            ),
        ));
    }
    
    let tuple_count = (lower - PAGE_HEADER_SIZE) / ITEM_ID_SIZE;
log::trace!("[page::get_tuple_count] Computing tuple_count: ({} - {}) / {} = {}", 
             lower, PAGE_HEADER_SIZE, ITEM_ID_SIZE, tuple_count);
    
    Ok(tuple_count)
}

/// Get slot entry (offset, length) for a given slot ID.
///
/// Uses the canonical 3-field slot format `[offset: u32, length: u16, flags: u16]`.
/// Returns `(0, 0)` for soft-deleted slots (those with `SLOT_FLAG_DELETED` set)
/// so that callers can uniformly treat `(0, 0)` as "skip this slot".
///
/// # Arguments
/// * `page`    - The page to read from
/// * `slot_id` - Zero-based slot index
///
/// # Returns
/// `(offset, length)` of the live tuple data, or `(0, 0)` for deleted slots.
///
/// # Errors
/// Returns an error only if `slot_id` is out of bounds or the page is structurally invalid.
pub fn get_slot_entry(page: &Page, slot_id: u32) -> std::io::Result<(u32, u32)> {
    let tuple_count = get_tuple_count(page)?;

    if slot_id >= tuple_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Slot ID {} out of bounds (tuple_count={})", slot_id, tuple_count),
        ));
    }

    // Canonical slot layout: [offset: u32 (4B)][length: u16 (2B)][flags: u16 (2B)]
    let slot_offset = PAGE_HEADER_SIZE as usize + (slot_id as usize * ITEM_ID_SIZE as usize);

    if slot_offset + ITEM_ID_SIZE as usize > page.data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Slot entry read would exceed page bounds",
        ));
    }

    let offset = u32::from_le_bytes(page.data[slot_offset..slot_offset + 4].try_into().unwrap());
    let length = u16::from_le_bytes(page.data[slot_offset + 4..slot_offset + 6].try_into().unwrap());
    let flags  = u16::from_le_bytes(page.data[slot_offset + 6..slot_offset + 8].try_into().unwrap());

    // Soft-deleted slot: signal caller to skip it.
    if flags & SLOT_FLAG_DELETED != 0 {
        log::trace!("[page::get_slot_entry] Slot {} is soft-deleted — returning (0, 0)", slot_id);
        return Ok((0, 0));
    }

    // Legacy dead-slot marker (offset=0 && length=0): treat the same as deleted.
    if offset == 0 && length == 0 {
        return Ok((0, 0));
    }

    let length_u32 = length as u32;
    if offset > PAGE_SIZE as u32 || length_u32 > PAGE_SIZE as u32 || offset + length_u32 > PAGE_SIZE as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Corrupted slot entry bounds: offset={}, length={}, page_size={}",
                offset, length_u32, PAGE_SIZE
            ),
        ));
    }

    log::trace!("[page::get_slot_entry] Slot {}: offset={}, length={}", slot_id, offset, length_u32);

    Ok((offset, length_u32))
}

/// Read slot `slot_index`
/// Returns: (offset, length, flags)
pub fn read_slot(page: &Page, slot_index: u32) -> (u32, u16, u16) {
    let base = (PAGE_HEADER_SIZE + slot_index * ITEM_ID_SIZE) as usize;

    let offset = u32::from_le_bytes(
        page.data[base..base + 4]
            .try_into()
            .unwrap(),
    );

    let length = u16::from_le_bytes(
        page.data[base + 4..base + 6]
            .try_into()
            .unwrap(),
    );

    let flags = u16::from_le_bytes(
        page.data[base + 6..base + 8]
            .try_into()
            .unwrap(),
    );

    (offset, length, flags)
}

/// Write slot `slot_index` with (offset, length, flags)
pub fn write_slot(
    page: &mut Page,
    slot_index: u32,
    offset: u32,
    length: u16,
    flags: u16,
) {
    let base = (PAGE_HEADER_SIZE + slot_index * ITEM_ID_SIZE) as usize;

    page.data[base..base + 4]
        .copy_from_slice(&offset.to_le_bytes());

    page.data[base + 4..base + 6]
        .copy_from_slice(&length.to_le_bytes());

    page.data[base + 6..base + 8]
        .copy_from_slice(&flags.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_free_space_detects_corrupted_pointers() {
        let mut page = Page::new();
        init_page(&mut page);

        page.data[0..4].copy_from_slice(&100u32.to_le_bytes());
        page.data[4..8].copy_from_slice(&50u32.to_le_bytes());

        let result = page_free_space(&page);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tuple_count_detects_invalid_alignment() {
        let mut page = Page::new();
        init_page(&mut page);

        page.data[0..4].copy_from_slice(&9u32.to_le_bytes());

        let result = get_tuple_count(&page);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_slot_entry_detects_out_of_bounds_tuple() {
        let mut page = Page::new();
        init_page(&mut page);

        let lower = PAGE_HEADER_SIZE + ITEM_ID_SIZE;
        page.data[0..4].copy_from_slice(&lower.to_le_bytes());

        let slot_offset = PAGE_HEADER_SIZE as usize;
        page.data[slot_offset..slot_offset + 4].copy_from_slice(&(PAGE_SIZE as u32 - 4).to_le_bytes());
        page.data[slot_offset + 4..slot_offset + 8].copy_from_slice(&16u32.to_le_bytes());

        let result = get_slot_entry(&page, 0);
        assert!(result.is_err());
    }
}
