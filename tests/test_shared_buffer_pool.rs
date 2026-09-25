//! Integration tests for the process-wide shared buffer pool
//! (ANALYSIS.md Tier 2 #9).
//!
//! Previously every `HeapManager` owned a private pool, so two managers on
//! one `.dat` file held independent caches — writes through one handle were
//! invisible (or even clobbered) by the other. They now share one pool keyed
//! by canonical file identity.

use std::path::PathBuf;
use storage_manager::heap::HeapManager;

/// Unique absolute path for an isolated test file.
fn temp_file(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rookdb_shp_p{}_{}", std::process::id(), tag));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{}.dat", tag))
}

#[test]
fn two_managers_share_one_pool() {
    let path = temp_file("share");
    let _ = std::fs::remove_file(&path);

    {
        let mut a = HeapManager::create(path.clone()).unwrap();
        let mut b = HeapManager::open(path.clone()).unwrap();

        // Same underlying pool object.
        assert!(
            Arc::ptr_eq(&a.pool, &b.pool),
            "managers on the same file must share one pool"
        );

        // Insert through A; B must observe the tuple WITHOUT any flush,
        // because both handles look at the same cached page.
        a.insert_tuple(b"hello world").unwrap();
        let got = b
            .get_tuple(1, 0)
            .expect("B must see A's insert via shared cache");
        assert_eq!(got, b"hello world");
    }

    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn registry_reuses_entry_and_sweeps_expired() {
    let path = temp_file("registry");
    let _ = std::fs::remove_file(&path);

    let first = HeapManager::create(path.clone()).unwrap();
    let weak_old = std::sync::Arc::downgrade(&first.pool);
    let second = HeapManager::open(path.clone()).unwrap();
    assert!(Arc::ptr_eq(&first.pool, &second.pool));

    // Drop all handles → Weak expires → fresh manager gets a NEW pool.
    drop(first);
    drop(second);
    let third = HeapManager::open(path.clone()).unwrap();
    let fourth = HeapManager::open(path.clone()).unwrap();
    assert!(Arc::ptr_eq(&third.pool, &fourth.pool));
    assert!(
        weak_old.upgrade().is_none(),
        "dropped pool must be released; a fresh one must be created"
    );

    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn data_survives_full_drop_cycle() {
    let path = temp_file("persist");
    let _ = std::fs::remove_file(&path);

    {
        let mut m = HeapManager::create(path.clone()).unwrap();
        m.insert_tuple(b"row-one").unwrap();
        m.insert_tuple(b"row-two").unwrap();
        // No explicit flush — Drop of the shared pool must persist.
    }

    let mut reopened = HeapManager::open(path.clone()).unwrap();
    assert_eq!(reopened.get_tuple(1, 0).unwrap(), b"row-one");
    assert_eq!(reopened.get_tuple(1, 1).unwrap(), b"row-two");

    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

// Arc is used only through the type alias in assertions.
use std::sync::Arc;

#[test]
fn create_replaces_stale_pool_for_same_path() {
    let path = temp_file("replace");
    let _ = std::fs::remove_file(&path);

    let mut old = HeapManager::create(path.clone()).unwrap();
    old.insert_tuple(b"old-data").unwrap();

    // Recreating the file must install a fresh pool, not reuse the old one.
    let mut fresh = HeapManager::create(path.clone()).unwrap();
    drop(old);

    // Fresh file starts empty — reading a stale cached row would be wrong.
    assert!(
        fresh.get_tuple(1, 0).is_err(),
        "recreated file must start empty"
    );

    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}
