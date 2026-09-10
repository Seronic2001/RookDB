//! Shared helpers for RookDB integration tests.
//!
//! The storage engine resolves every path (`database/base/...`,
//! `database/system/...`) relative to the process working directory, which
//! is shared by all `#[test]` functions in a binary. [`TestWorkspace`] gives
//! each test an isolated directory under the crate root, switches the process
//! into it, and — critically — cleans up even when the test panics: the
//! `Drop` implementation restores the original directory and removes the
//! whole workspace.
//!
//! Because `set_current_dir` is process-wide, tests within one binary must
//! not overlap. `TestWorkspace` therefore also holds the shared [`CWD_MUTEX`]
//! for as long as it lives, so simply creating a guard serialises the test.
//!
//! Workspace names embed the process ID so different test binaries (which
//! run in parallel as separate processes) never collide.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

/// Serialises tests within one binary (see type-level docs).
/// Cross-binary parallelism stays safe thanks to PID-scoped names below.
pub static CWD_MUTEX: Mutex<()> = Mutex::new(());

static SWEEP_STALE: std::sync::Once = std::sync::Once::new();

/// Remove workspace directories left behind by *dead* processes.
///
/// Guards cannot clean up when a test run is killed outright (SIGKILL /
/// timeout / IDE stop button), so this runs once per process before the
/// first workspace is created: any `database_<prefix>_p<PID>_<tag>` whose
/// PID no longer exists is stale by definition and gets removed. Live
/// processes' workspaces are never touched, keeping parallel binaries safe.
fn sweep_stale_workspaces() {
    SWEEP_STALE.call_once(|| {
        let Ok(entries) = std::fs::read_dir(".") else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };

            // Strict shape check — never touch anything we can't positively
            // attribute to a test workspace (e.g. real `database/` data).
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() < 4 || !name.starts_with("database_") {
                continue;
            }
            let Some(pid_str) = parts[2].strip_prefix('p') else { continue };
            let Ok(pid) = pid_str.parse::<u32>() else { continue };

            let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
            if !alive {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    });
}



pub struct TestWorkspace {
    /// Absolute path of the original CWD, restored on drop.
    prev_cwd: PathBuf,
    /// Absolute path of the workspace, removed on drop.
    path: PathBuf,
    /// Held for this workspace's lifetime so no other test in this binary
    /// can touch the shared working directory while it is active.
    _lock: Option<MutexGuard<'static, ()>>,
}

impl TestWorkspace {
    /// Create `<crate root>/database_<prefix>_p<pid>_<tag>/`, chdir into it
    /// and bootstrap the system catalog inside it.
    ///
    /// The returned guard must be bound (`let _ws = ...`) so it lives until
    /// the end of the test; dropping it undoes everything.
    pub fn new(prefix: &str, tag: &str) -> Self {
        sweep_stale_workspaces();
        // Ignore poisoning: a panicked predecessor leaves the mutex locked,
        // but the CWD bookkeeping itself stays consistent.
        let lock = CWD_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let prev_cwd = std::env::current_dir().expect("read cwd");
        let path = prev_cwd.join(format!(
            "database_{}_p{}_{}",
            prefix,
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("base")).expect("create workspace");

        std::env::set_current_dir(&path).expect("chdir into workspace");
        storage_manager::backend::executor::row_select::register_where_parser(rook_parser::parse_where_text);
        storage_manager::backend::cache::register_check_parser(rook_parser::parse_check_expr);
        storage_manager::catalog::init_catalog();

        Self {
            prev_cwd,
            path,
            _lock: Some(lock),
        }
    }

    /// Absolute path of the workspace (rarely needed; paths are relative).
    #[allow(dead_code)]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        // Leave the workspace before deleting it: removing the current
        // directory of a running process fails on some platforms. The mutex
        // guard field unlocks afterwards, letting the next test proceed.
        if std::env::set_current_dir(&self.prev_cwd).is_ok() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
        self._lock = None;
    }
}
