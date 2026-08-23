//! Typed error hierarchy for programmatic error handling
//! (ANALYSIS.md Tier 1 #2).
//!
//! Historically every fallible API returned `Result<T, String>`, forcing
//! callers to string-match to distinguish a constraint violation from an
//! I/O failure. [`RookError`] gives errors structure while remaining cheap
//! to adopt: `From<String>` lets existing string-producing code paths flow
//! into typed results through `?` unchanged, and `Display` renders the same
//! human-readable messages as before.
//!
//! ```ignore
//! match validate_row_insert(&catalog, "db", "t", &["NULL"]) {
//!     Err(RookError::ConstraintViolation { kind: ConstraintKind::NotNull, .. }) => { … }
//!     Err(e) if e.is_io() => { … }
//!     Err(e) => eprintln!("{}", e),
//!     Ok(()) => { … }
//! }
//! ```

use std::fmt;

/// The class of constraint that was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    NotNull,
    Unique,
    ForeignKey,
    Check,
}

impl ConstraintKind {
    /// Stable machine-readable name (suitable for logs / API responses).
    pub fn as_str(&self) -> &'static str {
        match self {
            ConstraintKind::NotNull => "NOT NULL",
            ConstraintKind::Unique => "UNIQUE",
            ConstraintKind::ForeignKey => "FOREIGN KEY",
            ConstraintKind::Check => "CHECK",
        }
    }
}

impl fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured error type for RookDB operations.
#[derive(Debug)]
pub enum RookError {
    /// A NOT NULL / UNIQUE / FOREIGN KEY / CHECK constraint was violated.
    ConstraintViolation {
        kind: ConstraintKind,
        table: String,
        column: Option<String>,
        message: String,
    },
    /// A referenced database / table / column does not exist.
    NotFound {
        entity: &'static str,
        name: String,
    },
    /// A name failed path-safety validation (see `name_validation`).
    InvalidIdentifier(String),
    /// A value could not be interpreted for its column's data type.
    TypeMismatch(String),
    /// Underlying storage I/O failure.
    Io(std::io::Error),
    /// Anything not yet covered by a structured variant.
    Internal(String),
}

impl RookError {
    /// Convenience constructor for constraint violations.
    pub fn constraint(
        kind: ConstraintKind,
        table: &str,
        column: Option<&str>,
        message: impl Into<String>,
    ) -> Self {
        RookError::ConstraintViolation {
            kind,
            table: table.to_string(),
            column: column.map(str::to_string),
            message: message.into(),
        }
    }

    /// `true` when this error represents any constraint violation.
    pub fn is_constraint_violation(&self) -> bool {
        matches!(self, RookError::ConstraintViolation { .. })
    }

    /// The violated constraint kind, if this is a constraint violation.
    pub fn constraint_kind(&self) -> Option<ConstraintKind> {
        match self {
            RookError::ConstraintViolation { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    /// `true` when the failure came from the storage layer.
    pub fn is_io(&self) -> bool {
        matches!(self, RookError::Io(_))
    }
}

impl fmt::Display for RookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RookError::ConstraintViolation { message, .. } => f.write_str(message),
            RookError::NotFound { entity, name } => {
                write!(f, "{} '{}' not found", entity, name)
            }
            RookError::InvalidIdentifier(name) => write!(f, "Invalid identifier '{}'", name),
            RookError::TypeMismatch(message) => f.write_str(message),
            RookError::Io(e) => write!(f, "I/O error: {}", e),
            RookError::Internal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RookError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Bridge legacy `String` errors into the typed hierarchy.
impl From<String> for RookError {
    fn from(s: String) -> Self {
        // Recognise the engine's canonical violation prefixes so legacy
        // string sites still produce structured variants.
        let lower = s.to_ascii_lowercase();
        if lower.starts_with("not null constraint") {
            return RookError::constraint(ConstraintKind::NotNull, "", None, s);
        }
        if lower.starts_with("unique constraint") {
            return RookError::constraint(ConstraintKind::Unique, "", None, s);
        }
        if lower.starts_with("foreign key constraint") {
            return RookError::constraint(ConstraintKind::ForeignKey, "", None, s);
        }
        if lower.starts_with("check constraint") {
            return RookError::constraint(ConstraintKind::Check, "", None, s);
        }
        RookError::Internal(s)
    }
}

impl From<&str> for RookError {
    fn from(s: &str) -> Self {
        RookError::from(s.to_string())
    }
}

impl From<std::io::Error> for RookError {
    fn from(e: std::io::Error) -> Self {
        RookError::Io(e)
    }
}

/// Result alias used by newly migrated APIs.
pub type RookResult<T> = Result<T, RookError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_legacy_message_text() {
        let e = RookError::constraint(
            ConstraintKind::NotNull,
            "users",
            Some("name"),
            "NOT NULL constraint violated: column 'name' cannot be null",
        );
        assert_eq!(
            e.to_string(),
            "NOT NULL constraint violated: column 'name' cannot be null"
        );
    }

    #[test]
    fn kind_accessors() {
        let e = RookError::constraint(ConstraintKind::Unique, "t", None, "boom");
        assert!(e.is_constraint_violation());
        assert_eq!(e.constraint_kind(), Some(ConstraintKind::Unique));
        assert!(!e.is_io());

        let io = RookError::from(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(io.is_io());
        assert_eq!(io.constraint_kind(), None);
    }

    #[test]
    fn from_string_detects_violation_prefixes() {
        let e: RookError = "UNIQUE constraint violated: value '1' dup".into();
        assert!(e.is_constraint_violation());
        assert_eq!(e.constraint_kind(), Some(ConstraintKind::Unique));

        let e: RookError = "some random failure".into();
        assert!(!e.is_constraint_violation());
        assert!(matches!(e, RookError::Internal(_)));
    }

    #[test]
    fn question_mark_auto_converts_strings() {
        fn inner() -> Result<(), String> {
            Err("CHECK constraint violated: 'x'".to_string())
        }
        fn outer() -> RookResult<()> {
            inner()?;
            Ok(())
        }
        let err = outer().unwrap_err();
        assert_eq!(err.constraint_kind(), Some(ConstraintKind::Check));
    }
}
