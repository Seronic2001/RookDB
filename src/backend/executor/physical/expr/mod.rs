//! Expression types and evaluation for the Volcano execution engine.
//!
//! Expressions evaluate against deserialised tuples and produce values.
//! This module also re-exports the predicate sub-module.

pub mod predicate;
pub mod convert;

#[cfg(test)]
pub mod tests;

pub use predicate::{Predicate, evaluate_predicate, ComparisonOp, BooleanTest};
pub use convert::{expr_from_ast, predicate_from_ast};

use std::cell::RefCell;
use std::rc::Rc;

use crate::types::value::DataValue;

use super::tuple::{Tuple, ColumnInfo};

// ── Expressions ───────────────────────────────────────────────────────────────

/// An expression that can be evaluated against a deserialised tuple.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A reference to a column by name, with optional table qualifier.
    ///
    /// Example: `Expr::Column { table: Some("t1"), column: "id".into() }`
    /// for a qualified reference `t1.id`, or `{ table: None, column: "name".into() }`
    /// for an unqualified reference `name`.
    Column {
        table: Option<String>,
        column: String,
    },
    Constant(DataValue),
    Null,
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    /// SQL `CAST(expr AS type)` — explicit type conversion.
    Cast(Box<Expr>, crate::types::datatype::DataType),
    /// A parameter whose value is set from an outer query row (correlated subquery binding).
    /// The `Rc<RefCell<Option<DataValue>>>` is shared between the predicate evaluator
    /// and the physical planner, allowing the value to be set per outer row.
    CorrelatedParam(Rc<RefCell<Option<DataValue>>>),
    /// `CASE WHEN cond1 THEN expr1 [WHEN cond2 THEN expr2] [ELSE exprN] END`.
    Case {
        when_then_pairs: Vec<(Expr, Expr)>,
        else_result: Option<Box<Expr>>,
    },
    /// Scalar function call: `UPPER(expr)`, `LENGTH(expr)`, `TRIM(expr)`, etc.
    /// Arguments are pre-evaluated `Expr` nodes that get evaluated against the
    /// current tuple before the function is applied.
    Function {
        name: String,
        args: Vec<Expr>,
    },
}

impl Expr {
    /// Evaluate this expression against a tuple and schema, returning the resulting value
    /// (or `None` for NULL).
    pub fn evaluate(&self, tuple: &Tuple, schema: &[ColumnInfo]) -> Result<Option<DataValue>, String> {
        match self {
            Expr::Column { table, column } => {
                // Look up the column by name in the schema (case-insensitive).
                // ColumnInfo.table is populated with actual table names by scan/join
                // operators, so table-qualified references like `t1.id` vs `t2.id`
                // resolve to different columns directly.
                //
                // Resolution rules:
                // 1. If Expr has a table qualifier (e.g. `t1.id`):
                //    Match only columns where BOTH the table name AND column name match.
                //    This correctly disambiguates `t1.id` from `t2.id` in join schemas.
                // 2. If Expr has no table qualifier (e.g. `id`):
                //    Match the FIRST column with the given name, regardless of table.
                let idx = match table {
                    Some(t) => {
                        // Table-qualified: match both table and column name
                        schema.iter().position(|ci| {
                            ci.name.eq_ignore_ascii_case(column)
                                && ci.table.as_ref()
                                    .map(|ct| ct.eq_ignore_ascii_case(t))
                                    .unwrap_or(false)
                        })
                    }
                    None => {
                        // Unqualified: match first column by name only
                        schema.iter()
                            .position(|ci| ci.name.eq_ignore_ascii_case(column))
                    }
                }
                .ok_or_else(|| format!(
                    "Column '{}' not found in tuple schema ({:?})",
                    column,
                    schema.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>()
                ))?;
                Ok(tuple.values.get(idx).ok_or_else(|| {
                    format!("Column '{}' index {} out of bounds (arity {})", column, idx, tuple.arity())
                })?.clone())
            }
            Expr::Constant(dv) => Ok(Some(dv.clone())),
            Expr::Null => Ok(None),
            Expr::Add(l, r) => {
                let lv = l.evaluate(tuple, schema)?;
                let rv = r.evaluate(tuple, schema)?;
                arithmetic_op(lv, rv, |a, b| Ok(a + b), |a, b| Ok(a + b), |a, b| Ok(a + b), |a, b| Ok(a + b), |a, b| Ok(a + b))
            }
            Expr::Sub(l, r) => {
                let lv = l.evaluate(tuple, schema)?;
                let rv = r.evaluate(tuple, schema)?;
                arithmetic_op(lv, rv, |a, b| Ok(a - b), |a, b| Ok(a - b), |a, b| Ok(a - b), |a, b| Ok(a - b), |a, b| Ok(a - b))
            }
            Expr::Mul(l, r) => {
                let lv = l.evaluate(tuple, schema)?;
                let rv = r.evaluate(tuple, schema)?;
                arithmetic_op(lv, rv, |a, b| Ok(a * b), |a, b| Ok(a * b), |a, b| Ok(a * b), |a, b| Ok(a * b), |a, b| Ok(a * b))
            }
            Expr::Div(l, r) => {
                let lv = l.evaluate(tuple, schema)?;
                let rv = r.evaluate(tuple, schema)?;
                arithmetic_op(
                    lv, rv,
                    |a, b| if b == 0 { Err("Division by zero".into())} else { Ok(a / b) },
                    |a, b| if b == 0 { Err("Division by zero".into())} else { Ok(a / b) },
                    |a, b| if b == 0 { Err("Division by zero".into())} else { Ok(a / b) },
                    |a, b| if b == 0.0 { Err("Division by zero".into())} else { Ok(a / b) },
                    |a, b| if b == 0.0 { Err("Division by zero".into())} else { Ok(a / b) },
                )
            }
            Expr::Cast(inner, target_type) => {
                let val = inner.evaluate(tuple, schema)?;
                match val {
                    None => Ok(None), // CAST(NULL AS type) → NULL
                    Some(dv) => {
                        let result = crate::types::functions::cast(&dv, target_type)
                            .map_err(|e| format!("CAST error: {}", e))?;
                        Ok(Some(result))
                    }
                }
            }
            Expr::CorrelatedParam(param_cell) => {
                Ok(param_cell.borrow().clone())
            }
            Expr::Case {
                when_then_pairs,
                else_result,
            } => {
                for (cond, res) in when_then_pairs {
                    let cond_val = cond.evaluate(tuple, schema)?;
                    match cond_val {
                        Some(DataValue::Bool(true)) => return res.evaluate(tuple, schema),
                        _ => continue, // false or NULL → try next WHEN
                    }
                }
                // No WHEN matched, use ELSE or NULL
                match else_result {
                    Some(else_expr) => else_expr.evaluate(tuple, schema),
                    None => Ok(None),
                }
            }
            Expr::Function { name, args } => {
                evaluate_scalar_function(name, args, tuple, schema)
            }
        }
    }
}

/// Helper to convert a DataValue to its raw string representation for functions like CONCAT.
fn value_to_raw_string(value: &DataValue) -> String {
    match value {
        DataValue::Char(s) => s.trim_end_matches(' ').to_string(),
        DataValue::Varchar(s) => s.clone(),
        DataValue::Date(d) => d.format("%Y-%m-%d").to_string(),
        DataValue::Time(t) => t.format("%H:%M:%S%.6f").to_string(),
        DataValue::Timestamp(ts) => ts.format("%Y-%m-%d %H:%M:%S%.6f").to_string(),
        DataValue::Bit(bits) => bits.clone(),
        _ => value.to_string(),
    }
}

/// Evaluate a scalar function call by dispatching to the implementations in
/// `types/functions.rs`. Returns an error for unknown functions or type mismatches.
fn evaluate_scalar_function(
    name: &str,
    args: &[Expr],
    tuple: &Tuple,
    schema: &[ColumnInfo],
) -> Result<Option<DataValue>, String> {
    // Evaluate all argument expressions against the tuple
    let evaluated: Result<Vec<Option<DataValue>>, String> = args
        .iter()
        .map(|arg| arg.evaluate(tuple, schema))
        .collect();
    let evaluated = evaluated?;

    let upper = name.to_ascii_uppercase();

    match upper.as_str() {
        // ── 0-argument functions ────────────────────────────────────────
        "CURRENT_DATE" => Ok(Some(crate::types::functions::current_date())),
        "CURRENT_TIME" => Ok(Some(crate::types::functions::current_time())),
        "CURRENT_TIMESTAMP" | "NOW" => Ok(Some(crate::types::functions::current_timestamp())),

        // ── 1-argument string functions ─────────────────────────────────
        "UPPER" | "UCASE" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "UPPER requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::upper(&dv)
                    .map(Some)
                    .map_err(|e| format!("UPPER error: {}", e)),
            }
        }
        "LOWER" | "LCASE" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "LOWER requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::lower(&dv)
                    .map(Some)
                    .map_err(|e| format!("LOWER error: {}", e)),
            }
        }
        "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "LENGTH requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => {
                    let len = crate::types::functions::length(&dv)
                        .map_err(|e| format!("LENGTH error: {}", e))?;
                    Ok(Some(DataValue::Int(len as i32)))
                }
            }
        }
        "POSITION" | "CHARINDEX" => {
            let mut iter = evaluated.into_iter();
            let substring = iter.next().flatten()
                .ok_or_else(|| "POSITION requires a non-NULL substring argument".to_string())?;
            let value = iter.next().flatten()
                .ok_or_else(|| "POSITION requires a non-NULL string argument".to_string())?;
            let pos = crate::types::functions::position(&value, &substring)
                .map_err(|e| format!("POSITION error: {}", e))?;
            Ok(Some(DataValue::Int(pos)))
        }
        "TRIM" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "TRIM requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::trim(&dv)
                    .map(Some)
                    .map_err(|e| format!("TRIM error: {}", e)),
            }
        }
        "LTRIM" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "LTRIM requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::ltrim(&dv)
                    .map(Some)
                    .map_err(|e| format!("LTRIM error: {}", e)),
            }
        }
        "RTRIM" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "RTRIM requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::rtrim(&dv)
                    .map(Some)
                    .map_err(|e| format!("RTRIM error: {}", e)),
            }
        }
        "ABS" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "ABS requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => crate::types::functions::abs(&dv)
                    .map(Some)
                    .map_err(|e| format!("ABS error: {}", e)),
            }
        }
        "FLOOR" => {
            let mut iter = evaluated.into_iter();
            let val = iter.next().flatten();
            let field_arg = iter.next().flatten();
            match (val, field_arg) {
                (Some(dv), None) => {
                    // Numeric FLOOR: FLOOR(3.7) → 3
                    crate::types::functions::floor(&dv)
                        .map(Some)
                        .map_err(|e| format!("FLOOR error: {}", e))
                }
                (Some(dv), Some(part_str)) => {
                    // Temporal FLOOR: FLOOR(date TO MONTH) → truncate date
                    let part = crate::types::functions::parse_date_part(&part_str)?;
                    crate::types::functions::date_trunc_floor(&dv, part)
                        .map(Some)
                        .map_err(|e| format!("FLOOR temporal truncation error: {}", e))
                }
                (None, _) => Ok(None),
            }
        }
        "CEIL" | "CEILING" => {
            let mut iter = evaluated.into_iter();
            let val = iter.next().flatten();
            let field_arg = iter.next().flatten();
            match (val, field_arg) {
                (Some(dv), None) => {
                    // Numeric CEILING: CEILING(3.7) → 4
                    crate::types::functions::ceiling(&dv)
                        .map(Some)
                        .map_err(|e| format!("CEILING error: {}", e))
                }
                (Some(dv), Some(part_str)) => {
                    // Temporal CEILING: CEILING(date TO MONTH) → round up
                    let part = crate::types::functions::parse_date_part(&part_str)?;
                    crate::types::functions::date_trunc_ceil(&dv, part)
                        .map(Some)
                        .map_err(|e| format!("CEILING temporal truncation error: {}", e))
                }
                (None, _) => Ok(None),
            }
        }

        "CONCAT" => {
            let mut res = String::new();
            for dv in evaluated.into_iter().flatten() {
                res.push_str(&value_to_raw_string(&dv));
            }
            Ok(Some(DataValue::Varchar(res)))
        }

        "MOD" => {
            let mut iter = evaluated.into_iter();
            let left = iter.next().flatten();
            let right = iter.next().flatten();
            match (left, right) {
                (None, _) | (_, None) => Ok(None),
                (Some(lv), Some(rv)) => {
                    arithmetic_op(
                        Some(lv),
                        Some(rv),
                        |a, b| if b == 0 { Err("Division by zero".to_string()) } else { Ok(a % b) },
                        |a, b| if b == 0 { Err("Division by zero".to_string()) } else { Ok(a % b) },
                        |a, b| if b == 0 { Err("Division by zero".to_string()) } else { Ok(a % b) },
                        |a, b| if b == 0.0 { Err("Division by zero".to_string()) } else { Ok(a % b) },
                        |a, b| if b == 0.0 { Err("Division by zero".to_string()) } else { Ok(a % b) },
                    )
                }
            }
        }

        "POWER" | "POW" => {
            let mut iter = evaluated.into_iter();
            let base = iter.next().flatten();
            let exponent = iter.next().flatten();
            match (base, exponent) {
                (None, _) | (_, None) => Ok(None),
                (Some(b), Some(e)) => {
                    let b_f64 = match b {
                        DataValue::SmallInt(v) => v as f64,
                        DataValue::Int(v) => v as f64,
                        DataValue::BigInt(v) => v as f64,
                        DataValue::Real(v) => v.0 as f64,
                        DataValue::DoublePrecision(v) => v.0,
                        DataValue::Numeric(v) => v.unscaled as f64 / 10_f64.powi(v.scale as i32),
                        _ => return Err("POWER requires numeric arguments".to_string()),
                    };
                    let e_f64 = match e {
                        DataValue::SmallInt(v) => v as f64,
                        DataValue::Int(v) => v as f64,
                        DataValue::BigInt(v) => v as f64,
                        DataValue::Real(v) => v.0 as f64,
                        DataValue::DoublePrecision(v) => v.0,
                        DataValue::Numeric(v) => v.unscaled as f64 / 10_f64.powi(v.scale as i32),
                        _ => return Err("POWER requires numeric arguments".to_string()),
                    };
                    let res = b_f64.powf(e_f64);
                    Ok(Some(DataValue::DoublePrecision(crate::types::value::OrderedF64(res))))
                }
            }
        }

        "SQRT" => {
            let val = evaluated.into_iter().next()
                .ok_or_else(|| "SQRT requires 1 argument".to_string())?;
            match val {
                None => Ok(None),
                Some(dv) => {
                    let num = match dv {
                        DataValue::SmallInt(v) => v as f64,
                        DataValue::Int(v) => v as f64,
                        DataValue::BigInt(v) => v as f64,
                        DataValue::Real(v) => v.0 as f64,
                        DataValue::DoublePrecision(v) => v.0,
                        DataValue::Numeric(v) => v.unscaled as f64 / 10_f64.powi(v.scale as i32),
                        _ => return Err("SQRT requires a numeric argument".to_string()),
                    };
                    if num < 0.0 {
                        return Err("cannot take square root of a negative number".to_string());
                    }
                    let res = num.sqrt();
                    Ok(Some(DataValue::DoublePrecision(crate::types::value::OrderedF64(res))))
                }
            }
        }

        // ── 2-argument functions ────────────────────────────────────────
        "SUBSTRING" | "SUBSTR" => {
            let mut iter = evaluated.into_iter();
            let val = iter.next().flatten()
                .ok_or_else(|| "SUBSTRING requires a non-NULL string argument".to_string())?;
            let start = match iter.next().flatten() {
                Some(DataValue::Int(s)) => s as usize,
                Some(DataValue::BigInt(s)) => s as usize,
                _ => return Err("SUBSTRING requires integer start position".to_string()),
            };
            let len = match iter.next().flatten() {
                Some(DataValue::Int(l)) => l as usize,
                Some(DataValue::BigInt(l)) => l as usize,
                // Default: rest of string (use string length via evaluating the value first)
                None => {
                    // Re-evaluate to get the actual string length for default
                    match &val {
                        DataValue::Varchar(s) | DataValue::Char(s) => s.chars().count(),
                        _ => return Err("SUBSTRING requires a string value".to_string()),
                    }
                }
                _ => return Err("SUBSTRING requires integer length".to_string()),
            };
            crate::types::functions::substring(&val, start, len)
                .map(Some)
                .map_err(|e| format!("SUBSTRING error: {}", e))
        }
        "ROUND" => {
            let mut iter = evaluated.into_iter();
            let val = iter.next().flatten()
                .ok_or_else(|| "ROUND requires a non-NULL numeric argument".to_string())?;
            let places = match iter.next().flatten() {
                Some(DataValue::Int(p)) => p,
                Some(DataValue::BigInt(p)) => p as i32,
                None => 0,
                _ => return Err("ROUND requires integer places".to_string()),
            };
            crate::types::functions::round(&val, places)
                .map(Some)
                .map_err(|e| format!("ROUND error: {}", e))
        }
        "COALESCE" => {
            let vals: Vec<Option<DataValue>> = evaluated;
            Ok(crate::types::functions::coalesce(&vals))
        }

        // ── DATE/TIME extract ───────────────────────────────────────────
        "EXTRACT" | "DATE_PART" => {
            if args.len() != 2 {
                return Err("EXTRACT requires 2 arguments: (part, value)".to_string());
            }
            let mut iter = evaluated.into_iter();
            let part_str = match iter.next().flatten() {
                Some(DataValue::Varchar(s)) | Some(DataValue::Char(s)) => s.to_uppercase(),
                _ => return Err("EXTRACT first argument must be a string (YEAR/MONTH/DAY/etc.)".to_string()),
            };
            let val = iter.next().flatten()
                .ok_or_else(|| "EXTRACT requires a non-NULL date/time value".to_string())?;

            let part = match part_str.as_str() {
                "YEAR" => crate::types::functions::DatePart::Year,
                "MONTH" => crate::types::functions::DatePart::Month,
                "DAY" => crate::types::functions::DatePart::Day,
                "HOUR" => crate::types::functions::DatePart::Hour,
                "MINUTE" => crate::types::functions::DatePart::Minute,
                "SECOND" => crate::types::functions::DatePart::Second,
                _ => return Err(format!("Unknown EXTRACT part: {}", part_str)),
            };
            let result = crate::types::functions::extract(part, &val)
                .map_err(|e| format!("EXTRACT error: {}", e))?;
            Ok(Some(DataValue::Int(result)))
        }

        // ── NULLIF ──────────────────────────────────────────────────────
        "NULLIF" => {
            let mut iter = evaluated.into_iter();
            let left = iter.next().flatten();
            let right = iter.next().flatten();
            // SQL NULLIF(expr1, expr2) is equivalent to:
            //   CASE WHEN expr1 = expr2 THEN NULL ELSE expr1 END
            // If expr1 is NULL, then expr1 = expr2 is UNKNOWN, so the ELSE
            // branch fires and returns expr1 (which is NULL).
            match (left, right) {
                (None, _) => Ok(None),  // NULLIF(NULL, anything) → NULL
                (Some(l), None) => Ok(Some(l)),  // NULLIF(val, NULL) → val (val = NULL is UNKNOWN → ELSE val)
                (Some(l), Some(r)) => {
                    crate::types::functions::nullif(l, r)
                        .map_err(|e| format!("NULLIF error: {}", e))
                }
            }
        }

        _ => Err(format!("Unknown scalar function: '{}'", name)),
    }
}

/// Helper: apply a binary arithmetic operation to two nullable values.
/// Supports INT, BIGINT, SMALLINT, DOUBLE PRECISION, and cross-type promotion.
fn arithmetic_op<FI, FB, FD, FF, FDbl>(
    left: Option<DataValue>,
    right: Option<DataValue>,
    int_op: FI,
    bigint_op: FB,
    smallint_op: FD,
    float_op: FF,
    double_op: FDbl,
) -> Result<Option<DataValue>, String>
where
    FI: FnOnce(i32, i32) -> Result<i32, String>,
    FB: FnOnce(i64, i64) -> Result<i64, String>,
    FD: FnOnce(i16, i16) -> Result<i16, String>,
    FF: FnOnce(f64, f64) -> Result<f64, String>,
    FDbl: FnOnce(f64, f64) -> Result<f64, String>,
{
    match (left, right) {
        (None, _) | (_, None) => Ok(None),
        (Some(DataValue::Int(a)), Some(DataValue::Int(b))) => {
            int_op(a, b).map(|v| Some(DataValue::Int(v)))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::BigInt(b))) => {
            bigint_op(a, b).map(|v| Some(DataValue::BigInt(v)))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::SmallInt(b))) => {
            smallint_op(a, b).map(|v| Some(DataValue::SmallInt(v)))
        }
        (Some(DataValue::Real(a)), Some(DataValue::Real(b))) => {
            float_op(a.0 as f64, b.0 as f64).map(|v| Some(DataValue::Real(
                crate::types::value::OrderedF32(v as f32)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a.0, b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        // Cross-type promotion
        (Some(DataValue::Real(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a.0 as f64, b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::Real(b))) => {
            double_op(a.0, b.0 as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Int(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a as f64, b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::Int(b))) => {
            double_op(a.0, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::Int(b))) => {
            int_op(a as i32, b).map(|v| Some(DataValue::Int(v)))
        }
        (Some(DataValue::Int(a)), Some(DataValue::SmallInt(b))) => {
            int_op(a, b as i32).map(|v| Some(DataValue::Int(v)))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::BigInt(b))) => {
            bigint_op(a as i64, b).map(|v| Some(DataValue::BigInt(v)))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::SmallInt(b))) => {
            bigint_op(a, b as i64).map(|v| Some(DataValue::BigInt(v)))
        }
        (Some(DataValue::Int(a)), Some(DataValue::BigInt(b))) => {
            bigint_op(a as i64, b).map(|v| Some(DataValue::BigInt(v)))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::Int(b))) => {
            bigint_op(a, b as i64).map(|v| Some(DataValue::BigInt(v)))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a as f64, b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::SmallInt(b))) => {
            double_op(a.0, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a as f64, b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::BigInt(b))) => {
            double_op(a.0, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        // Cross-type REAL ↔ INTEGER arithmetic (widen both to DoublePrecision)
        (Some(DataValue::Real(a)), Some(DataValue::Int(b))) => {
            double_op(a.0 as f64, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Int(a)), Some(DataValue::Real(b))) => {
            double_op(a as f64, b.0 as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Real(a)), Some(DataValue::SmallInt(b))) => {
            double_op(a.0 as f64, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::Real(b))) => {
            double_op(a as f64, b.0 as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Real(a)), Some(DataValue::BigInt(b))) => {
            double_op(a.0 as f64, b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::Real(b))) => {
            double_op(a as f64, b.0 as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        // Cross-type NUMERIC ↔ INTEGER/FLOAT: convert both to f64, use double_op
        (Some(DataValue::Numeric(a)), Some(DataValue::SmallInt(b))) => {
            double_op(a.unscaled as f64 / 10_f64.powi(a.scale as i32), b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::SmallInt(a)), Some(DataValue::Numeric(b))) => {
            double_op(a as f64, b.unscaled as f64 / 10_f64.powi(b.scale as i32)).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Numeric(a)), Some(DataValue::Int(b))) => {
            double_op(a.unscaled as f64 / 10_f64.powi(a.scale as i32), b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Int(a)), Some(DataValue::Numeric(b))) => {
            double_op(a as f64, b.unscaled as f64 / 10_f64.powi(b.scale as i32)).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Numeric(a)), Some(DataValue::BigInt(b))) => {
            double_op(a.unscaled as f64 / 10_f64.powi(a.scale as i32), b as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::BigInt(a)), Some(DataValue::Numeric(b))) => {
            double_op(a as f64, b.unscaled as f64 / 10_f64.powi(b.scale as i32)).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Numeric(a)), Some(DataValue::Real(b))) => {
            double_op(a.unscaled as f64 / 10_f64.powi(a.scale as i32), b.0 as f64).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Real(a)), Some(DataValue::Numeric(b))) => {
            double_op(a.0 as f64, b.unscaled as f64 / 10_f64.powi(b.scale as i32)).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::Numeric(a)), Some(DataValue::DoublePrecision(b))) => {
            double_op(a.unscaled as f64 / 10_f64.powi(a.scale as i32), b.0).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(DataValue::DoublePrecision(a)), Some(DataValue::Numeric(b))) => {
            double_op(a.0, b.unscaled as f64 / 10_f64.powi(b.scale as i32)).map(|v| Some(DataValue::DoublePrecision(
                crate::types::value::OrderedF64(v)
            )))
        }
        (Some(l), Some(r)) => Err(format!(
            "Arithmetic not supported between {:?} and {:?}",
            l, r
        )),
    }
}
