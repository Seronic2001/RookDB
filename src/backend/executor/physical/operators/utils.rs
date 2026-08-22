use crate::types::value::DataValue;

/// Normalise a DataValue into a string that can be used as a hash key.
/// Uses a simple `type_prefix|value` format that avoids ambiguity.
pub fn normalise_value_for_key(val: &DataValue) -> String {
    match val {
        DataValue::SmallInt(v) => format!("I16:{}", v),
        DataValue::Int(v) => format!("I32:{}", v),
        DataValue::BigInt(v) => format!("I64:{}", v),
        DataValue::Real(v) => format!("F32:{}", v.0),
        DataValue::DoublePrecision(v) => format!("F64:{}", v.0),
        DataValue::Bool(v) => format!("B:{}", v),
        DataValue::Char(s) => format!("C:{}", s),
        DataValue::Varchar(s) => format!("V:{}", s),
        DataValue::Date(d) => format!("D:{}", d),
        DataValue::Time(t) => format!("T:{}", t),
        DataValue::Timestamp(ts) => format!("TS:{}", ts),
        DataValue::Numeric(n) => format!("N:{}:{}", n.unscaled, n.scale),
        DataValue::Bit(b) => format!("BIT:{}", b),
    }
}
