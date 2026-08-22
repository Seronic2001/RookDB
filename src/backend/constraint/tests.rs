//! Tests for constraint validation.

#[cfg(test)]
mod tests {
    use super::super::validation;
    use crate::catalog::types::{Column, Constraints};
    use crate::types::DataType;

    #[test]
    fn test_check_not_null_passes_for_non_null_values() {
        let columns = vec![
            Column {
                name: "id".to_string(),
                data_type: DataType::Int,
                nullable: false,
                constraints: Constraints::default(),
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar(100),
                nullable: true,
                constraints: Constraints::default(),
            },
        ];
        let values = &["1", "Alice"];
        assert!(validation::check_not_null(&columns, values).is_ok());
    }

    #[test]
    fn test_check_not_null_rejects_null_for_non_nullable() {
        let columns = vec![
            Column {
                name: "id".to_string(),
                data_type: DataType::Int,
                nullable: false,
                constraints: Constraints::default(),
            },
        ];
        assert!(
            validation::check_not_null(&columns, &["null"]).is_err(),
            "Should reject NULL for NOT NULL column"
        );
        assert!(
            validation::check_not_null(&columns, &[""]).is_err(),
            "Should reject empty string for NOT NULL column"
        );
    }

    #[test]
    fn test_check_not_null_accepts_null_for_nullable() {
        let columns = vec![
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar(100),
                nullable: true,
                constraints: Constraints::default(),
            },
        ];
        assert!(
            validation::check_not_null(&columns, &["null"]).is_ok(),
            "Should accept NULL for nullable column"
        );
    }

    #[test]
    fn test_decode_values_for_constraint_with_int() {
        let columns = vec![
            Column {
                name: "age".to_string(),
                data_type: DataType::Int,
                nullable: true,
                constraints: Constraints::default(),
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Varchar(100),
                nullable: true,
                constraints: Constraints::default(),
            },
        ];
        let decoded = validation::decode_values_for_constraint(&columns, &["25", "Alice"]);

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, "age");
        match &decoded[0].1 {
            crate::backend::executor::delete::ColumnValue::Int(n) => assert_eq!(*n, 25),
            _ => panic!("Expected Int(25)"),
        }
        assert_eq!(decoded[1].0, "name");
        match &decoded[1].1 {
            crate::backend::executor::delete::ColumnValue::Text(s) => assert_eq!(s, "Alice"),
            _ => panic!("Expected Text(Alice)"),
        }
    }
}
