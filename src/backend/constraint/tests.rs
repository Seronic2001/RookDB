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
        assert!(validation::check_not_null("t", &columns, values).is_ok());
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
            validation::check_not_null("t", &columns, &["null"]).is_err(),
            "Should reject NULL for NOT NULL column"
        );
        assert!(
            validation::check_not_null("t", &columns, &[""]).is_err(),
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
            validation::check_not_null("t", &columns, &["null"]).is_ok(),
            "Should accept NULL for nullable column"
        );
    }

}
