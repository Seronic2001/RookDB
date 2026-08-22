use super::*;
use super::super::tuple::{Tuple, ColumnInfo};
use super::super::expr::{Expr, Predicate, ComparisonOp};

use crate::types::value::DataValue;
use crate::types::datatype::DataType;

/// A mock operator that yields a fixed set of tuples.
struct MockOperator {
    tuples: Vec<Tuple>,
    pos: usize,
    schema: Vec<ColumnInfo>,
}

impl MockOperator {
    fn new(tuples: Vec<Tuple>, schema: Vec<ColumnInfo>) -> Self {
        Self { tuples, pos: 0, schema }
    }
}

impl PhysicalOperator for MockOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.pos < self.tuples.len() {
            let t = self.tuples[self.pos].clone();
            self.pos += 1;
            Ok(Some(t))
        } else {
            Ok(None)
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.pos = 0;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Mock"
    }
}

fn int_schema() -> Vec<ColumnInfo> {
    vec![
        ColumnInfo { name: "a".into(), data_type: DataType::Int, table: None },
        ColumnInfo { name: "b".into(), data_type: DataType::Int, table: None },
    ]
}

fn int_tuples(data: Vec<(Option<i32>, Option<i32>)>) -> (Vec<Tuple>, Vec<ColumnInfo>) {
    let schema = int_schema();
    let tuples = data.into_iter().map(|(a, b)| {
        Tuple::new(
            vec![
                a.map(DataValue::Int),
                b.map(DataValue::Int),
            ],
            schema.clone(),
        )
    }).collect();
    (tuples, schema)
}

#[test]
fn test_mock_operator() {
    let (tuples, schema) = int_tuples(vec![(Some(1), Some(2)), (Some(3), Some(4))]);
    let mut op = MockOperator::new(tuples, schema);
    assert!(op.next().unwrap().is_some());
    assert!(op.next().unwrap().is_some());
    assert!(op.next().unwrap().is_none());
}

#[test]
fn test_filter_operator() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(2), Some(20)),
        (Some(3), Some(30)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut filter = FilterOperator::new(
        Box::new(child),
        Predicate::Compare(
            Expr::Column { table: None, column: "a".into() },
            ComparisonOp::GreaterThan,
            Expr::Constant(DataValue::Int(1)),
        ),
    );

    let t1 = filter.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(2)));
    let t2 = filter.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(3)));
    assert!(filter.next().unwrap().is_none());
}

#[test]
fn test_limit_operator() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(2), Some(20)),
        (Some(3), Some(30)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut limit = LimitOperator::new(Box::new(child), 2, 0);

    assert!(limit.next().unwrap().is_some());
    assert!(limit.next().unwrap().is_some());
    assert!(limit.next().unwrap().is_none());
}

#[test]
fn test_limit_with_offset() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(2), Some(20)),
        (Some(3), Some(30)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut limit = LimitOperator::new(Box::new(child), 2, 1);

    let t1 = limit.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(2)));
    let t2 = limit.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(3)));
    assert!(limit.next().unwrap().is_none());
}

#[test]
fn test_limit_zero() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(2), Some(20)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut limit = LimitOperator::new(Box::new(child), 0, 0);

    assert!(limit.next().unwrap().is_none());
}

#[test]
fn test_projection_operator() {
    let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
    let child = MockOperator::new(tuples, schema);
    let mut proj = ProjectionOperator::from_indices(
        Box::new(child),
        &[1],
        &["b".to_string()],
    ).unwrap();

    let t = proj.next().unwrap().unwrap();
    assert_eq!(t.values.len(), 1);
    assert_eq!(t.values[0], Some(DataValue::Int(10)));
    assert_eq!(t.column_info[0].name, "b");
    assert!(proj.next().unwrap().is_none());
}

#[test]
fn test_star_projection() {
    let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
    let child = MockOperator::new(tuples, schema);
    let mut proj = ProjectionOperator::star(Box::new(child));

    let t = proj.next().unwrap().unwrap();
    assert_eq!(t.values.len(), 2);
    assert!(proj.next().unwrap().is_none());
}

#[test]
fn test_distinct_operator() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(1), Some(10)),
        (Some(2), Some(20)),
        (Some(1), Some(10)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut distinct = DistinctOperator::new(Box::new(child));

    let t1 = distinct.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(1)));
    let t2 = distinct.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(2)));
    assert!(distinct.next().unwrap().is_none());
}

#[test]
fn test_distinct_operator_reset() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(1), Some(10)),
        (Some(2), Some(20)),
        (Some(1), Some(10)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut distinct = DistinctOperator::new(Box::new(child));

    // First pass: should get 2 unique tuples
    let t1 = distinct.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(1)));
    let t2 = distinct.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(2)));
    assert!(distinct.next().unwrap().is_none());

    // Reset: should re-scan child and produce the same result
    distinct.reset().unwrap();

    let r1 = distinct.next().unwrap().unwrap();
    assert_eq!(r1.values[0], Some(DataValue::Int(1)));
    let r2 = distinct.next().unwrap().unwrap();
    assert_eq!(r2.values[0], Some(DataValue::Int(2)));
    assert!(distinct.next().unwrap().is_none());
}

#[test]
fn test_sort_operator() {
    let (tuples, schema) = int_tuples(vec![
        (Some(3), Some(30)),
        (Some(1), Some(10)),
        (Some(2), Some(20)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut sort = SortOperator::new(Box::new(child), vec![(0, false)]);

    let t1 = sort.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(1)));
    let t2 = sort.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(2)));
    let t3 = sort.next().unwrap().unwrap();
    assert_eq!(t3.values[0], Some(DataValue::Int(3)));
    assert!(sort.next().unwrap().is_none());
}

#[test]
fn test_sort_descending() {
    let (tuples, schema) = int_tuples(vec![
        (Some(1), Some(10)),
        (Some(3), Some(30)),
        (Some(2), Some(20)),
    ]);
    let child = MockOperator::new(tuples, schema);
    let mut sort = SortOperator::new(Box::new(child), vec![(0, true)]);

    assert_eq!(sort.next().unwrap().unwrap().values[0], Some(DataValue::Int(3)));
    assert_eq!(sort.next().unwrap().unwrap().values[0], Some(DataValue::Int(2)));
    assert_eq!(sort.next().unwrap().unwrap().values[0], Some(DataValue::Int(1)));
    assert!(sort.next().unwrap().is_none());
}

fn name_schema() -> Vec<ColumnInfo> {
    vec![
        ColumnInfo { name: "name".into(), data_type: DataType::Varchar(100), table: None },
        ColumnInfo { name: "age".into(), data_type: DataType::Int, table: None },
        ColumnInfo { name: "salary".into(), data_type: DataType::DoublePrecision, table: None },
    ]
}

fn employee_tuples(data: Vec<(&str, Option<i32>, Option<f64>)>) -> (Vec<Tuple>, Vec<ColumnInfo>) {
    let schema = name_schema();
    let tuples = data.into_iter().map(|(name, age, salary)| {
        Tuple::new(
            vec![
                Some(DataValue::Varchar(name.to_string())),
                age.map(DataValue::Int),
                salary.map(|v| DataValue::DoublePrecision(crate::types::value::OrderedF64(v))),
            ],
            schema.clone(),
        )
    }).collect();
    (tuples, schema)
}

// NOTE: aggregate, join, set-operation and subquery operator tests arrive
// with the advanced-operators stage.

#[test]
fn test_single_row_operator() {
    let mut op = SingleRowOperator::new();
    let t1 = op.next().unwrap().unwrap();
    assert_eq!(t1.values.len(), 0);
    assert!(op.next().unwrap().is_none());

    op.reset().unwrap();
    let t2 = op.next().unwrap().unwrap();
    assert_eq!(t2.values.len(), 0);
}

#[test]
fn test_null_operator() {
    let schema = vec![ColumnInfo { name: "x".into(), data_type: DataType::Int, table: None }];
    let mut op = NullOperator::new(schema);
    assert!(op.next().unwrap().is_none());
}

#[test]
fn test_cte_scan_operator() {
    let (tuples, schema) = int_tuples(vec![(Some(1), Some(10)), (Some(2), Some(20))]);
    let mut op = CteScanOperator::new(tuples, schema);

    assert_eq!(op.estimate_cardinality(), 2);
    let t1 = op.next().unwrap().unwrap();
    assert_eq!(t1.values[0], Some(DataValue::Int(1)));
    let t2 = op.next().unwrap().unwrap();
    assert_eq!(t2.values[0], Some(DataValue::Int(2)));
    assert!(op.next().unwrap().is_none());

    op.reset().unwrap();
    let t3 = op.next().unwrap().unwrap();
    assert_eq!(t3.values[0], Some(DataValue::Int(1)));
}
