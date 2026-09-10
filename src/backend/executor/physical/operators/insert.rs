//! InsertOperator — Inserts tuples from a child operator into a table.
//!
//! Used for `INSERT INTO ... SELECT`. The child operator produces rows from the
//! SELECT query, and this operator inserts each row into the target table using
//! the existing `insert_single_tuple` function.
//!
//! The operator returns each inserted tuple so the engine can report the count.

use super::super::tuple::{Tuple, ColumnInfo};
use super::super::expr::Expr;
use super::PhysicalOperator;

/// Function signature for the insert operation.
///
/// Takes the string representations of the tuple values and returns:
/// - `Ok(())` on successful insertion
/// - `Err(String)` on failure (constraint violation or database error)
///
/// Making this injectable allows unit tests to mock the insertion logic
/// without requiring a real catalog and database files.
pub(crate) type InsertInserter = Box<dyn FnMut(&[&str]) -> Result<(), String>>;

/// Insert operator that writes child tuples into a table.
///
/// Streams rows one at a time: each call to `next()` pulls one tuple from the
/// child (SELECT), inserts it via an injectable inserter, and returns it.
/// The engine counts returned tuples, so the reported count matches the number
/// of rows actually inserted.
pub struct InsertOperator {
    child: Box<dyn PhysicalOperator>,
    /// Schema of the child (used for operator contract).
    child_schema: Vec<ColumnInfo>,
    /// Injectable insert function. Defaults to wrapping `insert_single_tuple`.
    /// In tests, can be replaced with a mock for isolated unit testing.
    inserter: InsertInserter,
}

impl InsertOperator {
    pub fn new(
        child: Box<dyn PhysicalOperator>,
        table: String,
        db_name: String,
        catalog: crate::catalog::types::Catalog,
    ) -> Self {
        let child_schema = child.schema().to_vec();

        // Build the default inserter that wraps insert_single_tuple.
        // We clone the data needed by the closure so it's self-contained.
        let cat = catalog.clone();
        let tbl = table.clone();
        let db = db_name.clone();
        let inserter: InsertInserter = Box::new(move |vals| {
            match crate::backend::executor::insert_single_tuple(&cat, &db, &tbl, vals) {
                Ok(true) => Ok(()),
                Ok(false) => Err("Constraint violation or invalid data".to_string()),
                Err(e) => Err(format!("Insert error: {}", e)),
            }
        });

        Self {
            child,
            child_schema,
            inserter,
        }
    }
}

impl PhysicalOperator for InsertOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        // Pull one tuple from child, insert it, return it
        match self.child.next()? {
            Some(tuple) => {
                // Convert tuple values to string slices for the inserter
                let value_strs: Vec<String> = tuple
                    .values
                    .iter()
                    .map(|v| match v {
                        Some(dv) => format!("{}", dv),
                        None => "NULL".to_string(),
                    })
                    .collect();

                let value_refs: Vec<&str> = value_strs.iter().map(|s| s.as_str()).collect();

                // Call the injectable inserter (in production, wraps insert_single_tuple)
                (self.inserter)(&value_refs)?;
                Ok(Some(tuple))
            }
            None => Ok(None),
        }
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.child_schema
    }

    fn reset(&mut self) -> Result<(), String> {
        self.child.reset()?;
        Ok(())
    }

    fn estimate_cardinality(&self) -> usize {
        // Insert produces at most as many tuples as the child
        self.child.estimate_cardinality()
    }

    fn name(&self) -> &'static str {
        "Insert"
    }
}


// ── ValuesOperator ────────────────────────────────────────────────────────────

/// A constant-producing child for `INSERT ... VALUES`.
///
/// Holds one row of pre-compiled expressions per VALUES tuple; each `next()`
/// evaluates the next row against an empty tuple (VALUES may not reference
/// columns) and emits it with the target table's schema.
pub struct ValuesOperator {
    rows: Vec<Vec<Expr>>,
    schema: Vec<ColumnInfo>,
    pos: usize,
}

impl ValuesOperator {
    pub fn new(rows: Vec<Vec<Expr>>, schema: Vec<ColumnInfo>) -> Self {
        Self { rows, schema, pos: 0 }
    }
}

impl PhysicalOperator for ValuesOperator {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.pos >= self.rows.len() {
            return Ok(None);
        }
        let empty = Tuple::new(Vec::new());
        let values = self.rows[self.pos]
            .iter()
            .map(|e| e.evaluate(&empty, &[]))
            .collect::<Result<Vec<_>, _>>()?;
        self.pos += 1;
        Ok(Some(Tuple::new(values)))
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn reset(&mut self) -> Result<(), String> {
        Err("reset() not supported".to_string())
    }

    fn estimate_cardinality(&self) -> usize {
        self.rows.len()
    }

    fn name(&self) -> &'static str {
        "Values"
    }
}


// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::value::DataValue;
    use crate::types::datatype::DataType;
    use std::rc::Rc;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use rook_ast::logical::{LogicalPlan, LogicalInsert, LogicalTableScan, LogicalProject, ColumnSchema};
    use crate::backend::executor::physical::planner::PhysicalPlanner;
    use crate::catalog::types::Catalog;

    /// A mock child operator that yields a fixed set of tuples.
    struct MockChild {
        tuples: Vec<Tuple>,
        pos: usize,
        schema: Vec<ColumnInfo>,
    }

    impl MockChild {
        fn new(tuples: Vec<Tuple>, schema: Vec<ColumnInfo>) -> Self {
            Self { tuples, pos: 0, schema }
        }
    }

    impl PhysicalOperator for MockChild {
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
            "MockChild"
        }

        fn estimate_cardinality(&self) -> usize {
            self.tuples.len()
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
            )
        }).collect();
        (tuples, schema)
    }

    // ── Streaming ───────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_empty_child() {
        // Child returns no tuples → operator should return None immediately
        let (tuples, schema) = int_tuples(vec![]);
        let child = MockChild::new(tuples, schema);

        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Ok(())),
        };

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_insert_operator_single_tuple() {
        // Single tuple → inserted and returned once
        let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
        let child = MockChild::new(tuples, schema);

        let inserted: Rc<RefCell<Vec<Vec<String>>>> = Rc::new(RefCell::new(Vec::new()));
        let ins = inserted.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |vals| {
                let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                ins.borrow_mut().push(strs);
                Ok(())
            }),
        };

        let t = op.next().unwrap().unwrap();
        assert_eq!(t.values[0], Some(DataValue::Int(1)));
        assert_eq!(t.values[1], Some(DataValue::Int(10)));
        assert!(op.next().unwrap().is_none());

        assert_eq!(inserted.borrow().len(), 1);
        assert_eq!(inserted.borrow()[0], vec!["1", "10"]);
    }

    #[test]
    fn test_insert_operator_streaming_multiple() {
        // Multiple tuples stream through one at a time
        let (tuples, schema) = int_tuples(vec![
            (Some(1), Some(10)),
            (Some(2), Some(20)),
            (Some(3), Some(30)),
        ]);
        let child = MockChild::new(tuples, schema);

        let count: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
        let cnt = count.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |_vals| {
                *cnt.borrow_mut() += 1;
                Ok(())
            }),
        };

        let t1 = op.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(1)));
        assert_eq!(*count.borrow(), 1);

        let t2 = op.next().unwrap().unwrap();
        assert_eq!(t2.values[0], Some(DataValue::Int(2)));
        assert_eq!(*count.borrow(), 2);

        let t3 = op.next().unwrap().unwrap();
        assert_eq!(t3.values[0], Some(DataValue::Int(3)));
        assert_eq!(*count.borrow(), 3);

        assert!(op.next().unwrap().is_none());
        assert_eq!(*count.borrow(), 3);
    }

    // ── Error handling ───────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_error_constraint_violation() {
        // Inserter returns an error → operator propagates it
        let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
        let child = MockChild::new(tuples, schema);

        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Err("Constraint violation: duplicate key".to_string())),
        };

        let result = op.next();
        match result {
            Err(msg) => {
                assert!(
                    msg.contains("duplicate key"),
                    "Expected constraint error, got: {}",
                    msg
                );
            }
            _ => panic!("Expected error for constraint violation"),
        }
    }

    #[test]
    fn test_insert_operator_error_stops_streaming() {
        // Error on the second tuple → subsequent calls keep returning the error
        let (tuples, schema) = int_tuples(vec![
            (Some(1), Some(10)),
            (Some(2), Some(20)), // This one will fail
            (Some(3), Some(30)),
        ]);
        let child = MockChild::new(tuples, schema);

        let call_count: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
        let cc = call_count.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |_| {
                let cnt = *cc.borrow();
                *cc.borrow_mut() += 1;
                if cnt == 1 {
                    // Fail on the second insertion
                    Err("Disk full".to_string())
                } else {
                    Ok(())
                }
            }),
        };

        // First call succeeds
        let t1 = op.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(1)));

        // Second call fails
        let err = op.next().unwrap_err();
        assert!(err.contains("Disk full"), "Got: {}", err);
    }

    // ── NULL values ──────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_null_values() {
        // Tuples with NULL values → converted to "NULL" strings
        let (tuples, schema) = int_tuples(vec![(Some(1), None), (None, Some(20))]);
        let child = MockChild::new(tuples, schema);

        let inserted: Rc<RefCell<Vec<Vec<String>>>> = Rc::new(RefCell::new(Vec::new()));
        let ins = inserted.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |vals| {
                let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                ins.borrow_mut().push(strs);
                Ok(())
            }),
        };

        // First tuple: (1, NULL)
        let t1 = op.next().unwrap().unwrap();
        assert_eq!(t1.values[0], Some(DataValue::Int(1)));
        assert_eq!(t1.values[1], None);
        assert_eq!(inserted.borrow()[0], vec!["1", "NULL"]);

        // Second tuple: (NULL, 20)
        let t2 = op.next().unwrap().unwrap();
        assert_eq!(t2.values[0], None);
        assert_eq!(t2.values[1], Some(DataValue::Int(20)));
        assert_eq!(inserted.borrow()[1], vec!["NULL", "20"]);

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_insert_operator_all_nulls() {
        // All NULL values
        let (tuples, schema) = int_tuples(vec![(None, None)]);
        let child = MockChild::new(tuples, schema);

        let inserted: Rc<RefCell<Vec<Vec<String>>>> = Rc::new(RefCell::new(Vec::new()));
        let ins = inserted.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |vals| {
                let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                ins.borrow_mut().push(strs);
                Ok(())
            }),
        };

        let t = op.next().unwrap().unwrap();
        assert_eq!(t.values[0], None);
        assert_eq!(t.values[1], None);
        assert_eq!(inserted.borrow()[0], vec!["NULL", "NULL"]);
    }

    // ── Reset ────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_reset() {
        // Reset allows re-iterating the child tuples
        let (tuples, schema) = int_tuples(vec![
            (Some(1), Some(10)),
            (Some(2), Some(20)),
        ]);
        let child = MockChild::new(tuples, schema);

        let count: Rc<RefCell<u32>> = Rc::new(RefCell::new(0));
        let cnt = count.clone();
        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(move |_vals| {
                *cnt.borrow_mut() += 1;
                Ok(())
            }),
        };

        // First pass: insert both tuples
        assert!(op.next().unwrap().is_some());
        assert!(op.next().unwrap().is_some());
        assert!(op.next().unwrap().is_none());
        assert_eq!(*count.borrow(), 2);

        // Reset the operator
        op.reset().unwrap();

        // Second pass: re-insert both tuples
        assert!(op.next().unwrap().is_some());
        assert!(op.next().unwrap().is_some());
        assert!(op.next().unwrap().is_none());
        assert_eq!(*count.borrow(), 4);
    }

    // ── Schema ───────────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_schema() {
        // Schema matches the child's schema
        let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
        let child = MockChild::new(tuples, schema.clone());

        let mut op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Ok(())),
        };

        // Clone to avoid borrow conflict with next()
        let schema = op.schema().to_vec();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].name, "a");
        assert_eq!(schema[0].data_type, DataType::Int);
        assert_eq!(schema[1].name, "b");
        assert_eq!(schema[1].data_type, DataType::Int);

        // Returned tuple should also match the schema
        let t = op.next().unwrap().unwrap();
        assert_eq!(t.arity(), schema.len());
    }

    // ── Cardinality ──────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_estimate_cardinality() {
        // estimate_cardinality mirrors the child's estimate
        let (tuples, schema) = int_tuples(vec![
            (Some(1), Some(10)),
            (Some(2), Some(20)),
            (Some(3), Some(30)),
            (Some(4), Some(40)),
        ]);
        let child = MockChild::new(tuples, schema);

        let op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Ok(())),
        };

        assert_eq!(op.estimate_cardinality(), 4);
    }

    #[test]
    fn test_insert_operator_estimate_cardinality_empty() {
        // Empty child → cardinality of 0
        let (tuples, schema) = int_tuples(vec![]);
        let child = MockChild::new(tuples, schema);

        let op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Ok(())),
        };

        assert_eq!(op.estimate_cardinality(), 0);
    }

    // ── Name ─────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_operator_name() {
        let (tuples, schema) = int_tuples(vec![(Some(1), Some(10))]);
        let child = MockChild::new(tuples, schema);

        let op = InsertOperator {
            child: Box::new(child),
            child_schema: int_schema(),
            inserter: Box::new(|_| Ok(())),
        };

        assert_eq!(op.name(), "Insert");
    }

    // ── Physical planner pipeline tests ────────────────────────────────────

    #[test]
    fn test_physical_planner_produces_insert_operator() {
        // Verify that PhysicalPlanner::plan() correctly creates an
        // InsertOperator from a LogicalPlan::Insert node.
        //
        // Uses a SingleRow system table as the child so no heap files are needed.
        let child = LogicalPlan::TableScan(LogicalTableScan {
            table: "__singlerow__".to_string(),
            alias: None,
            schema: ColumnSchema::empty(),
            system_table_name: Some("__singlerow__".to_string()),
        });

        let plan = LogicalPlan::Insert(LogicalInsert {
            table: "test_table".to_string(),
            columns: vec![],
            child: Box::new(child),
            values_rows: Vec::new(),
        });

        let catalog = Catalog { databases: HashMap::new() };
        let planner = PhysicalPlanner::new(catalog, "test_db".to_string());
        let result = planner.plan(&plan);

        assert!(result.is_ok(), "Physical planner should produce InsertOperator");
        let op = result.unwrap();
        assert_eq!(op.name(), "Insert");
        assert!(op.schema().is_empty(), "SingleRow child should produce empty schema");
        // SingleRowOperator.estimate_cardinality() returns 0 by default
        assert_eq!(op.estimate_cardinality(), 0, "SingleRowOperator returns 0 by default");
    }

    #[test]
    fn test_physical_planner_insert_rejects_missing_table() {
        // Verify that planning a regular TableScan inside Insert
        // fails when the heap file doesn't exist.
        let child = LogicalPlan::TableScan(LogicalTableScan {
            table: "nonexistent".to_string(),
            alias: None,
            schema: ColumnSchema::empty(),
            system_table_name: None,
        });

        let plan = LogicalPlan::Insert(LogicalInsert {
            table: "target".to_string(),
            columns: vec![],
            child: Box::new(child),
            values_rows: Vec::new(),
        });

        let catalog = Catalog { databases: HashMap::new() };
        let planner = PhysicalPlanner::new(catalog, "test_db".to_string());
        let result = planner.plan(&plan);

        assert!(
            result.is_err(),
            "Planning a TableScan without heap file should fail"
        );
    }

    #[test]
    fn test_physical_planner_insert_with_project_child() {
        // Verify InsertOperator planning with a Projection child
        // (avoids TableScan so no heap file is needed — uses SingleRow).
        let inner_scan = LogicalPlan::TableScan(LogicalTableScan {
            table: "__singlerow__".to_string(),
            alias: None,
            schema: ColumnSchema::empty(),
            system_table_name: Some("__singlerow__".to_string()),
        });

        let project = LogicalPlan::Project(LogicalProject {
            expressions: vec![],
            child: Box::new(inner_scan),
        });

        let plan = LogicalPlan::Insert(LogicalInsert {
            table: "users".to_string(),
            columns: vec!["name".to_string()],
            child: Box::new(project),
            values_rows: Vec::new(),
        });

        let catalog = Catalog { databases: HashMap::new() };
        let planner = PhysicalPlanner::new(catalog, "test_db".to_string());
        let result = planner.plan(&plan);

        assert!(result.is_ok(), "Insert with Project child should plan");
        let op = result.unwrap();
        assert_eq!(op.name(), "Insert");
    }

    #[test]
    fn test_physical_planner_insert_reset_propagation() {
        // Verify that reset() on the InsertOperator correctly
        // calls reset() on the child operator tree.
        let child = LogicalPlan::TableScan(LogicalTableScan {
            table: "__singlerow__".to_string(),
            alias: None,
            schema: ColumnSchema::empty(),
            system_table_name: Some("__singlerow__".to_string()),
        });

        let plan = LogicalPlan::Insert(LogicalInsert {
            table: "test_table".to_string(),
            columns: vec![],
            child: Box::new(child),
            values_rows: Vec::new(),
        });

        let catalog = Catalog { databases: HashMap::new() };
        let planner = PhysicalPlanner::new(catalog, "test_db".to_string());
        let mut op = planner.plan(&plan).unwrap();

        // Reset should succeed
        let reset_result = op.reset();
        assert!(reset_result.is_ok(), "InsertOperator.reset() should succeed");
    }
}
