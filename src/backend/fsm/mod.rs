#[allow(clippy::module_inception)]
pub mod fsm;

pub use fsm::{FSM, FSM_LEVELS, FSM_NODES_PER_PAGE, FSM_PAGE_SIZE, FSM_SLOTS_PER_PAGE, FSMPage};
