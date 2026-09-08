//! Pushdown query AST (binding-box model) for object-centric event data, plus its two
//! backends: an in-memory evaluator and a SQL translator, asserted by tests to agree.

/// Backend-agnostic algorithm rewrites on top of the query layer; see [`algos`] module docs.
pub mod algos;
/// In-memory evaluator; see [`eval`] module docs.
pub mod eval;
/// The query AST types.
pub mod model;
/// Pure `Query` -> SQL translation; see [`sql`] module docs.
pub mod sql;
pub use model::*;
