//! SQL seam: emits the SQL strings for each `QueryableOCEL` access pattern,
//! so alternative physical schemas can be plugged in without changing the
//! `DuckDbLinkedOCEL` query logic.

/// Emits the SQL for each `QueryableOCEL` access pattern, so alternative physical
/// schemas can be plugged in without changing the `DuckDbLinkedOCEL` logic.
pub(crate) trait DuckDbOcelSchema {
    /// Keyset page of events: params (`last_id`, limit). Columns: `id`, `ocel_type`, `"time"`.
    fn all_events_page_sql(&self) -> &'static str;
    /// Keyset page of objects: params (`last_id`, limit). Columns: `id`, `ocel_type`.
    fn all_objects_page_sql(&self) -> &'static str;
    /// Event type by id: param (id). Column: `ocel_type`.
    fn ev_type_sql(&self) -> &'static str;
    /// Object type by id: param (id). Column: `ocel_type`.
    fn ob_type_sql(&self) -> &'static str;
    /// Event timestamp by id: param (id). Column: `"time"`.
    fn ev_time_sql(&self) -> &'static str;
    /// E2O of an event: param (`event_id`). Columns: `qualifier`, `object_id`.
    fn e2o_sql(&self) -> &'static str;
    /// Reverse E2O of an object: param (`object_id`). Columns: `qualifier`, `event_id`.
    fn e2o_rev_sql(&self) -> &'static str;
    /// O2O of an object: param (`source_id`). Columns: `qualifier`, `target_id`.
    fn o2o_sql(&self) -> &'static str;
    /// Reverse O2O of an object: param (`target_id`). Columns: `qualifier`, `source_id`.
    fn o2o_rev_sql(&self) -> &'static str;
    /// Object ids of a given type: param (`ocel_type`). Column: `id`.
    fn obs_of_type_sql(&self) -> &'static str;
    /// Event ids of a given type: param (`ocel_type`). Column: `id`.
    fn evs_of_type_sql(&self) -> &'static str;
    /// Event attribute value by id: param (id). Column: the wide typed attribute column
    /// `"<name>"`. The name is a column identifier (quoted here), not a bind parameter.
    fn ev_attr_val_sql(&self, name: &str) -> String;
    /// Time-versioned object attribute values: params (id, name). Columns: `"time"`, `value`, `value_type`.
    fn ob_attr_vals_sql(&self) -> &'static str;
    /// Distinct event types (for loading the type-name list). Column: `ocel_type`.
    fn distinct_ev_types_sql(&self) -> &'static str;
    /// Distinct object types. Column: `ocel_type`.
    fn distinct_ob_types_sql(&self) -> &'static str;
    /// Event types for a batch of ids, `n` placeholders. Columns: `id`, `ocel_type`
    /// (id included so the caller can reorder results to match input order).
    /// Returns an owned String because arity varies.
    fn ev_types_batch_sql(&self, n: usize) -> String;
}

/// The Part 1 schema (EAV) schema.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSchema;

impl DuckDbOcelSchema for DefaultSchema {
    fn all_events_page_sql(&self) -> &'static str {
        r#"SELECT id, ocel_type, "time" FROM events WHERE id > ? ORDER BY id LIMIT ?"#
    }

    fn all_objects_page_sql(&self) -> &'static str {
        "SELECT id, ocel_type FROM objects WHERE id > ? ORDER BY id LIMIT ?"
    }

    fn ev_type_sql(&self) -> &'static str {
        "SELECT ocel_type FROM events WHERE id = ?"
    }

    fn ob_type_sql(&self) -> &'static str {
        "SELECT ocel_type FROM objects WHERE id = ?"
    }

    fn ev_time_sql(&self) -> &'static str {
        r#"SELECT "time" FROM events WHERE id = ?"#
    }

    fn e2o_sql(&self) -> &'static str {
        "SELECT qualifier, object_id FROM e2o WHERE event_id = ?"
    }

    fn e2o_rev_sql(&self) -> &'static str {
        "SELECT qualifier, event_id FROM e2o WHERE object_id = ?"
    }

    fn o2o_sql(&self) -> &'static str {
        "SELECT qualifier, target_id FROM o2o WHERE source_id = ?"
    }

    fn o2o_rev_sql(&self) -> &'static str {
        "SELECT qualifier, source_id FROM o2o WHERE target_id = ?"
    }

    fn obs_of_type_sql(&self) -> &'static str {
        "SELECT id FROM objects WHERE ocel_type = ?"
    }

    fn evs_of_type_sql(&self) -> &'static str {
        "SELECT id FROM events WHERE ocel_type = ?"
    }

    fn ev_attr_val_sql(&self, name: &str) -> String {
        format!(
            "SELECT {} FROM events WHERE id = ? LIMIT 1",
            super::tables::quote_ident(name)
        )
    }

    fn ob_attr_vals_sql(&self) -> &'static str {
        r#"SELECT "time", value, value_type FROM object_attribute_changes WHERE id = ? AND name = ? ORDER BY "time""#
    }

    fn distinct_ev_types_sql(&self) -> &'static str {
        "SELECT DISTINCT ocel_type FROM events"
    }

    fn distinct_ob_types_sql(&self) -> &'static str {
        "SELECT DISTINCT ocel_type FROM objects"
    }

    fn ev_types_batch_sql(&self, n: usize) -> String {
        if n == 0 {
            return "SELECT id, ocel_type FROM events WHERE 1=0".to_string();
        }
        let placeholders = std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",");
        format!("SELECT id, ocel_type FROM events WHERE id IN ({placeholders})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_placeholders(sql: &str) -> usize {
        sql.matches('?').count()
    }

    #[test]
    fn ev_attr_val_sql_selects_wide_column() {
        let sql = DefaultSchema.ev_attr_val_sql("amount");
        assert!(sql.contains("FROM events"));
        assert!(sql.contains("\"amount\"")); // quoted wide column, not an EAV subquery
        assert_eq!(count_placeholders(&sql), 1); // only the id binds
    }

    #[test]
    fn ob_attr_vals_sql_reads_change_table() {
        let sql = DefaultSchema.ob_attr_vals_sql();
        assert!(sql.contains("object_attribute_changes"));
        assert_eq!(count_placeholders(sql), 2);
    }

    #[test]
    fn ev_types_batch_sql_scales_placeholders_and_is_zero_safe() {
        assert_eq!(count_placeholders(&DefaultSchema.ev_types_batch_sql(3)), 3);
        let zero = DefaultSchema.ev_types_batch_sql(0);
        assert_eq!(count_placeholders(&zero), 0);
        assert!(zero.contains("1=0")); // empty batch -> vacuously-empty query
    }
}
