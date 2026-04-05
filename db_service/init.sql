CREATE TABLE IF NOT EXISTS utilization_logs (
    time TIMESTAMPTZ NOT NULL,
    equipment_id TEXT NOT NULL,
    equipment_class TEXT,
    current_state TEXT,
    current_activity TEXT,
    total_active_seconds DOUBLE PRECISION,
    total_idle_seconds DOUBLE PRECISION,
    utilization_percent DOUBLE PRECISION
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('utilization_logs', 'time', if_not_exists => TRUE);
