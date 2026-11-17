-- ForestGuard AI SQLite Schema

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT DEFAULT 'unknown',
    state TEXT NOT NULL,
    cnn_label TEXT,
    cnn_conf REAL,
    yolo_json TEXT,
    top_class TEXT,
    top_conf REAL,
    snapshot_path TEXT,
    alert_id INTEGER
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT,
    state TEXT,
    source TEXT DEFAULT 'unknown',
    snapshot_path TEXT,
    acknowledged INTEGER DEFAULT 0,
    is_test INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS counters (
    key TEXT PRIMARY KEY,
    value INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT
);

CREATE TABLE IF NOT EXISTS model_info (
    name TEXT PRIMARY KEY,
    version TEXT,
    trained_by TEXT,
    details_json TEXT,
    updated_at TEXT
);

-- Initialize default counters if not present
INSERT OR IGNORE INTO counters (key, value) VALUES ('frames_processed', 0);
INSERT OR IGNORE INTO counters (key, value) VALUES ('fire_detections', 0);
INSERT OR IGNORE INTO counters (key, value) VALUES ('smoke_detections', 0);
INSERT OR IGNORE INTO counters (key, value) VALUES ('no_fire_classifications', 0);
INSERT OR IGNORE INTO counters (key, value) VALUES ('alerts_generated', 0);
