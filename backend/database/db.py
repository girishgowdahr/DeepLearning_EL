import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from backend.config import settings

logger = logging.getLogger("forestguard.db")

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database with the schema."""
    conn = get_connection()
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        logger.info(f"Database initialized at {settings.DB_PATH}")
    finally:
        conn.close()

def increment_counter(key: str, amount: int = 1):
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO counters (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = value + ?
                """,
                (key, amount, amount)
            )
    finally:
        conn.close()

def get_all_counters() -> dict[str, int]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM counters")
        rows = cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        conn.close()

def insert_detection(record: dict) -> int:
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO detections (
                    ts, source, state, cnn_label, cnn_conf, 
                    yolo_json, top_class, top_conf, snapshot_path, alert_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("ts", datetime.now(timezone.utc).isoformat()),
                    record.get("source", "unknown"),
                    record.get("state", "NO_FIRE"),
                    record.get("cnn_label"),
                    record.get("cnn_conf"),
                    json.dumps(record.get("yolo", [])),
                    record.get("top_class"),
                    record.get("top_conf"),
                    record.get("snapshot_path"),
                    record.get("alert_id")
                )
            )
            return cursor.lastrowid
    finally:
        conn.close()

def get_detections(limit: int = 50, offset: int = 0, source: str = None, state: str = None) -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if state:
            query += " AND state = ?"
            params.append(state)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        result = []
        for r in rows:
            d = dict(r)
            if d.get("yolo_json"):
                try:
                    d["yolo"] = json.loads(d["yolo_json"])
                except Exception:
                    d["yolo"] = []
            result.append(d)
        return result
    finally:
        conn.close()

def get_detection_by_id(detection_id: int) -> dict | None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            if d.get("yolo_json"):
                try:
                    d["yolo"] = json.loads(d["yolo_json"])
                except Exception:
                    d["yolo"] = []
            return d
        return None
    finally:
        conn.close()

def insert_alert(alert: dict) -> int:
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (
                    ts, type, message, state, source, snapshot_path, acknowledged, is_test
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.get("ts", datetime.now(timezone.utc).isoformat()),
                    alert.get("type", "fire_detected"),
                    alert.get("message", ""),
                    alert.get("state", "FIRE_CONFIRMED"),
                    alert.get("source", "unknown"),
                    alert.get("snapshot_path"),
                    1 if alert.get("acknowledged") else 0,
                    1 if alert.get("is_test") else 0
                )
            )
            return cursor.lastrowid
    finally:
        conn.close()

def get_alerts(limit: int = 50, offset: int = 0, acknowledged: bool = None) -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(1 if acknowledged else 0)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

def acknowledge_alert(alert_id: int) -> bool:
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()

def get_setting(key: str, default=None):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value_json FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["value_json"])
        return default
    finally:
        conn.close()

def set_setting(key: str, value):
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO settings (key, value_json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = ?",
                (key, json.dumps(value), json.dumps(value))
            )
    finally:
        conn.close()

def get_all_settings() -> dict:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value_json FROM settings")
        rows = cursor.fetchall()
        return {r["key"]: json.loads(r["value_json"]) for r in rows}
    finally:
        conn.close()

def upsert_model_info(name: str, version: str, trained_by: str, details: dict):
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO model_info (name, version, trained_by, details_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET 
                    version = excluded.version,
                    trained_by = excluded.trained_by,
                    details_json = excluded.details_json,
                    updated_at = excluded.updated_at
                """,
                (name, version, trained_by, json.dumps(details), datetime.now(timezone.utc).isoformat())
            )
    finally:
        conn.close()

def get_all_model_info() -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_info")
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("details_json"):
                try:
                    d["details"] = json.loads(d["details_json"])
                except Exception:
                    d["details"] = {}
            result.append(d)
        return result
    finally:
        conn.close()
