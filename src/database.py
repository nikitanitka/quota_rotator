"""SQLite WAL-mode Database Layer for Gemini Key Pool Router.

Provides thread-safe, asynchronous CRUD for API keys and usage events logging.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo
from src.contracts import (
    INITIAL_PRELOADED_KEYS,
    RESET_TIMEZONE,
    mask_api_key,
    mask_client_api_key,
)

logger: logging.Logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite Database Manager running in WAL mode."""

    def __init__(self, db_path: str = "data/router.db") -> None:
        self.db_path: str = db_path
        self._lock: asyncio.Lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
            self._sync_init_tables()
        return self._conn

    def _sync_init_tables(self) -> None:
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_value TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'READY',
                is_disabled INTEGER NOT NULL DEFAULT 0,
                consecutive_429 INTEGER NOT NULL DEFAULT 0,
                cooldown_until REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                status_code INTEGER NOT NULL DEFAULT 200,
                is_dry_run INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_key_time ON usage_events(key_id, timestamp);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_model_time ON usage_events(model_id, timestamp);"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS client_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                key_value TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                rpm_limit INTEGER NOT NULL DEFAULT 60,
                rpd_limit INTEGER NOT NULL DEFAULT 1000,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS client_usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_key_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                tokens INTEGER NOT NULL DEFAULT 0,
                status_code INTEGER NOT NULL DEFAULT 200,
                FOREIGN KEY(client_key_id) REFERENCES client_keys(id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_client_usage_time ON client_usage_events(client_key_id, timestamp);"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                timestamp_iso TEXT NOT NULL,
                client_key_id INTEGER,
                client_key_name TEXT,
                google_key_id INTEGER,
                google_key_label TEXT,
                model_id TEXT NOT NULL,
                action TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                prompt_full_json TEXT,
                response_text TEXT NOT NULL DEFAULT '',
                response_full_json TEXT,
                tool_calls_json TEXT,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                status_code INTEGER NOT NULL DEFAULT 200,
                error_message TEXT,
                FOREIGN KEY(client_key_id) REFERENCES client_keys(id) ON DELETE SET NULL,
                FOREIGN KEY(google_key_id) REFERENCES api_keys(id) ON DELETE SET NULL
            );
            """
        )
        try:
            cursor.execute("ALTER TABLE request_logs ADD COLUMN tool_calls_json TEXT;")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                logger.debug("Column tool_calls_json migration note: %s", exc)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_logs_time ON request_logs(timestamp DESC);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_logs_client ON request_logs(client_key_id, timestamp DESC);"
        )
        self._conn.commit()
        self._sync_seed_if_needed()

    def _sync_seed_if_needed(self) -> None:
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM api_keys;")
        row = cursor.fetchone()
        count = int(row[0]) if row else 0
        if count == 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            for item in INITIAL_PRELOADED_KEYS:
                cursor.execute(
                    """
                    INSERT INTO api_keys (
                        key_value, label, status, is_disabled, consecutive_429,
                        cooldown_until, created_at, updated_at
                    ) VALUES (?, ?, 'READY', 0, 0, 0.0, ?, ?)
                    """,
                    (item["key_value"], item["label"], now_iso, now_iso),
                )
            self._conn.commit()

    def _row_to_key_dict(self, row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": int(row["id"]),
            "key_value": str(row["key_value"]),
            "label": str(row["label"]),
            "status": str(row["status"]),
            "is_disabled": bool(row["is_disabled"]),
            "consecutive_429": int(row["consecutive_429"]),
            "cooldown_until": float(row["cooldown_until"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _row_to_client_key_dict(self, row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "key_value": str(row["key_value"]),
            "key_prefix": str(row["key_prefix"]),
            "rpm_limit": int(row["rpm_limit"]),
            "rpd_limit": int(row["rpd_limit"]),
            "status": str(row["status"]),
            "total_requests": int(row["total_requests"]),
            "total_tokens": int(row["total_tokens"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "last_used_at": str(row["last_used_at"]) if row["last_used_at"] is not None else None,
        }

    def _sync_create_client_key(
        self,
        name: str,
        key_value: str,
        rpm_limit: int = 60,
        rpd_limit: int = 1000,
    ) -> dict[str, object]:
        conn = self._get_connection()
        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        prefix = mask_client_api_key(key_value)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO client_keys (
                name, key_value, key_prefix, rpm_limit, rpd_limit,
                status, total_requests, total_tokens, created_at, updated_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', 0, 0, ?, ?, NULL);
            """,
            (name.strip(), key_value.strip(), prefix, rpm_limit, rpd_limit, now_iso, now_iso),
        )
        conn.commit()
        inserted_id = int(cursor.lastrowid)
        result = self._sync_get_client_key_by_id(inserted_id)
        if result is None:
            raise RuntimeError(f"Failed to retrieve newly added client key id={inserted_id}")
        return result

    def _sync_get_all_client_keys(self) -> list[dict[str, object]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM client_keys ORDER BY id ASC;")
        return [self._row_to_client_key_dict(row) for row in cursor.fetchall()]

    def _sync_get_client_key_by_id(self, client_key_id: int) -> dict[str, object] | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM client_keys WHERE id = ?;", (client_key_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_client_key_dict(row)

    def _sync_get_client_key_by_token(self, token: str) -> dict[str, object] | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM client_keys WHERE key_value = ?;", (token.strip(),))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_client_key_dict(row)

    def _sync_update_client_key(
        self,
        client_key_id: int,
        name: str | None = None,
        rpm_limit: int | None = None,
        rpd_limit: int | None = None,
        status: str | None = None,
    ) -> dict[str, object] | None:
        current = self._sync_get_client_key_by_id(client_key_id)
        if current is None:
            return None

        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        new_name = current["name"] if name is None else name.strip()
        new_rpm_limit = current["rpm_limit"] if rpm_limit is None else rpm_limit
        new_rpd_limit = current["rpd_limit"] if rpd_limit is None else rpd_limit
        new_status = current["status"] if status is None else status.strip()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE client_keys
            SET name = ?, rpm_limit = ?, rpd_limit = ?, status = ?, updated_at = ?
            WHERE id = ?;
            """,
            (new_name, new_rpm_limit, new_rpd_limit, new_status, now_iso, client_key_id),
        )
        conn.commit()
        return self._sync_get_client_key_by_id(client_key_id)

    def _sync_delete_client_key(self, client_key_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM client_usage_events WHERE client_key_id = ?;", (client_key_id,))
        cursor.execute("DELETE FROM client_keys WHERE id = ?;", (client_key_id,))
        conn.commit()
        return cursor.rowcount > 0

    def _sync_count_client_keys(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM client_keys;")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _sync_get_client_usage(
        self, client_key_id: int, current_time: float, window_start_msk_iso: str
    ) -> tuple[int, int]:
        try:
            window_start_ts = datetime.fromisoformat(window_start_msk_iso).timestamp()
        except (ValueError, TypeError):
            try:
                window_start_ts = float(window_start_msk_iso)
            except (ValueError, TypeError):
                window_start_ts = 0.0

        cutoff_rpm = current_time - 60.0
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(CASE WHEN timestamp >= ? THEN 1 END),
                COUNT(CASE WHEN timestamp >= ? THEN 1 END)
            FROM client_usage_events
            WHERE client_key_id = ?;
            """,
            (cutoff_rpm, window_start_ts, client_key_id),
        )
        row = cursor.fetchone()
        if row:
            return int(row[0]), int(row[1])
        return 0, 0

    def _sync_record_client_usage(
        self,
        client_key_id: int,
        timestamp: float,
        tokens: int = 0,
        status_code: int = 200,
    ) -> None:
        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO client_usage_events (client_key_id, timestamp, tokens, status_code)
            VALUES (?, ?, ?, ?);
            """,
            (client_key_id, timestamp, tokens, status_code),
        )
        cursor.execute(
            """
            UPDATE client_keys
            SET total_requests = total_requests + 1,
                total_tokens = total_tokens + ?,
                last_used_at = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (tokens, now_iso, now_iso, client_key_id),
        )
        conn.commit()

    def _sync_get_all_keys(self) -> list[dict[str, object]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM api_keys ORDER BY id ASC;")
        return [self._row_to_key_dict(row) for row in cursor.fetchall()]

    def _sync_get_key_by_id(self, key_id: int) -> dict[str, object] | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM api_keys WHERE id = ?;", (key_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_key_dict(row)

    def _sync_add_key(self, key_value: str, label: str) -> dict[str, object]:
        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO api_keys (
                key_value, label, status, is_disabled, consecutive_429,
                cooldown_until, created_at, updated_at
            ) VALUES (?, ?, 'READY', 0, 0, 0.0, ?, ?)
            """,
            (key_value.strip(), label.strip(), now_iso, now_iso),
        )
        conn.commit()
        inserted_id = int(cursor.lastrowid)
        result = self._sync_get_key_by_id(inserted_id)
        if result is None:
            raise RuntimeError(f"Failed to retrieve newly added key id={inserted_id}")
        return result

    def _sync_bulk_create_keys(
        self, items: list[tuple[str, str]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Bulk insert keys into pool, skipping duplicates and validating length.

        Returns (added_records, item_results).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT key_value FROM api_keys;")
        existing_keys: set[str] = {str(row[0]) for row in cursor.fetchall()}

        now_iso = datetime.now(timezone.utc).isoformat()
        added_records: list[dict[str, object]] = []
        item_results: list[dict[str, object]] = []

        for key_val, label in items:
            cleaned_key = key_val.strip()
            cleaned_label = label.strip()
            masked = mask_api_key(cleaned_key)

            if len(cleaned_key) < 8:
                item_results.append(
                    {
                        "label": cleaned_label,
                        "masked_key": masked,
                        "status": "invalid",
                        "key_id": None,
                    }
                )
            elif cleaned_key in existing_keys:
                item_results.append(
                    {
                        "label": cleaned_label,
                        "masked_key": masked,
                        "status": "skipped_duplicate",
                        "key_id": None,
                    }
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO api_keys (
                        key_value, label, status, is_disabled, consecutive_429,
                        cooldown_until, created_at, updated_at
                    ) VALUES (?, ?, 'READY', 0, 0, 0.0, ?, ?);
                    """,
                    (cleaned_key, cleaned_label, now_iso, now_iso),
                )
                inserted_id = int(cursor.lastrowid)
                existing_keys.add(cleaned_key)
                rec = self._sync_get_key_by_id(inserted_id)
                if rec is not None:
                    added_records.append(rec)
                item_results.append(
                    {
                        "label": cleaned_label,
                        "masked_key": masked,
                        "status": "added",
                        "key_id": inserted_id,
                    }
                )

        conn.commit()
        return added_records, item_results

    def _sync_update_key(
        self,
        key_id: int,
        is_disabled: bool | None,
        reset_cooldown: bool | None,
        label: str | None,
    ) -> dict[str, object] | None:
        current = self._sync_get_key_by_id(key_id)
        if current is None:
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        new_is_disabled = current["is_disabled"] if is_disabled is None else is_disabled
        new_label = current["label"] if label is None else label.strip()
        new_cooldown_until = float(current["cooldown_until"])
        new_consecutive_429 = int(current["consecutive_429"])
        new_status = str(current["status"])

        if reset_cooldown:
            new_cooldown_until = 0.0
            new_consecutive_429 = 0
            if not new_is_disabled:
                new_status = "READY"

        if is_disabled is not None:
            if is_disabled:
                new_status = "DISABLED"
            else:
                if new_cooldown_until <= time.time():
                    new_status = "READY"
                else:
                    new_status = "COOLDOWN"

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE api_keys
            SET label = ?, is_disabled = ?, consecutive_429 = ?, cooldown_until = ?, status = ?, updated_at = ?
            WHERE id = ?;
            """,
            (
                new_label,
                1 if new_is_disabled else 0,
                new_consecutive_429,
                new_cooldown_until,
                new_status,
                now_iso,
                key_id,
            ),
        )
        conn.commit()
        return self._sync_get_key_by_id(key_id)

    def _sync_delete_key(self, key_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM usage_events WHERE key_id = ?;", (key_id,))
        cursor.execute("DELETE FROM api_keys WHERE id = ?;", (key_id,))
        conn.commit()
        return cursor.rowcount > 0

    def _sync_record_usage(
        self,
        key_id: int,
        model_id: str,
        action: str,
        timestamp: float,
        tokens_in: int,
        tokens_out: int,
        status_code: int,
        is_dry_run: bool,
    ) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO usage_events (
                key_id, model_id, action, timestamp, tokens_in, tokens_out, status_code, is_dry_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                key_id,
                model_id,
                action,
                timestamp,
                tokens_in,
                tokens_out,
                status_code,
                1 if is_dry_run else 0,
            ),
        )
        conn.commit()

    def _sync_get_sliding_window_usage(
        self, key_id: int, model_id: str, window_seconds: float, now: float | None = None
    ) -> tuple[int, int]:
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(id), COALESCE(SUM(tokens_in + tokens_out), 0)
            FROM usage_events
            WHERE key_id = ? AND model_id = ? AND timestamp >= ?;
            """,
            (key_id, model_id, cutoff),
        )
        row = cursor.fetchone()
        if row:
            return int(row[0]), int(row[1])
        return 0, 0

    def _sync_get_rpd_usage(
        self, key_id: int, model_id: str, window_start_timestamp: float
    ) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(id)
            FROM usage_events
            WHERE key_id = ? AND model_id = ? AND timestamp >= ?;
            """,
            (key_id, model_id, window_start_timestamp),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _sync_get_total_rpd_usage(
        self, key_id: int, window_start_timestamp: float
    ) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(id)
            FROM usage_events
            WHERE key_id = ? AND timestamp >= ?;
            """,
            (key_id, window_start_timestamp),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _sync_set_cooldown(
        self, key_id: int, cooldown_seconds: float, consecutive_429: int | None = None
    ) -> None:
        now = time.time()
        cooldown_until = now + cooldown_seconds
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        if consecutive_429 is not None:
            cursor.execute(
                """
                UPDATE api_keys
                SET cooldown_until = ?, consecutive_429 = ?, status = 'COOLDOWN', updated_at = ?
                WHERE id = ?;
                """,
                (cooldown_until, consecutive_429, now_iso, key_id),
            )
        else:
            cursor.execute(
                """
                UPDATE api_keys
                SET cooldown_until = ?, status = 'COOLDOWN', updated_at = ?
                WHERE id = ?;
                """,
                (cooldown_until, now_iso, key_id),
            )
        conn.commit()

    def _sync_reset_429_counter(self, key_id: int) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE api_keys
            SET consecutive_429 = 0, cooldown_until = 0.0, status = 'READY', updated_at = ?
            WHERE id = ? AND is_disabled = 0;
            """,
            (now_iso, key_id),
        )
        conn.commit()

    def _sync_record_request_log(
        self,
        timestamp: float,
        timestamp_iso: str,
        model_id: str,
        action: str,
        prompt_text: str,
        client_key_id: int | None = None,
        client_key_name: str | None = None,
        google_key_id: int | None = None,
        google_key_label: str | None = None,
        prompt_full_json: str | dict[str, object] | None = None,
        response_text: str = "",
        response_full_json: str | dict[str, object] | None = None,
        tool_calls: list[dict[str, object]] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        status_code: int = 200,
        error_message: str | None = None,
    ) -> int:
        p_json_str: str | None = None
        if isinstance(prompt_full_json, dict):
            p_json_str = json.dumps(prompt_full_json)
        elif isinstance(prompt_full_json, str):
            p_json_str = prompt_full_json

        r_json_str: str | None = None
        if isinstance(response_full_json, dict):
            r_json_str = json.dumps(response_full_json)
        elif isinstance(response_full_json, str):
            r_json_str = response_full_json

        t_json_str: str | None = None
        if tool_calls is not None:
            t_json_str = json.dumps(tool_calls)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO request_logs (
                timestamp, timestamp_iso, client_key_id, client_key_name,
                google_key_id, google_key_label, model_id, action,
                prompt_text, prompt_full_json, response_text, response_full_json,
                tool_calls_json, tokens_in, tokens_out, latency_ms, status_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                timestamp_iso,
                client_key_id,
                client_key_name,
                google_key_id,
                google_key_label,
                model_id,
                action,
                prompt_text,
                p_json_str,
                response_text,
                r_json_str,
                t_json_str,
                tokens_in,
                tokens_out,
                latency_ms,
                status_code,
                error_message,
            ),
        )
        inserted_id = int(cursor.lastrowid)

        # Rotation: delete older records if total count exceeds 2000
        cursor.execute(
            """
            DELETE FROM request_logs
            WHERE id NOT IN (
                SELECT id FROM request_logs
                ORDER BY id DESC
                LIMIT 2000
            );
            """
        )
        conn.commit()
        return inserted_id

    def _sync_get_request_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        client_key_id: int | None = None,
        google_key_id: int | None = None,
        search: str | None = None,
        since_timestamp: float | None = None,
    ) -> tuple[list[dict[str, object]], int, float]:
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions: list[str] = []
        params: list[object] = []

        if client_key_id is not None:
            conditions.append("client_key_id = ?")
            params.append(client_key_id)

        if google_key_id is not None:
            conditions.append("google_key_id = ?")
            params.append(google_key_id)

        if since_timestamp is not None:
            conditions.append("timestamp > ?")
            params.append(since_timestamp)

        if search is not None and search.strip():
            search_pattern = f"%{search.strip()}%"
            conditions.append("(prompt_text LIKE ? OR response_text LIKE ?)")
            params.append(search_pattern)
            params.append(search_pattern)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        count_sql = f"""
            SELECT COUNT(id), COALESCE(MAX(timestamp), 0.0)
            FROM request_logs
            {where_clause};
        """
        cursor.execute(count_sql, tuple(params))
        count_row = cursor.fetchone()
        total_count = int(count_row[0]) if count_row else 0
        latest_timestamp = float(count_row[1]) if count_row else 0.0

        select_sql = f"""
            SELECT *
            FROM request_logs
            {where_clause}
            ORDER BY timestamp DESC, id DESC
            LIMIT ? OFFSET ?;
        """
        cursor.execute(select_sql, (*params, limit, offset))
        rows = cursor.fetchall()

        records: list[dict[str, object]] = []
        for row in rows:
            tool_calls_list: list[dict[str, object]] = []
            raw_tc = None
            if "tool_calls_json" in row.keys():
                raw_tc = row["tool_calls_json"]
            if raw_tc and isinstance(raw_tc, str):
                try:
                    parsed_tc = json.loads(raw_tc)
                    if isinstance(parsed_tc, list):
                        for tc in parsed_tc:
                            if isinstance(tc, dict):
                                tool_calls_list.append(tc)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.debug("Failed to decode tool_calls_json: %s", exc)

            records.append(
                {
                    "id": int(row["id"]),
                    "timestamp": float(row["timestamp"]),
                    "timestamp_iso": str(row["timestamp_iso"]),
                    "client_key_id": int(row["client_key_id"]) if row["client_key_id"] is not None else None,
                    "client_key_name": str(row["client_key_name"]) if row["client_key_name"] is not None else None,
                    "google_key_id": int(row["google_key_id"]) if row["google_key_id"] is not None else None,
                    "google_key_label": str(row["google_key_label"]) if row["google_key_label"] is not None else None,
                    "model_id": str(row["model_id"]),
                    "action": str(row["action"]),
                    "prompt_text": str(row["prompt_text"]),
                    "prompt_full_json": str(row["prompt_full_json"]) if row["prompt_full_json"] is not None else None,
                    "response_text": str(row["response_text"]) if row["response_text"] is not None else "",
                    "response_full_json": str(row["response_full_json"]) if row["response_full_json"] is not None else None,
                    "tool_calls_json": str(raw_tc) if raw_tc is not None else None,
                    "tool_calls": tool_calls_list,
                    "tokens_in": int(row["tokens_in"]),
                    "tokens_out": int(row["tokens_out"]),
                    "latency_ms": float(row["latency_ms"]),
                    "status_code": int(row["status_code"]),
                    "error_message": str(row["error_message"]) if row["error_message"] is not None else None,
                }
            )

        return records, total_count, latest_timestamp

    def _sync_clear_request_logs(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM request_logs;")
        conn.commit()

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_init_tables)

    async def get_all_keys(self) -> list[dict[str, object]]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_all_keys)

    async def get_key_by_id(self, key_id: int) -> dict[str, object] | None:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_key_by_id, key_id)

    async def add_key(self, key_value: str, label: str) -> dict[str, object]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_add_key, key_value, label)

    async def bulk_create_keys(
        self, items: list[tuple[str, str]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Bulk insert keys into pool asynchronously with lock."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_bulk_create_keys, items)

    async def update_key(
        self,
        key_id: int,
        is_disabled: bool | None = None,
        reset_cooldown: bool | None = None,
        label: str | None = None,
    ) -> dict[str, object] | None:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_update_key, key_id, is_disabled, reset_cooldown, label
            )

    async def delete_key(self, key_id: int) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._sync_delete_key, key_id)

    async def record_usage(
        self,
        key_id: int,
        model_id: str,
        action: str,
        timestamp: float,
        tokens_in: int,
        tokens_out: int,
        status_code: int,
        is_dry_run: bool,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._sync_record_usage,
                key_id,
                model_id,
                action,
                timestamp,
                tokens_in,
                tokens_out,
                status_code,
                is_dry_run,
            )

    async def get_sliding_window_usage(
        self, key_id: int, model_id: str, window_seconds: float = 60.0
    ) -> tuple[int, int]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_sliding_window_usage, key_id, model_id, window_seconds
            )

    async def get_rpd_usage(
        self, key_id: int, model_id: str, window_start_timestamp: float
    ) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_rpd_usage, key_id, model_id, window_start_timestamp
            )

    async def get_total_rpd_usage(
        self, key_id: int, window_start_timestamp: float
    ) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_total_rpd_usage, key_id, window_start_timestamp
            )

    async def set_cooldown(
        self, key_id: int, cooldown_seconds: float, consecutive_429: int | None = None
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._sync_set_cooldown, key_id, cooldown_seconds, consecutive_429
            )

    async def reset_429_counter(self, key_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_reset_429_counter, key_id)

    async def create_client_key(
        self,
        name: str,
        key_value: str,
        rpm_limit: int = 60,
        rpd_limit: int = 1000,
    ) -> dict[str, object]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_create_client_key, name, key_value, rpm_limit, rpd_limit
            )

    async def get_all_client_keys(self) -> list[dict[str, object]]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_all_client_keys)

    async def get_client_key_by_id(self, client_key_id: int) -> dict[str, object] | None:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_client_key_by_id, client_key_id)

    async def get_client_key_by_token(self, token: str) -> dict[str, object] | None:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_client_key_by_token, token)

    async def update_client_key(
        self,
        client_key_id: int,
        name: str | None = None,
        rpm_limit: int | None = None,
        rpd_limit: int | None = None,
        status: str | None = None,
    ) -> dict[str, object] | None:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_update_client_key, client_key_id, name, rpm_limit, rpd_limit, status
            )

    async def delete_client_key(self, client_key_id: int) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._sync_delete_client_key, client_key_id)

    async def count_client_keys(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._sync_count_client_keys)

    async def get_client_usage(
        self, client_key_id: int, current_time: float, window_start_msk_iso: str
    ) -> tuple[int, int]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_client_usage, client_key_id, current_time, window_start_msk_iso
            )

    async def record_client_usage(
        self,
        client_key_id: int,
        timestamp: float,
        tokens: int = 0,
        status_code: int = 200,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._sync_record_client_usage, client_key_id, timestamp, tokens, status_code
            )

    async def record_request_log(
        self,
        timestamp: float,
        timestamp_iso: str,
        model_id: str,
        action: str,
        prompt_text: str,
        client_key_id: int | None = None,
        client_key_name: str | None = None,
        google_key_id: int | None = None,
        google_key_label: str | None = None,
        prompt_full_json: str | dict[str, object] | None = None,
        response_text: str = "",
        response_full_json: str | dict[str, object] | None = None,
        tool_calls: list[dict[str, object]] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        status_code: int = 200,
        error_message: str | None = None,
    ) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_record_request_log,
                timestamp=timestamp,
                timestamp_iso=timestamp_iso,
                model_id=model_id,
                action=action,
                prompt_text=prompt_text,
                client_key_id=client_key_id,
                client_key_name=client_key_name,
                google_key_id=google_key_id,
                google_key_label=google_key_label,
                prompt_full_json=prompt_full_json,
                response_text=response_text,
                response_full_json=response_full_json,
                tool_calls=tool_calls,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status_code=status_code,
                error_message=error_message,
            )

    async def get_request_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        client_key_id: int | None = None,
        google_key_id: int | None = None,
        search: str | None = None,
        since_timestamp: float | None = None,
    ) -> tuple[list[dict[str, object]], int, float]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_request_logs,
                limit=limit,
                offset=offset,
                client_key_id=client_key_id,
                google_key_id=google_key_id,
                search=search,
                since_timestamp=since_timestamp,
            )

    async def clear_request_logs(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_clear_request_logs)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None


_default_db: DatabaseManager | None = None


def get_database(db_path: str | None = None) -> DatabaseManager:
    """Return default singleton DatabaseManager instance."""
    global _default_db
    if _default_db is None:
        target_path = db_path if db_path is not None else "data/router.db"
        _default_db = DatabaseManager(db_path=target_path)
    elif db_path is not None and _default_db.db_path != db_path:
        _default_db = DatabaseManager(db_path=db_path)
    return _default_db


def set_database(db: DatabaseManager) -> None:
    """Override default singleton DatabaseManager instance (useful for tests)."""
    global _default_db
    _default_db = db


async def init_db(db_path: str = "data/router.db") -> DatabaseManager:
    """Initialize database and seed preloaded keys if table is empty."""
    db = get_database(db_path)
    await db.initialize()
    return db


async def get_all_keys() -> list[dict[str, object]]:
    """Fetch all keys from the database."""
    return await get_database().get_all_keys()


async def get_key_by_id(key_id: int) -> dict[str, object] | None:
    """Fetch single key by ID."""
    return await get_database().get_key_by_id(key_id)


async def add_key(key_value: str, label: str) -> dict[str, object]:
    """Insert new API key into the database."""
    return await get_database().add_key(key_value, label)


async def bulk_create_keys(
    items: list[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Bulk insert keys into pool, skipping duplicates and validating length."""
    return await get_database().bulk_create_keys(items)


async def update_key(
    key_id: int,
    is_disabled: bool | None = None,
    reset_cooldown: bool | None = None,
    label: str | None = None,
) -> dict[str, object] | None:
    """Update API key properties, status, or reset cooldown."""
    return await get_database().update_key(key_id, is_disabled, reset_cooldown, label)


async def delete_key(key_id: int) -> bool:
    """Delete key and associated usage events from the database."""
    return await get_database().delete_key(key_id)


async def record_usage(
    key_id: int,
    model_id: str,
    action: str,
    timestamp: float,
    tokens_in: int,
    tokens_out: int,
    status_code: int,
    is_dry_run: bool,
) -> None:
    """Log an API usage event."""
    await get_database().record_usage(
        key_id, model_id, action, timestamp, tokens_in, tokens_out, status_code, is_dry_run
    )


async def get_sliding_window_usage(
    key_id: int, model_id: str, window_seconds: float = 60.0
) -> tuple[int, int]:
    """Calculate RPM and TPM usage in sliding window."""
    return await get_database().get_sliding_window_usage(key_id, model_id, window_seconds)


async def get_rpd_usage(key_id: int, model_id: str, window_start_timestamp: float) -> int:
    """Calculate RPD usage for a specific model since window start."""
    return await get_database().get_rpd_usage(key_id, model_id, window_start_timestamp)


async def get_total_rpd_usage(key_id: int, window_start_timestamp: float) -> int:
    """Calculate total RPD usage across all models since window start."""
    return await get_database().get_total_rpd_usage(key_id, window_start_timestamp)


async def set_cooldown(
    key_id: int, cooldown_seconds: float, consecutive_429: int
) -> None:
    """Place key into cooldown with consecutive 429 count."""
    await get_database().set_cooldown(key_id, cooldown_seconds, consecutive_429)


async def reset_429_counter(key_id: int) -> None:
    """Reset key cooldown and consecutive 429 counter."""
    await get_database().reset_429_counter(key_id)


async def create_client_key(
    name: str, key_value: str, rpm_limit: int = 60, rpd_limit: int = 1000
) -> dict[str, object]:
    """Create a new client API key."""
    return await get_database().create_client_key(name, key_value, rpm_limit, rpd_limit)


async def get_all_client_keys() -> list[dict[str, object]]:
    """Retrieve all client API keys."""
    return await get_database().get_all_client_keys()


async def get_client_key_by_id(client_key_id: int) -> dict[str, object] | None:
    """Retrieve client API key by ID."""
    return await get_database().get_client_key_by_id(client_key_id)


async def get_client_key_by_token(token: str) -> dict[str, object] | None:
    """Retrieve client API key by raw token value."""
    return await get_database().get_client_key_by_token(token)


async def update_client_key(
    client_key_id: int,
    name: str | None = None,
    rpm_limit: int | None = None,
    rpd_limit: int | None = None,
    status: str | None = None,
) -> dict[str, object] | None:
    """Update client API key properties."""
    return await get_database().update_client_key(
        client_key_id, name, rpm_limit, rpd_limit, status
    )


async def delete_client_key(client_key_id: int) -> bool:
    """Delete client API key and associated usage events."""
    return await get_database().delete_client_key(client_key_id)


async def count_client_keys() -> int:
    """Count total client API keys in the database."""
    return await get_database().count_client_keys()


async def get_client_usage(
    client_key_id: int, current_time: float, window_start_msk_iso: str
) -> tuple[int, int]:
    """Get RPM and daily RPD usage for a client key."""
    return await get_database().get_client_usage(
        client_key_id, current_time, window_start_msk_iso
    )


async def record_client_usage(
    client_key_id: int,
    timestamp: float,
    tokens: int = 0,
    status_code: int = 200,
) -> None:
    """Record usage event for client API key and update stats."""
    await get_database().record_client_usage(
        client_key_id, timestamp, tokens, status_code
    )


async def record_request_log(
    timestamp: float,
    timestamp_iso: str,
    model_id: str,
    action: str,
    prompt_text: str,
    client_key_id: int | None = None,
    client_key_name: str | None = None,
    google_key_id: int | None = None,
    google_key_label: str | None = None,
    prompt_full_json: str | dict[str, object] | None = None,
    response_text: str = "",
    response_full_json: str | dict[str, object] | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0.0,
    status_code: int = 200,
    error_message: str | None = None,
) -> int:
    """Record a request log entry."""
    return await get_database().record_request_log(
        timestamp=timestamp,
        timestamp_iso=timestamp_iso,
        model_id=model_id,
        action=action,
        prompt_text=prompt_text,
        client_key_id=client_key_id,
        client_key_name=client_key_name,
        google_key_id=google_key_id,
        google_key_label=google_key_label,
        prompt_full_json=prompt_full_json,
        response_text=response_text,
        response_full_json=response_full_json,
        tool_calls=tool_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        status_code=status_code,
        error_message=error_message,
    )


async def get_request_logs(
    limit: int = 50,
    offset: int = 0,
    client_key_id: int | None = None,
    google_key_id: int | None = None,
    search: str | None = None,
    since_timestamp: float | None = None,
) -> tuple[list[dict[str, object]], int, float]:
    """Retrieve filtered request logs."""
    return await get_database().get_request_logs(
        limit=limit,
        offset=offset,
        client_key_id=client_key_id,
        google_key_id=google_key_id,
        search=search,
        since_timestamp=since_timestamp,
    )


async def clear_request_logs() -> None:
    """Clear all request logs."""
    await get_database().clear_request_logs()


