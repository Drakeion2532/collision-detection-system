# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Incident database (SQLite): every alert is logged with its snapshot for later review."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import ROOT

SCHEMA = """CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc TEXT NOT NULL, source TEXT, model TEXT,
  frame_index INTEGER, video_time_s REAL, confidence REAL,
  snapshot_path TEXT, status TEXT DEFAULT 'unverified')"""


class IncidentStore:
    def __init__(self, path: str | Path = ROOT / "results" / "incidents.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, inc, source="", model="") -> int:
        snap = None
        if inc.frame is not None:
            import cv2
            d = self.path.parent / "snapshots"
            d.mkdir(exist_ok=True)
            snap = str(d / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{inc.frame_index}.jpg")
            cv2.imwrite(snap, inc.frame)
        with self._conn() as c:
            cur = c.execute("INSERT INTO incidents (created_utc, source, model, frame_index, video_time_s, confidence, snapshot_path)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (datetime.now(timezone.utc).isoformat(), source, model, inc.frame_index,
                             inc.timestamp_s, inc.confidence, snap))
            return cur.lastrowid

    def set_status(self, incident_id: int, status: str):
        assert status in {"unverified", "confirmed", "false_alarm"}
        with self._conn() as c:
            c.execute("UPDATE incidents SET status=? WHERE id=?", (status, incident_id))

    def all(self) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM incidents ORDER BY id DESC")]
