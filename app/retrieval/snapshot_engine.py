"""
app/retrieval/snapshot_engine.py
Snapshotting engine for sharded vector memory states.
Provides atomic checkpoint persistence and sequence watermarking.
"""

import json
import os
import tempfile
from typing import Any


class SnapshotEngine:
    def __init__(self, snapshot_dir: str = "data/snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def get_latest_checkpoint_metadata(self) -> dict[str, Any]:
        meta_path = os.path.join(self.snapshot_dir, "checkpoint_meta.json")
        if not os.path.exists(meta_path):
            return {"last_checkpoint_seq": 0, "snapshot_file": None}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_snapshot(
        self, last_seq_id: int, shard_state: dict[str, list[str]]
    ) -> str:
        snapshot_filename = f"snapshot_seq_{last_seq_id}.json"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        meta_path = os.path.join(self.snapshot_dir, "checkpoint_meta.json")

        payload = {
            "checkpoint_seq": last_seq_id,
            "shard_state": shard_state,
        }

        # Atomic tempfile replacement pattern
        with tempfile.NamedTemporaryFile(
            "w", dir=self.snapshot_dir, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(payload, tf, indent=2)
            temp_name = tf.name

        os.replace(temp_name, snapshot_path)

        meta_payload = {
            "last_checkpoint_seq": last_seq_id,
            "snapshot_file": snapshot_filename,
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=self.snapshot_dir, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(meta_payload, tf, indent=2)
            temp_meta_name = tf.name

        os.replace(temp_meta_name, meta_path)
        return snapshot_path

    def load_latest_snapshot(self) -> dict[str, Any]:
        meta = self.get_latest_checkpoint_metadata()
        if not meta.get("snapshot_file"):
            return {"checkpoint_seq": 0, "shard_state": {}}

        snapshot_path = os.path.join(self.snapshot_dir, meta["snapshot_file"])
        if not os.path.exists(snapshot_path):
            return {"checkpoint_seq": 0, "shard_state": {}}

        with open(snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)
