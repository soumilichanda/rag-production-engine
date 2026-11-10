"""
app/retrieval/wal_gc.py
Garbage collector for checkpointed and finalized WAL log segments.
Enforces safe retention policies without interrupting real-time appends.
"""

import os
from typing import Any

from app.retrieval.snapshot_engine import SnapshotEngine
from app.retrieval.wal_segment import SegmentedWALWriter


class SegmentGarbageCollector:
    def __init__(self, wal_writer: SegmentedWALWriter, snapshot_engine: SnapshotEngine):
        self.wal_writer = wal_writer
        self.snapshot_engine = snapshot_engine

    def inspect_segment_max_seq(self, segment_idx: int) -> int:
        max_seq = 0
        try:
            for seq_id, _, _ in self.wal_writer.replay_segment(segment_idx):
                max_seq = max(max_seq, seq_id)
        except (OSError, ValueError):
            pass
        return max_seq

    def collect_garbage(self) -> dict[str, Any]:
        meta = self.snapshot_engine.get_latest_checkpoint_metadata()
        last_checkpoint_seq = meta.get("last_checkpoint_seq", 0)

        deleted_segments: list[int] = []
        with self.wal_writer.lock:
            segments = self.wal_writer.list_segments()
            # Never delete the active write segment (the latest index)
            for seg_idx in segments[:-1]:
                max_seq = self.inspect_segment_max_seq(seg_idx)
                if max_seq > 0 and max_seq <= last_checkpoint_seq:
                    seg_path = self.wal_writer._segment_filename(seg_idx)
                    try:
                        os.unlink(seg_path)
                        deleted_segments.append(seg_idx)
                    except OSError:
                        pass

        return {
            "checkpoint_seq": last_checkpoint_seq,
            "deleted_segments": deleted_segments,
            "remaining_segments": self.wal_writer.list_segments(),
        }
