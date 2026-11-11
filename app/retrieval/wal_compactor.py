"""
app/retrieval/wal_compactor.py
Dynamic WAL segment compactor coalescing historical log segments
and eliminating redundant mutation overhead while retaining CRC32 frame invariants.
Also preserves backward compatibility for single-log WALCompactor.
"""

import json
import os
import struct
import threading
import zlib
from typing import Any

from app.retrieval.wal_segment import SegmentedWALWriter


class DynamicSegmentCompactor:
    HEADER_STRUCT = struct.Struct(">IIQB")

    def __init__(
        self, wal_writer: SegmentedWALWriter, min_segments_to_compact: int = 3
    ):
        self.wal_writer = wal_writer
        self.min_segments_to_compact = min_segments_to_compact
        self.lock = threading.RLock()

    def compact_segments(self) -> dict[str, Any]:
        with self.lock, self.wal_writer.lock:
            segments = self.wal_writer.list_segments()
            if len(segments) <= self.min_segments_to_compact:
                return {
                    "compacted": False,
                    "reason": f"Insufficient segments ({len(segments)} <= {self.min_segments_to_compact})",
                    "coalesced_segments": [],
                }

            target_segments = segments[:-1]
            latest_chunk_state: dict[str, dict[str, Any]] = {}
            deleted_chunk_ids: set[str] = set()

            for seg_idx in target_segments:
                for seq_id, op_type, data in self.wal_writer.replay_segment(seg_idx):
                    chunk_id = data.get("chunk_id") or data.get("chunk")
                    if not chunk_id:
                        continue

                    if op_type == SegmentedWALWriter.OP_DELETE:
                        deleted_chunk_ids.add(chunk_id)
                        latest_chunk_state.pop(chunk_id, None)
                    elif op_type == SegmentedWALWriter.OP_CLEAR:
                        deleted_chunk_ids.clear()
                        latest_chunk_state.clear()
                    elif op_type == SegmentedWALWriter.OP_INSERT:
                        deleted_chunk_ids.discard(chunk_id)
                        latest_chunk_state[chunk_id] = {
                            "seq_id": seq_id,
                            "op_type": op_type,
                            "data": data,
                        }

            compacted_file = os.path.join(
                self.wal_writer.wal_dir, f"wal_compacted_{target_segments[0]:05d}.tmp"
            )
            written_records = 0

            with open(compacted_file, "wb") as f:
                surviving = sorted(
                    latest_chunk_state.values(), key=lambda x: x["seq_id"]
                )
                for item in surviving:
                    payload_bytes = json.dumps(item["data"]).encode("utf-8")
                    crc = zlib.crc32(payload_bytes)
                    length = len(payload_bytes)
                    header = self.HEADER_STRUCT.pack(
                        length, crc, item["seq_id"], item["op_type"]
                    )
                    f.write(header + payload_bytes)
                    written_records += 1
                f.flush()
                os.fsync(f.fileno())

            first_target_path = self.wal_writer._segment_filename(target_segments[0])
            os.replace(compacted_file, first_target_path)

            for seg_idx in target_segments[1:]:
                path = self.wal_writer._segment_filename(seg_idx)
                try:
                    os.unlink(path)
                except OSError:
                    pass

            return {
                "compacted": True,
                "coalesced_segments": target_segments,
                "surviving_records": written_records,
                "active_segments": self.wal_writer.list_segments(),
            }


class WALCompactor:
    """Legacy backward-compatible adapter linking WriteAheadLog with SnapshotEngine."""

    def __init__(self, wal: Any, snapshot_engine: Any):
        self.wal = wal
        self.snapshot_engine = snapshot_engine

    def execute_checkpoint_and_compact(
        self, current_shards: dict[str, list[str]]
    ) -> int:
        with self.wal.lock:
            latest_seq = self.wal._current_seq
            if latest_seq == 0:
                return 0
            self.snapshot_engine.create_snapshot(latest_seq, current_shards)
            self.wal.truncate()
            return latest_seq
