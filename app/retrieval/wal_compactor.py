"""
app/retrieval/wal_compactor.py
Dynamic WAL segment compactor coalescing historical log segments
and eliminating redundant mutation overhead while retaining CRC32 frame invariants.
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
            # Shield active tail segment from compaction
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
                        if chunk_id in deleted_chunk_ids:
                            deleted_chunk_ids.remove(chunk_id)
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
                # Write back active surviving state ordered by original seq_id
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

            # Replace first target segment with compacted log and unlink subsequent targets
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
