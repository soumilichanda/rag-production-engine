"""
app/retrieval/wal_segment.py
Segmented Write-Ahead Logging (WAL) engine with size-bounded monotonic
segment rollover, CRC32 checksum frames, and multi-segment replay iterator.
"""

import json
import os
import struct
import threading
import zlib
from collections.abc import Iterator
from typing import Any


class SegmentedWALWriter:
    # Frame format: Length(4B) + CRC32(4B) + SeqID(8B) + OpType(1B)
    HEADER_STRUCT = struct.Struct(">IIQB")
    OP_INSERT = 1
    OP_DELETE = 2
    OP_CLEAR = 3

    def __init__(self, wal_dir: str = "data/wal_segments", max_segment_bytes: int = 10 * 1024 * 1024):
        self.wal_dir = wal_dir
        self.max_segment_bytes = max_segment_bytes
        os.makedirs(self.wal_dir, exist_ok=True)
        self.lock = threading.RLock()
        self._current_segment_idx = self._find_latest_segment_idx()
        self._current_seq = self._recover_latest_seq()

    def _segment_filename(self, index: int) -> str:
        return os.path.join(self.wal_dir, f"wal_{index:05d}.log")

    def _find_latest_segment_idx(self) -> int:
        segments = self.list_segments()
        if not segments:
            return 1
        return max(segments)

    def list_segments(self) -> list[int]:
        indexes = []
        for f in os.listdir(self.wal_dir):
            if f.startswith("wal_") and f.endswith(".log"):
                try:
                    idx = int(f.replace("wal_", "").replace(".log", ""))
                    indexes.append(idx)
                except ValueError:
                    continue
        return sorted(indexes)

    def _recover_latest_seq(self) -> int:
        latest = 0
        try:
            for seq_id, _, _ in self.replay_all():
                latest = max(latest, seq_id)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return latest

    @property
    def next_seq_id(self) -> int:
        with self.lock:
            self._current_seq += 1
            return self._current_seq

    def append(self, op_type: int, payload: dict[str, Any]) -> int:
        payload_bytes = json.dumps(payload).encode("utf-8")
        seq_id = self.next_seq_id
        crc = zlib.crc32(payload_bytes)
        length = len(payload_bytes)

        header = self.HEADER_STRUCT.pack(length, crc, seq_id, op_type)
        frame = header + payload_bytes

        with self.lock:
            curr_path = self._segment_filename(self._current_segment_idx)
            curr_size = os.path.getsize(curr_path) if os.path.exists(curr_path) else 0

            # Monotonic segment roll-over on size threshold breach
            if curr_size + len(frame) > self.max_segment_bytes and curr_size > 0:
                self._current_segment_idx += 1
                curr_path = self._segment_filename(self._current_segment_idx)

            with open(curr_path, "ab") as f:
                f.write(frame)
                f.flush()
                os.fsync(f.fileno())

        return seq_id

    def replay_segment(self, segment_idx: int) -> Iterator[tuple[int, int, dict[str, Any]]]:
        path = self._segment_filename(segment_idx)
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            while True:
                header_bytes = f.read(self.HEADER_STRUCT.size)
                if not header_bytes or len(header_bytes) < self.HEADER_STRUCT.size:
                    break

                length, expected_crc, seq_id, op_type = self.HEADER_STRUCT.unpack(header_bytes)
                payload_bytes = f.read(length)

                if len(payload_bytes) != length:
                    raise ValueError(f"Truncated WAL frame at seq_id={seq_id} in segment {segment_idx}")

                computed_crc = zlib.crc32(payload_bytes)
                if computed_crc != expected_crc:
                    raise ValueError(f"CRC32 mismatch at seq_id={seq_id} in segment {segment_idx}")

                data = json.loads(payload_bytes.decode("utf-8"))
                yield (seq_id, op_type, data)

    def replay_all(self) -> Iterator[tuple[int, int, dict[str, Any]]]:
        with self.lock:
            segments = self.list_segments()
            for seg_idx in segments:
                yield from self.replay_segment(seg_idx)
