"""
app/retrieval/wal.py
Distributed Write-Ahead Logging (WAL) engine with CRC32 integrity frames
and cold-start recovery replay mechanics for vector shard mutations.
"""

import os
import struct
import zlib
import json
import threading
from typing import Iterator, Dict, Any, Tuple


class WriteAheadLog:
    # Frame format: Length(4B) + CRC32(4B) + SeqID(8B) + OpType(1B)
    HEADER_STRUCT = struct.Struct(">IIQB")
    OP_INSERT = 1
    OP_DELETE = 2
    OP_CLEAR = 3

    def __init__(self, log_path: str = "data/wal.log"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.lock = threading.RLock()
        self._current_seq = self._recover_latest_seq()

    def _recover_latest_seq(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        latest_seq = 0
        try:
            for seq_id, _, _ in self.replay_entries():
                latest_seq = max(latest_seq, seq_id)
        except Exception:
            pass
        return latest_seq

    @property
    def next_seq_id(self) -> int:
        with self.lock:
            self._current_seq += 1
            return self._current_seq

    def append(self, op_type: int, payload: Dict[str, Any]) -> int:
        payload_bytes = json.dumps(payload).encode("utf-8")
        seq_id = self.next_seq_id
        crc = zlib.crc32(payload_bytes)
        length = len(payload_bytes)

        header = self.HEADER_STRUCT.pack(length, crc, seq_id, op_type)

        with self.lock:
            with open(self.log_path, "ab") as f:
                f.write(header + payload_bytes)
                f.flush()
                os.fsync(f.fileno())

        return seq_id

    def replay_entries(self) -> Iterator[Tuple[int, int, Dict[str, Any]]]:
        if not os.path.exists(self.log_path):
            return

        with self.lock:
            with open(self.log_path, "rb") as f:
                while True:
                    header_bytes = f.read(self.HEADER_STRUCT.size)
                    if not header_bytes or len(header_bytes) < self.HEADER_STRUCT.size:
                        break

                    length, expected_crc, seq_id, op_type = self.HEADER_STRUCT.unpack(header_bytes)
                    payload_bytes = f.read(length)

                    if len(payload_bytes) != length:
                        raise ValueError(f"Truncated WAL entry detected at seq_id={seq_id}")

                    computed_crc = zlib.crc32(payload_bytes)
                    if computed_crc != expected_crc:
                        raise ValueError(f"CRC32 checksum mismatch at seq_id={seq_id}: corrupt WAL frame")

                    data = json.loads(payload_bytes.decode("utf-8"))
                    yield (seq_id, op_type, data)

    def truncate(self) -> None:
        with self.lock:
            with open(self.log_path, "wb") as f:
                f.truncate(0)
            self._current_seq = 0
