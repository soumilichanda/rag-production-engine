"""
app/retrieval/wal_compactor.py
Log compaction controller linking WriteAheadLog truncation with SnapshotEngine checkpoints.
"""

from app.retrieval.snapshot_engine import SnapshotEngine
from app.retrieval.wal import WriteAheadLog


class WALCompactor:
    def __init__(self, wal: WriteAheadLog, snapshot_engine: SnapshotEngine):
        self.wal = wal
        self.snapshot_engine = snapshot_engine

    def execute_checkpoint_and_compact(
        self, current_shards: dict[str, list[str]]
    ) -> int:
        with self.wal.lock:
            latest_seq = self.wal._current_seq
            if latest_seq == 0:
                return 0

            # 1. Snapshot memory state at current sequence ID
            self.snapshot_engine.create_snapshot(latest_seq, current_shards)

            # 2. Safely truncate the WAL log file
            self.wal.truncate()
            return latest_seq
