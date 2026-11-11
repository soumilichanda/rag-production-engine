"""
app/retrieval/wal_journal.py
Journal-driven partition synchronizer with cold-boot crash recovery.
"""

from typing import Any

from app.retrieval.wal import WriteAheadLog


class ShardMutationJournal:
    def __init__(self, wal_path: str = "data/wal.log"):
        self.wal = WriteAheadLog(log_path=wal_path)

    def record_mutation(
        self, op: str, chunk_id: str, shard_id: str, data: dict[str, Any] | None = None
    ) -> int:
        op_map = {
            "insert": WriteAheadLog.OP_INSERT,
            "delete": WriteAheadLog.OP_DELETE,
            "clear": WriteAheadLog.OP_CLEAR,
        }
        op_code = op_map.get(op.lower(), WriteAheadLog.OP_INSERT)
        payload = {
            "chunk_id": chunk_id,
            "shard_id": shard_id,
            "data": data or {},
        }
        return self.wal.append(op_code, payload)

    def recover(self, active_shards: dict[str, list[str]]) -> int:
        replayed_count = 0
        for seq_id, op_type, entry in self.wal.replay_entries():
            shard_id = entry.get("shard_id")
            chunk_id = entry.get("chunk_id")
            if shard_id not in active_shards:
                active_shards[shard_id] = []

            if op_type == WriteAheadLog.OP_INSERT:
                if chunk_id not in active_shards[shard_id]:
                    active_shards[shard_id].append(chunk_id)
            elif op_type == WriteAheadLog.OP_DELETE:
                if chunk_id in active_shards[shard_id]:
                    active_shards[shard_id].remove(chunk_id)
            elif op_type == WriteAheadLog.OP_CLEAR:
                active_shards[shard_id].clear()

            replayed_count += 1
        return replayed_count
