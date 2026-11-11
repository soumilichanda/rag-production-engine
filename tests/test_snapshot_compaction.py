import os

import pytest

from app.retrieval.snapshot_engine import SnapshotEngine
from app.retrieval.wal import WriteAheadLog
from app.retrieval.wal_compactor import WALCompactor


@pytest.fixture
def temp_environment(tmp_path):
    wal_file = str(tmp_path / "test_wal.log")
    snap_dir = str(tmp_path / "snapshots")
    return wal_file, snap_dir


def test_snapshot_creation_and_recovery(temp_environment):
    _wal_file, snap_dir = temp_environment
    engine = SnapshotEngine(snapshot_dir=snap_dir)

    mock_state = {"shard_0": ["doc_1", "doc_2"], "shard_1": ["doc_3"]}
    snap_path = engine.create_snapshot(last_seq_id=42, shard_state=mock_state)

    assert os.path.exists(snap_path)
    loaded = engine.load_latest_snapshot()
    assert loaded["checkpoint_seq"] == 42
    assert loaded["shard_state"]["shard_0"] == ["doc_1", "doc_2"]


def test_compaction_truncates_wal(temp_environment):
    wal_file, snap_dir = temp_environment
    wal = WriteAheadLog(log_path=wal_file)
    engine = SnapshotEngine(snapshot_dir=snap_dir)
    compactor = WALCompactor(wal=wal, snapshot_engine=engine)

    wal.append(WriteAheadLog.OP_INSERT, {"chunk_id": "c1"})
    wal.append(WriteAheadLog.OP_INSERT, {"chunk_id": "c2"})
    assert wal._current_seq == 2

    shards = {"shard_0": ["c1", "c2"]}
    checkpoint_seq = compactor.execute_checkpoint_and_compact(shards)
    assert checkpoint_seq == 2

    entries = list(wal.replay_entries())
    assert len(entries) == 0

    recovered = engine.load_latest_snapshot()
    assert recovered["checkpoint_seq"] == 2
    assert recovered["shard_state"] == shards
