import os
import pytest
from app.retrieval.wal import WriteAheadLog
from app.retrieval.wal_journal import ShardMutationJournal


@pytest.fixture
def temp_wal_path(tmp_path):
    return str(tmp_path / "test_wal.log")


def test_wal_append_and_replay_integrity(temp_wal_path):
    wal = WriteAheadLog(log_path=temp_wal_path)
    seq1 = wal.append(WriteAheadLog.OP_INSERT, {"chunk_id": "c1", "content": "alpha"})
    seq2 = wal.append(WriteAheadLog.OP_INSERT, {"chunk_id": "c2", "content": "beta"})

    assert seq1 == 1
    assert seq2 == 2

    entries = list(wal.replay_entries())
    assert len(entries) == 2
    assert entries[0][0] == 1
    assert entries[0][2]["chunk_id"] == "c1"
    assert entries[1][0] == 2
    assert entries[1][2]["chunk_id"] == "c2"


def test_wal_crc_corruption_detection(temp_wal_path):
    wal = WriteAheadLog(log_path=temp_wal_path)
    wal.append(WriteAheadLog.OP_INSERT, {"chunk_id": "c_secure", "val": 100})

    # Intentionally flip bytes in the payload to simulate disk corruption
    with open(temp_wal_path, "r+b") as f:
        f.seek(-2, os.SEEK_END)
        f.write(b"\xFF\xFF")

    corrupted_wal = WriteAheadLog(log_path=temp_wal_path)
    with pytest.raises(ValueError, match="CRC32 checksum mismatch"):
        list(corrupted_wal.replay_entries())


def test_journal_cold_boot_recovery(temp_wal_path):
    journal = ShardMutationJournal(wal_path=temp_wal_path)
    journal.record_mutation("insert", "doc_101", "shard_A")
    journal.record_mutation("insert", "doc_102", "shard_A")
    journal.record_mutation("insert", "doc_201", "shard_B")
    journal.record_mutation("delete", "doc_101", "shard_A")

    # Reconstruct empty state from log
    mock_shards = {}
    replayed = journal.recover(mock_shards)

    assert replayed == 4
    assert mock_shards["shard_A"] == ["doc_102"]
    assert mock_shards["shard_B"] == ["doc_201"]
