import pytest
from app.retrieval.wal_segment import SegmentedWALWriter
from app.retrieval.wal_gc import SegmentGarbageCollector
from app.retrieval.snapshot_engine import SnapshotEngine


@pytest.fixture
def temp_segmented_env(tmp_path):
    wal_dir = str(tmp_path / "wal_segments")
    snap_dir = str(tmp_path / "snapshots")
    return wal_dir, snap_dir


def test_monotonic_segment_rollover(temp_segmented_env):
    wal_dir, _ = temp_segmented_env
    # 120 bytes max capacity triggers rapid roll-over
    writer = SegmentedWALWriter(wal_dir=wal_dir, max_segment_bytes=120)

    seq1 = writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "doc_alpha"})
    seq2 = writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "doc_beta_with_longer_payload"})
    seq3 = writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "doc_gamma"})

    segments = writer.list_segments()
    assert len(segments) >= 2
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3

    # Ensure full monotonic replay matches sequence order across segment boundaries
    replayed = list(writer.replay_all())
    assert len(replayed) == 3
    assert [item[0] for item in replayed] == [1, 2, 3]


def test_segment_retention_garbage_collection(temp_segmented_env):
    wal_dir, snap_dir = temp_segmented_env
    writer = SegmentedWALWriter(wal_dir=wal_dir, max_segment_bytes=100)
    snap_engine = SnapshotEngine(snapshot_dir=snap_dir)
    gc = SegmentGarbageCollector(wal_writer=writer, snapshot_engine=snap_engine)

    # Populate entries to force at least 3 segments
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "first_item_in_seg_1"})
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "second_item_forces_seg_2"})
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk": "third_item_forces_seg_3"})

    segments_before = writer.list_segments()
    assert len(segments_before) >= 3

    # Create checkpoint at seq_id = 2
    snap_engine.create_snapshot(last_seq_id=2, shard_state={"shard_0": ["c1", "c2"]})

    # Run GC
    res = gc.collect_garbage()
    assert len(res["deleted_segments"]) >= 1
    assert 1 in res["deleted_segments"]

    # Active segment remains intact
    assert writer.list_segments()[-1] in res["remaining_segments"]
