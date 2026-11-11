import pytest
import numpy as np
from app.retrieval.wal_segment import SegmentedWALWriter
from app.retrieval.wal_compactor import DynamicSegmentCompactor
from app.retrieval.sharded_index import ShardedVectorIndex
from app.retrieval.shard_merger import BackgroundShardMerger
from app.schemas.payloads import DocumentChunk


@pytest.fixture
def temp_wal_dir(tmp_path):
    return str(tmp_path / "wal_segments")


def test_wal_segment_compaction(temp_wal_dir):
    # Small capacity forces generation of multiple segments
    writer = SegmentedWALWriter(wal_dir=temp_wal_dir, max_segment_bytes=100)
    compactor = DynamicSegmentCompactor(wal_writer=writer, min_segments_to_compact=2)

    # Segment 1 & 2: duplicate chunk overrides and deletes
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk_id": "c1", "val": "v1"})
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk_id": "c2", "val": "v2"})
    writer.append(SegmentedWALWriter.OP_DELETE, {"chunk_id": "c1"})
    # Forces segment rollover
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk_id": "c3", "val": "v3"})
    writer.append(SegmentedWALWriter.OP_INSERT, {"chunk_id": "active_tail", "val": "v4"})

    segments_before = writer.list_segments()
    assert len(segments_before) >= 3

    res = compactor.compact_segments()
    assert res["compacted"] is True
    assert len(res["coalesced_segments"]) >= 2

    # Verify post-compaction replay contains surviving c3 and c2, but not deleted c1
    replayed = list(writer.replay_all())
    replayed_chunks = [entry[2].get("chunk_id") for entry in replayed]
    assert "c1" not in replayed_chunks
    assert "c3" in replayed_chunks
    assert "active_tail" in replayed_chunks


@pytest.mark.asyncio
async def test_shard_merger_consolidation():
    index = ShardedVectorIndex(dimension=4, num_shards=4)
    merger = BackgroundShardMerger(index=index, min_threshold=3)

    c1 = DocumentChunk(chunk_id="doc_1", text="Sample 1")
    c2 = DocumentChunk(chunk_id="doc_2", text="Sample 2")

    # Manually populate source shard 1
    index.shards[1].insert(c1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    index.shards[1].insert(c2, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    assert len(index.shards[1].chunks) == 2
    assert len(index.shards[0].chunks) == 0

    merge_res = await merger.merge_sparse_shards(source_id=1, target_id=0)
    assert merge_res["status"] == "success"
    assert merge_res["records_moved"] == 2
    assert len(index.shards[1].chunks) == 0
    assert len(index.shards[0].chunks) == 2
