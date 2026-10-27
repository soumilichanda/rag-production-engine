import pytest
from app/retrieval/context_compressor import CrossEncoderContextCompressor


def test_compression_trims_length():
    compressor = CrossEncoderContextCompressor(max_tokens_per_chunk=10)
    query = "database latency"
    text = "Database latency was reduced by half. We also bought coffee for the kitchen. Server metrics improved significantly."
    result = compressor.compress_chunk(query, text)

    assert "Database latency" in result
    assert len(result.split()) <= 15


def test_compress_candidates_preserves_metadata():
    compressor = CrossEncoderContextCompressor(max_tokens_per_chunk=20)
    candidates = [{"chunk_id": "c1", "text": "High throughput async workers. Extra filler text here."}]
    res = compressor.compress_candidates("async workers", candidates)

    assert res[0]["chunk_id"] == "c1"
    assert "compressed_text" in res[0]
