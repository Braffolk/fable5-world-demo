import struct
import zlib

import numpy as np
import pytest

from assetgen.config import load_base
from assetgen.cook.chunkio import (
    FMT,
    MAGIC,
    ChunkMeta,
    read_chunk,
    read_chunk_any,
    read_chunk_v2,
    write_chunk,
    write_chunk_v2,
)
from assetgen.cook.encode import (
    decode_quant16,
    decode_records,
    decode_u8_planes,
    delta2d,
    encode_quant16,
    encode_quant16_checked,
    encode_quantized16,
    encode_records,
    encode_u8_planes,
    undelta2d,
    quantize16_checked,
)

CFG = load_base().encode
RNG = np.random.default_rng(7)


def synth_terrain(res=257):
    y, x = np.mgrid[0:res, 0:res].astype(np.float64)
    return (
        40 * np.sin(x / 37) * np.cos(y / 53)
        + 8 * np.sin(x / 5.1 + y / 7.3)
        + RNG.normal(0, 0.05, (res, res))
        + 90.0
    )


def test_delta2d_round_trip():
    q = RNG.integers(0, 65536, (129, 129), dtype=np.uint16)
    assert np.array_equal(undelta2d(delta2d(q)), q)


def test_quant16_round_trip_within_half_step():
    arr = synth_terrain()
    payload, qoffset = encode_quant16(CFG, arr, 0.01)
    out = decode_quant16(CFG, payload, arr.shape[0], qoffset, 0.01)
    # half a quantization step + float32 dequantization rounding
    assert np.max(np.abs(out - arr)) <= 0.0052


def test_quant16_compresses_smooth_terrain():
    arr = synth_terrain(513)
    payload, _ = encode_quant16(CFG, arr, 0.01)
    # synthetic terrain carries deliberately white 5cm noise; real LiDAR DTM does much better
    assert len(payload) < arr.size * 1.5, "delta filter should keep smooth terrain well under 1.5 B/texel"


def test_quant16_handles_nan():
    arr = synth_terrain()
    arr[:10, :10] = np.nan
    payload, qoffset = encode_quant16(CFG, arr, 0.01)
    out = decode_quant16(CFG, payload, arr.shape[0], qoffset, 0.01)
    assert np.isfinite(out).all()


def test_u8_planes_round_trip():
    a = RNG.integers(0, 12, (256, 256)).astype(np.uint8)
    b = RNG.integers(0, 255, (256, 256)).astype(np.uint8)
    payload = encode_u8_planes(CFG, [a, b])
    a2, b2 = decode_u8_planes(CFG, payload, 256, 2)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_records_round_trip():
    cols = [
        RNG.integers(0, 65536, 1000).astype(np.uint16),
        RNG.integers(0, 65536, 1000).astype(np.uint16),
        RNG.integers(0, 6, 1000).astype(np.uint8),
    ]
    payload = encode_records(CFG, cols)
    out = decode_records(CFG, payload, 1000, ["u2", "u2", "u1"])
    for c, o in zip(cols, out):
        assert np.array_equal(c, o)


def test_chunkio_round_trip(tmp_path):
    arr = synth_terrain()
    payload, qoffset = encode_quant16(CFG, arr, 0.01)
    meta = ChunkMeta(
        layer="height", lod=0, enc=1, cx=152, cz=94, res=arr.shape[0], count=0,
        origin_e=680000.0, origin_n=6443000.0, qoffset=qoffset, qscale=0.01,
    )
    p = tmp_path / "t.bin"
    write_chunk(p, meta, payload)
    meta2, payload2 = read_chunk(p)
    assert payload2 == payload
    assert (meta2.layer, meta2.lod, meta2.enc, meta2.cx, meta2.cz, meta2.res, meta2.count) == (
        meta.layer, meta.lod, meta.enc, meta.cx, meta.cz, meta.res, meta.count)
    # qoffset/qscale/origins survive the header's float32 fields within f32 precision
    assert meta2.qoffset == pytest.approx(meta.qoffset, abs=1e-3)
    assert meta2.qscale == pytest.approx(meta.qscale, rel=1e-6)
    assert meta2.origin_e == meta.origin_e and meta2.origin_n == meta.origin_n


def test_chunkio_detects_corruption(tmp_path):
    payload, qoffset = encode_quant16(CFG, synth_terrain(65), 0.01)
    meta = ChunkMeta("height", 0, 1, 0, 0, 65, 0, 0.0, 0.0, qoffset, 0.01)
    p = tmp_path / "t.bin"
    write_chunk(p, meta, payload)
    blob = bytearray(p.read_bytes())
    blob[70] ^= 0xFF
    p.write_bytes(blob)
    with pytest.raises(ValueError):
        read_chunk(p)


def test_lac1_wire_bytes_remain_unchanged(tmp_path):
    payload = b"lac1-golden-payload"
    meta = ChunkMeta("height", 0, 1, -2, 3, 17, 0, 364544.0, 6629376.0, -1.0, 0.01)
    path = tmp_path / "golden.lac"
    write_chunk(path, meta, payload)
    expected_header = struct.pack(
        FMT, MAGIC, 0, 0, 1, 0, -2, 3, 17, 0,
        364544.0, 6629376.0, -1.0, 0.01, len(payload), zlib.crc32(payload),
    )
    assert path.read_bytes() == expected_header + payload


def test_lac2_signed_lod_round_trip(tmp_path):
    payload = b"lac2-payload"
    meta = ChunkMeta("height", -2, 1, -7, 11, 2049, 0, 367744.0, 6634112.0, 72.0, 0.002)
    path = tmp_path / "fine.lac"
    write_chunk_v2(path, meta, payload)

    decoded, decoded_payload = read_chunk_v2(path)
    version, any_meta, any_payload = read_chunk_any(path)
    assert version == 2
    assert decoded_payload == any_payload == payload
    assert (decoded.layer, decoded.lod, decoded.cx, decoded.cz) == ("height", -2, -7, 11)
    assert any_meta == decoded
    with pytest.raises(ValueError, match="bad magic"):
        read_chunk(path)


def test_lac2_detects_crc_and_trailing_bytes(tmp_path):
    path = tmp_path / "fine.lac"
    write_chunk_v2(path, ChunkMeta("height", -1, 1, 0, 0, 17, 0, 0, 0, 0, 0.005), b"payload")
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="length/crc"):
        read_chunk_v2(path)


def test_checked_quant16_uses_wire_float32_qscale():
    arr = np.linspace(10.0, 70.0, 257 * 257, dtype=np.float64).reshape(257, 257)
    payload, qoffset, effective = encode_quant16_checked(CFG, arr, 0.001)
    assert effective == float(np.float32(0.001))
    assert effective != 0.001
    decoded = decode_quant16(CFG, payload, arr.shape[0], qoffset, effective)
    assert np.max(np.abs(decoded - arr)) <= effective * 0.5 + 1e-5


def test_checked_quantization_split_is_byte_identical():
    arr = synth_terrain(65)
    payload, qoffset, qscale = encode_quant16_checked(CFG, arr, 0.002)
    q, qoffset2, qscale2 = quantize16_checked(arr, 0.002)
    assert (qoffset2, qscale2) == (qoffset, qscale)
    assert encode_quantized16(CFG, q) == payload


def test_checked_quantization_shared_qoffset_preserves_shared_sample():
    a = np.array([[10.0, 10.126], [10.0, 10.126]], dtype=np.float64)
    b = np.array([[10.126, 10.25], [10.126, 10.25]], dtype=np.float64)
    pa, oa, qa = encode_quant16_checked(CFG, a, 0.002, qoffset=8.0)
    pb, ob, qb = encode_quant16_checked(CFG, b, 0.002, qoffset=8.0)
    da = decode_quant16(CFG, pa, 2, oa, qa)
    db = decode_quant16(CFG, pb, 2, ob, qb)
    assert da[0, 1] == db[0, 0]


def test_checked_quant16_rejects_overflow_and_nonfinite():
    with pytest.raises(OverflowError):
        encode_quant16_checked(CFG, np.array([[0.0, 70.0]]), 0.001)
    with pytest.raises(ValueError, match="finite"):
        encode_quant16_checked(CFG, np.array([[0.0, np.nan]]), 0.002)
