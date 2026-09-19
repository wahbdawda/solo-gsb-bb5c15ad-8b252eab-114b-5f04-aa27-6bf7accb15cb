"""Tests for the /api/v1/image-digest endpoint and fingerprint logic."""

import hashlib

from fastapi.testclient import TestClient

from app.hex_parser import parse_intel_hex
from app.image_digest import MAGIC, compute_image_digest
from app.main import app

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def linear_base(value: int) -> str:
    return rec(0x04, 0, value.to_bytes(2, "big"))


def segment_base(value: int) -> str:
    return rec(0x02, 0, value.to_bytes(2, "big"))


def linear_entry(value: int) -> str:
    return rec(0x05, 0, value.to_bytes(4, "big"))


def digest_of(text: str, include_entry: bool = True):
    parsed = parse_intel_hex(text)
    assert parsed.ok, "test fixture must parse"
    return compute_image_digest(parsed.segments, parsed.entry_address, include_entry)


def frame(start: int, data: bytes) -> bytes:
    return b"S" + start.to_bytes(8, "big") + len(data).to_bytes(8, "big") + data


# ---------------------------------------------------------------------------
# pure digest logic
# ---------------------------------------------------------------------------

def test_record_fragmentation_does_not_change_fingerprint():
    whole = "\n".join([data_rec(0x0000, bytes(range(16))), EOF])
    fragmented = "\n".join(
        [data_rec(0x0000 + i, bytes(range(16))[i : i + 4]) for i in range(0, 16, 4)]
        + [EOF]
    )
    assert digest_of(whole).image_sha256 == digest_of(fragmented).image_sha256
    assert digest_of(whole).manifest == digest_of(fragmented).manifest


def test_record_order_and_base_record_style_do_not_change_fingerprint():
    # Same sparse image: 0x10000..0x10003 and 0x10010..0x10011, reached via
    # an 04 linear-base record vs an 02 segment-base record, records shuffled.
    via_linear = "\n".join(
        [linear_base(0x0001), data_rec(0x0000, b"\x01\x02\x03\x04"),
         data_rec(0x0010, b"\xAA\xBB"), EOF]
    )
    via_segment = "\n".join(
        [segment_base(0x1000), data_rec(0x0010, b"\xAA\xBB"),
         data_rec(0x0000, b"\x01\x02\x03\x04"), EOF]
    )
    result_a = digest_of(via_linear)
    result_b = digest_of(via_segment)
    assert result_a.image_sha256 == result_b.image_sha256
    assert [(s.start, s.length) for s in result_a.segments] == [(0x10000, 4), (0x10010, 2)]
    assert result_a.total_bytes == 6


def test_gap_placement_does_not_change_fingerprint():
    # Identical sparse bytes; one text writes the two runs left-to-right,
    # the other right-to-left with an extra identical rewrite in between.
    forward = "\n".join(
        [data_rec(0x0000, b"\x01\x02"), data_rec(0x0010, b"\x03"), EOF]
    )
    shuffled = "\n".join(
        [data_rec(0x0010, b"\x03"), data_rec(0x0000, b"\x01\x02"),
         data_rec(0x0001, b"\x02"), EOF]
    )
    assert digest_of(forward).image_sha256 == digest_of(shuffled).image_sha256


def test_different_bytes_or_addresses_change_fingerprint():
    base = digest_of("\n".join([data_rec(0x0000, b"\x01\x02"), EOF]))
    other_data = digest_of("\n".join([data_rec(0x0000, b"\x01\x03"), EOF]))
    other_addr = digest_of("\n".join([data_rec(0x0001, b"\x01\x02"), EOF]))
    assert base.image_sha256 != other_data.image_sha256
    assert base.image_sha256 != other_addr.image_sha256


def test_known_answer_framing_with_entry():
    text = "\n".join(
        [data_rec(0x0000, b"\x01\x02"), linear_entry(0x08000101), EOF]
    )
    result = digest_of(text)
    expected = hashlib.sha256(
        MAGIC + frame(0, b"\x01\x02") + b"E" + (0x08000101).to_bytes(8, "big")
    ).hexdigest()
    assert result.image_sha256 == expected
    assert result.segments[0].sha256 == hashlib.sha256(frame(0, b"\x01\x02")).hexdigest()
    assert result.entry_address == 0x08000101
    assert result.manifest == (
        "FWDIG1\n"
        f"segment 0x00000000 2 {result.segments[0].sha256}\n"
        "entry 0x08000101"
    )


def test_include_entry_false_excludes_entry_from_hash():
    text = "\n".join(
        [data_rec(0x0000, b"\x01\x02"), linear_entry(0x08000101), EOF]
    )
    with_entry = digest_of(text, include_entry=True)
    without_entry = digest_of(text, include_entry=False)
    assert with_entry.image_sha256 != without_entry.image_sha256
    # excluding the entry matches the digest of the same image with no
    # entry record at all
    no_entry_record = digest_of("\n".join([data_rec(0x0000, b"\x01\x02"), EOF]))
    assert without_entry.image_sha256 == no_entry_record.image_sha256
    # the entry is still reported and the manifest carries no entry line
    assert without_entry.entry_address == 0x08000101
    assert "entry" not in without_entry.manifest


def test_include_entry_flag_is_irrelevant_without_entry():
    text = "\n".join([data_rec(0x0000, b"\x01\x02"), EOF])
    assert (
        digest_of(text, include_entry=True).image_sha256
        == digest_of(text, include_entry=False).image_sha256
    )


def test_empty_image_fingerprint_is_deterministic():
    result = digest_of(EOF)
    assert result.image_sha256 == hashlib.sha256(MAGIC).hexdigest()
    assert result.total_bytes == 0
    assert result.segments == []
    assert result.manifest == "FWDIG1"


def test_segments_hashed_in_ascending_address_order():
    text = "\n".join(
        [data_rec(0x0010, b"\x03"), data_rec(0x0000, b"\x01\x02"), EOF]
    )
    result = digest_of(text)
    expected = hashlib.sha256(
        MAGIC + frame(0x0000, b"\x01\x02") + frame(0x0010, b"\x03")
    ).hexdigest()
    assert result.image_sha256 == expected
    assert [s.start for s in result.segments] == [0x0000, 0x0010]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_digest_success_matches_manual_hash():
    hex_text = "\n".join(
        [linear_base(0x0800), data_rec(0x0000, bytes(range(16))),
         linear_entry(0x08000101), EOF]
    )
    resp = client.post(
        "/api/v1/image-digest", json={"hex_text": hex_text, "include_entry": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["hex_ok"] is True
    assert body["include_entry"] is True
    assert body["entry_address"] == 0x08000101
    assert body["total_bytes"] == 16
    assert body["segment_count"] == 1
    expected = hashlib.sha256(
        MAGIC
        + frame(0x08000000, bytes(range(16)))
        + b"E" + (0x08000101).to_bytes(8, "big")
    ).hexdigest()
    assert body["image_sha256"] == expected
    segment = body["segments"][0]
    assert (segment["start"], segment["end"], segment["size"]) == (
        0x08000000,
        0x08000010,
        16,
    )
    assert segment["sha256"] == hashlib.sha256(
        frame(0x08000000, bytes(range(16)))
    ).hexdigest()
    assert body["manifest"].splitlines() == [
        "FWDIG1",
        f"segment 0x08000000 16 {segment['sha256']}",
        "entry 0x08000101",
    ]
    assert body["error_count"] == 0
    assert all(d["status"] == "ok" for d in body["diagnostics"])


def test_api_include_entry_defaults_to_true():
    hex_text = "\n".join(
        [data_rec(0x0000, b"\x01\x02"), linear_entry(0x08000101), EOF]
    )
    default = client.post("/api/v1/image-digest", json={"hex_text": hex_text}).json()
    explicit = client.post(
        "/api/v1/image-digest", json={"hex_text": hex_text, "include_entry": True}
    ).json()
    excluded = client.post(
        "/api/v1/image-digest", json={"hex_text": hex_text, "include_entry": False}
    ).json()
    assert default["include_entry"] is True
    assert default["image_sha256"] == explicit["image_sha256"]
    assert excluded["image_sha256"] != default["image_sha256"]
    assert excluded["entry_address"] == 0x08000101
    assert excluded["manifest"].splitlines() == [
        "FWDIG1",
        f"segment 0x00000000 2 {excluded['segments'][0]['sha256']}",
    ]


def test_api_fragmented_and_gap_variants_share_fingerprint():
    whole = "\n".join(
        [data_rec(0x0000, b"\x01\x02"), data_rec(0x0010, b"\x03\x04"), EOF]
    )
    fragmented = "\n".join(
        [data_rec(0x0011, b"\x04"), data_rec(0x0010, b"\x03"),
         data_rec(0x0001, b"\x02"), data_rec(0x0000, b"\x01"), EOF]
    )
    body_a = client.post("/api/v1/image-digest", json={"hex_text": whole}).json()
    body_b = client.post("/api/v1/image-digest", json={"hex_text": fragmented}).json()
    assert body_a["image_sha256"] == body_b["image_sha256"]
    assert body_a["manifest"] == body_b["manifest"]
    assert body_a["total_bytes"] == body_b["total_bytes"] == 4


def test_api_invalid_hex_returns_only_diagnostics():
    bad = data_rec(0, b"\x01")[:-2] + "00"  # corrupt checksum, also missing EOF
    resp = client.post("/api/v1/image-digest", json={"hex_text": bad})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["image_sha256"] is None
    assert body["segments"] == []
    assert body["segment_count"] == 0
    assert body["manifest"] is None
    assert body["error_count"] >= 1
    messages = " ".join(d["message"] for d in body["diagnostics"])
    assert "checksum" in messages
    assert "end-of-file" in messages


def test_api_empty_image_is_hashed():
    body = client.post("/api/v1/image-digest", json={"hex_text": EOF}).json()
    assert body["ok"] is True
    assert body["total_bytes"] == 0
    assert body["segments"] == []
    assert body["entry_address"] is None
    assert body["image_sha256"] == hashlib.sha256(MAGIC).hexdigest()
    assert body["manifest"] == "FWDIG1"
