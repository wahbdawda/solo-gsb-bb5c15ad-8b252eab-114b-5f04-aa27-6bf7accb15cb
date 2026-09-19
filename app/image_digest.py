"""Firmware content fingerprinting over assembled sparse images.

The shared Intel HEX parser assembles an image into contiguous,
address-ordered segments; this module encodes that canonical form into
an unambiguous binary stream and hashes it with SHA-256:

* the stream starts with the versioned tag ``FWDIG1`` so the framing
  is domain-separated from raw image bytes and can evolve;
* every segment contributes one frame, in ascending address order:
  ``b"S" + start(8-byte big-endian) + length(8-byte big-endian) + data``;
* when ``include_entry`` is true and the image defines an entry
  address, a final frame ``b"E" + entry(8-byte big-endian)`` commits
  the whole-image hash to that entry as well.

Because only the assembled segments are encoded — never the record
layout — HEX texts that describe the same sparse image with different
record fragmentation, base-address record usage, write order or gap
placement produce byte-identical streams and therefore the same
fingerprint. The per-segment SHA-256 (over each segment's own frame)
and a normalized text manifest mirroring the hashed frames are
returned alongside the whole-image digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .hex_parser import Segment

MAGIC = b"FWDIG1\x00"
"""Versioned stream tag prepended to every canonical byte stream."""

MANIFEST_HEADER = "FWDIG1"
"""First manifest line; matches :data:`MAGIC` without the NUL terminator."""

_SEGMENT_TAG = b"S"
_ENTRY_TAG = b"E"


@dataclass
class SegmentDigest:
    """SHA-256 of one segment's canonical frame, with its address span."""

    start: int
    length: int
    sha256: str

    @property
    def end(self) -> int:
        """Exclusive end address."""
        return self.start + self.length


@dataclass
class DigestResult:
    """Whole-image fingerprint plus the normalized manifest."""

    image_sha256: str
    segments: list[SegmentDigest] = field(default_factory=list)
    total_bytes: int = 0
    entry_address: int | None = None
    include_entry: bool = True
    manifest: str = ""


def _segment_frame(segment: Segment) -> bytes:
    """Encode one segment as tag + start + length + raw bytes."""
    return (
        _SEGMENT_TAG
        + segment.start.to_bytes(8, "big")
        + len(segment.data).to_bytes(8, "big")
        + segment.data
    )


def _entry_frame(entry_address: int) -> bytes:
    """Encode the entry address as its own tagged frame."""
    return _ENTRY_TAG + entry_address.to_bytes(8, "big")


def _fmt(addr: int) -> str:
    return f"0x{addr:08X}"


def compute_image_digest(
    segments: list[Segment],
    entry_address: int | None,
    include_entry: bool = True,
) -> DigestResult:
    """Fingerprint an assembled image.

    ``segments`` must be the parser's contiguous, address-ordered
    output. The entry address is folded into the whole-image hash only
    when ``include_entry`` is true and an entry exists; it is always
    reported back verbatim. The manifest lists exactly the frames that
    were hashed, in hash order: one ``segment`` line per segment
    (ascending address), then an ``entry`` line when applicable.
    """
    hasher = hashlib.sha256()
    hasher.update(MAGIC)

    digests: list[SegmentDigest] = []
    manifest_lines = [MANIFEST_HEADER]
    for segment in segments:
        frame = _segment_frame(segment)
        hasher.update(frame)
        frame_sha256 = hashlib.sha256(frame).hexdigest()
        digests.append(
            SegmentDigest(
                start=segment.start,
                length=len(segment.data),
                sha256=frame_sha256,
            )
        )
        manifest_lines.append(
            f"segment {_fmt(segment.start)} {len(segment.data)} {frame_sha256}"
        )

    if include_entry and entry_address is not None:
        hasher.update(_entry_frame(entry_address))
        manifest_lines.append(f"entry {_fmt(entry_address)}")

    return DigestResult(
        image_sha256=hasher.hexdigest(),
        segments=digests,
        total_bytes=sum(d.length for d in digests),
        entry_address=entry_address,
        include_entry=include_entry,
        manifest="\n".join(manifest_lines),
    )
