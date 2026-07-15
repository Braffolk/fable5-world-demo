"""Read one logical immutable object from verified disjoint local byte ranges."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RangePiece:
    start: int
    end: int
    path: Path
    sha256: str
    source: str
    headers_path: Path | None = None
    headers_sha256: str | None = None

    @property
    def bytes(self) -> int:
        return self.end - self.start + 1


class RangeStore:
    def __init__(self, pieces: list[RangePiece]) -> None:
        self.pieces = sorted(pieces, key=lambda piece: (piece.start, piece.end))
        for piece in self.pieces:
            if piece.end < piece.start:
                raise ValueError(f"invalid byte range: {piece.start}-{piece.end}")
            if piece.path.stat().st_size != piece.bytes:
                raise ValueError(f"range size differs from file size: {piece.path}")
        for previous, current in zip(self.pieces, self.pieces[1:]):
            if current.start <= previous.end:
                raise ValueError(
                    f"retained byte ranges overlap: {previous.start}-{previous.end} and "
                    f"{current.start}-{current.end}"
                )

    def covering_pieces(self, start: int, end: int) -> list[RangePiece]:
        if end < start:
            raise ValueError("empty or reversed requested range")
        result: list[RangePiece] = []
        cursor = start
        for piece in self.pieces:
            if piece.end < cursor:
                continue
            if piece.start > cursor:
                break
            result.append(piece)
            cursor = min(end + 1, piece.end + 1)
            if cursor > end:
                return result
        raise ValueError(f"retained byte coverage has a gap at {cursor} within {start}-{end}")

    def read_chunks(self, start: int, end: int, *, chunk_bytes: int = 8 << 20) -> Iterator[bytes]:
        cursor = start
        for piece in self.covering_pieces(start, end):
            piece_start = max(cursor, piece.start)
            piece_end = min(end, piece.end)
            with piece.path.open("rb") as source:
                source.seek(piece_start - piece.start)
                remaining = piece_end - piece_start + 1
                while remaining:
                    block = source.read(min(chunk_bytes, remaining))
                    if not block:
                        raise ValueError(f"unexpected EOF in retained range: {piece.path}")
                    remaining -= len(block)
                    cursor += len(block)
                    yield block
        if cursor != end + 1:
            raise ValueError(f"retained byte coverage ended at {cursor - 1}; expected {end}")
