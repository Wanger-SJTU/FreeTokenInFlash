"""GGUF O_DIRECT reads: unaligned offsets/lengths, EOF-adjacent tails, no page cache."""

from __future__ import annotations

import os

import pytest

from freetoken.models.gguf.reader import GgufDirectReader


@pytest.fixture()
def sample(tmp_path):
    path = tmp_path / "weights.gguf"
    data = os.urandom(1 << 20) + b"tail-data"  # size not a block multiple
    path.write_bytes(data)
    return str(path), data


def _read(path: str, offset: int, nbytes: int) -> bytes:
    reader = GgufDirectReader(path)
    out = bytearray(nbytes)
    reader.read_into(memoryview(out), offset, nbytes)
    reader.close()
    return bytes(out)


@pytest.mark.parametrize("offset,nbytes", [
    (0, 4096),
    (123, 5000),          # unaligned head, unaligned length
    (4096 + 7, 3),        # tiny read inside one block
    (1 << 20, 9),         # EOF-adjacent: the aligned window runs past the file
    (1000, 1 << 19),      # multi-chunk
])
def test_direct_reads_match_the_file(sample, offset, nbytes):
    path, data = sample
    assert _read(path, offset, nbytes) == data[offset : offset + nbytes]


def test_buffered_fallback_when_direct_is_off(sample):
    path, data = sample
    reader = GgufDirectReader(path)
    reader._direct = 0  # force the preadv path
    out = bytearray(3000)
    reader.read_into(memoryview(out), 777, 3000)
    reader.close()
    assert bytes(out) == data[777 : 777 + 3000]
