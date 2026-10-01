"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import queue
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    ``PINNED`` (cudaHostRegister'd) is required for anything the GPU dereferences:
    the decode gather kernels and the DMA prefill copies. ``LOCKED`` (VirtualLock /
    mlock: resident for the CPU executor, but with no device address) and
    ``PAGEABLE`` are reserved for platforms where the pin quota cannot cover every
    layer (Windows/WDDM); their movement paths are not implemented here -- layers
    with either class must be served by the CPU executor.

    ``PAGEABLE`` layers produced by a --cpu-mem budget are FILE-BACKED (spill
    tmpfile + shared mmap): touching a dropped page faults it back from disk, so
    the kernel's page cache pages them in and out on demand. The CPU executor is
    their only reader.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"


# HostBank backing stores. "anon" is the classic lazy anonymous mmap (pin-able,
# contents undefined after release). "file" is a shared mmap over an unlinked
# spill tmpfile: MADV_DONTNEED'd pages re-fault from the file, so the OS page
# cache -- not the pin quota -- governs how much RAM the bank's layer holds.
ANON_BACKING = "anon"
FILE_BACKING = "file"


_DEFAULT_CHUNK = 8 << 20

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []


class HostBank:
    """A lazily-allocated, page-aligned host buffer + its torch view, registered on demand.

    Allocate -> fill (write real data, or O_DIRECT read into it) -> ``pin()``. The mmap is
    rounded up to the O_DIRECT block so chunked reads are always aligned; ``tensor`` views
    exactly ``nbytes``.

    ``backing="file"`` allocates over an unlinked spill tmpfile in ``spill_dir`` instead
    of anonymous memory (the --cpu-mem budget's over-budget layers): the buffer is never
    pinned; after filling, ``drop_pages()`` returns the pages to the OS and any later
    access faults them back from the file, so resident RAM tracks the page cache rather
    than the bank's size."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "backing")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        backing: str = ANON_BACKING,
        spill_dir: str | None = None,
    ):
        assert backing in (ANON_BACKING, FILE_BACKING), backing
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self.backing = backing
        if backing == ANON_BACKING:
            self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
        else:
            self._buf = _mmap_spill_tmpfile(asize, spill_dir)
        _LIVE_BUFFERS.append(self._buf)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._pinned = False

    @property
    def residency(self) -> HostResidency:
        return HostResidency.PINNED if self._pinned else HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled, resident) buffer -- pin-after-fill."""
        assert self.backing == ANON_BACKING, (
            "file-backed (spilled) banks must never be registered: the GPU movement "
            "paths cannot dereference them and pinning would defeat the RAM budget"
        )
        if self._pinned:
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def drop_pages(self) -> None:
        """Return the buffer's resident pages to the OS. FILE-BACKED banks only: the
        (already written) pages live in the page cache / spill file, so later access
        re-faults the same data. This is the after-fill step that replaces pin() for
        over-budget layers."""
        assert self.backing == FILE_BACKING, "drop_pages is only meaningful for file-backed banks"
        self._buf.madvise(mmap.MADV_DONTNEED)

    def release(self) -> None:
        """Drop the buffer's resident pages (address space stays valid; contents
        become undefined). Only for buffers that are done being read -- the
        converter releases each layer after writing it out."""
        assert not self._pinned, "cannot release a pinned bank"
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """Make the buffer resident without a device mapping (VirtualLock/mlock).

        Platform-specific and not implemented here. A locked bank layer is
        CPU-executor-readable but must never be handed to the GPU movement
        paths.
        """
        raise NotImplementedError("HostBank.lock() is platform-specific and not implemented")


def _mmap_spill_tmpfile(asize: int, spill_dir: str | None) -> mmap.mmap:
    """Shared mmap over an unlinked spill tmpfile: pages the kernel drops are re-faulted
    from disk, so the bank caps at the page cache instead of pinned/committed anon RAM.
    Unlinking immediately hands the space back at process exit; the mapping itself keeps
    the storage alive (POSIX semantics)."""
    import platform

    if platform.system() != "Linux":
        raise NotImplementedError(
            "file-backed host banks (--cpu-mem spill) are Linux-only (MADV_DONTNEED "
            "on a shared mapping)"
        )
    fd, path = tempfile.mkstemp(prefix="freetoken_expert_", suffix=".spill", dir=spill_dir)
    try:
        os.ftruncate(fd, asize)
        buf = mmap.mmap(fd, asize, flags=mmap.MAP_SHARED)
    finally:
        os.close(fd)
        os.unlink(path)  # storage survives until the mapping closes (process lifetime)
    return buf


def alloc_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    backing: str = ANON_BACKING,
    spill_dir: str | None = None,
) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype, backing=backing, spill_dir=spill_dir) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    num_layers: int,
    backing: list[str] | str = ANON_BACKING,
    spill_dir: str | None = None,
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name.

    ``backing`` is either a constant or a ``num_layers``-long list of per-layer values:
    ``"pinned"`` layers get pin-able anonymous banks, anything else (the budget plan's
    ``"pageable"``) gets file-backed spill banks whose pages the kernel pages on demand."""
    per_layer = [backing] * num_layers if isinstance(backing, str) else list(backing)
    assert len(per_layer) == num_layers, (len(per_layer), num_layers)
    return {
        name: [
            HostBank(
                shape,
                dtype,
                backing=ANON_BACKING if b == "pinned" else FILE_BACKING,
                spill_dir=spill_dir,
            )
            for b in per_layer
        ]
        for name, (shape, dtype) in specs.items()
    }


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Pin (register) every anonymous bank after it has been filled -- pin-after-fill.
    File-backed (spilled) banks are dropped instead: their after-fill step returns the
    pages to the page cache."""
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_bank in bank:
                if layer_bank.backing == FILE_BACKING:
                    layer_bank.drop_pages()
                else:
                    layer_bank.pin()
        elif bank.backing == FILE_BACKING:
            bank.drop_pages()
        else:
            bank.pin()


class PinPipeline:
    """Pin filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a
    queue of completed banks; submitters never block. Total load time becomes
    ~max(read, pin) instead of their sum. Use as a context manager: a clean exit
    drains the queue and re-raises the first pin failure; an exceptional exit
    still joins the thread but lets the original exception propagate.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            bank = self._q.get()
            if bank is None:
                return
            if self._exc is not None:
                continue  # drain without pinning after a failure
            try:
                if bank.backing == FILE_BACKING:
                    bank.drop_pages()  # spilled layer: after-fill step is an unpin
                else:
                    bank.pin()
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank) -> None:
        self._q.put(bank)

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer."""
        for bank in banks.values():
            self.submit(bank)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "pin_banks",
    "read_file_into",
]
