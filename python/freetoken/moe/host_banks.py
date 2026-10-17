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

    ``SPLIT`` layers (the --cpu-mem expert budget) are one tensor straddling both
    backings: experts ``[0, pinned_experts)`` live in a registered anonymous
    prefix (GPU-gatherable), the rest in a file-backed suffix (CPU executor only,
    page-cache-governed, exactly like PAGEABLE).
    """

    PINNED = "pinned"
    SPLIT = "split"
    LOCKED = "locked"
    PAGEABLE = "pageable"


# HostBank backing stores. "anon" is the classic lazy anonymous mmap (pin-able,
# contents undefined after release). "file" is a shared mmap over an unlinked
# spill tmpfile: MADV_DONTNEED'd pages re-fault from the file, so the OS page
# cache -- not the pin quota -- governs how much RAM the bank's layer holds.
# "split" concatenates the two at consecutive addresses (MAP_FIXED) so the bank
# stays ONE contiguous [E, ...] tensor: an anonymous, pin-able prefix covering
# the first ``split_experts`` experts, then the file-backed suffix.
ANON_BACKING = "anon"
FILE_BACKING = "file"
SPLIT_BACKING = "split"


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
    than the bank's size.

    ``backing="split"`` (the --cpu-mem expert budget) keeps the tensor contiguous but
    straddles two mappings: experts ``[0, split_experts)`` of the leading dim are an
    anonymous, pin-able prefix; the remainder is file-backed exactly like ``"file"``.
    ``pin()`` registers ONLY the anonymous prefix (``anon_bytes``; page-aligned), and
    ``drop_pages()`` returns only the file suffix to the page cache. Callers must treat
    experts ``>= split_experts`` as never GPU-fetchable."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "backing",
                 "split_experts", "anon_bytes")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        backing: str = ANON_BACKING,
        spill_dir: str | None = None,
        split_experts: int | None = None,
    ):
        assert backing in (ANON_BACKING, FILE_BACKING, SPLIT_BACKING), backing
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self.backing = backing
        self.split_experts = None
        self.anon_bytes = asize
        if backing == ANON_BACKING:
            self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
        elif backing == FILE_BACKING:
            self._buf = _mmap_spill_tmpfile(asize, spill_dir)
        else:
            num_experts = shape[0]
            assert split_experts is not None and 0 < split_experts < num_experts, (
                f"split backing needs 0 < split_experts < {num_experts}, got {split_experts}"
            )
            slab = self.nbytes // num_experts  # bytes per leading-dim row (one expert)
            self.split_experts = split_experts
            self.anon_bytes = min(((split_experts * slab + _BLK - 1) // _BLK) * _BLK, asize)
            self._buf = _mmap_split(asize, self.anon_bytes, spill_dir)
        _LIVE_BUFFERS.append(self._buf)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._pinned = False

    @property
    def residency(self) -> HostResidency:
        if self.backing == SPLIT_BACKING:
            return HostResidency.SPLIT
        return HostResidency.PINNED if self._pinned else HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled, resident) buffer -- pin-after-fill.

        SPLIT banks register only the anonymous prefix; the file-backed suffix has no
        device alias and must never be dereferenced by GPU movement paths."""
        assert self.backing in (ANON_BACKING, SPLIT_BACKING), (
            "file-backed (spilled) banks must never be registered: the GPU movement "
            "paths cannot dereference them and pinning would defeat the RAM budget"
        )
        if self._pinned:
            return
        from freetoken.kernel.pinned import host_register

        pin_bytes = self.anon_bytes if self.backing == SPLIT_BACKING else len(self._buf)
        try:
            host_register(self.addr, pin_bytes)
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {pin_bytes / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def drop_pages(self) -> None:
        """Return the buffer's resident pages to the OS. FILE-BACKED banks only: the
        (already written) pages live in the page cache / spill file, so later access
        re-faults the same data. This is the after-fill step that replaces pin() for
        over-budget layers. For SPLIT banks it drops only the file-backed suffix --
        the registered anonymous prefix stays pinned/resident."""
        if self.backing == FILE_BACKING:
            self._buf.madvise(mmap.MADV_DONTNEED)
            return
        assert self.backing == SPLIT_BACKING, (
            "drop_pages is only meaningful for file-backed (or split) banks"
        )
        _madvise(self.addr + self.anon_bytes, len(self._buf) - self.anon_bytes, mmap.MADV_DONTNEED)

    def release(self) -> None:
        """Drop the buffer's resident pages (address space stays valid; contents
        become undefined). Only for buffers that are done being read -- the
        converter releases each layer after writing it out."""
        assert not self._pinned, "cannot release a pinned bank"
        if isinstance(self._buf, mmap.mmap):
            self._buf.madvise(mmap.MADV_DONTNEED)
        else:  # ctypes char-array view (split backing)
            _madvise(self.addr, len(self._buf), mmap.MADV_DONTNEED)

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


def _ctypes_mmap():
    """libc mmap(2) with explicit argtypes (Python's mmap module cannot target a fixed
    address, which MAP_FIXED needs)."""
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.mmap
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    fn.restype = ctypes.c_void_p
    return fn


def _madvise(addr: int, length: int, advice: int) -> None:
    """madvise(2) on a raw address range (the split bank's file-backed suffix)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), advice) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"madvise({hex(addr)}, {length}): {os.strerror(err)}")


_MAP_FAILED = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1


def _mmap_split(asize: int, anon_bytes: int, spill_dir: str | None):
    """One contiguous buffer straddling two backings (Linux): reserve ``asize`` PROT_NONE,
    then MAP_FIXED an anonymous (pin-able) prefix of ``anon_bytes`` and a shared spill-
    tmpfile suffix over the rest. Returns a ctypes char-array view of the whole range;
    the mappings live for the process lifetime (never munmap'd, like every HostBank)."""
    import platform

    if platform.system() != "Linux":
        raise NotImplementedError(
            "split host banks (--cpu-mem expert mode) are Linux-only (MAP_FIXED)"
        )
    prot_none = 0
    prot_rw = mmap.PROT_READ | mmap.PROT_WRITE
    map_shared = mmap.MAP_SHARED
    map_fixed = getattr(mmap, "MAP_FIXED", 0x10)
    map_anon = getattr(mmap, "MAP_ANONYMOUS", getattr(mmap, "MAP_ANON", 0x20))
    fn = _ctypes_mmap()

    def _call(addr, length, prot, flags, fd=-1, offset=0):
        res = fn(ctypes.c_void_p(addr) if addr is not None else None, length, prot, flags, fd, offset)
        if not res or res == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap(len={length}, flags={flags:#x}): {os.strerror(err)}")
        return res

    base = _call(None, asize, prot_none, mmap.MAP_PRIVATE | map_anon)
    suffix_len = asize - anon_bytes
    try:
        _call(base, anon_bytes, prot_rw, map_shared | map_anon | map_fixed)
        if suffix_len > 0:  # page rounding can grow the prefix to the whole bank
            fd, path = tempfile.mkstemp(prefix="freetoken_expert_", suffix=".spill", dir=spill_dir)
            try:
                os.ftruncate(fd, suffix_len)
                _call(base + anon_bytes, suffix_len, prot_rw, map_shared | map_fixed, fd, 0)
            finally:
                os.close(fd)
                os.unlink(path)
    except OSError:
        _madvise(base, asize, mmap.MADV_DONTNEED)  # best-effort release of the reservation
        raise
    return (ctypes.c_char * asize).from_address(base)


def alloc_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    backing: str = ANON_BACKING,
    spill_dir: str | None = None,
) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype, backing=backing, spill_dir=spill_dir) for name, (shape, dtype) in specs.items()}


def parse_layer_backing(value: str) -> tuple[str, int | None]:
    """Parse one per-layer backing-plan value: ``"pinned"`` / ``"pageable"`` / ``"split"``
    or ``"split:<k>"`` (the --cpu-mem-mode expert plan; ``k`` = experts in the layer's
    anonymous prefix). Raw backing constants (anon/file/split) pass through unchanged."""
    if value in (ANON_BACKING, FILE_BACKING, SPLIT_BACKING):
        return value, None
    if value.startswith("split"):
        if ":" in value:
            return SPLIT_BACKING, int(value.split(":", 1)[1])
        return SPLIT_BACKING, None
    if value == "pinned":
        return ANON_BACKING, None
    return FILE_BACKING, None


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    num_layers: int,
    backing: list[str] | str = ANON_BACKING,
    spill_dir: str | None = None,
    split_experts: int | None = None,
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name.

    ``backing`` is either a constant or a ``num_layers``-long list of per-layer plan
    values: ``"pinned"`` layers get pin-able anonymous banks, ``"pageable"`` (the layer
    budget's spill) gets file-backed banks whose pages the kernel pages on demand, and
    ``"split"`` / ``"split:<k>"`` (the expert budget) gets the straddling layout with the
    first ``k`` leading-dim rows (experts) in the pin-able anonymous prefix. A bare
    ``"split"`` entry needs ``split_experts`` (one count for every split layer)."""
    per_layer = [backing] * num_layers if isinstance(backing, str) else list(backing)
    assert len(per_layer) == num_layers, (len(per_layer), num_layers)
    banks: dict[str, list[HostBank]] = {}
    for name, (shape, dtype) in specs.items():
        row = []
        for b in per_layer:
            real_backing, k = parse_layer_backing(b) if isinstance(b, str) else (b, None)
            if real_backing == SPLIT_BACKING and k is None:
                k = split_experts
            row.append(
                HostBank(shape, dtype, backing=real_backing, spill_dir=spill_dir, split_experts=k)
            )
        banks[name] = row
    return banks


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Pin (register) every anonymous bank after it has been filled -- pin-after-fill.
    File-backed (spilled) banks are dropped instead: their after-fill step returns the
    pages to the page cache. Split banks do both: register the anonymous prefix, then
    drop the file-backed suffix's pages."""
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_bank in bank:
                _finish_bank(layer_bank)
        else:
            _finish_bank(bank)


# Backpressure for spill fills: writing file-backed banks dirties page cache faster
# than one NVMe under read+write contention can write back, and reclaim failing on a
# dirty-cache pileup ends in a global OOM kill (observed loading a 21G GGUF onto 18G
# of spill tmpfiles on a 14G-RAM box). _await_writeback_drain blocks the pipeline
# thread until Dirty shrinks below the limit; the bounded PinPipeline queue turns that
# into reader backpressure (the fill loop's tracker sink blocks a few banks behind).
_SPILL_DIRTY_LIMIT = int(os.getenv("FREETOKEN_SPILL_DIRTY_LIMIT", 1 << 30))  # bytes


def _await_writeback_drain(limit: int = _SPILL_DIRTY_LIMIT, timeout_s: float = 300.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with open("/proc/meminfo") as f:
                dirty = next(
                    (int(line.split()[1]) * 1024 for line in f if line.startswith("Dirty:")),
                    0,
                )
        except (OSError, StopIteration):
            return  # not Linux / no meminfo: nothing to throttle on
        if dirty <= limit:
            return
        time.sleep(0.05)


def _finish_bank(bank: HostBank) -> None:
    if bank.backing == FILE_BACKING:
        _await_writeback_drain()
        bank.drop_pages()
    elif bank.backing == SPLIT_BACKING:
        bank.pin()
        _await_writeback_drain()
        bank.drop_pages()
    else:
        bank.pin()


class PinPipeline:
    """Pin filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a
    queue of completed banks. The queue is bounded (4 banks ~= 2 layers): with
    file-backed fills the thread also waits for writeback to drain, and the bound
    propagates that backpressure to the reader (the loaders' tracker sink blocks
    in submit) so dirty page cache cannot run arbitrarily far ahead of the disk.
    Total load time stays ~max(read, pin+writeback). Use as a context manager: a
    clean exit drains the queue and re-raises the first pin failure; an
    exceptional exit still joins the thread but lets the original exception
    propagate.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=4)
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
                _finish_bank(bank)
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
    "parse_layer_backing",
    "pin_banks",
    "read_file_into",
    "ANON_BACKING",
    "FILE_BACKING",
    "SPLIT_BACKING",
]
