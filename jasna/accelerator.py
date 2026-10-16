from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
import ctypes
from enum import StrEnum
import functools
import os
import sys
import logging
from pathlib import Path
import time
from typing import Any

import torch

# BasicVSR++ meets a new convolution batch for every clip length. FAST never
# writes the user FindDb, and the Windows ROCm wheels ship no system FindDb
# (ROCm/TheRock#7958), so every miss falls back to slow heuristic kernels forever.
# HYBRID runs a real find once per new problem and keeps the result across runs
# (issue #481: 4x faster primary restoration once the DB is seeded).
logger = logging.getLogger(__name__)

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
#
# Expandable segments release memory by unmapping virtual address ranges rather
# than by a device-synchronizing free, so a VramOffloader empty_cache() could
# pull pages out from under kernels another thread still had in flight — issue
# #252 caught a restorer fp16 GEMM faulting with "Page not present". The env
# names cover the versions in the field; whichever one the build reads wins.
#
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL opts in to the AOTriton flash-attention
# kernels on Radeon parts that PyTorch ships as "not officially validated". It is
# what makes FLASH/EFFICIENT SDPA usable on the consumer cards this build targets;
# see aotriton_image_dir() below for the second half of that story.
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:

    # HYBRID, not FAST: BasicVSR++ meets a new convolution batch for every clip
    # length, and FAST never writes the user FindDb while the Windows ROCm wheels
    # ship no system FindDb (ROCm/TheRock#7958) - every miss fell back to slow
    # heuristic kernels on every run. HYBRID runs a real find once per problem and
    # persists it (upstream issue #481: 4x faster primary once the DB is seeded).
    environ.setdefault("MIOPEN_FIND_MODE", "HYBRID")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")



# --- AOTriton flash-attention availability -----------------------------------
#
# Windows ROCm wheels split the AOTriton payload across two device packages and
# neither one is sufficient on its own:
#
#   amd-torch-device-<exact arch>    -> torch/.kpack/torch_<arch>.kpack
#                                       (the torch kernels)
#   amd-torch-device-<arch family>   -> torch/lib/aotriton.images/<family>/...
#                                       (the SDPA kernel images)
#
# With only the exact-arch package installed the images are absent and the first
# FLASH/EFFICIENT SDPA launch submits a null kernel: hipModuleLaunchKernel is
# called with a null function pointer and the process dies with
# hipErrorInvalidImage (ROCm/TheRock#7315). Disabling those two backends keeps the
# run alive on the numerically identical MATH backend, but it is much slower — on
# RF-DETR's DINOv2 backbone, which is attention bound, MATH costs 38.3 ms/frame
# against 18.8 ms/frame with FLASH at 576x576 (RX 7900 XT, gfx1100). So the
# backends are only switched off when the images for *this* GPU are really
# missing.
_AOTRITON_IMAGES_DIRNAME = "aotriton.images"
_AOTRITON_IMAGE_GLOB = "*/flash/attn_fwd.zip"
#: image directories are named amd-<family>, e.g. amd-gfx110x / amd-gfx120x
_AOTRITON_IMAGES_DIR_PREFIX = "amd-"
_FORCE_MATH_ENV = "JASNA_FORCE_MATH_SDP"
_TRUTHY = {"1", "true", "yes", "on"}


def _image_sets(root: Path) -> list[Path]:
    """Image directories under ``root`` that carry a complete flash kernel set."""
    if not root.is_dir():
        return []
    return sorted({marker.parent.parent for marker in root.glob(_AOTRITON_IMAGE_GLOB)})


def _arch_dir_names(arch: str) -> tuple[str, ...]:
    """Image directory names (without the ``amd-`` prefix) that may serve ``arch``.

    The images live under a directory named after the architecture *family*:
    ``gfx1101`` is served by ``amd-gfx110x``, ``gfx1201`` by ``amd-gfx120x``. An
    exact-architecture directory, when upstream ships one, wins.

    The family name is derived from the arch (drop the last digit, append ``x``)
    and never from the family *package* name. RDNA4 is the trap here: its family
    package is ``amd-torch-device-gfx12-0`` (wheel file ``gfx12_0``) while the
    images inside it live in ``amd-gfx120x``, so looking up ``gfx120x`` as a
    package name 404s on AMD's index and looking up ``gfx12_0`` as a directory
    name finds nothing. ``setup-rocm101-gpu.ps1`` builds the same name the same
    way, so the two stay in step.

    Verified against every family wheel up to ROCm 10.1: ``amd-gfx110x`` (from
    amd_torch_device_gfx110x), ``amd-gfx115x`` (gfx115x) and ``amd-gfx120x``
    (gfx12_0) all follow it. The ``amd-`` prefix is dropped here so that a
    directory named without it still matches (see ``_image_dir_matches``).
    """
    arch = (arch or "").split(":")[0].strip().lower()
    if not arch.startswith("gfx"):
        return ()
    names = [arch]
    if len(arch) > 1:
        names.append(arch[:-1] + "x")
    return tuple(dict.fromkeys(names))


def _image_dir_matches(dirname: str, name: str) -> bool:
    """True when ``dirname`` is the image directory ``name`` (prefix optional).

    Accepts both ``amd-gfx110x`` (what upstream ships) and a bare ``gfx110x``, so
    a hand-assembled or differently-packaged set is not silently ignored.
    """
    return dirname == name or dirname.rsplit("-", 1)[-1] == name


def _device_gcn_arch() -> str:
    """``gfxNNNN`` for the GPU jasna runs on, or an empty string when unreadable.

    Uses :func:`preferred_gpu_index` rather than device 0: on a machine whose
    device 0 is an integrated GPU (gfx1103/gfx1036 and friends) device 0 reports
    an arch the pipeline never runs on, so the answer would be wrong on every
    launch.
    """
    try:
        name = str(torch.cuda.get_device_properties(preferred_gpu_index()).gcnArchName)
    except Exception:  # pragma: no cover - depends on the runtime
        return ""
    return name.split(":")[0].strip()


def aotriton_image_dir(lib_dir: Path | str | None = None, *, arch: str | None = None) -> Path | None:
    """AOTriton image directory usable by this GPU, or ``None`` when there is none.

    ``lib_dir`` overrides ``torch/lib`` (used by the tests); ``arch`` overrides the
    device query.

    An installed set only counts when it *covers this GPU*. A portable package is
    assembled for one family (usually ``amd-gfx110x``, for RDNA3), and a card of
    another family finds that set installed but useless: returning it would leave
    FLASH/EFFICIENT SDPA enabled, AOTriton would then find no image for this arch
    and submit a null kernel - the crash path of ROCm/TheRock#7315. So a set that
    does not cover this GPU is treated as absent and the MATH backend is used.

    A single installed set is always assumed to be ours: with nothing to choose
    between, the device arch is never queried, so a machine that hides its arch
    does not silently lose the fast path. Only several sets force a device query
    (and are never guessed between when the arch stays unknown).
    """
    base = Path(lib_dir) if lib_dir is not None else Path(torch.__file__).resolve().parent / "lib"
    sets = _image_sets(base / _AOTRITON_IMAGES_DIRNAME)
    if not sets:
        return None
    if arch is None and len(sets) == 1:
        return sets[0]

    resolved_arch = arch if arch is not None else _device_gcn_arch()
    wanted = _arch_dir_names(resolved_arch)
    if wanted:
        for name in wanted:
            for candidate in sets:
                if _image_dir_matches(candidate.name, name):
                    return candidate
        logger.debug(
            "AOTriton image sets %s do not cover %s (looked for %s); keeping the MATH backend",
            [c.name for c in sets],
            resolved_arch,
            " / ".join(_AOTRITON_IMAGES_DIR_PREFIX + n for n in wanted),
        )
        return None

    if len(sets) == 1:
        return sets[0]
    logger.debug(
        "Several AOTriton image sets found but the GPU arch is unknown; keeping the MATH backend"
    )
    return None


def _math_sdp_forced() -> bool:
    return os.environ.get(_FORCE_MATH_ENV, "").strip().lower() in _TRUTHY


#: Device-visibility restrictions apply in the HIP runtime's own enumeration, while
#: HSA-level tools (``hipInfo``, ``rocm-smi``) still list every agent. A machine whose
#: iGPU is device 0 and whose card is device 1 therefore looks like a one-GPU machine to
#: PyTorch: the pipeline runs on the iGPU and the MIGraphX engine dies with
#: ``RUNTIME_EXCEPTION ... Failed to call function`` (gfx103x has no device code).
#: jasna needs the whole list to find the discrete Radeon, unless the user pins a device
#: on purpose.
_VISIBLE_DEVICES_ENV = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
_KEEP_VISIBLE_DEVICES_ENV = "JASNA_KEEP_HIP_VISIBLE_DEVICES"


def _device_pin_file() -> str | None:
    """A stale ``sitecustomize.py`` that pins device visibility, if one is installed.

    ``sitecustomize`` is imported by ``site`` before the application runs, so a helper
    that writes ``HIP_VISIBLE_DEVICES`` from a cached index re-applies the pin inside
    every interpreter: clearing the variable at the entry point changes nothing, and the
    symptom looks like a driver problem.
    """
    import site

    directories = [*site.getsitepackages(), site.getusersitepackages()]
    for directory in directories:
        candidate = Path(directory) / "sitecustomize.py"
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "HIP_VISIBLE_DEVICES" in text:
            return str(candidate)
    return None


def allow_all_devices(environ: MutableMapping[str, str] | None = None) -> None:
    """Drop a device-visibility variable so every GPU is enumerated again.

    Must happen before the first ``torch.cuda`` call: the HIP runtime reads
    ``HIP_VISIBLE_DEVICES`` when it initialises, so clearing it later changes nothing.
    Set ``JASNA_KEEP_HIP_VISIBLE_DEVICES=1`` (e.g. a workstation that pins one card per
    job) to keep the restriction and select with ``--device`` instead.
    """
    env = os.environ if environ is None else environ
    if str(env.get(_KEEP_VISIBLE_DEVICES_ENV, "")).strip().lower() in _TRUTHY:
        return
    for name in _VISIBLE_DEVICES_ENV:
        value = env.pop(name, None)
        if value is not None:
            logger.info(
                "%s=%s restricted this machine's visible GPUs; listing all of them so "
                "jasna can pick the discrete Radeon (set %s=1 to keep the restriction)",
                name, value, _KEEP_VISIBLE_DEVICES_ENV,
            )
            pin = _device_pin_file()
            if pin:
                logger.warning(
                    "%s is pinned again at every interpreter start by %s; rename it to "
                    "sitecustomize.py.disabled (or delete it) to keep the pin gone",
                    name, pin,
                )


_VRAM_LIMIT_GB_ENV = "JASNA_VRAM_LIMIT_GB"
_VRAM_FRACTION_ENV = "JASNA_VRAM_FRACTION"
#: Small fixed headroom subtracted from the *discrete* card's VRAM when no
#: explicit budget is set: 500 MB covers the compositor's basic buffers without
#: giving away gigabytes of the card to the desktop.
_VRAM_DEFAULT_HEADROOM_BYTES = 500 * 1024 * 1024
#: Fraction of the budget above which the allocator's reserve counts as hoarded
#: cold blocks and a clip boundary triggers a defragmenting release
#: (JASNA_VRAM_DEFRAG_THRESHOLD overrides).
_VRAM_DEFRAG_THRESHOLD_DEFAULT = 0.75
_VRAM_DEFRAG_THRESHOLD_ENV = "JASNA_VRAM_DEFRAG_THRESHOLD"
#: Minimum seconds between two defragmenting releases. empty_cache forces the
#: next clips to re-allocate their workspaces, so back-to-back cleanups churn
#: (reported as "slower towards the end") without reclaiming anything new.
_VRAM_DEFRAG_COOLDOWN_S = 60.0
_last_defrag_monotonic = 0.0
#: The fraction apply_vram_budget last installed (None = no budget applied).
_vram_budget_fraction: float | None = None


def apply_vram_budget() -> None:
    """Cap the CUDA caching allocator so long jobs stop ballooning.

    The allocator caches every freed block and never returns VRAM on its own,
    so a long 4K job creeps toward the whole card; once Windows starts demoting
    pages to shared memory (PCIe slow path) the frame rate collapses.
    ``set_per_process_memory_fraction`` makes the allocator reclaim its own
    cached blocks at the cap instead of growing past it.

    The budget is derived from the discrete card that ``preferred_gpu_index``
    picks (the iGPU is skipped): that card's VRAM minus 500 MB. Tune with
    ``JASNA_VRAM_LIMIT_GB`` (absolute GB) or ``JASNA_VRAM_FRACTION`` (0-1].
    """
    global _vram_budget_fraction
    if not torch.cuda.is_available():
        return
    try:
        index = preferred_gpu_index()
        total = torch.cuda.get_device_properties(index).total_memory
        if not total:
            return
        limit = os.environ.get(_VRAM_LIMIT_GB_ENV, "").strip()
        frac_env = os.environ.get(_VRAM_FRACTION_ENV, "").strip()
        if limit:
            fraction = float(limit) * (1024 ** 3) / total
        elif frac_env:
            fraction = float(frac_env)
        else:
            fraction = (total - _VRAM_DEFAULT_HEADROOM_BYTES) / total
        if not 0.05 <= fraction <= 0.99:
            logger.warning(
                "VRAM budget %s resolves to fraction %.2f, outside 0.05-0.99; ignored",
                limit or frac_env or "default", fraction,
            )
            return
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:  # noqa: BLE001
            name = "GPU"
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        _vram_budget_fraction = fraction
        logger.info(
            "VRAM budget: device %d (%s) capped at %.1f GB of %.1f GB (%.0f%%); "
            "tune with %s / %s",
            index, name,
            total * fraction / (1024 ** 3), total / (1024 ** 3), fraction * 100,
            _VRAM_LIMIT_GB_ENV, _VRAM_FRACTION_ENV,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM budget not applied: %s", exc)


#: Captured CUDA/HIP graph pools, registered by the restorers that own them.
#: empty_cache() cannot spare a selected pool, so while any graph is registered
#: release_vram_cache() skips the release entirely - freeing the pool under a
#: captured graph can leave replay reading re-mapped pages, which corrupts the
#: graph's output silently. Owners compare hip_graph_pool_generation() against
#: the value they captured at and re-capture when a release slipped through
#: another path (a direct torch.cuda.empty_cache elsewhere).
_graph_pool_owners: set[int] = set()
_hip_graph_pool_generation = 0


def register_graph_pool(token: int) -> None:
    """Mark a captured graph's pool as exempt from cold-cache releases."""
    _graph_pool_owners.add(int(token))


def unregister_graph_pool(token: int) -> None:
    _graph_pool_owners.discard(int(token))


def hip_graph_pool_generation() -> int:
    """Bumped whenever a cold-cache release runs; owners compare and re-capture."""
    return _hip_graph_pool_generation


def release_vram_cache() -> None:
    """Return every cached-but-unused block to the driver (defragmentation).

    Live allocations - the resident detector and secondary restorer, the
    current clip's tensors - are untouched. Only the allocator's cold cache is
    released, which is what otherwise grows until Windows starts demoting the
    hot pages to shared memory and the frame rate collapses.

    Skipped while a captured HIP graph is registered: empty_cache() cannot
    exclude the graph's static buffers, and freeing the pool under a captured
    graph can silently corrupt its replays. Owners re-capture on the next clip
    if a release ran through another path.
    """
    global _hip_graph_pool_generation
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    if _graph_pool_owners:
        _hip_graph_pool_generation += 1
        logger.debug(
            "VRAM cold-cache release skipped: %d captured HIP graph pool(s) registered",
            len(_graph_pool_owners),
        )
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def maybe_release_vram_cache() -> bool:
    """Defragment when the reserve creeps toward the budget; cheap no-op otherwise.

    Called at clip boundaries. The check itself only reads allocator stats; the
    actual release runs once the reserve crosses the defrag threshold. The freed
    blocks are the allocator's cold cache - the part that does not need to sit
    in VRAM - which the driver may then hold in system memory; the
    speed-critical resident models stay untouched and, with the reserve back
    under the budget, un-demoted.
    """
    global _last_defrag_monotonic
    if not torch.cuda.is_available():
        return False
    try:
        threshold = float(
            os.environ.get(_VRAM_DEFRAG_THRESHOLD_ENV, "").strip()
            or _VRAM_DEFRAG_THRESHOLD_DEFAULT
        )
        if not 0.1 <= threshold <= 1.0:
            threshold = _VRAM_DEFRAG_THRESHOLD_DEFAULT
        index = preferred_gpu_index()
        total = torch.cuda.get_device_properties(index).total_memory
        allowed = int(total * _vram_budget_fraction) if _vram_budget_fraction \
            else int(total - _VRAM_DEFAULT_HEADROOM_BYTES)
        reserved = torch.cuda.memory_reserved(index)
        if allowed and reserved > allowed * threshold:
            import time as _time

            now = _time.monotonic()
            if now - _last_defrag_monotonic < _VRAM_DEFRAG_COOLDOWN_S:
                return False
            _last_defrag_monotonic = now
            logger.info(
                "VRAM defrag: %.2f GB reserved exceeds %.0f%% of the %.2f GB budget; "
                "returning the cold cache to the driver",
                reserved / (1024 ** 3), threshold * 100, allowed / (1024 ** 3),
            )
            release_vram_cache()
            logger.info(
                "VRAM defrag: reserve %.2f GB -> %.2f GB",
                reserved / (1024 ** 3), torch.cuda.memory_reserved(index) / (1024 ** 3),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM defrag check failed: %s", exc)
    return False


#: Restored-video-second cadences for the two sweeps that keep a long pass from
#: drifting. A whole-file render is a single pass, so the clip-boundary defrag
#: above only fires when the allocator's reserve crosses a fraction of the *whole
#: card* - a threshold a job that shares the GPU with a 4K desktop and a
#: remote-desktop capture (several GB the budget never accounts for) rarely
#: reaches. The reserve then drifts until Windows demotes pages to shared memory
#: and the frame rate collapses. On a schedule keyed to how much *video* the pass
#: has already produced (not wall clock, which a stalled or paused run would
#: distort), the sweeps land at the same point of the film however fast the
#: machine is.
#:
#: The light sweep returns the allocator's cold cache; 1200 = every 20 minutes of
#: restored video. The deep sweep additionally pushes stranded clip results out
#: and drops the cross-process cache; 3600 = first at the 60-minute mark, then
#: every 60 minutes of restored video. 0 disables each.
_VRAM_LIGHT_VIDEO_ENV = "JASNA_VRAM_LIGHT_VIDEO_S"
_VRAM_LIGHT_VIDEO_DEFAULT_S = 1200.0
_VRAM_DEEP_VIDEO_ENV = "JASNA_VRAM_DEEP_VIDEO_S"
_VRAM_DEEP_VIDEO_DEFAULT_S = 3600.0


def _positive_float_env(name: str, default: float) -> float:
    """Parse an env var as a positive float; blank/garbage keeps the default, <=0 disables."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %g", name, raw, default)
        return default
    return value if value > 0 else 0.0


def light_video_seconds() -> float:
    """Restored-video seconds between light cold-cache releases; 0 disables them."""
    return _positive_float_env(
        _VRAM_LIGHT_VIDEO_ENV, _VRAM_LIGHT_VIDEO_DEFAULT_S
    )


def deep_video_seconds() -> float:
    """Restored-video seconds between deep consolidations; 0 disables them."""
    return _positive_float_env(
        _VRAM_DEEP_VIDEO_ENV, _VRAM_DEEP_VIDEO_DEFAULT_S
    )


class VramReleaser:
    """Runs the "tidy up and carry on" sweeps while one long pass runs.

    A whole-file render is one ``run_restoration_pass``, so without this the only
    mid-run defrag is the clip-boundary one, gated on the allocator's reserve
    exceeding a fraction of the entire card - a threshold a job sharing the GPU
    with a 4K desktop and a remote-desktop capture (several GB the budget never
    accounts for) never reaches. The reserve then drifts until Windows demotes our
    hot pages to system memory over PCIe and the frame rate collapses.

    Both sweeps are keyed on ``progress``, not on the clock: the caller passes a
    callable returning how many seconds of *video* the pass has already produced.
    Every ``light_video`` seconds of it the cold cache is returned; every
    ``deep_video`` seconds the heavier ``deep_hook`` additionally sweeps stranded
    clip results and drops the cross-process cache. Keying on restored video
    (rather than elapsed time) puts the sweeps at the same point of the film
    whatever the machine's speed and however much the pass stalls on I/O or encode
    back-pressure. A deep sweep supersedes the light one due at the same point.

    Neither sweep touches live allocations, stops the pass or reopens the output,
    so progress is unchanged and the single output file stays one file. ``tick`` is
    cheap and safe to call from the pass's supervisor loop; without a progress
    source it is a no-op.
    """

    def __init__(
        self,
        *,
        deep_video: float | None = None,
        light_video: float | None = None,
        progress: Callable[[], float] | None = None,
        deep_hook: Callable[[], str] | None = None,
    ) -> None:
        self._deep_video = (
            deep_video_seconds() if deep_video is None else float(deep_video)
        )
        self._light_video = (
            light_video_seconds() if light_video is None else float(light_video)
        )
        self._progress = progress
        self._deep_hook = deep_hook
        # Each sweep fires when restored video reaches its mark, then advances by
        # another interval. None when there is nothing to key on (0 or no source).
        self._next_deep: float | None = (
            self._deep_video
            if progress is not None and self._deep_video > 0
            else None
        )
        self._next_light: float | None = (
            self._light_video
            if progress is not None and self._light_video > 0
            else None
        )
        self._enabled = self._next_deep is not None or self._next_light is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def deep_video(self) -> float:
        return self._deep_video

    @property
    def light_video(self) -> float:
        return self._light_video

    def _video_position(self) -> float | None:
        """Restored-video seconds so far, or None if it cannot be read yet."""
        if self._progress is None:
            return None
        try:
            return float(self._progress())
        except Exception:  # noqa: BLE001 - a bad reading is not a sweep trigger
            return None

    def _run_light(self, position: float) -> None:
        release_vram_cache()
        logger.info(
            "VRAM cold-cache release at %.1f min of restored video", position / 60.0
        )

    def _run_deep(self, position: float) -> None:
        detail = ""
        if self._deep_hook is not None:
            try:
                detail = self._deep_hook()
            except Exception as exc:  # noqa: BLE001 - a failed tidy-up must not end the pass
                logger.warning("VRAM deep consolidation failed: %r", exc)
                release_vram_cache()
                detail = f"hook_failed={exc!r}"
        else:
            release_vram_cache()
        logger.info(
            "VRAM deep consolidation at %.1f min of restored video%s",
            position / 60.0, f": {detail}" if detail else "",
        )

    def tick(self) -> bool:
        """Fire the due sweep (deep first) once restored video passes its mark."""
        if not self._enabled:
            return False
        position = self._video_position()
        if position is None:
            return False
        if self._next_deep is not None and position >= self._next_deep:
            # Skip straight to the next un-passed mark so a stalled poll or a coarse
            # update cannot fire the sweep twice for the same stretch.
            while self._next_deep <= position:
                self._next_deep += self._deep_video
            # A deep sweep subsumes the light one due at the same stretch.
            if self._next_light is not None:
                while self._next_light <= position:
                    self._next_light += self._light_video
            self._run_deep(position)
            return True
        if self._next_light is not None and position >= self._next_light:
            while self._next_light <= position:
                self._next_light += self._light_video
            self._run_light(position)
            return True
        return False


#: Seconds between memory-trend log lines; 0 (the default) disables them. Opt-in
#: because the line is for diagnosing a slow *long* run, not for every job.
_MEM_TREND_ENV = "JASNA_MEM_TREND_S"
_MEM_TREND_DEFAULT_S = 0.0


def mem_trend_seconds() -> float:
    """Seconds between memory-trend log lines; 0 disables them."""
    return _positive_float_env(_MEM_TREND_ENV, _MEM_TREND_DEFAULT_S)


class MemoryTrend:
    """Logs the allocator and caller-supplied counters on a fixed schedule.

    A working set that only creeps over an hour is invisible in a short window, so
    a long run needs its own trajectory. Every ``period`` seconds this writes one
    ``MEM TREND`` line with the allocator's reserved / allocated / peak bytes - the
    split that separates "the allocator is holding freed blocks" from "the live
    set really grew" - plus whatever extra counters the caller passes (buffer
    sizes, RSS). ``tick`` is cheap and safe from the pass's supervisor loop; with
    no period it never fires, and without CUDA it still logs the extra counters.
    """

    def __init__(self, period: float, *, extra=None) -> None:
        self._period = float(period)
        self._extra = extra
        self._next = time.monotonic() + self._period if self._period > 0 else None

    @property
    def enabled(self) -> bool:
        return self._next is not None

    @property
    def period(self) -> float:
        return self._period

    def sample(self) -> list[str]:
        """The current counters as ``key=value`` strings (also used by the caller)."""
        parts: list[str] = []
        mib = 1024 ** 2
        try:
            index = preferred_gpu_index()
            parts.append(f"reserved={torch.cuda.memory_reserved(index) / mib:.0f}MiB")
            parts.append(f"allocated={torch.cuda.memory_allocated(index) / mib:.0f}MiB")
            parts.append(f"peak={torch.cuda.max_memory_allocated(index) / mib:.0f}MiB")
        except Exception:  # noqa: BLE001 - no CUDA: the extra counters still help
            parts.append("cuda=unavailable")
        if self._extra is not None:
            try:
                parts.extend(f"{key}={value}" for key, value in self._extra().items())
            except Exception as exc:  # noqa: BLE001
                parts.append(f"extra_failed={exc!r}")
        return parts

    def tick(self) -> bool:
        """Write one trend line once the period elapses."""
        if self._next is None:
            return False
        now = time.monotonic()
        if now < self._next:
            return False
        self._next = now + self._period
        logger.info("MEM TREND %s", " ".join(self.sample()))
        return True


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    allow_all_devices()
    apply_vram_budget()
    if torch.version.hip:
        apply_rocm_env_defaults(os.environ)
        if _math_sdp_forced():
            logger.info("%s is set: using the MATH SDPA backend", _FORCE_MATH_ENV)
        elif aotriton_image_dir() is None:
            logger.warning(
                "No AOTriton SDPA images for this GPU were found, so FLASH/EFFICIENT "
                "attention cannot be launched (ROCm/TheRock#7315). Falling back to the "
                "MATH backend: correct but noticeably slower. Install the image package "
                "for this architecture, e.g. 'amd-torch-device-gfx110x' for gfx1100/1101/"
                "1102/1103, or 'amd-torch-device-gfx12-0' for gfx1200/1201 (its images then "
                "land in amd-gfx120x), to get the fast path back."
            )
        else:
            return
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:  # pragma: no cover - depends on the torch build
            pass


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None and resolved.type == "cpu":
        return AcceleratorVendor.CPU
    if torch.version.hip:
        return AcceleratorVendor.AMD
    if torch.version.cuda:
        return AcceleratorVendor.NVIDIA
    return AcceleratorVendor.CPU


def is_nvidia_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.NVIDIA


def is_amd_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.AMD


#: RDNA 3 / 4 *discrete* cards. Integrated Radeons (gfx103x, gfx1103, gfx115x) share
#: system memory and carry no MIGraphX device code at all, so a desktop whose driver
#: order puts the iGPU first would otherwise run the whole pipeline - and the detection
#: engine - on the wrong device.
DISCRETE_RADEON_ARCHS = frozenset({
    "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201",
})

#: ``preferred_gpu_index`` reports the device list once per process (see below).
_device_choice_logged = False


def hip_device_archs() -> list[str]:
    """``gcnArchName`` of every CUDA/HIP device, in the runtime's device order."""
    try:
        count = int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - no GPU / no usable runtime
        return []
    archs: list[str] = []
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            arch = str(getattr(props, "gcnArchName", "") or "").split(":")[0].strip()
        except Exception:  # pragma: no cover - depends on the runtime
            arch = ""
        archs.append(arch)
    return archs


def preferred_gpu_index() -> int:
    """Device index jasna should run on: the first *discrete* Radeon, otherwise 0.

    Some driver builds enumerate a Ryzen iGPU as device 0 and the discrete card as
    device 1. Everything that hard-coded ``cuda:0`` then targeted the iGPU: the MIGraphX
    execution provider aborts on ``gfx103x`` (``RUNTIME_EXCEPTION ... Failed to call
    function``, there is no device code for it) and the rest of the pipeline runs on the
    slowest device in the machine. Selecting the discrete Radeon keeps the caller on the
    card the user actually bought. Machines without one - a Strix Halo APU, an NVIDIA
    card, plain CPU - keep device 0, so this is a no-op for them.
    """
    global _device_choice_logged
    if not torch.version.hip:
        return 0
    archs = hip_device_archs()
    choice = 0
    for index, arch in enumerate(archs):
        if arch in DISCRETE_RADEON_ARCHS:
            choice = index
            break
    if not _device_choice_logged:
        # Once per process: this is the line to look at when a machine behaves as if it
        # had one GPU, or as if the engine ran on the wrong one.
        _device_choice_logged = True
        logger.info("ROCm devices %s -> running on device %d", archs or "none", choice)
    return choice


def preferred_device() -> torch.device:
    """The device jasna should use: :func:`preferred_gpu_index` resolved to a device."""
    return torch.device(f"cuda:{preferred_gpu_index()}")


def device_module(device: torch.device | str):
    return torch.get_device_module(torch.device(device))


def device_context(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return nullcontext()
    return device_module(resolved).device(resolved)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    try:
        return device_module(stream.device).stream(stream)
    except (TypeError, ValueError):
        # Also supports lightweight stream doubles used by callers/tests.
        return torch.cuda.stream(stream)


def new_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).Stream(resolved)


def current_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).current_stream(resolved)


def new_event(device: torch.device | str):
    return device_module(torch.device(device)).Event()


def set_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).set_device(resolved)


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.accelerator.synchronize()
        return
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).synchronize(resolved)


def empty_cache(device: torch.device | str | None = None) -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.empty_cache()
        return
    if device is not None:
        module = device_module(torch.device(device))
        if hasattr(module, "empty_cache"):
            module.empty_cache()


def ipc_collect(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def reset_peak_memory_stats(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(torch.device(device))


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    module = device_module(torch.device(device))
    return module.mem_get_info(torch.device(device))


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@functools.cache
def _nvml_device(uuid: str):
    nvml = ctypes.WinDLL("nvml.dll")
    _nvml_check(nvml, nvml.nvmlInit_v2())
    handle = ctypes.c_void_p()
    _nvml_check(nvml, nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode(), ctypes.byref(handle)))
    return nvml, handle


def _nvml_check(nvml, status: int) -> None:
    if status:
        nvml.nvmlErrorString.restype = ctypes.c_char_p
        raise RuntimeError(f"NVML: {nvml.nvmlErrorString(status).decode()}")


def _nvml_free_bytes(device: torch.device) -> int:
    nvml, handle = _nvml_device(str(torch.cuda.get_device_properties(device).uuid))
    memory = _NvmlMemory()
    _nvml_check(nvml, nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
    return memory.free


def device_free_bytes(device: torch.device) -> int:
    """VRAM free on the whole GPU. On Windows, CUDA counts only this process's memory (WDDM
    hides every other process's), so NVML's device-wide figure caps it there."""
    free = torch.cuda.mem_get_info(device)[0]
    if sys.platform == "win32":
        return min(free, _nvml_free_bytes(device))
    return free


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))
