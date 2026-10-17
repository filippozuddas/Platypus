"""
The production search chain for one cadence, as a library.

``scripts/inference.py`` is a thin driver over these functions; the
injection-recovery harness calls the same ones, so a measured sensitivity is
the sensitivity of the search that is actually run.

Stages, in order:

1. :class:`CadenceWindows` — the six observations of one cadence, staged as
   disk-backed memmaps; every sliding window is preprocessed from its own
   channels only (per-observation ``bandpass_correct`` + ``core_transform``).
2. :func:`score_windows` — one scalar anomaly score per window and method.
3. :func:`select_candidates` — per-cadence thresholds (FAR quantile, Gaussian
   references), frequency-adjacency clustering, one anomaly map per cluster
   peak, the OFF-noise ceiling (cluster maps + a random background probe,
   floored at ``thresh_5``), ON/OFF diagnostics and short-list membership.
4. :func:`rank_for_export` — the order in which candidates are shown to a human.

Rationale for each rule: ``docs/04_candidate-filtering.md``.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import bandpass_correct, core_transform
from src.models.autoencoder import build_autoencoder
from src.search.candidates import (
    cluster_candidates, full_row_hits, off_noise_ceiling, on_off_contrast,
)

MAD_SCALE = 1.4826
OFF_ROWS = (1, 3, 5)

Log = Callable[[str], None]


@dataclass(frozen=True)
class SelectionRule:
    """Parameters of the candidate-selection stages (production defaults).

    far_quantile: per-cadence quantile of the window scores that forms
        candidates (0.99 = a 1% false-alarm quota).
    off_ceiling_probe: random background windows added to the OFF-noise pool.
    min_off_pool: minimum pooled OFF cells for a robust ceiling estimate;
        below it the ceiling falls back to ``thresh_5``.
    leak_frac: ``full_row_hits`` magnitude gate for an OFF row to count as leak.
    """

    far_quantile: float = 0.99
    off_ceiling_probe: int = 300
    min_off_pool: int = 30
    leak_frac: float = 0.3


def enable_determinism() -> Dict[str, object]:
    """Make GPU scoring bit-reproducible across runs; returns the settings.

    By default PyTorch lets cuDNN run float32 convolutions in TF32 (~3
    significant digits) and pick algorithms whose summation order can change
    between processes: two runs of the same search then differ by ~1e-4
    relative on almost every score (measured on Voyager-1, 2026-10-17). This
    forces full FP32 and deterministic kernels, at some cost in speed. Call
    before the model is moved to the GPU (cuBLAS reads its workspace setting
    when the CUDA context starts).
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: an op without a deterministic kernel warns instead of failing;
    # bit-identity of two runs is what verifies the setting.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def robust_stats(scores: np.ndarray) -> Tuple[float, float]:
    """Median and Gaussian-equivalent sigma (``MAD * 1.4826``)."""
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median, mad * MAD_SCALE


def preprocess_window(obs: Sequence[np.ndarray], f_start: int, fchans: int, tchans: int,
                      preproc: Dict) -> np.ndarray:
    """One ``(tchans, fchans)`` model input: each observation normalised on its
    own slice ``[f_start, f_start + fchans)``, then stacked along time."""
    return normalize_frames([o[:, f_start:f_start + fchans] for o in obs], tchans, preproc)


def normalize_frames(frames: Sequence[np.ndarray], tchans: int, preproc: Dict) -> np.ndarray:
    """Per-observation ``bandpass_correct`` + ``core_transform``, stacked along time."""
    method = preproc.get("bandpass_method", "polynomial")
    poly_degree = preproc.get("poly_degree", 3)
    mad_epsilon = preproc.get("mad_epsilon", 1e-6)
    normed = [
        core_transform(bandpass_correct(f, method=method, poly_degree=poly_degree), mad_epsilon)
        for f in frames
    ]
    return np.concatenate(normed, axis=0)[:tchans, :]


# Worker-process state for the spawn pool. Workers are started with 'spawn'
# (never forked), because the main process already holds a CUDA context and
# forking after CUDA init can deadlock silently; spawned workers inherit no
# memory, so they reopen the observation memmaps by path.
_worker_obs: Optional[List[np.ndarray]] = None
_worker_args: Optional[Tuple[int, int, Dict]] = None


def _worker_init(file_specs, fchans: int, tchans: int, preproc: Dict) -> None:
    global _worker_obs, _worker_args
    _worker_obs = [np.memmap(path, dtype=dtype, mode="r", shape=shape)
                   for path, shape, dtype in file_specs]
    _worker_args = (fchans, tchans, preproc)


def _worker_window(f_start: int) -> np.ndarray:
    fchans, tchans, preproc = _worker_args
    return preprocess_window(_worker_obs, f_start, fchans, tchans, preproc)


class CadenceWindows:
    """The observations of one cadence and its sliding-window grid.

    Each observation is written once to ``scratch_dir`` as a raw float32 file
    and memory-mapped, so spawned preprocessing workers can open it by path
    (``multiprocessing.shared_memory`` lives in /dev/shm, typically far too
    small in Docker for a ~26 GB cadence). Use as a context manager, or call
    :meth:`close`, to delete the scratch files.
    """

    def __init__(self, obs_arrays: Sequence[np.ndarray], scratch_dir: Path, tag: str,
                 *, fchans: int, tchans: int, stride: int, preproc: Dict):
        self.fchans, self.tchans, self.stride, self.preproc = fchans, tchans, stride, preproc
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.paths: List[Path] = []
        self.file_specs = []
        for i, arr in enumerate(obs_arrays):
            path = scratch_dir / f"{tag}_obs{i}.f32"
            arr.tofile(str(path))
            self.paths.append(path)
            self.file_specs.append((str(path), arr.shape, arr.dtype))
        self._obs = [np.memmap(p, dtype=d, mode="r", shape=s) for p, s, d in self.file_specs]
        self.nchans = int(self._obs[0].shape[1])
        n_windows = max(0, (self.nchans - fchans) // stride + 1)
        self.f_starts = np.array([i * stride for i in range(n_windows)])

    def window(self, f_start: int) -> np.ndarray:
        return preprocess_window(self._obs, f_start, self.fchans, self.tchans, self.preproc)

    def raw_block(self, c0: int, c1: int) -> np.ndarray:
        """Copy of the raw data in channels ``[c0, c1)``: ``(n_obs, ntime, c1 - c0)``."""
        if c0 < 0 or c1 > self.nchans:
            raise ValueError(f"channels [{c0}, {c1}) outside [0, {self.nchans})")
        return np.stack([np.array(o[:, c0:c1]) for o in self._obs])

    def iter_windows(self, f_starts: Sequence[int], num_workers: int,
                     chunksize: int) -> Iterator[Tuple[int, np.ndarray]]:
        """``(f_start, window)`` in input order, preprocessed by a spawn pool."""
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers, initializer=_worker_init,
                      initargs=(self.file_specs, self.fchans, self.tchans, self.preproc)) as pool:
            yield from zip(f_starts, pool.imap(_worker_window, f_starts, chunksize=chunksize))

    def close(self) -> None:
        self._obs = []
        for path in self.paths:
            path.unlink(missing_ok=True)

    def __enter__(self) -> "CadenceWindows":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_model(checkpoint_path: Path, model_config: dict, input_shape: tuple, device: str):
    """Build the backbone from ``model_config`` and load a Lightning checkpoint."""
    model = build_autoencoder(input_shape, model_config, loss="mse")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def read_cadence_meta(h5_path: Path) -> dict:
    """Target name, start frequency, signed channel width and date from an .h5 header."""
    with h5py.File(str(h5_path), "r") as f:
        attrs = dict(f["data"].attrs) if "data" in f and f["data"].attrs else {}
        if not attrs:
            for key in f.keys():
                if hasattr(f[key], "attrs") and len(f[key].attrs) > 0:
                    attrs = dict(f[key].attrs)
                    break
            if not attrs and len(f.attrs) > 0:
                attrs = dict(f.attrs)

    source = attrs.get("source_name", b"unknown")
    if isinstance(source, bytes):
        source = source.decode("utf-8", errors="replace")
    source = source.strip().replace(" ", "_")

    fch1 = float(attrs.get("fch1", 0.0))
    foff = float(attrs.get("foff", 0.0))

    tstart_mjd = float(attrs.get("tstart", 0.0))
    if tstart_mjd > 0:
        from astropy.time import Time
        date_str = Time(tstart_mjd, format="mjd").iso[:10].replace("-", "")
    else:
        date_str = "nodate"

    return {"source": source, "fch1_mhz": fch1, "foff_mhz": foff, "date": date_str}


def _to_tensor(batch: Sequence[np.ndarray], device: str) -> torch.Tensor:
    return torch.from_numpy(np.array(batch)).float().unsqueeze(1).to(device)


def score_windows(model, windows: CadenceWindows, methods: Sequence[str], *,
                  batch_size: int, num_workers: int, device: str,
                  log: Log = print) -> Dict[str, np.ndarray]:
    """Score every window of ``windows.f_starts`` with each method.

    Returns ``{method: (n_windows,) float64}`` aligned with ``windows.f_starts``.
    """
    import time

    f_starts = [int(f) for f in windows.f_starts]
    n_windows = len(f_starts)
    scores: Dict[str, List[float]] = {m: [] for m in methods}
    batch: List[np.ndarray] = []
    processed = 0
    chunksize = max(1, min(256, n_windows // (num_workers * 4)))
    log(f"  Pool: {num_workers} workers (spawn), chunksize={chunksize}")

    t0 = time.time()
    for _, window in windows.iter_windows(f_starts, num_workers, chunksize):
        batch.append(window)
        if len(batch) == batch_size or processed == n_windows - 1:
            x = _to_tensor(batch, device)
            for m in methods:
                with torch.no_grad():
                    s = model.anomaly_score(x, method=m).cpu().numpy()
                scores[m].extend(float(v) for v in s)
            batch = []
        processed += 1
        if processed % 5000 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            log(f"  {processed}/{n_windows} snippets  "
                f"({rate:.0f}/s, ETA {(n_windows - processed) / rate:.0f}s)")
    elapsed = time.time() - t0
    log(f"  Scored {n_windows} snippets in {elapsed:.1f}s ({n_windows / max(elapsed, 1):.0f}/s)")
    return {m: np.array(v) for m, v in scores.items()}


@dataclass
class MethodResult:
    """Candidates of one scoring method on one cadence.

    ``clusters`` is the candidate table (one row per cluster, the CSV schema);
    ``amaps`` holds the anomaly map of each cluster's peak window, aligned
    with ``clusters`` rows (``None`` for backbones without ``anomaly_map``).
    """

    method: str
    clusters: pd.DataFrame
    amaps: List[Optional[np.ndarray]]
    thresholds: Dict[str, float] = field(default_factory=dict)


def select_candidates(scores: np.ndarray, f_starts: np.ndarray, *, model, windows: CadenceWindows,
                      method: str, rule: SelectionRule, freq_step_hz: float, fch1_mhz: float,
                      probe_seed: int, batch_size: int, device: str,
                      amap_cache: Optional["AmapCache"] = None, stale: frozenset = frozenset(),
                      log: Log = print) -> MethodResult:
    """Thresholds, clustering, cluster anomaly maps, OFF-noise ceiling and
    short-list membership for one method's window scores on one cadence.

    ``amap_cache`` reuses anomaly maps of windows whose data are unchanged
    (:class:`CadenceSearch`); ``stale`` lists the ``f_start`` of windows whose
    data differ from the cached ones, which are always recomputed and never
    stored. Maps are computed exactly as without a cache (cluster peaks one
    window at a time, the probe in its own batches), so the result is the
    same either way.
    """
    stride, fchans = windows.stride, windows.fchans
    median, mad_sigma = robust_stats(scores)
    # Gaussian 3s/5s are diagnostic references only: heavy-tailed disagreement
    # scores put them inside the OFF noise tail. thresh_5 still floors the
    # OFF-noise ceiling below.
    thresh_3 = median + 3 * mad_sigma
    thresh_5 = median + 5 * mad_sigma
    n_3s = (scores > thresh_3).sum()
    n_5s = (scores > thresh_5).sum()

    # Operating point: empirical quantile of this cadence's own score pool.
    far_thresh = float(np.quantile(scores, rule.far_quantile))
    n_far = (scores > far_thresh).sum()
    log(f"  {method}: median={median:.4f}  MAD_s={mad_sigma:.4f}  "
        f"3s(ref)={thresh_3:.4f}->{n_3s}  5s(ref)={thresh_5:.4f}->{n_5s}  "
        f"FAR{100 * (1 - rule.far_quantile):.0f}%={far_thresh:.4f}->{n_far}")

    # Frequency-adjacency dedup: one wide/strong line triggers many adjacent
    # overlapping windows (stride < fchans); collapse them into one candidate.
    clusters = cluster_candidates(f_starts, scores, far_thresh, stride, fchans,
                                  freq_step_hz, fch1_mhz=fch1_mhz)
    log(f"  {method}: {len(clusters)} distinct candidates after "
        f"frequency-adjacency dedup (from {n_far} raw threshold-crossings)")

    thresholds = {"median": median, "mad_sigma": mad_sigma, "thresh_3": thresh_3,
                  "thresh_5": thresh_5, "far_thresh": far_thresh}
    amaps: List[Optional[np.ndarray]] = [None] * len(clusters)
    if not hasattr(model, "anomaly_map") or len(clusters) == 0:
        return MethodResult(method, clusters, amaps, thresholds)

    off_pool = []
    for i, (_, crow) in enumerate(clusters.iterrows()):
        fs_c = int(crow["f_start_peak"])
        cacheable = amap_cache is not None and fs_c not in stale
        if cacheable and fs_c in amap_cache.peak:
            amap_c = amap_cache.peak[fs_c]
        else:
            x_c = torch.from_numpy(windows.window(fs_c)).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                amap_c = model.anomaly_map(x_c)[0].cpu().numpy()
            if cacheable:
                amap_cache.peak[fs_c] = amap_c
        amaps[i] = amap_c
        off_idx = [r for r in OFF_ROWS if r < amap_c.shape[0]]
        if off_idx:
            off_pool.append(amap_c[off_idx, :].ravel())

    # Background probe spread over the whole cadence, so the ceiling reflects
    # the cadence-wide OFF noise floor rather than only OFF cells co-located
    # with already-flagged candidates (a small, detection-biased sample).
    n_bg = 0
    if rule.off_ceiling_probe > 0:
        rng = np.random.default_rng(probe_seed)
        n_bg = min(rule.off_ceiling_probe, len(f_starts))
        bg_fstarts = rng.choice(f_starts, size=n_bg, replace=False)
        probe_cacheable = amap_cache is not None and not stale.intersection(bg_fstarts.tolist())
        if probe_cacheable and amap_cache.probe is not None:
            off_pool.extend(amap_cache.probe)
        else:
            probe_cells = []
            bg_windows = [windows.window(int(fs)) for fs in bg_fstarts]
            for bstart in range(0, len(bg_windows), batch_size):
                x_bg = torch.from_numpy(np.stack(bg_windows[bstart:bstart + batch_size])) \
                    .float().unsqueeze(1).to(device)
                with torch.no_grad():
                    amap_bg = model.anomaly_map(x_bg).cpu().numpy()
                off_idx = [r for r in OFF_ROWS if r < amap_bg.shape[1]]
                if off_idx:
                    probe_cells.append(amap_bg[:, off_idx, :].ravel())
            off_pool.extend(probe_cells)
            if probe_cacheable:
                amap_cache.probe = probe_cells

    # Robust OFF-noise-core ceiling, floored at thresh_5: the pool is small and
    # detection-biased and can compute below the Gaussian reference, which let
    # scattered marginal cells count as row hits (Voyager-1, 2026-07-16).
    # The floor only affects short-list membership, never which clusters exist.
    off_pool_n = sum(len(p) for p in off_pool)
    if off_pool and off_pool_n >= rule.min_off_pool:
        off_ceiling = max(off_noise_ceiling(np.concatenate(off_pool)), thresh_5)
    else:
        off_ceiling = thresh_5
    log(f"  {method}: OFF-noise-core ceiling={off_ceiling:.4f} "
        f"(pooled from {len(clusters)} clusters + {n_bg} background snippets, "
        f"{off_pool_n} cells, vs Gaussian 3s(ref)={thresh_3:.4f}, 5s(ref)={thresh_5:.4f})")

    columns: Dict[str, list] = {k: [] for k in (
        "on_off_contrast", "on_mean", "off_mean", "n_on_hits", "n_off_hits",
        "n_on_hits_full", "n_off_hits_full", "off_leak", "in_short_list")}
    for amap_c in amaps:
        stats = on_off_contrast(amap_c, threshold=off_ceiling)
        for k in ("on_off_contrast", "on_mean", "off_mean", "n_on_hits", "n_off_hits"):
            columns[k].append(stats[k])
        fr = full_row_hits(amap_c, threshold=off_ceiling, leak_frac=rule.leak_frac)
        for k in ("n_on_hits_full", "n_off_hits_full", "off_leak", "in_short_list"):
            columns[k].append(fr[k])
    for k, v in columns.items():
        clusters[k] = v
    # Persist the thresholds themselves (constant per cadence), so a candidate's
    # short-list verdict can be audited without the run log.
    clusters["off_ceiling"] = off_ceiling
    clusters["thresh_3"] = thresh_3
    clusters["thresh_5"] = thresh_5

    thresholds.update(off_ceiling=off_ceiling, n_probe=n_bg, off_pool_cells=off_pool_n)
    return MethodResult(method, clusters, amaps, thresholds)


@dataclass
class AmapCache:
    """Anomaly maps of unmodified windows, reused across re-searches of one cadence.

    ``peak``: cluster-peak maps by ``f_start``. ``probe``: the background
    probe's pooled OFF cells, valid while no probe window is modified.
    """

    peak: Dict[int, np.ndarray] = field(default_factory=dict)
    probe: Optional[List[np.ndarray]] = None


@dataclass(frozen=True)
class Patch:
    """New raw values for channels ``[c0, c0 + width)`` of some observations.

    ``data`` maps an observation index to its ``(ntime, width)`` replacement;
    observations not listed are unchanged (an ON-only injection lists no OFF).
    """

    c0: int
    data: Dict[int, np.ndarray]

    @property
    def width(self) -> int:
        return next(iter(self.data.values())).shape[1] if self.data else 0

    @property
    def c1(self) -> int:
        return self.c0 + self.width


class PatchedWindows:
    """A cadence's windows with a :class:`Patch` applied, without copying the cadence.

    Same interface as :class:`CadenceWindows` (``f_starts``, ``window``, ...);
    ``affected`` indexes the windows whose channels overlap the patch.
    """

    def __init__(self, base: CadenceWindows, patch: Patch):
        self.base, self.patch = base, patch
        self.fchans, self.tchans, self.stride, self.preproc = (
            base.fchans, base.tchans, base.stride, base.preproc)
        self.nchans, self.f_starts = base.nchans, base.f_starts
        ntime = base._obs[0].shape[0]
        if patch.c0 < 0 or patch.c1 > self.nchans:
            raise ValueError(f"patch channels [{patch.c0}, {patch.c1}) outside [0, {self.nchans})")
        for i, arr in patch.data.items():
            if not 0 <= i < len(base._obs) or arr.shape != (ntime, patch.width):
                raise ValueError(f"patch for observation {i} has shape {arr.shape}, "
                                 f"expected ({ntime}, {patch.width})")
        overlap = (self.f_starts < patch.c1) & (self.f_starts + self.fchans > patch.c0)
        self.affected = np.flatnonzero(overlap) if patch.width else np.array([], dtype=int)

    def window(self, f_start: int) -> np.ndarray:
        c0, c1 = self.patch.c0, self.patch.c1
        lo, hi = max(c0, f_start), min(c1, f_start + self.fchans)
        if lo >= hi:
            return self.base.window(f_start)
        frames = []
        for i, obs in enumerate(self.base._obs):
            frame = obs[:, f_start:f_start + self.fchans]
            if i in self.patch.data:
                frame = np.array(frame)
                frame[:, lo - f_start:hi - f_start] = self.patch.data[i][:, lo - c0:hi - c0]
            frames.append(frame)
        return normalize_frames(frames, self.tchans, self.preproc)


class CadenceSearch:
    """One cadence searched once in full, then re-searched after modifications.

    The background search scores every window and caches anomaly maps.
    :meth:`search` applies a :class:`Patch` (e.g. one injected signal),
    re-scores only the windows it overlaps and re-runs every selection stage on
    the whole cadence — FAR threshold, clustering (including merges with
    existing clusters), probe, OFF-noise ceiling, ON/OFF and short list — so
    the result is the search of the modified cadence. Preprocessing is
    window-local, which is what makes reusing the other windows exact.

    On GPU, re-scored windows run in smaller batches than in the full search;
    with :func:`enable_determinism` the difference is at float32 rounding level.
    """

    def __init__(self, model, windows: CadenceWindows, *, method: str, rule: SelectionRule,
                 freq_step_hz: float, fch1_mhz: float, probe_seed: int, batch_size: int,
                 device: str, num_workers: int, log: Log = print):
        self.model, self.windows, self.method, self.rule = model, windows, method, rule
        self._select_args = dict(method=method, rule=rule, freq_step_hz=freq_step_hz,
                                 fch1_mhz=fch1_mhz, probe_seed=probe_seed,
                                 batch_size=batch_size, device=device)
        self.batch_size, self.device = batch_size, device
        self.scores = score_windows(model, windows, [method], batch_size=batch_size,
                                    num_workers=num_workers, device=device, log=log)[method]
        self.cache = AmapCache()
        self.background = select_candidates(
            self.scores, windows.f_starts, model=model, windows=windows,
            amap_cache=self.cache, log=log, **self._select_args)

    def search(self, patch: Patch, log: Log = lambda _: None) -> Tuple[MethodResult, np.ndarray]:
        """Candidates and window scores of the cadence modified by ``patch``."""
        patched = PatchedWindows(self.windows, patch)
        scores = self.scores.copy()
        idx = patched.affected
        for b in range(0, len(idx), self.batch_size):
            chunk = idx[b:b + self.batch_size]
            x = _to_tensor([patched.window(int(self.windows.f_starts[i])) for i in chunk], self.device)
            with torch.no_grad():
                s = self.model.anomaly_score(x, method=self.method).cpu().numpy()
            scores[chunk] = [float(v) for v in s]
        stale = frozenset(int(f) for f in self.windows.f_starts[idx])
        result = select_candidates(
            scores, self.windows.f_starts, model=self.model, windows=patched,
            amap_cache=self.cache, stale=stale, log=log, **self._select_args)
        return result, scores


def rank_for_export(result: MethodResult, ignore_short_list: bool = False,
                    log: Log = print) -> Tuple[pd.DataFrame, List[Optional[np.ndarray]]]:
    """Order in which candidates are shown to a human.

    With anomaly maps: short-listed candidates only, by ``on_off_contrast``
    descending (peak score alone rewards persistent RFI over target-only
    events); ``ignore_short_list`` ranks every candidate the same way, as a
    diagnostic. Without maps: the clustering order (``peak_score``).
    """
    clusters, amaps, method = result.clusters, result.amaps, result.method
    has_amap = len(amaps) > 0 and amaps[0] is not None
    if has_amap and not ignore_short_list:
        short_idx = np.nonzero(clusters["in_short_list"].to_numpy())[0]
        order = short_idx[np.argsort(-clusters["on_off_contrast"].to_numpy()[short_idx])]
        log(f"  {method}: {len(order)}/{len(clusters)} candidates "
            f"in short list after ON/OFF full-row filter")
    elif has_amap:
        order = np.argsort(-clusters["on_off_contrast"].to_numpy())
        log(f"  {method}: ignore_short_list set, ranking every candidate "
            f"by on_off_contrast regardless of off_leak")
    else:
        return clusters, amaps
    return clusters.iloc[order].reset_index(drop=True), [amaps[i] for i in order]
