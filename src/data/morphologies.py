"""Signal morphologies behind one interface, for the sensitivity sweeps.

Every injection benchmark in this project has so far injected exactly one thing:
a linearly-drifting narrowband carrier (``NarrowbandDriftingGenerator``). That
makes the central thesis — that an unsupervised autoencoder is sensitive to
*arbitrary* signal morphologies, unlike narrowband-only turboSETI — untested by
construction: the search has only ever been shown signals of the one class the
traditional pipeline already finds.

This module exposes a family of morphologies through a single interface so that
``scripts/inject_recover.py`` (scorer-level) and ``scripts/pipeline_sensitivity.py``
(end-to-end through the ON/OFF rule stage) can sweep all of them by name,
without either script knowing how any one of them is rendered.

    injector = build_morphology("narrowband_accel", data_cfg, seed=123)
    site = injector.sample_site(fchans, total_tchans)   # morphology, frozen
    for snr in snr_list:                                # amplitude only
        windows, info = injector.inject(obs_windows, site, snr)

The ``sample_site`` / ``inject`` split is load-bearing and matches the existing
convention in ``NarrowbandDriftingGenerator.sample_cadence_signal_params``: the
morphology is drawn ONCE per injection site and reused across the whole SNR
sweep, so SNR is the sweep's single independent variable. Re-sampling per SNR
would confound amplitude with shape.

Two families implement the interface:

* :class:`SetigenMorphology` — signals expressible as setigen's
  ``(path, t_profile, f_profile)`` decomposition, rendered ON-only through
  ``_SetigenInjector.inject_on_only_cadence``. Covers the drifting, accelerating
  and sinusoidal classes plus the (frequency extent) x (time structure)
  factorial: ``narrowband_drift`` / ``narrowband_pulsed`` /
  ``wideband_continuous`` / ``wideband_pulsed``. That factorial exists because a
  single pulsed cell cannot say whether a deficit comes from the pulsing or from
  the bandwidth, and the two imply different fixes.
* :class:`DirectArrayMorphology` — signals that decomposition cannot express,
  because their extent in frequency varies with time and is not separable into a
  path times a profile. Covers the smiley-face and random-2D-shape classes:
  one complete 2D template is rasterised per site and added to EACH ON
  observation — three copies per cadence, OFF frames untouched — following
  ``BroadbandTransientGenerator``'s direct-array approach.

  These exist because they are the only morphologies in the sweep that a
  narrowband matched filter cannot express *at all*. The three narrowband
  classes and the pulsed beacon are all still tracks in the time-frequency
  plane; an arbitrary 2D shape is not, so it is the actual test of the project's
  central claim — sensitivity to arbitrary morphology, which is what justifies
  an autoencoder search over turboSETI. A dispersed broadband sweep (the
  ``0001`` product's signal class) would also belong here and is NOT implemented.

**SNR is not commensurable across morphologies.** Each family normalises
amplitude by its own convention (setigen's frame-integrated
``Frame.get_intensity`` for the first, per-transient energy for the second), and
a pulsed signal's "snr" is a per-pulse peak rather than an integrated level.
Compare the *shape* of survival curves across morphologies — where they fall
off, which pipeline stage kills them — never absolute SNR between two of them.

The 2D-template classes are **peak-matched** by default
(``info["snr_convention"] = "peak"``): the template's brightest pixels reach
``Frame.get_intensity(snr)``, the same per-pixel level a narrowband carrier of
that nominal SNR reaches. This is the convention that makes them *consistent
with the rest of the sweep*, not a favour to them — setigen's ``f_profile`` is
unit-HEIGHT, not unit-area, so ``wideband_*`` already injects roughly
``width_in_channels`` times a carrier's total power at the same nominal SNR.

An energy-matched alternative is kept (``SHAPE_SNR_CONVENTION = "integrated"``),
which scales the template so its total added power equals a carrier's. It was
the original default and is a defensible question to ask — "is the shape
recovered at equal transmitted energy?" — but on this geometry it is not a
*measurable* one: a shape covers hundreds of pixels against a carrier's 16 per
observation, so per-pixel amplitude collapses and the answer is "no" at every
SNR, by arithmetic rather than by anything the model does. Report the convention
whenever these morphologies' numbers are quoted; SNR is not commensurable across
conventions any more than it is across morphologies.

**Band-excursion discipline.** setigen's non-linear paths take a coefficient,
not an excursion, and are evaluated on the cadence's ABSOLUTE timeline (~1728 s
for the 0000 product), not one observation. Feeding them a narrowband drift rate
(~0.3 Hz/s) would sweep ~448 kHz across a ~2.9 kHz window: the signal leaves the
frame within a few bins and the sweep silently measures "no signal present"
rather than "signal not detected". So the non-linear morphologies here sample a
**total excursion as a fraction of the band** and solve for the coefficient,
and place ``start_channel`` so the whole track stays in-band.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import setigen as stg
    from astropy import units as u
except ImportError:  # pragma: no cover - mirrors synthetic.py's optional import
    stg = None
    u = None

from src.data.synthetic import (
    NarrowbandParams,
    NarrowbandDriftingGenerator,
    WidebandParams,
    WidebandPulsedGenerator,
)

ON_INDICES = (0, 2, 4)


@dataclass
class Site:
    """One frozen injection site: the morphology, minus its amplitude.

    ``payload`` carries whatever the owning morphology needs to render (setigen
    profiles and a path builder, or a 2D template factory). ``meta`` is the
    human/CSV-facing description and is written verbatim into the results table,
    so every row can be traced back to the shape that produced it.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class SetigenMorphology:
    """Morphologies expressible as setigen ``(path, t_profile, f_profile)``.

    Delegates the ON-only cadence assembly — absolute-timeline ``ts`` shifting,
    one constant physical amplitude across ON frames, byte-identical OFF frames —
    to ``_SetigenInjector.inject_on_only_cadence``, so all morphologies in this
    family share exactly the same injection semantics as the validated
    narrowband path and differ only in the three profile functions.
    """

    def __init__(self, name: str, generator, sampler: Callable[..., Site]):
        self.name = name
        self._gen = generator
        self._sampler = sampler

    @property
    def rng(self) -> np.random.Generator:
        return self._gen.rng

    def sample_site(self, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
        return self._sampler(self._gen, fchans, total_tchans, n_obs)

    def inject(self, obs_windows: np.ndarray, site: Site, snr: float,
               on_indices: Tuple[int, ...] = ON_INDICES) -> Tuple[np.ndarray, dict]:
        out, info = self._gen.inject_on_only_cadence(
            obs_windows,
            snr=snr,
            drift_rate=site.payload["drift_rate"],
            start_channel=site.payload["start_channel"],
            f_profile=site.payload["f_profile"],
            t_profile_builder=site.payload["t_profile_builder"],
            on_indices=on_indices,
            path_builder=site.payload.get("path_builder"),
        )
        info.update(site.meta)
        info["morphology"] = self.name
        return out, info


class DirectArrayMorphology:
    """A complete 2D template rendered inside each ON observation.

    One full copy of the shape per ON observation — three per cadence — with the
    OFF frames left byte-identical, exactly like every other morphology here.
    Ruled by Vishal, 2026-07-27: *"the signal should only be there during ONs,
    not split between ON-OFFs... one smiley face per ON, so a total of three of
    them on all three."*

    The consequence to keep in mind when reading these classes' numbers: the
    anomaly map is (6, 64) over a (96, 1024) input, so one map cell is 16 rows x
    16 channels and one observation is exactly 16 rows. A shape inside one
    observation therefore occupies exactly **one map row**, and the model has no
    vertical resolution with which to tell a circle from a bar. What is measured
    is the response to an unfamiliar 2D pattern, not to its shape. In exchange
    these classes are ON-only like all the others, so their end-to-end survival is
    a real sensitivity number rather than ~0 by construction — which a canvas-wide
    variant (swept in `morphologies_v4`/`v6`, now superseded) could never be.

    Amplitude calibration borrows the setigen frame only for its noise
    statistics (``Frame.get_intensity``), so the SNR scale is the same one the
    setigen morphologies use before the energy matching described in the module
    docstring is applied. No setigen path or profile is involved.
    """

    def __init__(self, name: str, generator, sampler: Callable[..., Site]):
        self.name = name
        self._gen = generator
        self._sampler = sampler

    @property
    def rng(self) -> np.random.Generator:
        return self._gen.rng

    def sample_site(self, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
        return self._sampler(self._gen, fchans, total_tchans, n_obs)

    def inject(self, obs_windows: np.ndarray, site: Site, snr: float,
               on_indices: Tuple[int, ...] = ON_INDICES) -> Tuple[np.ndarray, dict]:
        obs = np.asarray(obs_windows, dtype=float)
        n_obs, tchans_per_obs, fchans = obs.shape
        template = site.payload["template"]
        col0 = int(site.payload["start_channel"])
        t_rows, t_cols = template.shape
        if t_rows != tchans_per_obs:
            raise ValueError(
                f"{self.name}: template has {t_rows} rows but one observation is "
                f"{tchans_per_obs}. The site was sampled for a different geometry."
            )

        # Same reference-frame convention as inject_on_only_cadence: intensity is
        # computed ONCE from the first ON frame and reused, so the injected shape
        # has one constant physical amplitude across the cadence instead of being
        # renormalised to each observation's local noise.
        ref_frame = self._gen._make_frame(obs[on_indices[0]])
        intensity = float(ref_frame.get_intensity(snr=snr))
        mass = float(template.sum())
        convention = site.payload.get("snr_convention", SHAPE_SNR_CONVENTION)
        if convention == "peak":
            # Brightest pixels reach a carrier's per-pixel level. Consistent with
            # how setigen's unit-height f_profile already treats wideband signals.
            amplitude = intensity
        elif convention == "integrated":
            # Total added power equals a carrier spanning the whole canvas.
            amplitude = intensity * t_rows / mass
        else:
            raise ValueError(f"{self.name}: unknown snr_convention {convention!r}")

        # One complete copy of the shape inside EACH ON observation, OFF frames
        # untouched (Vishal, 2026-07-27: "one smiley face per ON, three in total,
        # not split between ON-OFFs"). The same template every time: the shape is
        # the signal, so it must repeat on-source exactly as a carrier does.
        out = obs.copy()
        for i in on_indices:
            out[i, :, col0:col0 + t_cols] += amplitude * template

        info = {
            "snr": snr,
            "drift_rate": 0.0,
            "start_channel": col0,
            "intensity": intensity,
            "on_indices": tuple(on_indices),
            "peak_amplitude": amplitude * float(template.max()),
            "template_mass": mass,
            "template_cols": int(t_cols),
            "template_rows": int(t_rows),
            "span": "on_observation",
            "snr_convention": convention,
        }
        info.update(site.meta)
        info["morphology"] = self.name
        return out.astype(np.float32), info


def _render_curve(points: np.ndarray, shape: Tuple[int, int], sigma: float,
                  aspect: float = 1.0) -> np.ndarray:
    """Rasterise a curve as a Gaussian stroke ``sigma`` ROWS thick.

    ``points`` is ``(N, 2)`` in ``(row, col)`` pixel coordinates. Every pixel
    takes ``exp(-d^2 / 2 sigma^2)`` on its distance ``d`` to the nearest sample.
    Soft edges rather than a binary mask: a hard-edged shape is mostly staircase
    artifact, and the model would be scored on the aliasing as much as on the
    morphology.

    ``aspect`` (channels per row-equivalent, see :data:`SHAPE_DISPLAY_ASPECT`)
    makes the distance anisotropic, so the stroke is ``sigma`` rows thick
    vertically and ``sigma * aspect`` channels thick horizontally — i.e. it has
    *uniform* apparent thickness all the way round the curve. With the isotropic
    version the top and bottom of a circle came out ~5x thinner than its sides
    and rendered as hairlines.
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    d2 = ((rows[..., None] - points[:, 0]) ** 2
          + ((cols[..., None] - points[:, 1]) / aspect) ** 2).min(axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _normalise(template: np.ndarray) -> np.ndarray:
    """Scale a rasterised template to peak exactly 1.0.

    Without this, ``snr_convention="peak"`` would be approximate: an outline's
    brightest pixel sits wherever the curve happens to pass closest to a pixel
    centre, typically 0.85-1.0, so the effective SNR would carry a silent
    few-percent jitter that varies with the shape rather than with the sweep.
    """
    peak = float(template.max())
    return np.clip(template / peak, 0.0, 1.0) if peak > 0 else template


def _polar_shape_field(shape: Tuple[int, int], centre: Tuple[float, float],
                       radii: Tuple[float, float], radial) -> np.ndarray:
    """Signed distance in ROW units (negative inside) to a star-shaped curve.

    ``radial(theta) -> normalised radius`` lets one expression cover both a plain
    ellipse (``radial`` constant) and a Fourier-perturbed blob.

    Scaling the normalised distance by the ROW radius — not by the mean of the two
    radii — is what makes the returned distance "row-equivalents" in both
    directions: at the top of the ellipse it is a distance in rows, and at the
    side it is a distance in channels divided by the aspect ratio. A stroke width
    expressed in rows then has uniform apparent thickness all the way round,
    matching :func:`_render_curve`.
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    dy = (rows - centre[0]) / radii[0]
    dx = (cols - centre[1]) / radii[1]
    rho = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    return (rho - radial(theta)) * radii[0]


# --------------------------------------------------------------------------
# Samplers. One per morphology; each returns a Site with everything frozen
# except amplitude.
# --------------------------------------------------------------------------

def _sample_drifting(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 1 — linear narrowband drift. The historical baseline.

    Delegates verbatim to the existing, validated sampler so this morphology is
    bit-identical to every previous benchmark: the sweeps remain comparable to
    the 79.11/95.56/100.0 numbers rather than merely similar.
    """
    drift_rate, start_channel, f_profile, t_builder, meta = \
        gen.sample_cadence_signal_params(fchans, total_tchans)
    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_builder,
                 "path_builder": None},
        meta={**meta, "path": "constant",
              "freq_extent": "narrowband", "time_structure": "continuous"},
    )


def _band_and_duration(gen, fchans: int, total_tchans: int) -> Tuple[float, float]:
    """Usable bandwidth (Hz) and absolute cadence duration (s).

    ``total_tchans`` is the whole cadence's time extent, because that is the
    domain setigen paths are evaluated over once ``inject_on_only_cadence``
    shifts each ON frame onto the absolute timeline.
    """
    p = gen.params
    return float(fchans) * p.df, float(total_tchans) * p.dt


def _sample_accelerating(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 2a — quadratically accelerating drift (``squared_path``).

    Physically: a transmitter whose line-of-sight acceleration is non-negligible
    over the cadence, e.g. a close-in planetary rotator. This is the class a
    linear-drift matched filter such as turboSETI degrades on, so it is one of
    the cleanest places to look for a sensitivity gap in either direction.

    Parameterised by total excursion rather than by the coefficient
    ``squared_path`` actually takes — see the module docstring. Given an
    excursion ``df`` over cadence duration ``T``, ``0.5*a*T^2 = df`` gives
    ``a = 2*df/T^2``. The track is monotonic, so placing ``f_start`` at the
    low (or high) edge with a margin keeps the whole sweep in-band.
    """
    band_hz, duration_s = _band_and_duration(gen, fchans, total_tchans)
    frac = float(gen.rng.uniform(0.05, 0.40))
    sign = 1.0 if gen.rng.random() < 0.5 else -1.0
    excursion_hz = frac * band_hz
    accel = 2.0 * excursion_hz / (duration_s ** 2)

    margin = 0.05 * band_hz
    span_chans = excursion_hz / gen.params.df
    lo = margin / gen.params.df
    hi = fchans - lo - span_chans
    if hi <= lo:  # excursion too wide for the band; fall back to centred
        start_channel = int(fchans // 2)
    elif sign > 0:
        start_channel = int(gen.rng.uniform(lo, hi))
    else:
        start_channel = int(gen.rng.uniform(lo + span_chans, fchans - lo))

    # Width follows the MEAN drift over the cadence, not the coefficient: the
    # signal's instantaneous linewidth is set by how fast it sweeps, and
    # `accel` (Hz/s^2) is not a rate.
    mean_drift = sign * excursion_hz / duration_s
    width = gen._calculate_eti_width(abs(mean_drift))
    f_profile, f_name = gen._select_f_profile(width)

    def path_builder(f_start_hz, _drift_rate):
        return stg.squared_path(f_start=f_start_hz,
                                drift_rate=sign * accel * u.Hz / u.s)

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.constant_t_profile(level=intensity)

    return Site(
        payload={"drift_rate": mean_drift, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": path_builder},
        meta={"path": "squared", "accel_hz_s2": sign * accel,
              "excursion_frac": frac, "mean_drift": mean_drift,
              "width": width, "f_profile": f_name, "t_profile": "constant",
              "start_channel": int(start_channel)},
    )


def _sample_sinusoidal(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 2b — sinusoidally modulated drift (``sine_path``).

    Physically: the Doppler signature of a transmitter on a rotating or orbiting
    body, which is periodic rather than monotonic. Unlike the accelerating case
    this track *returns* to earlier frequencies, so a frequency-adjacency
    clustering stage sees it as several disconnected candidates — which makes it
    a direct test of the clustering stage, not only of the scorer.

    Amplitude is a fraction of the band and the period a fraction of the cadence
    (2-5 cycles, so the periodicity is actually visible within the window); the
    centre is placed so ``+/- amplitude`` plus the linear term stays in-band.
    """
    band_hz, duration_s = _band_and_duration(gen, fchans, total_tchans)
    amp_frac = float(gen.rng.uniform(0.05, 0.20))
    amplitude_hz = amp_frac * band_hz
    n_cycles = float(gen.rng.uniform(2.0, 5.0))
    period_s = duration_s / n_cycles

    max_lin = 0.10 * band_hz / duration_s
    drift_rate = float(gen.rng.uniform(-max_lin, max_lin))
    lin_excursion = abs(drift_rate) * duration_s

    half_span_chans = (amplitude_hz + lin_excursion) / gen.params.df
    lo, hi = half_span_chans, fchans - half_span_chans
    start_channel = int(gen.rng.uniform(lo, hi)) if hi > lo else int(fchans // 2)

    # Peak instantaneous drift of the sinusoid, which is what sets linewidth.
    peak_drift = 2.0 * np.pi * amplitude_hz / period_s + abs(drift_rate)
    width = gen._calculate_eti_width(peak_drift)
    f_profile, f_name = gen._select_f_profile(width)

    def path_builder(f_start_hz, dr):
        return stg.sine_path(f_start=f_start_hz, drift_rate=dr * u.Hz / u.s,
                             period=period_s * u.s, amplitude=amplitude_hz * u.Hz)

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.constant_t_profile(level=intensity)

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": path_builder},
        meta={"path": "sine", "amplitude_hz": amplitude_hz,
              "amplitude_frac": amp_frac, "period_s": period_s,
              "n_cycles": n_cycles, "linear_drift": drift_rate,
              "peak_drift": peak_drift, "width": width,
              "f_profile": f_name, "t_profile": "periodic-none",
              "start_channel": int(start_channel)},
    )


#: Pulses inside one ON observation. THE parameter of a pulsed morphology on this
#: product, and the one the v2 analysis showed dominates its detectability
#: (survival fell 70% -> 7% from the shortest to the longest sampled period).
#: Sampling this rather than a period in bins is what keeps the class inside the
#: regime the product can actually resolve — see :func:`_sample_pulse_timing`.
PULSES_PER_OBS_RANGE = (2.0, 6.0)

#: Floor on pulse width, in time bins. A `periodic_gaussian_t_profile` narrower
#: than the sampling interval is a gaussian evaluated at one point: whether the
#: sampled peak reaches its nominal amplitude then depends on where the pulse
#: centre falls relative to a bin centre, so the injected SNR carries a large,
#: silent, per-site error. 1.5 bins keeps every pulse resolved.
MIN_PULSE_WIDTH_BINS = 1.5

#: Duty cycle range. Bounded below so a pulse is a pulse rather than a spike, and
#: above so consecutive pulses stay separated.
DUTY_RANGE = (0.30, 0.60)


def _sample_pulse_timing(gen, p, total_tchans: int, n_obs: int) -> Dict[str, Any]:
    """Sample a pulse train from PULSES PER OBSERVATION, not from a period in bins.

    Shared by the wideband and narrowband pulsed morphologies so the two differ
    in frequency extent and in nothing else — that is the whole point of having
    both (see :func:`_sample_narrowband_pulsed`). ``p`` supplies ``dt``.

    **Why not the generator's own period range.** ``WidebandParams`` defaults
    (``period_bins_min=8``, ``period_bins_max_frac=0.5``, ``duty_max=0.33``) were
    tuned for the 0002 product: dt ~1.07 s, 64-bin frames. On 0000 dt is 18.25 s
    and an ON observation is 16 bins, so those same numbers give a period of 8-16
    bins — **1 to 2 pulses per observation** — and a pulse width that can fall to
    a single bin, i.e. to the sampling limit. A sweep run there does not measure
    whether a blinking beacon is detectable; it measures where the pulses land
    relative to the sampling grid. That is how `narrowband_pulsed` came to score
    2.6% at SNR 15 in `morphologies_v3`: not a model result.

    So the timing is derived from the product's own geometry instead of inherited
    from another product's config block. ``period = tchans_per_obs / pulses``,
    with ``pulses`` in :data:`PULSES_PER_OBS_RANGE` and the width floored at
    :data:`MIN_PULSE_WIDTH_BINS`.

    Declared scope restriction, unchanged: this tests beacons whose period is
    short relative to a single observation. Longer-period trains are a genuinely
    different morphology — they degenerate towards "present in some ON blocks
    only" — and belong to a separate experiment where the number of illuminated ON
    blocks is the measured variable rather than a nuisance.
    """
    tchans_per_obs = max(1, int(total_tchans // max(1, n_obs)))
    pulses = float(gen.rng.uniform(*PULSES_PER_OBS_RANGE))
    period_bins = max(1e-9, tchans_per_obs / pulses)

    duty = float(gen.rng.uniform(*DUTY_RANGE))
    pulse_width_bins = max(MIN_PULSE_WIDTH_BINS, duty * period_bins)
    # Report the duty the injection actually has, not the one sampled: the floor
    # can raise it, and a CSV column that disagrees with the data is worse than
    # no column.
    duty = pulse_width_bins / period_bins

    return {
        "period_s": period_bins * p.dt,
        "pulse_width_s": pulse_width_bins * p.dt,
        "period_bins": period_bins,
        "pulse_width_bins": pulse_width_bins,
        "tchans_per_obs": tchans_per_obs,
        "pulses_per_obs": tchans_per_obs / period_bins,
        "duty": duty,
        "seed": int(gen.rng.integers(0, 2 ** 31)),
    }


def _periodic_t_profile_builder(timing: Dict[str, Any]):
    def t_profile_builder(intensity: float, n_bins: int):
        return stg.periodic_gaussian_t_profile(
            pulse_width=timing["pulse_width_s"] * u.s,
            period=timing["period_s"] * u.s,
            pulse_direction="up",
            amplitude=intensity,
            level=0.0,
            min_level=0.0,
            seed=timing["seed"],
        )
    return t_profile_builder


def _sample_narrowband_pulsed(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 3, narrowband arm — a blinking carrier.

    Vishal's brief says "pulsating signal with changing periodicity" and does
    NOT say wideband; ``wideband_pulsed`` supplies both at once and therefore
    cannot attribute its own result. This arm takes the drift, linewidth and
    frequency profile from the *narrowband* sampler — bit-identical to
    ``narrowband_drift`` — and replaces only the time profile with the same
    pulse train ``wideband_pulsed`` uses.

    Together the four classes form a 2x2 in (frequency extent) x (time
    structure): ``narrowband_drift`` / ``narrowband_pulsed`` /
    ``wideband_continuous`` / ``wideband_pulsed``. That factorial is what
    separates "the pipeline is penalising pulsed signals" from "the pipeline is
    penalising diffuse ones" — two claims the existing single-cell measurement
    conflates, and which imply different fixes.
    """
    drift_rate, start_channel, f_profile, _, meta = \
        gen.sample_cadence_signal_params(fchans, total_tchans)
    timing = _sample_pulse_timing(gen, gen.aux_params, total_tchans, n_obs)

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile,
                 "t_profile_builder": _periodic_t_profile_builder(timing),
                 "path_builder": None},
        meta={**meta, "path": "constant", "t_profile": "periodic_gaussian",
              "freq_extent": "narrowband", "time_structure": "pulsed",
              "period_s": timing["period_s"],
              "pulse_width_s": timing["pulse_width_s"],
              "period_bins": timing["period_bins"],
              "pulse_width_bins": timing["pulse_width_bins"],
              "tchans_per_obs": timing["tchans_per_obs"],
              "pulses_per_obs": timing["pulses_per_obs"],
              "duty": timing["duty"], "snr_convention": "pulse_peak"},
    )


def _sample_wideband_continuous(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """The fourth cell of the 2x2 — a wide, steady band with no time structure.

    Physically the least motivated of the six (a broadband carrier that never
    switches off looks like receiver gain), but it is the control that makes the
    other three interpretable: it isolates frequency extent with the pulse train
    removed, exactly as ``narrowband_pulsed`` isolates the pulse train with the
    frequency extent removed.
    """
    p = gen.params
    frac = float(gen.rng.uniform(p.frac_low, p.frac_high))
    width_hz = max(1.0, frac * fchans) * p.df
    f_profile, f_name = gen._select_f_profile(width_hz)
    drift_rate = float(gen.rng.uniform(-p.drift_jitter, p.drift_jitter))
    start_channel = int(gen.rng.integers(1, max(2, fchans - 1)))

    def t_profile_builder(intensity: float, n_bins: int):
        return stg.constant_t_profile(level=intensity)

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile, "t_profile_builder": t_profile_builder,
                 "path_builder": None},
        meta={"path": "constant", "width": width_hz, "width_frac": frac,
              "f_profile": f_name, "t_profile": "constant",
              "freq_extent": "wideband", "time_structure": "continuous",
              "start_channel": int(start_channel)},
    )


def _sample_pulsed(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 3, wideband arm — wide-band periodic pulse train (radar/beacon-like).

    Reuses ``WidebandPulsedGenerator``'s frequency-extent and pulse sampling,
    but renders through the shared ON-only cadence path rather than the
    generator's own single-frame ``inject_signal``, so the OFF observations stay
    untouched exactly as for every other morphology here.

    Timing comes from :func:`_sample_pulse_timing`, shared verbatim with
    ``narrowband_pulsed`` — the two pulsed cells of the 2x2 must differ in
    frequency extent and in NOTHING else, or the factorial cannot attribute a
    deficit to the bandwidth. (An earlier version inlined a copy of the timing
    draw here to preserve the `morphologies_v2` RNG stream. That is moot now that
    the timing ranges themselves have changed: both pulsed cells are superseded
    from `morphologies_v4` on, so the duplication bought nothing and is gone.)

    ``snr`` is a per-pulse peak level here, not a frame-integrated SNR — the
    train is off for most of the frame, so integrated SNR is roughly
    ``snr * duty``. See the module docstring on cross-morphology comparison.
    """
    p = gen.params
    frac = float(gen.rng.uniform(p.frac_low, p.frac_high))
    width_hz = max(1.0, frac * fchans) * p.df
    f_profile, f_name = gen._select_f_profile(width_hz)

    timing = _sample_pulse_timing(gen, p, total_tchans, n_obs)
    drift_rate = float(gen.rng.uniform(-p.drift_jitter, p.drift_jitter))
    start_channel = int(gen.rng.integers(1, max(2, fchans - 1)))

    return Site(
        payload={"drift_rate": drift_rate, "start_channel": start_channel,
                 "f_profile": f_profile,
                 "t_profile_builder": _periodic_t_profile_builder(timing),
                 "path_builder": None},
        meta={"path": "constant", "width": width_hz, "width_frac": frac,
              "f_profile": f_name, "t_profile": "periodic_gaussian",
              "freq_extent": "wideband", "time_structure": "pulsed",
              "period_s": timing["period_s"],
              "pulse_width_s": timing["pulse_width_s"],
              "period_bins": timing["period_bins"],
              "pulse_width_bins": timing["pulse_width_bins"],
              "tchans_per_obs": timing["tchans_per_obs"],
              "pulses_per_obs": timing["pulses_per_obs"],
              "duty": timing["duty"],
              "start_channel": int(start_channel), "snr_convention": "pulse_peak"},
    )


#: Channels per time bin that make a shape look ROUND in a plotted waterfall.
#: A (96, 1024) block drawn at ~2:1 (width:height) compresses frequency by
#: 1024/W px and stretches time by 96/(W/2) px, so one time bin occupies the
#: screen space of 1024/(2*96) ≈ 5.33 channels. A shape with equal row and
#: channel radii would render as a thin vertical sliver — which is exactly what
#: the first version produced. Aspect is jittered around this so the sweep sees
#: a range of eccentricities rather than one canonical shape.
SHAPE_DISPLAY_ASPECT = 1024.0 / (2.0 * 96.0)

#: Amplitude convention for the 2D-template morphologies: ``"peak"`` (default,
#: consistent with setigen's unit-height f_profiles) or ``"integrated"``. See the
#: module docstring — this changes what the SNR axis means, so it belongs in any
#: report of these morphologies' numbers.
SHAPE_SNR_CONVENTION = "peak"


def _template_geometry(gen, fchans: int, extent_rows: int,
                       r_row_frac: Tuple[float, float]) -> Tuple[float, float, float, int]:
    """Sample a shape's pixel size within ``extent_rows``, and the width to hold it.

    ``extent_rows`` is ONE observation (16 bins for the 0000 product): Vishal's
    2026-07-27 ruling is that a shape must be complete inside each ON
    observation, three copies per cadence, never split across an ON/OFF boundary.

    That bounds the vertical radius hard, and the consequence is worth stating
    where it is made: one anomaly-map cell is 16 rows x 16 channels, so a shape
    inside one observation occupies exactly ONE map row. The model has no
    vertical resolution with which to tell a circle from a bar, and the class
    measures the response to an unfamiliar 2D pattern rather than to its shape.
    The gain in exchange is that these morphologies are ON-only like every other
    class, so their end-to-end survival is meaningful instead of ~0 by
    construction.

    Stroke width scales with the radius rather than being absolute: at a 6-row
    radius a fixed 2-3 row stroke would be most of the shape.
    """
    r_row_max = max(2.0, 0.5 * extent_rows - 1.5)
    r_row = float(gen.rng.uniform(r_row_frac[0] * extent_rows,
                                  r_row_frac[1] * extent_rows))
    r_row = float(np.clip(r_row, 2.0, r_row_max))
    sigma = float(np.clip(0.16 * r_row, 0.7, 2.0))
    r_row = float(np.clip(r_row, 2.0, max(2.0, 0.5 * extent_rows - 1.5 * sigma - 0.5)))
    r_col = r_row * SHAPE_DISPLAY_ASPECT * float(gen.rng.uniform(0.8, 1.2))

    # Clamp the WIDTH RADIUS, not the finished template width. Clipping t_cols
    # after the fact (the previous version) let r_col exceed the template, so the
    # sides of the shape fell outside it: a circle rendered as two horizontal
    # arcs with no closure, and a filled blob as a rectangle.
    col_budget = 0.5 * (max(3.0, fchans * 0.5) - 6.0 * sigma - 4.0)
    r_col = min(r_col, col_budget)

    t_cols = int(np.ceil(2.0 * r_col + 6.0 * sigma + 4.0))
    t_cols = int(np.clip(t_cols, 3, fchans))
    return sigma, r_row, r_col, t_cols


def _sample_smiley(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 4 — a smiley face, with variations.

    Not a plausible astrophysical or engineered signal, and that is the point:
    it is a shape with no track-like structure at all, so a narrowband matched
    filter has nothing to match. Recovery here measures whether the anomaly
    score responds to *unfamiliar structure* rather than to a line.

    Varied per site: overall size, frequency/time aspect ratio, stroke width,
    eye size, and mouth curvature — including negative curvature, i.e. a frown,
    so the sweep is not measuring one memorised shape.

    Drawn on the whole cadence canvas, so it crosses the observation boundaries
    and appears in the OFF frames too — see :class:`DirectArrayMorphology` for
    why that is the only geometry the model can resolve, and what it means for
    how the result must be read.
    """
    tchans_per_obs = max(1, int(total_tchans // max(1, n_obs)))
    sigma, r_row, r_col, t_cols = _template_geometry(
        gen, fchans, tchans_per_obs, (0.30, 0.42))

    shape = (tchans_per_obs, t_cols)
    cy = 0.5 * (tchans_per_obs - 1)
    cx = 0.5 * (t_cols - 1)

    aspect = r_col / r_row

    face = np.exp(-_polar_shape_field(shape, (cy, cx), (r_row, r_col),
                                      lambda th: 1.0) ** 2 / (2.0 * sigma ** 2))

    eye_sigma = sigma * float(gen.rng.uniform(1.2, 1.9))
    eyes = _render_curve(
        np.array([[cy - 0.35 * r_row, cx - 0.38 * r_col],
                  [cy - 0.35 * r_row, cx + 0.38 * r_col]]), shape, eye_sigma,
        aspect=aspect)

    curvature = float(gen.rng.uniform(-1.0, 1.0))
    u = np.linspace(-1.0, 1.0, 192)
    mouth = _render_curve(
        np.stack([cy + 0.22 * r_row + curvature * (1.0 - u ** 2) * 0.42 * r_row,
                  cx + u * 0.50 * r_col], axis=1), shape, sigma, aspect=aspect)

    template = _normalise(np.maximum(np.maximum(face, eyes), mouth))
    start_channel = int(gen.rng.integers(0, max(1, fchans - t_cols)))

    return Site(
        payload={"template": template, "start_channel": start_channel},
        meta={"path": "template", "shape_kind": "smiley", "span": "on_observation",
              "r_row": r_row, "r_col": r_col, "aspect": r_col / r_row,
              "stroke_sigma": sigma, "eye_sigma": eye_sigma,
              "mouth_curvature": curvature,
              "expression": "smile" if curvature > 0 else "frown",
              "template_cols": t_cols, "start_channel": start_channel},
    )


def _sample_random_2d(gen, fchans: int, total_tchans: int, n_obs: int = 6) -> Site:
    """Trick 5 — a random closed 2D blob, filled or outlined.

    The radius is a Fourier series in the polar angle,
    ``r(theta) = 1 + sum_k a_k cos(k theta + phi_k)``, with the coefficients
    bounded so the radius stays positive and the curve stays star-shaped (hence
    closed and non-self-intersecting). Unlike the smiley this samples a
    *distribution* of shapes rather than variations on one, so it is the
    morphology-agnostic end of the sweep: no two sites inject the same object.

    Drawn on the whole cadence canvas, like the smiley.
    """
    tchans_per_obs = max(1, int(total_tchans // max(1, n_obs)))
    sigma, r_row, r_col, t_cols = _template_geometry(
        gen, fchans, tchans_per_obs, (0.30, 0.42))

    n_modes = int(gen.rng.integers(2, 6))
    orders = gen.rng.choice(np.arange(2, 8), size=n_modes, replace=False)
    coeffs = gen.rng.uniform(0.05, 0.30, size=n_modes)
    coeffs = coeffs * (0.55 / max(0.55, coeffs.sum()))  # keep r(theta) > 0
    phases = gen.rng.uniform(0.0, 2.0 * np.pi, size=n_modes)

    # `_template_geometry` sizes an unperturbed ellipse, but r(theta) reaches
    # `1 + sum(a_k)` at its widest. Shrink the base radii by that factor so the
    # *perturbed* extent is the one that fits the canvas — otherwise the blob is
    # clipped at the canvas edge and becomes a different morphology.
    swell = 1.0 + float(coeffs.sum())
    r_row, r_col = r_row / swell, r_col / swell

    def radial(theta):
        out = np.ones_like(theta)
        for k, a, phi in zip(orders, coeffs, phases):
            out = out + a * np.cos(k * theta + phi)
        return out

    shape = (tchans_per_obs, t_cols)
    signed = _polar_shape_field(
        shape, (0.5 * (tchans_per_obs - 1), 0.5 * (t_cols - 1)),
        (r_row, r_col), radial)

    # Always filled. The outline variant was dropped after visual review: on this
    # canvas an outlined blob reads as a pair of thin arcs, visually and
    # (more importantly) structurally indistinguishable from a drifting curve —
    # which `narrowband_sine` and `narrowband_accel` already cover. A filled blob
    # is the thing this morphology is for: extent in both axes at once.
    edge = np.exp(-signed ** 2 / (2.0 * sigma ** 2))
    template = _normalise(np.where(signed < 0.0, 1.0, edge))
    start_channel = int(gen.rng.integers(0, max(1, fchans - t_cols)))

    return Site(
        payload={"template": template, "start_channel": start_channel},
        meta={"path": "template", "span": "on_observation",
              "shape_kind": "random_filled",
              "r_row": r_row, "r_col": r_col, "aspect": r_col / r_row,
              "stroke_sigma": sigma, "n_modes": n_modes,
              "mode_orders": ",".join(str(int(k)) for k in orders),
              "template_cols": t_cols, "start_channel": start_channel},
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# family, params dataclass, generator, sampler, auxiliary params (or None).
#
# ORDER IS LOAD-BEARING and new entries go at the END. `pipeline_sensitivity.py`
# seeds each site with `100000 * m_idx`, where `m_idx` is the position in the
# list it was given; reordering the first four would stop them reproducing the
# `morphologies_v2` sweep already on disk.
#
# The 2D-template morphologies borrow NarrowbandDriftingGenerator purely for its
# seeded rng, frame geometry and SNR calibration — none of its path/profile
# machinery is reached. `narrowband_pulsed` takes its geometry from the
# narrowband params and only its pulse *timing* ranges from the wideband block,
# which is what the auxiliary slot is for.
_MORPHOLOGIES = {
    "narrowband_drift": (SetigenMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_drifting, None),
    "narrowband_accel": (SetigenMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_accelerating, None),
    "narrowband_sine": (SetigenMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_sinusoidal, None),
    "wideband_pulsed": (SetigenMorphology, WidebandParams, WidebandPulsedGenerator, _sample_pulsed, None),
    "narrowband_pulsed": (SetigenMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_narrowband_pulsed, WidebandParams),
    "wideband_continuous": (SetigenMorphology, WidebandParams, WidebandPulsedGenerator, _sample_wideband_continuous, None),
    "smiley_face": (DirectArrayMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_smiley, None),
    "random_2d": (DirectArrayMorphology, NarrowbandParams, NarrowbandDriftingGenerator, _sample_random_2d, None),
}

MORPHOLOGIES = tuple(_MORPHOLOGIES)

#: The (frequency extent) x (time structure) factorial. Reporting these four
#: together is what lets a pulsed-signal deficit be attributed to the pulsing
#: rather than to the bandwidth — the two are confounded in any single cell.
FACTORIAL_2X2 = ("narrowband_drift", "narrowband_pulsed",
                 "wideband_continuous", "wideband_pulsed")

#: The 2D-template classes. ON-only like every other morphology (one complete
#: shape per ON observation), so their end-to-end numbers are directly comparable
#: with the rest of the sweep. Grouped only because they share a renderer and a
#: peak-matched amplitude convention.
SHAPE_MORPHOLOGIES = ("smiley_face", "random_2d")


def build_morphology(name: str, data_cfg: dict, seed: Optional[int] = None):
    """Construct a morphology injector by name.

    ``data_cfg`` is the parsed product config (e.g. ``configs/data/gbt_fine.yaml``);
    each morphology reads its geometry and sampling ranges from the block its
    params dataclass owns, so frame geometry stays a single source of truth.
    """
    if name not in _MORPHOLOGIES:
        raise ValueError(
            f"Unknown morphology '{name}'. Available: {', '.join(MORPHOLOGIES)}."
        )
    family, params_cls, gen_cls, sampler, aux_cls = _MORPHOLOGIES[name]
    generator = gen_cls(params_cls.from_config(data_cfg), seed=seed)
    # Samplers that mix two params blocks (geometry from one, ranges from
    # another) read the second here rather than reaching for the config.
    generator.aux_params = aux_cls.from_config(data_cfg) if aux_cls else None
    return family(name, generator, sampler)
