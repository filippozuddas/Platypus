"""Morphology injectors: ON-only semantics and in-band containment.

The containment tests are the load-bearing ones. setigen's non-linear paths take
a coefficient evaluated over the cadence's absolute timeline, so an
under-constrained parameterisation sweeps the signal straight out of the window
and every downstream sensitivity number silently becomes "no signal present"
rather than "signal not detected" — a failure that looks exactly like a real
negative result. See the module docstring of src/data/morphologies.py.

All morphologies share one contract: ON-only injection, OFF observations
returned bit-identical. The 2D-template classes additionally have to fit a
complete copy of the shape inside a single observation — see
``SHAPE_MORPHOLOGIES`` and the tests at the bottom.
"""

import numpy as np
import pytest
import yaml

from src.data.morphologies import MORPHOLOGIES, SHAPE_MORPHOLOGIES, build_morphology

ON, OFF = (0, 2, 4), (1, 3, 5)

# Every morphology is ON-only, including the two 2D-template classes: Vishal
# ruled (2026-07-27) that a shape must be complete inside each ON observation,
# three per cadence, never split across an ON/OFF boundary. A canvas-wide variant
# existed briefly and is gone.
ON_ONLY_MORPHOLOGIES = MORPHOLOGIES


@pytest.fixture(scope="module")
def data_cfg():
    with open("configs/data/gbt_fine.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def background():
    """Flat unit background: any excess is unambiguously injected signal."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=10.0, scale=0.1, size=(6, 16, 1024))


@pytest.mark.parametrize("name", ON_ONLY_MORPHOLOGIES)
def test_off_observations_untouched(name, data_cfg, background):
    """OFF observations must come back bit-for-bit, as a real signal would vanish.

    Compared in float32 because the injector returns float32: against a float64
    background every element differs by ~5e-7 from the cast alone, which is not
    signal. Asserting in the returned dtype keeps this an exact-equality test
    rather than a tolerance that could hide a genuinely leaking injection.
    """
    inj = build_morphology(name, data_cfg, seed=1)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    expected = background.astype(np.float32)
    for i in OFF:
        np.testing.assert_array_equal(out[i], expected[i])


@pytest.mark.parametrize("name", ON_ONLY_MORPHOLOGIES)
def test_signal_lands_in_band(name, data_cfg, background):
    """Every ON observation must actually receive power.

    Not just the cadence as a whole: a non-linear track that leaves the band
    partway through would still add energy to the first ON block while being
    absent from the last, which is the exact silent failure this guards.
    """
    inj = build_morphology(name, data_cfg, seed=1)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    # Peak excess, not summed excess: summing 16384 float32 cells carries ~0.02
    # of accumulation error, a noise floor comparable to a faint injection. A
    # single-pixel maximum has no such floor.
    expected = background.astype(np.float32)
    for i in ON:
        peak = float((out[i] - expected[i]).max())
        assert peak > 0, f"{name}: ON observation {i} received no power"


@pytest.mark.parametrize("name", ON_ONLY_MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(48))
def test_in_band_across_seeds(name, data_cfg, background, seed):
    """Containment must hold for the whole sampled parameter range, not one draw.

    The excursion/amplitude fractions are sampled per site, so a single seed
    exercises one point in that range; the sweep will draw hundreds.
    """
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=50.0)

    expected = background.astype(np.float32)
    for i in ON:
        peak = float((out[i] - expected[i]).max())
        assert peak > 0, f"{name}: ON observation {i} empty at seed {seed}"


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_morphology_frozen_across_snr(name, data_cfg, background):
    """Amplitude is the only thing an SNR sweep may vary.

    Re-injecting the same Site at two SNRs must move the same pixels: if the
    shape were re-sampled per SNR, the sweep would confound amplitude with
    morphology and its curve would be uninterpretable.
    """
    inj = build_morphology(name, data_cfg, seed=3)
    site = inj.sample_site(fchans=1024, total_tchans=96)

    lo, _ = inj.inject(background, site, snr=10.0)
    hi, _ = inj.inject(background, site, snr=40.0)

    expected = background.astype(np.float32)
    support_lo = (lo[list(ON)] - expected[list(ON)]) > 1e-9
    support_hi = (hi[list(ON)] - expected[list(ON)]) > 1e-9
    # The louder injection may light up marginally more of the profile tail, so
    # require containment rather than equality.
    assert support_lo.sum() > 0
    assert np.all(support_hi[support_lo])


@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_info_is_traceable(name, data_cfg, background):
    """Every row of the results CSV must identify the shape that produced it."""
    inj = build_morphology(name, data_cfg, seed=5)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    _, info = inj.inject(background, site, snr=20.0)

    assert info["morphology"] == name
    assert info["snr"] == 20.0
    assert "path" in info and "start_channel" in info


PULSED_MORPHOLOGIES = ("narrowband_pulsed", "wideband_pulsed")


@pytest.mark.parametrize("name", PULSED_MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(24))
def test_pulse_train_stays_resolvable(name, data_cfg, background, seed):
    """Pulses must be resolved by the product's time sampling, and be plural.

    The failure this guards is not a crash. Inheriting ``WidebandParams``' period
    range — tuned for a product with 1.07 s bins — gave 1-2 pulses per 16-bin
    observation and pulse widths down to a single 18.25 s bin, i.e. a gaussian
    evaluated at one point. Detection then depended on where the pulse centre fell
    relative to a bin centre, and `narrowband_pulsed` scored 2.6% at SNR 15 for
    reasons that had nothing to do with the model. A sweep in that regime looks
    perfectly healthy and reports a number that means nothing.
    """
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    _, info = inj.inject(background, site, snr=30.0)

    assert info["pulse_width_bins"] >= 1.5, "pulse narrower than the sampling limit"
    assert info["pulses_per_obs"] >= 2.0, "fewer than 2 pulses per ON observation"
    assert 0.25 <= info["duty"] <= 0.85, f"duty {info['duty']:.2f} out of range"


def test_pulsed_cells_of_the_factorial_share_their_timing_distribution(
        data_cfg, background):
    """The 2x2's two pulsed cells must differ in bandwidth and nothing else.

    The contract is on the *distribution*, not on individual sites: the two
    samplers consume different numbers of random draws before reaching the timing
    (one runs the full narrowband parameter sampler first, the other a bandwidth
    draw), so the same seed gives different realised periods — and that is fine,
    because the sweep averages over 500 sites per cell. What must not happen is
    the two drawing from different ranges, which would make every
    narrowband-vs-wideband comparison also a comparison of pulse periods.
    """
    stats = {}
    for name in PULSED_MORPHOLOGIES:
        pulses, duties = [], []
        for seed in range(40):
            inj = build_morphology(name, data_cfg, seed=seed)
            site = inj.sample_site(fchans=1024, total_tchans=96)
            _, info = inj.inject(background, site, snr=30.0)
            pulses.append(info["pulses_per_obs"])
            duties.append(info["duty"])
        stats[name] = (float(np.mean(pulses)), float(np.mean(duties)),
                       min(pulses), max(pulses))

    nb, wb = stats["narrowband_pulsed"], stats["wideband_pulsed"]
    assert nb[0] == pytest.approx(wb[0], rel=0.2), f"pulses/obs differ: {stats}"
    assert nb[1] == pytest.approx(wb[1], rel=0.2), f"duty differs: {stats}"
    # Both must exercise the declared range rather than a corner of it.
    for name, (_, _, lo, hi) in stats.items():
        assert lo < 3.0 and hi > 5.0, f"{name} covers only {lo:.1f}-{hi:.1f}"


def _occupancy(name, data_cfg, background, seeds=range(8)):
    """Median (instantaneous channel width, temporal modulation) of an ON observation.

    The channel count is taken from the single brightest time row, not from the
    union over time. For a drifting signal the union is the whole track — at
    ``drift_median`` 0.3 Hz/s over a 292 s observation that is ~31 channels, an
    order of magnitude more than the ~2-channel line — so a union-based metric
    measures drift rate, not frequency extent, and cannot express the factorial's
    frequency axis at all. It also made this test's verdict depend on the pulse
    timing: a train that lit only 1-2 time bins per observation swept fewer
    channels and so looked "narrower" for reasons unrelated to bandwidth.

    Time structure is a modulation index (std/mean of the per-row peak power)
    rather than a count of occupied rows. At the top of the sampled range — 6
    pulses in a 16-bin observation at duty 0.6 — the gaps between pulses are
    narrower than one bin, so "rows above a threshold" saturates at 16 for a pulse
    train too, and a count cannot express that axis either.

    Widths are measured at 10% of the relevant peak, so a gaussian frequency
    profile's tails do not count as extent. Medianed over seeds because both axes
    are sampled per site.
    """
    cols, mods = [], []
    for seed in seeds:
        inj = build_morphology(name, data_cfg, seed=seed)
        site = inj.sample_site(fchans=1024, total_tchans=96)
        out, _ = inj.inject(background, site, snr=50.0)
        excess = out[ON[0]] - background.astype(np.float32)[ON[0]]
        row_peaks = excess.max(axis=1)
        brightest = excess[int(np.argmax(row_peaks))]
        cols.append(int((brightest > 0.1 * brightest.max()).sum()))
        mods.append(float(row_peaks.std() / max(row_peaks.mean(), 1e-12)))
    return float(np.median(cols)), float(np.median(mods))


def test_factorial_separates_frequency_extent_from_time_structure(data_cfg, background):
    """The 2x2 must actually vary the two axes independently.

    If it does not, a pulsed-signal deficit cannot be attributed: the existing
    single ``wideband_pulsed`` cell varies both at once, which is the reason
    these two arms were added. Guarding the factorial here means a later tweak
    to either sampler cannot silently collapse it back into one cell.
    """
    nb_cont_cols, nb_cont_mod = _occupancy("narrowband_drift", data_cfg, background)
    nb_puls_cols, nb_puls_mod = _occupancy("narrowband_pulsed", data_cfg, background)
    wb_cont_cols, wb_cont_mod = _occupancy("wideband_continuous", data_cfg, background)
    wb_puls_cols, wb_puls_mod = _occupancy("wideband_pulsed", data_cfg, background)

    # Frequency axis: wideband is far wider instantaneously, at both time structures.
    assert wb_cont_cols > 10 * nb_cont_cols, f"{wb_cont_cols} vs {nb_cont_cols}"
    assert wb_puls_cols > 10 * nb_puls_cols, f"{wb_puls_cols} vs {nb_puls_cols}"
    # Time axis: a pulse train modulates its amplitude, a continuous signal does
    # not. The margin is modest rather than large because the continuous
    # narrowband class samples a scintillating time profile half the time, which
    # puts a floor on its own modulation — the axis is still separated, just not
    # by an order of magnitude.
    assert nb_puls_mod > 1.3 * nb_cont_mod, f"{nb_puls_mod:.3f} vs {nb_cont_mod:.3f}"
    assert wb_puls_mod > 1.3 * wb_cont_mod, f"{wb_puls_mod:.3f} vs {wb_cont_mod:.3f}"
    assert nb_puls_mod > 0.25 and wb_puls_mod > 0.25


@pytest.mark.parametrize("name", SHAPE_MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(12))
def test_shape_is_complete_in_every_on_observation(name, data_cfg, background, seed):
    """One whole copy of the shape per ON observation, identical in all three.

    Vishal, 2026-07-27: *"one smiley face per ON, so a total of three of them on
    all three"* — not one shape split across the cadence. A version that differed
    between ON observations, or that leaked into an OFF, would be a different
    experiment wearing the same name.
    """
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=30.0)

    excess = [out[i] - background.astype(np.float32)[i] for i in ON]
    for other in excess[1:]:
        np.testing.assert_allclose(excess[0], other, rtol=0, atol=1e-5)
    assert float(excess[0].max()) > 0


@pytest.mark.parametrize("name", SHAPE_MORPHOLOGIES)
def test_shape_rows_are_not_all_identical(name, data_cfg, background):
    """Vertical structure must actually exist — the point of the canvas geometry.

    A shape whose rows were all equal would be a vertical bar, indistinguishable
    from a wideband carrier, and every "shape" number would be measuring
    bandwidth. Comparing the top and middle bands of the canvas is the cheapest
    check that the rasteriser is drawing a 2D object.
    """
    inj = build_morphology(name, data_cfg, seed=7)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    template = site.payload["template"]

    per_row_mass = template.sum(axis=1)
    assert per_row_mass.std() > 0.05 * per_row_mass.mean()


@pytest.mark.parametrize("name", SHAPE_MORPHOLOGIES)
def test_amplitude_follows_the_declared_convention(name, data_cfg, background):
    """The SNR axis must mean what ``info["snr_convention"]`` says it means.

    Both conventions are legitimate questions but they differ by a factor of the
    template's pixel count — thousands. A silent switch would move every shape
    number by orders of magnitude while the CSV column still read the same, which
    is the worst kind of error to make in a comparison table.
    """
    inj = build_morphology(name, data_cfg, seed=11)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, info = inj.inject(background, site, snr=25.0)

    excess = out - background.astype(np.float32)
    total_tchans = background.shape[0] * background.shape[1]
    if info["snr_convention"] == "peak":
        assert float(excess.max()) == pytest.approx(info["intensity"], rel=1e-3)
    else:
        assert float(excess.sum()) == pytest.approx(
            info["intensity"] * total_tchans, rel=1e-3)


@pytest.mark.parametrize("name", SHAPE_MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(24))
def test_shape_template_fits_one_observation(name, data_cfg, seed):
    """No clipping at the observation edge or the band edge, for any draw.

    An ON observation is 16 time bins. A shape sampled taller would be truncated
    into a pair of bars — a different morphology than the CSV claims.
    """
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    template = site.payload["template"]

    assert template.shape[0] == 16
    assert 0 <= site.payload["start_channel"]
    assert site.payload["start_channel"] + template.shape[1] <= 1024
    # Structure must be interior: mass on the first/last row means the shape was
    # cut off rather than sized to fit.
    assert template[0].max() < 0.5 and template[-1].max() < 0.5
    # An outline's peak depends on how close the curve passes to a pixel centre,
    # so this is a "the shape is actually there" floor, not a normalisation.
    assert template.max() > 0.5
    # It must span several map rows, which is the entire reason for the change:
    # 16 rows = 1 map row, and the old geometry never exceeded that.
    # The shape must use most of the observation it is given: a tiny blob in the
    # middle of 16 rows would be a point source, not a 2D morphology.
    lit = np.where(template.max(axis=1) > 0.1)[0]
    assert (lit[-1] - lit[0]) >= 6, f"{name}: only {lit[-1]-lit[0]} rows tall at seed {seed}"


def test_unknown_morphology_names_alternatives(data_cfg):
    with pytest.raises(ValueError, match="narrowband_drift"):
        build_morphology("does_not_exist", data_cfg)
