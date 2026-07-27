"""Morphology injectors: ON-only semantics and in-band containment.

The containment tests are the load-bearing ones. setigen's non-linear paths take
a coefficient evaluated over the cadence's absolute timeline, so an
under-constrained parameterisation sweeps the signal straight out of the window
and every downstream sensitivity number silently becomes "no signal present"
rather than "signal not detected" — a failure that looks exactly like a real
negative result. See the module docstring of src/data/morphologies.py.
"""

import numpy as np
import pytest
import yaml

from src.data.morphologies import CANVAS_MORPHOLOGIES, MORPHOLOGIES, build_morphology

ON, OFF = (0, 2, 4), (1, 3, 5)

# The canvas morphologies (smiley_face, random_2d) are painted across all six
# observations, so they appear in the OFF frames BY DESIGN and are excluded from
# the ON-only contract below. That is not a relaxation of the contract — it is a
# different one, checked in its own tests further down. See
# src/data/morphologies.py::DirectArrayMorphology for why the canvas geometry is
# the only one the (6,64) anomaly map can resolve.
ON_ONLY_MORPHOLOGIES = tuple(m for m in MORPHOLOGIES if m not in CANVAS_MORPHOLOGIES)


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


@pytest.mark.parametrize("name", MORPHOLOGIES)
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


@pytest.mark.parametrize("name", MORPHOLOGIES)
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


def _occupancy(name, data_cfg, background, seeds=range(8)):
    """Median (occupied channels, occupied time rows) of an ON observation.

    "Occupied" is measured at 10% of the injection's own peak, so a gaussian
    frequency profile's tails do not count as extent. Medianed over seeds
    because both axes are sampled per site.
    """
    cols, rows = [], []
    for seed in seeds:
        inj = build_morphology(name, data_cfg, seed=seed)
        site = inj.sample_site(fchans=1024, total_tchans=96)
        out, _ = inj.inject(background, site, snr=50.0)
        excess = out[ON[0]] - background.astype(np.float32)[ON[0]]
        mask = excess > 0.1 * excess.max()
        cols.append(int(mask.any(axis=0).sum()))
        rows.append(int(mask.any(axis=1).sum()))
    return float(np.median(cols)), float(np.median(rows))


def test_factorial_separates_frequency_extent_from_time_structure(data_cfg, background):
    """The 2x2 must actually vary the two axes independently.

    If it does not, a pulsed-signal deficit cannot be attributed: the existing
    single ``wideband_pulsed`` cell varies both at once, which is the reason
    these two arms were added. Guarding the factorial here means a later tweak
    to either sampler cannot silently collapse it back into one cell.
    """
    nb_cont_cols, nb_cont_rows = _occupancy("narrowband_drift", data_cfg, background)
    nb_puls_cols, nb_puls_rows = _occupancy("narrowband_pulsed", data_cfg, background)
    wb_cont_cols, wb_cont_rows = _occupancy("wideband_continuous", data_cfg, background)
    wb_puls_cols, wb_puls_rows = _occupancy("wideband_pulsed", data_cfg, background)

    # Frequency axis: wideband occupies far more channels, at both time structures.
    assert wb_cont_cols > 10 * nb_cont_cols
    assert wb_puls_cols > 10 * nb_puls_cols
    # Time axis: a pulse train leaves gaps, a continuous signal does not.
    assert nb_cont_rows == background.shape[1]
    assert wb_cont_rows == background.shape[1]
    assert nb_puls_rows < nb_cont_rows
    assert wb_puls_rows < wb_cont_rows


@pytest.mark.parametrize("name", CANVAS_MORPHOLOGIES)
def test_canvas_shape_reaches_every_observation(name, data_cfg, background):
    """A canvas shape must cross the observation boundaries, OFF frames included.

    This is the opposite of the ON-only contract, and it is deliberate: a shape
    confined to one observation occupies exactly one row of the (6,64) anomaly
    map, so the model cannot resolve its vertical structure at all. If this test
    ever passes only on the ON rows, the shape has silently gone back to being a
    bandwidth test wearing a morphology's name.
    """
    inj = build_morphology(name, data_cfg, seed=7)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, _ = inj.inject(background, site, snr=30.0)

    expected = background.astype(np.float32)
    for i in range(background.shape[0]):
        peak = float((out[i] - expected[i]).max())
        assert peak > 0, f"{name}: observation {i} received no power"


@pytest.mark.parametrize("name", CANVAS_MORPHOLOGIES)
def test_canvas_rows_are_not_all_identical(name, data_cfg, background):
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


@pytest.mark.parametrize("name", CANVAS_MORPHOLOGIES)
def test_energy_matches_equivalent_carrier(name, data_cfg, background):
    """Total injected power over the canvas = intensity * total_tchans.

    This is the declared SNR convention (module docstring). If it silently became
    per-pixel, an extended shape would inject orders of magnitude more power than
    the narrowband class it is compared against, and the cross-morphology
    survival curves would be meaningless.
    """
    inj = build_morphology(name, data_cfg, seed=11)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    out, info = inj.inject(background, site, snr=25.0)

    total_tchans = background.shape[0] * background.shape[1]
    added = float((out - background.astype(np.float32)).sum())
    assert added == pytest.approx(info["intensity"] * total_tchans, rel=1e-3)


@pytest.mark.parametrize("name", CANVAS_MORPHOLOGIES)
@pytest.mark.parametrize("seed", range(24))
def test_canvas_template_fits_the_block(name, data_cfg, seed):
    """No clipping at the canvas edge or the band edge, for any draw."""
    inj = build_morphology(name, data_cfg, seed=seed)
    site = inj.sample_site(fchans=1024, total_tchans=96)
    template = site.payload["template"]

    assert template.shape[0] == 96
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
    # A random blob's Fourier perturbation can pull its vertical radius in, so the
    # bar is "more than one map row" (16), not the nominal height.
    lit = np.where(template.max(axis=1) > 0.1)[0]
    assert (lit[-1] - lit[0]) > 24, f"{name}: spans ~1 anomaly-map row at seed {seed}"


def test_unknown_morphology_names_alternatives(data_cfg):
    with pytest.raises(ValueError, match="narrowband_drift"):
        build_morphology("does_not_exist", data_cfg)
