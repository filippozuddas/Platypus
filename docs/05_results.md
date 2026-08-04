# Results and current limitations

Validation results for the production UDMA scorer, and an honest list of what is
known to be wrong or unverified.

**Last updated:** 2026-07-27
**Model:** [`02_udma-architecture.md`](02_udma-architecture.md) ·
**Bars:** §9 of that document, pre-registered before any run

---

## 1. Acceptance bars — all passed

Full training run, ~560k real cadence snippets, 56/9/28 split. Evaluated on 3
independent seeds, on both train and never-seen validation.

| Bar | Pre-registered criterion | Result | |
|---|---|---|---|
| **B1** matched-energy separation | AUC ≥ 0.80 on all seeds and above the prior probe (0.770–0.788) | **0.884 / 0.887 / 0.901** | ✅ +0.10–0.12 |
| **B2** mid-SNR detection | TPR@10%FP ≥ 70% @ SNR 12, ≥ 40% @ SNR 10 | **100% / 100%** (probe: 35–60%, 4–15%) | ✅ |
| **B3** low-SNR detection | TPR@10%FP > 0% at SNR 5 and 7 (probe was 0%) | **68–92% @ SNR 5, 97–100% @ SNR 7** | ✅ |
| **B4** generalization | val AUC within ±0.03 of train | Δ = 0.000–0.024 | ✅ |
| **B5** harness sanity | energy alone must not already win | margin ≥ +0.30 over trivial | ✅ |

All five bars passed, on all three seeds, on both splits. B3 is the one that
mattered most: the low-SNR operating point is precisely what every prior scorer
failed to reach, and what UDMA was built to unlock.

UDMA therefore replaces both the reconstruction scorers (AE / MemAE / ViT-MAE)
and the pixel-space disagreement probe as the production scorer.

## 2. Qualitative behaviour

**Anomaly maps localize.** On pure noise, real RFI, and injected signal, the
`(6, 64)` maps put the peak at the true position — the RFI line on its actual
time interval, the injected signal on the correct time/frequency cell. The
global-attention smearing accepted as a design risk did **not** materialize. On
a broadband RFI feature that drifts along a curve, the map peak *follows the
curve*, confirming the mechanism captures non-trivial morphology rather than a
static hot pixel.

**Known RFI is suppressed.** A narrowband line present continuously through a
cadence is almost entirely suppressed by the students, which saw it often in
training as normal; residual survives only where the line departs from the
learned pattern. This is the intended "suppress the known, flag the unexpected"
behaviour.

**Blind to ON/OFF occupancy, by construction.** UDMA does not distinguish
ON-only from persistent signals (AUC ~0.49, chance). This is correct: there is
no cadence objective in training, and ON/OFF discrimination is the downstream
stage's job ([`04_candidate-filtering.md`](04_candidate-filtering.md)).

## 3. Injection-recovery sensitivity

10 real cadences, 1440 injections. The top-k score gives a smooth, monotonic
detection curve:

| SNR | 3 | 15 | 20 | 50 |
|---|---|---|---|---|
| Detection | 2.8% | 64% | 89% | 100% |

Mean-based scoring is structurally weaker at every SNR, confirming that grid
dilution is a real effect even on a 384-position grid — this is why `topk` is
the production aggregation.

Two measurement caveats worth knowing, both discovered later:

- Early weak injection-recovery numbers were a **pooled-threshold artifact**, not
  a model problem. Between-cadence variance dominates (η² 0.94 for topk), so a
  threshold pooled across cadences is invalid. With a per-cadence threshold the
  detection curve recovers to 21/64/82/94/98% at SNR 10/15/20/30/50.
- Detection rates measured against a quiet-only baseline mislead badly at small
  n: one arm read 100% at n=50 and **0% at n=500**. Sample ≥500 per stratum.

## 4. Does the architecture earn its complexity?

Component attribution at matched energy against **real RFI**: UDMA beats trivial
statistics by **+0.07–0.09 AUC**. The memory unit earns its place.

Two caveats that matter for interpretation:

- Against **clean** negatives the comparison gives the opposite answer. Only
  matched-energy-versus-real-RFI is a meaningful benchmark here — against empty
  sky, any brightness detector looks good.
- Equal-weight fusion **dilutes `map_ss`**, which is by far the strongest single
  term at SNR 10. Weight tuning is an open, unexploited lever.

## 5. Search on real data

A full inference pass over the Exotica `0000.h5` held-out set (364 cadences) has
been run. Short-list behaviour on 280k held-out candidates:

- **73% are rejected at the `off_ceiling` gate before the ON/OFF logic engages.**
  The gate, not the ON/OFF rules, is the binding constraint on the false-alarm
  rate for the majority of cadences. Two caveats, both from the full 364-cadence
  audit in [`06_threshold-audit.md`](06_threshold-audit.md): the gate's value is
  always set by the `thresh_5` floor, never by `off_noise_ceiling()` (0/364), so
  the OFF-noise machinery is inert in production; and on 32.7% of cadences the
  top-1% `far_thresh` pre-cut sits *above* the gate (median 3.1x, max 162x) and
  is itself the binding, signal-killing cut.
- The accepted short list has a median ON/OFF contrast of only **1.21** — weak.

A real-signal control is available: on a Voyager-1 cadence the pipeline ranks
the true carrier first, with an ON/OFF contrast of ~32 and a clean 3/3 short
list with no noise entries.

## 6. Morphology sensitivity — what the search is complete to

`outputs/sensitivity/morphologies_v7`, 2026-07-27. 20 held-out cadences × 25
injection sites × 8 signal classes × 6 SNR levels (including a 0 control),
n = 500 per class per SNR. Production model frozen at `6d0d1ba/epoch=057`,
`--preproc_mode per_obs`. Sites are drawn once per cadence and **shared across
classes**, so the comparison between rows is paired. Every class is injected
ON-only; the two 2D-shape classes place one complete copy of the shape inside
each ON observation, three per cadence.

This is the first measurement of the *pipeline* rather than the *scorer*: every
earlier benchmark stopped at "does the score clear a threshold".

**Completeness at SNR 30**, as a percentage of injections:

| class | model (`s0`) | reaches a human | with the [`06`](06_threshold-audit.md) fix |
|---|---|---|---|
| `smiley_face` | 100.0 | 79.2 | **99.2** |
| `random_2d` | 100.0 | 75.8 | **95.4** |
| `narrowband_accel` | 99.8 | 80.0 | **99.0** |
| `narrowband_drift` | 98.4 | 78.6 | **96.4** |
| `narrowband_sine` | 99.2 | 71.2 | 87.6 |
| `wideband_pulsed` | 96.8 | 68.4 | 82.6 |
| `narrowband_pulsed` | 90.6 | 70.0 | 83.4 |
| `wideband_continuous` | 60.0 | 36.2 | 43.8 |

At SNR 15, end-to-end as shipped: `random_2d` 80.0 · `smiley_face` 79.0 ·
`narrowband_sine` 59.4 · `narrowband_accel` 55.0 · `wideband_pulsed` 53.8 ·
`narrowband_drift` 44.8 · `wideband_continuous` 23.6 · `narrowband_pulsed` 9.2.

Four findings, in order of consequence.

**6.1 The FAR 1% pre-cut caps completeness, and on some cadences it is total.**
The 2D-shape classes make this unmissable: the model detects them **100%** of the
time at SNR 20, 30 and 50, and stage 1 passes **exactly 80.0%** at all three.
80.0% of 500 injections is 400 — sixteen cadences out of twenty. On the other
**four cadences a signal the model always sees never enters the candidate list
at any SNR**, because RFI has already filled the 1% budget. This is not a curve
saturating; it is a subset of the survey where the search is blind by
construction, consistent with the 32.7% found across all 364 cadences in
[`06_threshold-audit.md`](06_threshold-audit.md). Across the other classes stage 1
lands at 78–84% at SNR 50 regardless of morphology; for `narrowband_accel` the
end-to-end figure *equals* stage 1 exactly (83.8 = 83.8). Replacing the quota with
an absolute threshold recovers 13–20 points on every class.

**6.2 The effect of pulsing depends on bandwidth, and changes sign.** The 2×2 in
(frequency extent) × (time structure), `s0` at SNR 30:

| | continuous | pulsed | effect of pulsing |
|---|---|---|---|
| narrowband | 98.4 | 90.6 | −7.8 |
| wideband | 60.0 | 96.8 | **+36.8** |

A wide-band beacon is detected *better* when it blinks, despite a pulse train
carrying roughly `duty` (~0.4) of the integrated energy of a continuous signal at
the same nominal SNR. `wideband_continuous` is the worst cell of the four: a wide
steady band is smooth and the model reconstructs it. **The detector responds to
structure, not to energy** — the same mechanism documented in
[`01_scoring-history.md`](01_scoring-history.md). A single pulsed cell cannot
show this; the factorial exists because the two axes were confounded.

**6.3 The `off_leak` losses belong to spatially extended responses.** Stage 3
completeness peaks at SNR 30 and falls at SNR 50 for `narrowband_sine`
(88.0 → 76.0), `wideband_pulsed` (82.6 → 76.6) and `wideband_continuous`
(44.6 → 43.2). It is flat for `drift`, `accel` and `narrowband_pulsed` — the last
has `s2` = `s3` exactly at every SNR, i.e. zero leak. The discriminant is neither
pulsing nor nominal bandwidth but how *spatially extended* the map response is:
`narrowband_sine` is narrowband yet wanders over 5–20% of the band and behaves
like a wideband class.

The shape classes provide the controlled version of that claim, since they differ
from each other in nothing but fill: the stroke-drawn `smiley_face` loses 0.8
points at stage 3 (100.0 → 99.2), the filled `random_2d` loses 4.6 (100.0 → 95.4).
Same renderer, same geometry, same sites, same cadences.

**6.4 The scorer handles arbitrary morphology as well as a carrier — but the map
does not represent it.** `smiley_face` and `random_2d` reach `s0` = 100% from
SNR 20 upward, above every track-like class. That is the project's central claim
measured directly.

Two things must be said with it. First, the shapes are **peak-matched**, so at
equal nominal SNR they carry more total power than a carrier — the number is not
a like-for-like comparison against `narrowband_drift`. Second, and the reason the
result still stands: `wideband_continuous` lights roughly **ten times more pixels**
than a shape (≈300 channels × 16 rows against ≈40 × 10) and scores 69.0 against
100.0 at SNR 50, 42.2 against 96.8 at SNR 15. With an order of magnitude more
energy it does far worse. Extent is not what earns detection; structure is.

What the model does *not* do is characterise the shape. In the anomaly maps a
smiley and a drifting carrier produce the same object — a narrow bright stripe in
each ON row. One map cell is 16 rows × 16 channels and one observation is exactly
16 rows, so a shape inside an observation occupies a single map row and there is
no representational room for its geometry. **The pipeline detects arbitrary
morphology; it cannot report it.**

Three things these numbers are not:

- **SNR is not commensurable across classes.** setigen's profiles are
  unit-height, so a wideband signal at "the same SNR" has a carrier's per-pixel
  level while carrying hundreds of times its power; the 2D shapes are
  peak-matched for consistency with that. Compare the *shape* of a class's curve
  and its within-class SNR trend, never absolute levels between two classes.
- **The parameter curves in the companion figures are observational.** Morphology
  parameters are sampled per site, not gridded, so a bin's occupants differ in
  every other parameter too.
- **Review budget is not binding.** Completeness is identical at plot caps
  30/50/100/200, and the median rank of a surviving injection is 0.

### 6.5 A methodological note worth keeping

Three separate results in this section were wrong on first measurement, all in
the same way: **the harness was measuring its own parameterization, not the
model.** 2D shapes sized to one observation *and* scaled to equal total energy
sat at ~0.1 sigma per pixel and scored 6.4% while measuring nothing; pulse timing
inherited from a config block tuned for a product with 17× finer time resolution
put 1–2 pulses per observation at the sampling limit, scoring 2.6% where the
corrected regime scores 18.0%; and a test measured linewidth as the union over
time bins, which for a drifting signal is the whole track.

Each produced a low, clean, entirely credible number. `scripts/debug/preview_morphology_templates.py`
exists because of the first: render what the injector actually injects, and look
at it, before spending hours on a sweep. A canvas-wide shape variant was also
built and swept before being reversed on the mentor's instruction that a shape
must be complete inside each ON observation — that one was a scoping decision
rather than an error, but it cost a sweep too.

## 7. Known limitations

Listed because they are real, not because they are resolved.

**Short-list threshold does not scale.** The `thresh_5` floor was validated at
Voyager scale (~2046 snippets per cadence, ~15% band coverage). At full-band
Exotica scale (~1000 clusters per cadence) the multiple-comparisons burden is far
higher and false positives reappear. Fixable post-hoc from the saved CSVs
without re-running the scan. See
[`04_candidate-filtering.md`](04_candidate-filtering.md) §3.1.

**`off_ceiling_probe` default is mistuned for production.** The default of 300
was set on the narrowband Voyager file; on full-band Exotica cadences it covers
only ~0.2% of the band. Production runs should use ~2000–3000.

**Detection and localization are won by different models.** The distilled CNN
teacher detects better (93.3/98.4/100.0 at SNR 15/20/30 vs 80.9/94.7/99.8) but
does not localize; the domain-matched teacher localizes but detects worse. There
is currently no single checkpoint that wins both. See
[`03_teacher-localization.md`](03_teacher-localization.md) §5.

**One historical benchmark is unreproducible.** A control re-run with verified
identical checkpoint, config, and code produced 80.9/94.7/99.8% where the
recorded figures were 48.9/68.0/79.3%. The only unverifiable variable is the
exact cadence list of the original run, whose log was lost. The new numbers are
canonical; the old ones should not be cited.

**Patch-geometry refinement is deferred**, documented rather than fixed. §6.4
gives it a concrete cost, and a narrower one than expected: one anomaly-map cell
is 16 rows × 16 channels and one observation is exactly 16 rows, so structure
confined to a single observation occupies exactly one map row. That does **not**
stop detection — the shape classes score `s0` = 100% — but it removes any room to
represent what was detected. The pipeline can flag arbitrary morphology and
cannot report it, which matters for what a candidate table is allowed to claim,
and would be the first thing a finer patch grid buys.

## 8. Not yet addressed

- **`0001.h5` (broadband transients)** — the natural next extension of the
  "broader morphology" goal in the project specification, once `0000.h5` is
  consolidated. Generators and DM-sweep sizing exist; no model has been trained.
- **`0002.h5` (wideband / modulated)** — frame geometry still to be determined.
- **turboSETI cross-comparison** — the optional final comparison against
  traditional narrowband SETI, and goal #5 of the project specification.
