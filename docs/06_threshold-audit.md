# Threshold audit

Why a third of the search was never actually searched, and what it costs to fix.

**Last updated:** 2026-07-21
**Applies to:** the heldout Exotica scan, run `exotica_heldout_topk001`
(364 cadences, production model `20260707_093113_6d0d1ba/epoch=057`)
**Status:** findings 1 and 2 measured on 150/364 cadences, finding 3 on all 364.
Percentages will move; the mechanisms will not.

---

## 1. The two cuts

A cadence is sliced into ~130,000 overlapping frequency windows. The model
scores each one. Two cuts then run in series:

| | What it is | How it is set |
|---|---|---|
| **Stage 1** | Keep the top 1% of snippets by score | `far_thresh = quantile(scores, 0.99)` — **relative** |
| **Stage 3** | Keep those bright on ON rows and dark on OFF rows | `off_ceiling = max(off_noise_ceiling, thresh_5)` — **absolute** |

(Stage 2 is clustering: adjacent above-threshold snippets are merged into one
candidate. It makes no decisions.)

The intended division of labour is clear: stage 1 caps volume, stage 3 decides
what is a candidate. Stage 1 was never meant to be a judge.

Both live in the same units — the scalar score is a top-k reduction over the
same anomaly map that stage 3 thresholds — so they are directly comparable.
That comparison had never been made.

---

## 2. Why we looked

Every benchmark this project has run — `det@3σ`, `topk_frac` sweeps, teacher
comparisons — measures the **scorer**. None measures the **pipeline**. A model
that detects at 98% is irrelevant if a downstream cut discards the detection.

`scripts/pipeline_sensitivity.py` closes that gap: it injects synthetic signals
into real cadences at known SNR and records which stage each one dies at. On a
2-cadence smoke test:

| Cadence | `far_thresh` | `off_ceiling` |
|---|---|---|
| HIP114176 | 0.211 | 0.226 |
| 3C125 | **0.986** | 0.245 |

On HIP114176 the two cuts nearly coincide and nothing is lost. On 3C125 —
a heavily RFI-contaminated cadence — the 1% cut sits **4x above** the ceiling,
and *every* injection died at stage 1, including at SNR 50. Those signals were
morphologically perfect and would have passed stage 3 unanimously. They never
reached it.

That prompted the audit.

---

## 3. Finding 1 — the relative cut is a fixed quota, not a threshold

`scripts/debug/threshold_regime.py`, 150 cadences:

| | |
|---|---|
| `off_ceiling` binds (stage 1 inert) | 108 (72%) |
| **`far_thresh` binds (stage 1 is the gate)** | **42 (28%)** |
| `far_thresh` more than 2x above the ceiling | 24 (16%) |

Ratio `far_thresh / off_ceiling`: median 0.94, p90 **4.99**, max **88.8**.

The worst cases are extreme. On `cad75_MESSIER67_2214MHz` the 1% cut sits at
21.0 against a ceiling of 0.24 — only the most violent RFI enters the candidate
list; anything resembling a real signal is discarded before any ON/OFF
reasoning happens.

**It tracks frequency band, not source.** 2214 MHz appears under two different
targets, 1689 MHz under three. 2401 MHz is the ISM band; 1689 and 2214 MHz are
satellite downlink bands. This is an RFI-environment property, which is why it
was invisible in per-source analyses.

### The underlying reason

`scripts/debug/candidate_volume.py` makes the mechanism unmistakable. Cluster
counts under the current stage 1, across wildly different cadences:

```
695, 695, 742, 681, 583, 567, 742, 650, 611, 499, 409 ...
```

Nearly constant — and necessarily so. A 1% quantile selects 1% of snippets
whatever the data contains. **It is a budget, not a threshold.**

The consequences run both ways, and both are bad:

- **Quiet cadence:** the quota must fill, so it fills with noise. This is the
  failure mode observed during manual review — "the model flags noise snippets,
  and their anomaly maps show random peaks that fool the filter." The peaks are
  real; the quota is what forces them into the list.
- **Noisy cadence:** the quota fills with the strongest RFI, and genuine
  signals are excluded a priori. Invisible without injection testing, because
  no candidate list ever showed what was missing.

An absolute threshold has neither behaviour: nothing above it means nothing
passes.

---

## 4. Finding 2 — the OFF-noise ceiling has never been used

`off_ceiling = max(off_noise_ceiling(OFF cells), thresh_5)`.

On **150 cadences out of 150**, the winning term is `thresh_5`.

`off_noise_ceiling` is the robust, data-driven estimate built in July
([`04_candidate-filtering.md`](04_candidate-filtering.md) §3) specifically to
replace a Gaussian threshold that sat far below the real OFF noise floor. It is
measured per cadence from the OFF rows, which carry no target signal by
construction.

`thresh_5` is `median + 5·MAD_sigma` of the score distribution — a Gaussian
statistic, added as a safety floor after a single narrowband Voyager file
produced an anomalously low ceiling
([`04_candidate-filtering.md`](04_candidate-filtering.md) §3.1).

The floor wins everywhere. So the sophisticated measurement is inert, and the
entire search is in fact gated by the Gaussian threshold the design set out to
abandon, reintroduced through the back door as a guard rail.

Two readings, not yet distinguished:

1. **The floor is too high** — it was tuned on one narrowband file and does not
   generalise to full-band cadences.
2. **The ceiling is too low** — it takes the 0.999 quantile of a 3σ-clipped
   core, and that clipping may be removing genuine noise tail.

Resolving this is cheap: the probe maps saved by
`scripts/recompute_anomaly_maps.py` contain everything needed, offline.

### Pooling is not the issue

Production pools OFF cells from candidate maps *and* background probes.
`recover_off_ceiling.py` and `threshold_regime.py` pool differently. Checked
explicitly: median ratio between the two poolings is **1.003**, and **0/150**
cadences change verdict. The definitions can be unified without consequence.

---

## 5. Finding 3 — the fix is affordable

The stated fix follows directly from §1: **stage 1 must never be stricter than
stage 3.**

```
stage-1 threshold  =  min(far_thresh, off_ceiling)
```

Nothing that could survive the ON/OFF filter is discarded before reaching it.
Where `far_thresh` is already below the ceiling — two thirds of cadences —
nothing changes at all.

`scripts/debug/candidate_volume.py`, all 364 cadences:

| | |
|---|---|
| Cadences affected | 119 (32.7%) |
| Total clusters, now | 280,193 |
| Total clusters, fixed | 412,182 (**×1.47**) |
| Per-cadence multiplier, affected only | median 3.1, p90 6.2, max 13.7 |
| New clusters per affected cadence | median 1,251, p90 2,939, max 8,475 |

A 47% increase overall, concentrated entirely in the third of cadences where
the search is currently not happening. That is not added noise — it is
recovered bandwidth.

### The population also improves

The score is the mean of the hottest cells of the anomaly map. If the score
exceeds the ceiling, at least some map cells must exceed it too. So every newly
admitted candidate has at least one cell above threshold.

The candidates being displaced have the opposite property. Of the 280,193
current clusters, **123,282 (44%) have zero ON-row cells above the ceiling** —
they are in the top 1% by score yet cannot pass stage 3 under any
circumstances. They exist only because the quota had to be filled.

**Consequence to keep in mind:** the short list will therefore grow by *more*
than 1.47x, since the new candidates are qualitatively more likely to pass. The
2.9% stage-3 admission rate measured on the current population is a lower bound
for the new one.

---

## 6. What is not yet measured

**Manual review load.** Cluster counts are machine numbers; the human budget is
the real constraint. Measurable offline on the 150 cadences that already have
maps: rerun `full_row_hits` at the new threshold and count short-list entries.
No HDD access, no model inference. **This is the open question that could still
sink the fix.**

**Which threshold is correct** (§4). Also offline, from the saved probe maps.

**Where the plot cap should sit.** The 30-plots-per-cadence default is a review
budget, not a filter — 2,452 short-listed candidates were never rendered.
`pipeline_sensitivity.py` reports completeness against caps of 30/50/100/200 so
the trade is explicit. Its smoke test already showed that weak survivors do not
rank near the top: at SNR 15 the 90th percentile of rank was 10 and the 99th
was 14.

---

## 7. Tools

All three read cached arrays only. None touches the HDD-backed observation
files, none loads the model, all run against a partially complete recompute.

| Script | Answers |
|---|---|
| `scripts/recompute_anomaly_maps.py` | Re-derives and caches the three anomaly-map components + probe maps + per-cadence thresholds, so every downstream question becomes an offline array operation. ~11 h once, resumable. |
| `scripts/debug/threshold_regime.py` | Which cut binds, per cadence; which term wins inside `off_ceiling`; pooling sensitivity. |
| `scripts/debug/candidate_volume.py` | Cluster volume under the proposed stage-1 rule, all 364 cadences. |
| `scripts/debug/recover_off_ceiling.py` | Single-cadence and single-candidate breakdown, row by row. |
| `scripts/pipeline_sensitivity.py` | End-to-end injection survival by stage; the measurement that started this. |

---

## 8. Why this matters for the write-up

A null result is only publishable if the search's sensitivity is known. "We
found nothing" with an unmeasured pipeline is an absence of result; with a
measured survival curve it is a quantified upper limit.

Independently of what the Exotica scan turns up, this audit produces a
transferable methodological finding: **a relative quantile is unsuitable as the
first cut of a SETI search.** It cannot adapt to the RFI environment, and its
failure is asymmetric — it fabricates candidates where there is nothing and
suppresses them where the environment is hostile, which is precisely where a
weak signal most needs protection. The failure is silent in both directions and
only becomes visible under injection testing.

---

Related: [`04_candidate-filtering.md`](04_candidate-filtering.md) (§3 threshold
design, §3.1 the Voyager-scale caveat this audit confirms, §5 the short-list
rule) · [`05_results.md`](05_results.md) (scorer-level validation)
