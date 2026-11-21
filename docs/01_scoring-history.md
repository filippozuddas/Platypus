# Anomaly scoring — what failed, why, and how UDMA was reached

The single most important piece of context in this repository. Six scorer
families were built and evaluated on GBT `0000.fil`; five failed. This document
records the mechanism behind each failure so they are not re-attempted.

**Related:** [`02_udma-architecture.md`](02_udma-architecture.md) ·
[`03_teacher-localization.md`](03_teacher-localization.md) ·
[`05_results.md`](05_results.md)

---

## 1. The core mechanism

> **Reconstruction error measures predictability, not anomalousness.**

This one sentence explains every failure below. A smooth, predictable signal —
a narrowband line drifting in frequency — can *lower* the mean reconstruction
error below the pure-noise baseline, while a static RFI spike raises it. The
scorer was ranking the wrong thing, and no amount of retraining fixes a scoring
rule.

The corollary, established repeatedly: **the representation is not the
bottleneck.** Supervised probes on the frozen encoder embedding reach AUC
0.75–0.85, so the morphology *is* present in the features. The failure is in
how those features are turned into a scalar.

## 2. The five failures

Evaluated at matched energy against real RFI — the only benchmark that
discriminates here (clean negatives flatter every method).

| Approach | Mechanism | Outcome | Diagnosis |
|---|---|---|---|
| **Plain AE** (MSE recon) | encoder-decoder, recon error as score | R² AUC 0.42–0.50 — *below chance* | The AE **copies** the injected line through; there is no residual to measure. Scaling 200k → 1M snippets makes it **worse**: the model learns to redraw real RFI lines better too. |
| **MemAE** (Gong 2019) | decoder constrained to learned normal prototypes | R² AUC 0.648 | Genuinely better than the AE, and the gap is **architectural** (verified) — but it only narrowly beats trivial energy/peak statistics. |
| **ViT-MAE / CNN-MAE** (75% masking) | reconstruct from masked patches | R² AUC 0.61, collapses to an energy detector | Masking makes the task "predict noise from noise", which is unsolvable — the model converges to the mean. |
| **Mahalanobis / GMM on embedding** | statistical distance in latent space | AUC ~0.5 (chance) | The embedding *contains* morphology (supervised AUC 0.746–0.845), but the injected signal sits too close in density to normal noise for a one-class fit to isolate without labels. |
| **Occupancy scorer** (ON/OFF cadence pattern) | spectral-occupancy vetting over the cadence | Pre-registered bars **passed** | **Withdrawn on principle**, not on performance: the winning arm was not encoder-based, violating the architectural constraint. |

### 2.1 A retracted result worth remembering

A Mahalanobis-on-embedding score was reported at **AUC 0.927**, then retracted
(2026-06-30). It was a lucky-partition artifact: the original split had
n_rfi=75 / n_quiet=79, the re-run n_rfi=81 / n_quiet=64 — *same seed*, different
split. With identical parameters the re-run gave **AUC 0.518**.

The lesson is now standard practice in this repo: pre-register acceptance bars
before the run, pin checkpoints by explicit name (never `last.ckpt`), fix the
seed, and sample ≥500 per stratum. A related guardrail — SNR-sweep
detection rates computed on a quiet-only baseline are badly misleading at small
n: one ConvAE arm read 100% detection at n=50 and **0% at n=500**.

## 3. The probe that opened the way

Zero training, pixel space: `‖AE(x) − MemAE(x)‖²`
(`DisagreementPair` in `scripts/debug/encode_separation_test.py`).

| Test | Result |
|---|---|
| Matched-energy AUC vs real RFI (topk, n≈475/class) | **0.770 / 0.776 / 0.788** (seeds 42/7/123) |
| Same, on **val** (unseen observations) | **0.781** — inductive, not memorization |
| TPR@10%FP | ~100% @ SNR≥20 · 88–97% @ 15 · 35–60% @ 12 · **0% @ ≤7** |
| ON-only vs persistent, same morphology | 0.423 / 0.451 — **blind to ON/OFF occupancy** |

This was the **first unsupervised scorer to beat trivial statistics at matched
energy**, after five families at chance. Interpretation: `‖AE − MemAE‖²` is a
morphological *novelty* detector — it suppresses in-distribution RFI (both
students agree) and flags unseen morphology (the AE copies it, the MemAE
redraws it from normal prototypes).

**Why that justified building full UDMA.** The residual gap was the low-SNR
operating point (0% TPR@10%FP below SNR 10: the RFI topk tail crushes the
threshold, so the old max-error-RFI failure mode survives in the operating
point even though it no longer survives in the ranking). In Qi et al., the
pixel-space `ss` term is the *minor* one; the main effect is the two
teacher-student terms, where students regress **teacher features** rather than
pixels. Moving the score out of pixel space — where the RFI tail lives — is the
direct attack on §1's mechanism.

## 4. Where UDMA landed

| Metric | Disagreement probe | UDMA |
|---|---|---|
| Matched-energy AUC vs real RFI | 0.77–0.79 | **0.88–0.90** |
| TPR@10%FP | 100% @ SNR≥20 | 100% @ SNR≥12 |
| Low-SNR floor | 0% below SNR 10 | **non-zero at SNR 5–7** |

UDMA passed every pre-registered acceptance bar and is the production scorer.
See [`02_udma-architecture.md`](02_udma-architecture.md) for the full
specification and the bars themselves, and [`05_results.md`](05_results.md)
for the measured outcome.

## 5. Why `topk_mse` exists

`src/models/losses.py:topk_mse` averages squared error over only the top
`frac` of pixels instead of the whole frame.

Motivated by the MemAE noise-floor diagnostic: a real injected signal can
**triple the local peak** squared error while the whole-frame mean barely
moves, because ~1e5 pixels of incompressible background noise dilute a residual
elevated over only a few dozen of them.

Restricting the average to the top fraction preserves the local signal while
still averaging over enough pixels to avoid being a single-outlier statistic —
unlike a raw per-pixel max, which is already confounded by heavy-tailed
background residual.

Production `topk_frac` is **0.01** (swept; see
[`03_teacher-localization.md`](03_teacher-localization.md) §7).

## 6. Mechanism confirmed at the memory level

Direct causal evidence for §2's "the AE copies the line" claim
(`scripts/debug/injection_memory_addressing.py`, 2026-07-07): injected
narrowband/drift signals route to MemAE slots **321** and **85** — explicit
narrowband and drift prototypes — and the decoder redraws them as normal.

That is the recon-scorer's low-SNR recovery failure observed at the level of
the mechanism, not just the metric.

A visualization note: the Qi et al. Fig. 14 method (decode a broadcast memory
item) produces a checkerboard artifact for this repo's *per-position* memory
variant and does not transfer — confirmed against a random-vector control. Use
real-patch galleries instead (`scripts/debug/visualize_memae_memory.py`).

## 7. Standing constraints this history produced

- **Encoder-based only.** Non-ML stages are acceptable as a downstream filter,
  but may not *be* or *replace* the primary detection model. This is what
  retired the occupancy scorer despite it passing its bars.
- **No ON/OFF training objective.** Relaxed for `0000.fil` (scoring only, never
  as a training objective); the full ban resumes when `0001.fil` work restarts.
- **Solve `0000.fil` first.** Pivoting to `0001.fil` is not an escape hatch from
  a `0000.fil` plateau.

## 8. The retired backbones

Everything in §2 and §3 was built as a selectable backbone before UDMA existed,
and the code and configs are still in the tree. They are kept as baselines and
as parts of UDMA, not as search models; `build_autoencoder()`
(`src/models/autoencoder.py`) still dispatches to all of them.

| Backbone | Config | Code | What it is | Status |
|---|---|---|---|---|
| **Autoencoder** | `convae.yaml` | `Autoencoder` | Deterministic CNN encoder–decoder (4 blocks, filters 32→256, 16× spatial reduction, spatial bottleneck); score is the full reconstruction error. Modelled on the CNN backbone of Ma et al., without ContrastiveVAE, Random Forest or cadence labels. | Retired as a scorer (§2: copies the injected line). Its encoder trunk is the students' trunk in UDMA. |
| **CNN-MAE** | `convae_mae.yaml` | `MAE` | The same stack with ~75% of the patches zeroed in training and the loss on the masked positions only, against the "too-good reconstruction" failure of a plain AE. | Retired (§2: collapses to an energy detector). |
| **VAE** | `convae_vae.yaml` | `VAE` | The same stack with a variational bottleneck: reconstruction plus `beta`·KL. A single `variational` flag, kept as a fair-comparison variant. | Never the deciding comparison: no result is recorded in this repository. |
| **MemAE** | `memae.yaml` | `MemoryUnit` in `src/models/memory.py` | CNN AE with a content-addressable memory of normal prototypes (Gong et al. 2019) between encoder and decoder. | Retired as a scorer (§2: R² AUC 0.648, narrowly above trivial statistics). The memory unit is student B of UDMA. |
| **ViT-MAE** | `vit_mae.yaml`, plus `vit_mae_ssim.yaml` and `vit_dae.yaml` (denoising variant) | `src/models/vit_mae.py` | SSAST-style Vision Transformer masked autoencoder: patch tokens, Transformer encoder, partitioned reconstruction at inference. | Retired as a scorer (§2). Its encoder is the frozen UDMA teacher. |

Two points that are easy to lose:

- **The backbones were not compared on a common architecture.** An early goal
  was parity across backbones (one conv stack, one flag each). It was dropped:
  each backbone could be tuned on its own and the best one on the
  injection-recovery test would win. For the scorer families in §2 the question
  was closed by the mechanism, not by tuning.
- **AE versus VAE is orthogonal to faithfulness to Ma et al.** What the
  project drops from that work is the supervised classifier and the cadence
  labels, which bind the search to known signal classes; the encoder–decoder
  backbone is kept. The goal of searching a *broader range* of signals is the
  reason the scorer is unsupervised, not a reason to use their VAE.

Training configs for the retired backbones are `configs/training/{default,
memae_gbt_fine,bvae_gbt_fine,dae_gbt_fine,srt_real}.yaml`. They point at
`configs/data/gbt_fine.yaml` (the pre-night-split layout) and are not expected
to reproduce on the current manifests.
