# Teacher choice and the localization problem

Why UDMA currently uses a **ViT-MAE teacher trained on radio spectrograms**
instead of the paper's out-of-domain distilled CNN, the record of the July 2026
experiments behind that choice, and the November 2026 re-examination that
overturned their main conclusion (§4.2).

> **Status (2026-11-19).** The July conclusion "only domain-matched training
> localizes; the receptive field is not the lever" does not hold. The July probe
> never tested a teacher confined to one observation, and its smallest ratio is
> reproduced by an untrained network. A ResNet-18 applied to each observation
> separately passes every teacher gate. Whether a teacher distilled from it gives
> a usable search chain (localization and detection after student training) is
> **not yet measured**. The production teacher is unchanged. (2026-11-21: the
> code for that distillation, `scripts/distill_teacher.py --per_observation` and
> `configs/model/udma_cnn_obs.yaml`, is written and tested; the first
> training attempts ran out of GPU memory and none has completed yet.)

Rationale behind `scripts/debug/teacher_row_leakage_test.py`,
`scripts/debug/teacher_sensitivity_test.py`, and the teacher block of the UDMA
model configs (`configs/model/udma_nightsplit.yaml`, `udma_cnn_obs.yaml`).

**Related:** [`02_udma-architecture.md`](02_udma-architecture.md) §11 ·
[`04_candidate-filtering.md`](04_candidate-filtering.md)

---

## 1. The deviation from the paper

Qi et al. 2024 use a 3-layer CNN teacher (RF 7×7) *distilled from a frozen,
generic pretrained network P*, à la Bergmann's "Uninformed Students". The
teacher's feature space is anchored **out of domain by construction**: spectrum
data enters only as distillation input, never as a learning target.

This project instead uses a ViT-MAE trained self-supervised **on radio
spectrograms** — the exact opposite. The reference teacher (`057f87c`) was trained
on SRT cadences, not on GBT data: in the radio domain, but on another telescope
and with no contact with the search data. The deviation was deliberate but needed justifying, because
all three structural deviations from the paper (teacher, memory variant,
student bottleneck) push in the same direction: toward collapse of the
student-disagreement signal.

## 2. The symptom: diffuse anomaly maps

The distilled CNN teacher (read at ResNet-18 `layer3`) produced **diffuse** UDMA
anomaly maps. An ON-only signal (Voyager-1) bled into the OFF rows, collapsing
`on_off_contrast` from ~32 (ViT-MAE teacher) to ~2.3 and breaking every ON/OFF
short-list filter downstream.

The mechanism is specifically **cross-observation leakage**. The `(96, 1024)`
input stacks 6 observations of 16 time bins each. For a teacher feature at an
OFF row to move when signal is injected only into ON observations, the
teacher's *vertical* receptive field must cross the 16-bin observation
boundary:

| Stage | Grid | Vertical RF | OFF feature rows that see ON pixels |
|---|---|---|---|
| stem (conv1+maxpool) | (24, 256) | 11 px | 8 of 12 |
| layer1 | (24, 256) | 43 px | 12 of 12 |
| layer2 | (12, 128) | 99 px (whole frame) | 6 of 6 |
| layer3 (deployed) | (6, 64) | 211 px (whole frame) | 3 of 3 |
| layer3, applied per observation | (6, 64) | 16 px | 0 of 3 |

The last column counts, for each stage, the feature rows of OFF observations
whose receptive field holds at least one ON pixel; it is measured by perturbing
one input row at a time (2026-11, random weights: which rows can move does not
depend on the weights). An earlier version of this table read the stem as
"< 1 observation, no cross-observation leak in theory". That is wrong: the stem
has stride 4 and its rows are centred on the observation boundary, so 8 of the
12 OFF rows see ON pixels. **No ResNet-18 stage is confined to one
observation**; the distilled teacher T (4 blocks of two 3×3 convolutions, 91 px)
and the students (whole frame) are not either.

## 3. The probe

`teacher_row_leakage_test.py` measures leakage **directly on the teacher's
features** — no students, no distillation, no training. Inject an ON-only line
into a quiet frame, take the feature displacement `‖P(x+s) − P(x)‖` per
(row, col) cell, and compare the response in ON feature-rows vs OFF
feature-rows. A localizing teacher keeps displacement in ON rows (high ON/OFF
ratio); a leaky one spreads it (ratio → 1).

The test is clean because:

1. `inject_narrowband_on_only` renders the signal **only** into obs 0/2/4 —
   OFF frames are byte-identical.
2. `preprocess_raw` normalizes **per observation** (the `gbt_fine` normalization
   fix), so an ON injection cannot shift OFF pixels via renormalization.

Any OFF-row feature motion is therefore pure receptive-field leakage. Feature
rows map to observations by the stage's vertical downsampling: obs `i` occupies
rows `[i*(nh/6) : (i+1)*(nh/6))`, ON = obs {0, 2, 4}. No column window or
morphology assumption is used (row max over the full frequency axis).

Cost: `n_frames × len(snr_list) × len(stages)` frozen forwards — minutes on
GPU, **no training**.

## 4. Three hypotheses (2026-07-16) and their re-examination (2026-11)

### 4.1 July 2026 reading — superseded on the receptive field

The probe supports several `--architecture` modes, each isolating one variable.
All three ran as zero-training controls. The table is kept as it was recorded;
§4.2 explains why its first row and its conclusion do not hold.

| Hypothesis | Control | Result |
|---|---|---|
| **Receptive field** is the lever | ResNet-18 RF sweep across stem/layer1/layer2/layer3 | **Refuted** — ratio stayed ~2–3 at *every* depth, including the small-RF stem |
| **Architecture** (CNN vs Transformer) is the lever | `--architecture udma_student` on two trained students sharing one trunk but different teacher targets | **Refuted** — the old-target student localizes, the new-target one does not; same architecture both times |
| **Pretraining objective** (classification vs reconstruction) is the lever | `--architecture convnextv2_fcmae` — same ImageNet domain, same CNN family, but masked-patch reconstruction pretraining | **Refuted** — ConvNeXtV2-FCMAE ≈ ResNet, stayed diffuse |

The `convnextv2_fcmae` control matters most: the resnet18-vs-vit_mae comparison
confounds *domain* (ImageNet vs GBT) with *objective* (classification vs
reconstruction), changing both at once. FCMAE holds domain wrong while fixing
the objective. It stayed diffuse. (It was chosen over a ViT-MAE-on-ImageNet
alternative because it is fully convolutional, so the non-224×224 input needs
no positional-embedding interpolation.)

July conclusion (**superseded**): "only domain-matched training ever
localizes; domain matching is necessary regardless of architecture, receptive
field, or pretraining objective. Stop hunting backbones."

### 4.2 Re-examination (2026-11)

**Why the July reading fails.** The probe is differential and the OFF pixels are
byte-identical, so an OFF feature row can move *only* through its receptive
field. "Generic features fire on OFF texture" cannot produce OFF motion in this
test. And no stage in the sweep was confined (§2), so a flat ratio across
stages says nothing about a confined teacher.

**Checks, all without training.**

1. **Instrument check.** The probe now prints the response of every feature
   row. On the July cache and arguments it reproduces the July numbers digit for
   digit, and the four stem OFF rows whose receptive field holds no ON pixel
   (rows 6, 14, 22, 23) respond with exactly 0. All OFF response sits in the
   boundary rows.
2. **Untrained control.** A ResNet-18 with random weights gives, on the same
   cache, a stem ratio of 2.94 / 2.80 / 2.69 at SNR 10 / 20 / 40, against
   3.17 / 3.09 / 2.98 for ImageNet weights. The July "~3 at the stem" is set by
   geometry, not by feature content. Deeper stages: layer1 random 1.8 vs
   ImageNet 2.1–2.2; layer2–3 random 1.0–1.3 vs ImageNet 1.7–2.2. Trained
   features leak somewhat less than random ones, but stay near 2 because the
   receptive field crosses the boundary anyway.
3. **ConvNeXt V2, re-read.** The July run already contained a generic ImageNet
   network with high ratios: infinite at the non-overlapping 4×4 stem (OFF
   response exactly 0) and 11–12 at `stage0`; only `stage2` (2.2–2.3) had been
   read.
4. **Reference student.** The student of the current reference model gives
   ON/OFF 25.6–27.3 (each OFF row at 3–4% of the ON mean).

**Confined teacher candidate.** `ResNetTeacher(confine="obs")` applies ResNet-18
to each 16 × 1024 observation on its own; `confine="tile"` to a 16 × 64 crop per
cell (a patch descriptor as in Bergmann et al. 2020). Gate results on the
night-split training cache, same 200 + 200 frames for every variant:

| Gate (threshold) | full frame | `obs` | `tile` | `obs`, random weights |
|---|---|---|---|---|
| G1 relative variation (≥ 0.05) | 0.618 | 0.211 | 0.146 | 0.260 |
| G2-ON at SNR 20 (≥ 0.80) | 0.999 | 0.999 | 0.999 | 0.999 |
| G3a at SNR 20 (≥ 0.70) | 0.833 | **0.952** | 0.925 | 0.877 |
| G3b (≥ 0.60; trivial baseline 0.772) | 0.741 | **0.880** | 0.769 | 0.726 |
| Frame AUC vs real RFI, SNR 20 (reported) | 0.369 | **0.638** | 0.506 | 0.234 |

G2-ON is G2 read on ON-observation tokens only: with a confined teacher every
OFF token has zero displacement, which would otherwise inflate the AUC.

Reading:

- Confinement does not blind ResNet-18; on these gates it does better than the
  full-frame target.
- The gates do not separate pretrained from untrained features: the random
  confined networks pass them too. ImageNet weights stand out only in `obs`, on
  G3b (the only value above the trivial baseline) and on the frame AUC against
  real RFI. One run per variant; the uncertainty of these differences is not
  measured.
- The price of `obs` is a poorer target: content rank 16 vs 68, 14 of 256
  dimensions dead (within the 10% bound).
- `tile` adds nothing over `obs`.

**What is still open.** The anomaly map is teacher minus students, and the
students see the whole frame, so they can reintroduce OFF response. Part of the
July detection advantage (§5) may also have come from the leak itself. Both need
one distillation and student-training cycle evaluated on the full search chain;
that run has not been done.

**What survives from July.** Students follow the localization of their target:
the same student trunk gives ON/OFF 24 with the ViT-MAE target and 2 with the
full-frame ResNet target. What changed is the reading of why the ResNet target
leaks: its receptive field crosses the observation boundary. The ViT-MAE builds
its tokens from 16 × 16 patches inside one observation, so its target is nearly
confined by construction; attention adds a few percent of OFF response.

## 5. Detection and localization are orthogonal

A separate result complicates the picture. The distilled CNN teacher, despite
failing to localize, is the **best detector yet built**:

| Teacher | det@3σ_cad, SNR 15 / 20 / 30 |
|---|---|
| CNN (distilled, `d13`) | **93.33 / 98.44 / 100.0** |
| ViT-MAE (old, domain-matched) | 80.9 / 94.7 / 99.8 |

The same checkpoint that wins detection cannot localize. Different teachers win
each axis, so the open design question is **detection-tuning on ViT-MAE vs. a
two-model pipeline**, not "which teacher is better".

Since §4.2 a third option is open: a distilled CNN teacher whose target is
confined to one observation. Whether it keeps this detection advantage once its
map no longer leaks into the OFF rows is untested. Note also that this table
comes from the July Tier-A benchmark (detection at 3σ per cadence), which does
not go through clustering and the short list.

## 6. What the teacher actually buys

Component attribution (2026-07-20, `udma_component_attribution.py`) at matched
energy against **real RFI**: UDMA beats trivial statistics by +0.07–0.09 AUC.
The memory unit earns its place.

Two caveats worth keeping visible:

- Clean negatives give the **opposite** answer. Only matched-energy-vs-real-RFI
  is a meaningful benchmark here.
- Equal-weight fusion of the three maps dilutes `ss`, which is by far the best
  single term at SNR 10.

## 7. Related closed threads

- **`mem_slots` = 30 (paper-faithful Park variant).** Killed 2026-07-15. The
  mechanistic signal was strong throughout (`val_ss` 2.2–2.8×) but never
  converted to a Tier-A detection gain — flat or negative at two checkpoints
  9 epochs apart. `mem_slots` stays at 500.
- **`topk_frac`.** Swept over both students; the optimum is 0.01, not the 0.02
  that was in the production config. No model switch — just the config bump.
- **Benchmark reproducibility.** A Fase-0 control re-run reproduced
  80.9/94.7/99.8% from an identical checkpoint/config/code where the historical
  record said 48.9/68.0/79.3%. The old cadence list is unrecoverable, so the
  new number is canonical.
