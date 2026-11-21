# Platypus

Unsupervised anomaly detection for technosignatures in GBT observations of
Breakthrough Listen Exotica Catalog sources. This glossary fixes the vocabulary
for the scoring model and the candidate pipeline, where several terms were being
used for more than one thing.

## Language

**Cadence**:
One ABACAD observing sequence — three ON-target observations interleaved with
three OFF-target ones, all of the same target.
_Avoid_: scan, sequence, pointing

**Observation**:
One time-contiguous telescope observation in a cadence, identified as ON or OFF
relative to that cadence's target.
_Avoid_: cadence, independent sample

**Band cadence**:
One recorder node's frequency coverage of a six-observation cadence; one row
of the current manifests. Multiple rows may cover the same sequence and band.
_Avoid_: independent cadence, observing session

**Observing session**:
A group of nearby observing sequences kept together when splitting data.
The chosen operational grouping is an observing night; independence is not
guaranteed merely by assigning a session identifier.
_Avoid_: manifest row, observation, calendar day

**Search unit**:
The observation context used to find a candidate: a complete cadence for the
initial narrowband/wideband searches, one observation for isolated transients.
It is distinct from the group used to split the data.
_Avoid_: independent sample, training split

**Anomaly map**:
The per-snippet grid of anomaly response produced by the scoring model. In the
current `0000` pipeline it has one row band per observation of the cadence;
that geometry has not been established for `0001` or `0002`.
_Avoid_: error map, heatmap, residual map

**Anomaly score**:
The single scalar per snippet obtained by reducing the anomaly map. What the
first candidate cut thresholds.
_Avoid_: reconstruction error, loss, anomaly value

**ON/OFF discrimination**:
Whether a model's anomaly response stays confined to the ON observations when
only the ON observations contain signal. Measured as a ratio of ON to OFF
response; a domain-matched teacher reaches ~35, a generic out-of-domain one ~2.
The generic value is set by receptive-field geometry, not by what the features
encode (`docs/03`, §4.2).
_Avoid_: localization, ON/OFF localization, cadence localization

**Map resolution**:
Whether the anomaly map can represent *where within an observation* the response
sits, and therefore what shape it had. Limited by the map's cell geometry, and
limited identically for every teacher.
_Avoid_: localization, spatial localization, morphology characterisation

**Teacher**:
The frozen network whose intermediate features the students are trained to
regress. Supplies the target; never scores directly.
_Avoid_: backbone, reference model, P

**Student**:
One of the two trainable networks regressing the teacher's features — one plain,
one memory-augmented. Their disagreement is the anomaly response.
_Avoid_: branch, head, learner

**Completeness**:
The fraction of injected signals that survive the whole candidate pipeline and
reach a human reviewer. A property of the pipeline, not of the model.
_Avoid_: recall, detection rate, sensitivity

**Detection**:
Whether the anomaly score separates signal from background, independently of any
downstream filtering. A property of the model.
_Avoid_: recovery, completeness

**Candidate**:
One cluster of adjacent frequency windows whose score clears the first cut,
represented by its peak window.
_Avoid_: detection, hit, event

**Selection rule**:
A named bundle of the candidate-selection stages and short-list gates, chosen
with `--rule` (`RULES` in `src/search/pipeline.py`). The current one is
`hybrid_v4`.
_Avoid_: threshold, filter, cut

**Short list**:
The candidates that pass the ON/OFF gates of the selection rule and, after the
veto, go to the reviewed PDF. The full candidate table is larger and is kept.
_Avoid_: final list, candidates (when the filtered set is meant)

**Cross-target veto**:
Dropping a candidate that overlaps clusters of at least two other targets of the
same observing night and compute node: a signal from one direction on the sky
cannot be at the same channels toward other targets. Applied after the night is
scanned.
_Avoid_: RFI filter, deduplication

**BL band / extended band**:
The BL band is the receiver's published range (receiver notches excluded) used
for the main completeness figure; the extended band is the rest of the recorded
band, which is searched and labelled but reported separately.
_Avoid_: usable band, clean band
