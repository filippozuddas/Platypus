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

**Anomaly map**:
The per-snippet grid of anomaly response produced by the scoring model, one row
band per observation of the cadence.
_Avoid_: error map, heatmap, residual map

**Anomaly score**:
The single scalar per snippet obtained by reducing the anomaly map. What the
first candidate cut thresholds.
_Avoid_: reconstruction error, loss, anomaly value

**ON/OFF discrimination**:
Whether a model's anomaly response stays confined to the ON observations when
only the ON observations contain signal. Measured as a ratio of ON to OFF
response; a domain-matched teacher reaches ~35, a generic out-of-domain one ~2.
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
