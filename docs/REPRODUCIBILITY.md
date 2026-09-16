# Reproducibility and scope

This is a frozen-inference demonstration release. It contains no detector/ReID weights, image datasets, raw ReID descriptor caches, credentials, manuscript author placeholders, or internal submission archives. Full benchmark evaluation requires the original datasets and independently obtained host dependencies. The repository alone does not reproduce host tracking, feature extraction from images, or the entire model-selection study.

## What is preserved

The core source was copied from the implementation used in the research audit. Only the package namespace, an internal feature-list identifier, two error messages and the location of the unchanged box-IoU helper were adapted. The original frozen model is copied byte for byte. `RELEASE_PROVENANCE.json` records hashes and replacements. The public inference and command-line wrappers are new; the example replay and source-equivalence checks verify their agreement with the frozen results.

The release is associated with the manuscript titled *TrackRelink: Learned Offline Tracklet Repair with Fixed Detections*. The manuscript PDF is not included because its author and declaration metadata are being finalized. No acceptance or publication status is implied.

## Inputs and action

Each host ID forms one tracklet containing all its observations. Internal temporal gaps are not split. Frames must be strictly increasing within a tracklet after sorting; duplicate observations of the same ID at the same frame are invalid. MOT boxes use `x,y,width,height`, and must have positive width and height. Confidence and optional trailing columns are retained.

Candidate generation uses positive endpoint gaps with at most 60 missing frames, normalized forward center distance at most 4.0 and absolute log area ratio at most 1.4. The feature schema comprises 15 geometry, ten appearance, eight reciprocal and six additional motion fields. Competitors are drawn from all geometrically gated candidates, before ground-truth filtering. The model NPZ stores the complete ordered field names.

Appearance inputs are matrices keyed by numeric track ID. Each row is a descriptor from a host-matched detection, in temporal order. All nonempty matrices must have the same width. Missing tracks use the original unavailable-appearance representation; this is supported mechanically but does not establish accuracy without appearance. No ground-truth labels or target refitting enter inference. The included examples replay archived 39-dimensional feature vectors rather than re-extracting descriptors.

The score is `sigmoid(((x - mean) / scale) @ coefficients + intercept)`. The threshold is 0.766583780987215. Candidates are visited in descending score, breaking ties by predecessor then successor ID. Each tracklet has at most one outgoing and one incoming selected link; chains are possible. A connected chain receives its minimum original ID, which need not be the chronologically first ID. No interpolation, deletion, splitting or box correction occurs. The command-line wrapper refuses to overwrite an input or existing output.

`repair_model.py` includes grouped splits, fitting, video weights and threshold-selection utilities. It is not an end-to-end training recipe. The research fit uses video-disjoint training selection; results from the evaluation videos must not be used to tune the bundled threshold.

## Real-example replay

Run `python examples/replay_figure1.py` after installation. It loads every stored candidate for each of the two Figure 1 sequences, checks the frozen scores and complete constrained selection, regenerates relabeled IDs from the host outputs and compares all saved output rows. It verifies preservation of the original non-ID string fields. The archived output uses its historical numeric serialization, so the cross-archive comparison is numerical.

The displayed snapshots do not necessarily coincide with tracklet endpoints: DT0010 uses illustrative observations at frames 410 and 535, while the candidate's endpoint gap is 42 missing frames. “Correct link” and “correct rejection” are posthoc audited labels for these examples, not a general action-accuracy guarantee.

## Interpreting results

The numerical results use the same 25 DanceTrack evaluation videos for both hosts. The second host is StrongSORT-derived, with shared detections/appearance inputs and implementation qualifications; it is not a claim to reproduce every official StrongSORT setting. The same explored dataset on two hosts is not an independent dataset replication.

Pooled HOTA must not be obtained by averaging per-video HOTA. The stored `combined` entries are pooled evaluator outputs; paired intervals resample videos and concern macro differences. The first-host comparison uses 20,000 bootstrap draws, seed 2027; the second uses 10,000, seed 20260915. The headline full-method gains are descriptive; the original follow-up and second-host primary contrasts are not retroactively changed by the release.

The primary first-host follow-up compared reciprocal scoring with its protocol-specific hard-gate control. The second-host primary contrast was reciprocal minus pairwise; its interval crosses zero. Full-minus-host was secondary on the second host. Pairwise repair also helps both hosts. The whole-tracklet label/action mismatch, unresolved and conflicting actions, negative boundary-gate result, unequal comparator support and explored data limit claims. Calibration probabilities remain a separate diagnostic and are not the frozen repair score.
