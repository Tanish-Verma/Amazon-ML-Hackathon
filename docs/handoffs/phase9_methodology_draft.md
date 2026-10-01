# Phase 9 methodology — drafting guide, not the official template

Transfer these sections into the original `Documentation_template.md` after checking the official problem statement. Replace all `TODO` entries with measured outputs; do not submit placeholders.

## Methodology
Country-partitioned, CPU-only business entity resolution. Unicode-aware normalization; four sparse retrieval channels (name trigrams, address, name tokens, exact digit keys); reciprocal rank fusion of deep retrieval; vectorized 16-feature cheap reranker; top-50 final candidates; Phase 4 expanded features; Phase 5 LightGBM classifier with calibrated scores, tuned thresholds and exclusive S2/S3 assignment. Verify the exact production model/feature set before submission.

## Blocking measurements
Phase 3 report: initial macro recall 0.7102; India four-channel RRF configuration D at depth 1000: 97.93% recall on 5,000 queries / 18,451 true pairs; later reported broad-pool recall India 98.8%, US 99.3%, post-rerank K50 India 97.4%, US 99.3%. Specify the validation populations and run dates for each figure. **TODO:** actual full-test candidate counts and reduction ratio = candidate pairs / possible within-country S1×(S2+S3) pairs; do not confuse sample measurements with final-test measurements.

## Features and model
Existing 16 vectorized features from `src/rerank_vec.py`; Phase 4 adds rank/margin, candidate count, reverse rank, rare address overlap, city/region agreement. **TODO:** verify exact deployed feature list and all real-data per-feature AUC values. **TODO:** final LightGBM parameters, validation split, probability calibration, threshold optimization, assignment rule and saved model checksum.

## Results and error analysis
**TODO:** actual macro-F0.5 overall and per-country, bootstrap intervals, singleton precision, false-positive and false-negative examples anonymized or minimized as appropriate. **TODO:** Phase 7 US→India, India→US, dictionary ablation and France unlabeled drift; do not report a France test score without labels.

## Reproducibility
**TODO:** actual Python and dependency versions, exact git commit, clean-room commands, hardware, measured wall time, peak RSS, official validator result and SHA-256 of the two output TSVs.

## Model licenses
**TODO:** verify the exact license of the *installed* LightGBM and scikit-learn versions from official upstream sources, record URLs and retrieval date. Both are conventionally MIT licensed; check the actual shipped versions. No pretrained embeddings are in the final CPU-only pipeline per Phase 7 handoff. LightGBM tree count is not a neural-network parameter count; report exact tree/leaf counts or serialized model size instead. For logistic regression, report feature count and coefficients if requested. Do not invent parameter counts.
