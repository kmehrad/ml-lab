# EDA findings — BirdSpecies (MLSP 2013 Birds)

Source: `notebooks/01_eda.ipynb` (run after downloading the competition data).
Metric is **flattened-pair AUC**, so per-species prevalence and cross-species
score comparability matter.

> **TODO (fill after running the notebook on the downloaded data):**
> - Recordings: __ total (train __ / test __ from `CVfolds_2.txt`)
> - Audio props confirmed (10 s, 16 kHz, mono): __
> - Species prevalence (train fold) — min / median / max count: __ / __ / __
> - Label cardinality per recording — mean / min / max species: __ / __ / __
> - Recordings with 0 species: __
> - Segments per recording — min / median / max: __ / __ / __; no-segment recs: __
> - Missing values (histogram / segment features): __
> - Train vs test feature-distribution shift (histogram bins, segment counts): __
> - Histogram-bin correlation / near-zero-variance bins: __
> - Species co-occurrence highlights: __

## Implications for modeling

_(populated after EDA)_
