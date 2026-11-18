# Tab-LIT: Numeric Literal Prediction with Tabular Foundation Models

Reproduction package for the paper's literal prediction results on the shipped
splits. Each knowledge graph becomes one table with a row per entity. A frozen
tabular foundation model then predicts each numeric attribute in context from
that table, and nothing is trained. The package also reproduces the
language-model comparison: Qwen3-8B answering directly, and its answers added
to the table as a column.

The repository holds only what running and reproducing the tables below needs. Any
setting that is the same in every reported run is a constant in the code, not a
flag; [Fixed settings](#fixed-settings) lists them. The literal-disjoint
evaluation is not part of this package.

## What is in it

Backbones (`src/runner.py`), each decoded at its predictive median:

| Name | Model | Package |
|---|---|---|
| `tabpfn` | TabPFN-3.5 | `tabpfn` 9.2.0 |
| `tabicl` | TabICLv2 | `tabicl` 2.2.0 |
| `kumo` | Kumo-Tabular, large | `structured-data-models` 0.2.1 |
| `nori` | Nori-1.5 | `synthefy-nori` 0.22.0 |
| `gbt` | gradient-boosted trees, absolute-error loss | `scikit-learn`; needs no weights, so it is also the smoke test |

The conversion (`src/featurizer.py`) has five blocks, and `full` uses all of
them:

| Block | Feature group | Columns |
|---|---|---|
| struct | `structural` | degrees, PageRank, 14 Laplacian eigenvectors |
| assoc | `association` | 128-d SVD of the (relation, neighbor) incidence matrix |
| value | `neighbor_literal` | mean, max, min and count of the neighbors' values, per attribute |
| offset | `mrap_prior` | relation-offset estimates of each attribute, from training values only |
| own | `own_literal` | the entity's own training values; the target's column is masked |

`full_llm` adds one more group, `llm_answer`: a column `llm::<a>` per attribute
holding Qwen3-8B's answer for that cell, asked from the entity's name alone.
When predicting attribute `a`, every `llm::<b>` with `b` other than `a` is
masked, because which cells were asked reveals which attributes the entity has.
The block-ablation arms are also in `CONVERSIONS`.

The language model (`src/llm.py`) is asked three ways for each test row: from
the entity's name; from up to 30 of its one-hop training triples, with the
entity anonymized as X; and from those triples with up to three known values of
each neighbor. Decoding is greedy with thinking disabled. A reply with no
parsable number is asked once more with the reply prefilled as "My best
estimate is"; if it still does not parse, the training median is used. The
replies ship in `llm/`, so nothing needs the 16 GB model unless those files are
deleted.

The MrAP rerun (`src/mrap.py`) scores MrAP on the same test rows, following its
released protocol: offsets are estimated from all literals, test included, over
all triples.

Datasets: DB15K, FB15k-237 and YAGO15k, read from `$TABLIT_KG_ROOT/<dataset>`
(default `../KGs`). Each dataset needs `train/valid/test.txt` triples and
`literals/{train,valid,test}.txt` (entity, attribute, value), used as shipped.
The `KGs/` folder of the LitEm repository below has all three in this layout,
byte for byte the files the results were computed on. FB15k-237 also needs
`entity2text.txt` (Freebase id, label) to name its entities in prompts, as
distributed with KG-BERT; it is read only when the language model is asked
again.

## Install

```bash
conda env create -f environment.yml && conda activate tab-lit
# or: pip install -r requirements.txt
```

TabPFN 9 downloads its weights after a one-time license acceptance; for an
unattended run, set `TABPFN_TOKEN` (<https://ux.priorlabs.ai/account>). The
other models download from Hugging Face without a token. A GPU is needed in
practice. The paper ran everything on one RTX 3090 (24 GB).

The datasets and the MrAP script come from the LitEm repository:

```bash
git clone -b paper-reproduction https://github.com/dice-group/literal-embeddings
export TABLIT_KG_ROOT=$PWD/literal-embeddings/KGs
export MRAP_DIR=$PWD/literal-embeddings/MrAP
```

## Run

From inside the repository:

```bash
# one fit, no weights needed
python -m src.main run --dataset DB15K --backbone gbt

# the tables below from the shipped results, no GPU needed
python -m src.main report

# everything in the tables, then rewrite them in this file
MRAP_DIR=... ./run_all.sh

# or step by step
python -m src.main llm            # replies to llm/, direct scores to results/
python -m src.main experiment --backbones tabpfn tabicl kumo nori gbt
python -m src.main experiment --backbones kumo tabpfn gbt --conversion full_llm
python -m src.main mrap
python -m src.main report --readme
```

`experiment` writes `results/<dataset>_<backbone>_<conversion>.csv`, with one
row per attribute, seed and split (test and validation), and a manifest with the
package versions and the time per seed. Existing files are skipped unless you
pass `--overwrite`. `llm` reads its replies from `llm/cache/` and asks Qwen3-8B
only for prompts missing there.

On one RTX 3090, TabPFN-3.5, TabICLv2, Kumo-Tabular and GBT take about 25
minutes together on the three datasets. Nori-1.5 takes hours, most of it on
YAGO15k, whose largest attributes do not fit in 24 GB and fall back to the CPU.
Asking Qwen3-8B again takes about an hour.

`report` reads only `results/` and recomputes nothing, so it runs without a
GPU, weights or datasets. It refuses to pool two files that were scored on
different test rows for the same attribute and seed. `pytest tests` checks the
feature groups, the target masking, answer parsing and the table code without a
GPU.

## Fixed settings

| Setting | Fixed to | Where |
|---|---|---|
| Point prediction | predictive median of every backbone | `src/runner.py` |
| Target | raw values, no transform | `src/runner.py` |
| Context size | at most 10,000 training rows per attribute, drawn per seed | `build_predictor` |
| Ensemble size | TabPFN-3.5: 4, TabICLv2: 8 | backbone classes |
| Kumo-Tabular size | `large` | `KumoTabularLiteralPredictor` |
| Missing values | passed to the backbone as NaN | `src/runner.py` |
| Laplacian eigenvectors | 14 | `build_featuriser` |
| Association SVD | 128 dimensions, keyed by (relation, neighbor, direction) | `AssociationFeaturiser` |
| Block width | any block but `assoc` wider than 64 columns is reduced to 64 by PCA | `CompositeFeaturiser` |
| Offset block | up to 12 (relation, direction) channels per attribute, each with at least 15 edges; 5-fold cross-fitting | `MrAPPriorFeaturiser` |
| Configuration selection | none: one conversion applied to every attribute | `run_experiment` |
| Evaluated attributes | more than 40 test rows | `MIN_TEST_ROWS` |
| Language model | Qwen3-8B, greedy, at most 24 new tokens, one retry | `src/llm.py` |
| Prompt context | at most 30 one-hop facts, 3 values per neighbor | `src/llm.py` |

`--conversion` is the one design flag kept although the main table reads only
`full`: it runs the block ablation and the answer column.

## Reproducibility

TabPFN-3.5 and TabICLv2 are deterministic for a given seed. Rerunning them here
gave the same per-attribute MAE as the runs behind the paper's main table,
within 0.005%. The MrAP rerun matches to floating-point precision. The prompts
`src/llm.py` builds are byte-identical to the ones the paper's replies were
generated from, and the shipped replies parse to the same values.

The paper's GBT rows were run under numpy 1.26 and scipy 1.16, not the
versions pinned here. The feature table differs slightly between the two
(SVD and eigenvector numerics), and GBT follows it. Under those versions the
package reproduces the paper's GBT rows exactly; under the pinned ones a GBT
cell moves by a median of 1% to 2% per attribute.

Kumo-Tabular is not deterministic on the GPU. Two runs of the same seeds
differed by a median of 0.7% to 3.1% in per-attribute MAE, depending on the
dataset, and by at most 9.5% (on FB15k-237). Read a Kumo-Tabular cell to about
two significant figures.

The seed fixes the context subsample, the cross-fitting folds, the association
SVD and the backbone's own randomness. The splits do not depend on it.

## Results

Means over seeds 1 to 3, on attributes with more than 40 test rows. A cell is
left empty (`--`) unless all three seeds ran. LitEm values are quoted from the
LitEm paper (Table 2 for FB15k-237 and YAGO15k, Table 4 for DB15K), not rerun.
MrAP is rerun as described above. Package versions per run are in
`results/*_manifest.json`. Dates are in years, coordinates in degrees, heights
in meters. In the first table, bold marks the lowest MAE and italics the second
lowest.

The second table gives the median over the 21 attributes of MASE, the MAE
divided by that of predicting the training median, so 1.00 is no better than the
median. The direct answers are from one greedy run. Qwen3-8B was trained on web
text that may state the true values, so the answer-column rows measure what the
model remembers, not what the graph supports, and are kept apart from the first
table.

<!-- BEGIN GENERATED TABLES -->
### Shipped splits: MAE

| Attribute | n | LitEm | MrAP | TabPFN-3.5 | TabICLv2 | Kumo-Tabular |
|---|---:|---:|---:|---:|---:|---:|
| **FB15k-237** | | | | | | |
| date_of_birth | 454 | *11.935* | 14.551 | 12.703 | 15.706 | **10.595** |
| latitude | 335 | 2.596 | 1.707 | *1.447* | 2.376 | **1.324** |
| longitude | 324 | 4.424 | 4.163 | *3.202* | 3.330 | **3.043** |
| height_meters | 276 | 0.0760 | 0.0823 | *0.0645* | 0.0650 | **0.0636** |
| pop._number (10^6) | 220 | *2.330* | 24.356 | 2.356 | 2.701 | **1.562** |
| loc.area (10^6) | 214 | **1.370** | 1.894 | 1.507 | 1.396 | *1.377* |
| release_date | 180 | **3.449** | 5.827 | *3.511* | 3.819 | 3.579 |
| date_of_death | 123 | 15.752 | *14.402* | 17.890 | **13.365** | 15.197 |
| org.date_founded | 112 | *47.762* | 62.594 | 48.768 | 49.703 | **45.951** |
| loc.date_founded | 71 | 149.2 | 206.6 | *143.3* | 145.7 | **141.8** |
| **YAGO15k** | | | | | | |
| BornOnDate | 837 | 13.242 | 17.738 | *10.944* | 11.578 | **10.460** |
| CreatedOnDate | 651 | 57.206 | 67.857 | **43.457** | 48.128 | *43.962* |
| Latitude | 309 | 1.703 | 1.965 | **1.223** | 1.378 | *1.248* |
| Longitude | 294 | 5.598 | 5.915 | *3.681* | 4.455 | **3.023** |
| diedOnDate | 169 | 30.135 | 36.085 | 21.992 | *21.800* | **19.963** |
| DestroyedOnDate | 58 | 21.673 | 39.724 | **18.179** | 21.195 | *19.361* |
| **DB15K** | | | | | | |
| birthDate | 502 | 13.212 | 18.214 | *11.319* | 11.650 | **11.091** |
| longitude | 193 | 11.776 | 16.992 | *7.869* | 11.062 | **7.180** |
| latitude | 188 | 3.758 | 6.299 | *3.108* | 3.525 | **2.669** |
| deathDate | 121 | 15.825 | 17.422 | *11.272* | 11.821 | **11.069** |
| foundingDate | 86 | 40.270 | 44.468 | *30.401* | **30.182** | 32.143 |

Lower MAE than both LitEm and MrAP: TabPFN-3.5 15/21; TabICLv2 15/21; Kumo-Tabular 18/21.

### A language model as baseline and as a column: MASE

| Table or prompt | Learner | Median MASE | MASE < 1 | Attributes |
|---|---|---:|---:|---:|
| name only | Qwen3-8B | 0.689 | 15 | 21 |
| one-hop triples | Qwen3-8B | 1.056 | 9 | 21 |
| triples, neighbor values | Qwen3-8B | 1.597 | 8 | 21 |
| conversion | Kumo-Tabular | 0.430 | 21 | 21 |
|  | TabPFN-3.5 | 0.450 | 21 | 21 |
|  | GBT | 0.553 | 21 | 21 |
| + answer column | Kumo-Tabular | -- | 0 | 0 |
|  | TabPFN-3.5 | -- | 0 | 0 |
|  | GBT | 0.430 | 21 | 21 |

Attributes where the answer column lowers MASE: GBT 20/21.
<!-- END GENERATED TABLES -->

## Layout

- `src/featurizer.py`: `KGIndex`, the feature groups, `CONVERSIONS`,
  `build_featuriser()`, `load_literal_splits()`
- `src/runner.py`: the shared per-attribute loop, the target masking, the
  backbones, and `run_experiment()`
- `src/llm.py`: prompts, answer parsing, generation, and scoring of the direct
  answers
- `src/mrap.py`: the MrAP rerun
- `src/evaluation.py`: metrics, the LitEm values, and the two tables
- `src/main.py`: the command line (`run`, `experiment`, `mrap`, `llm`, `report`)
- `llm/`: Qwen3-8B's replies, `<dataset>_answers.csv` for every (entity,
  attribute) cell and `<dataset>_direct.csv` for the three prompts on the test
  rows
- `results/`: one CSV per dataset, backbone and conversion, with a manifest
  each, plus the MrAP rerun and the scored direct answers; the tables above are
  built from these
