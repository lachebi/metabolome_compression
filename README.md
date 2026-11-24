# Should we embed in metabolomics?

This is the code for a benchmark of PCA, UMAP and a variational autoencoder (VAE) as compressed representations of the plasma metabolome. The models predict 14 binary outcomes at 6 months, 18 months and 6 years, for 42 outcome-timepoint pairs.

The study asks three questions:
1. Is the raw metabolome predictive of the outcome? This is answered with a label-permutation test.
2. Does an embedding with k = 2, 3, 4, 8 or 16 keep the predictive signal? This is answered with nested cross-validation and three classifiers, and with repeated nested cross-validation using logistic regression.
3. Do the metabolites behind a prediction survive compression? This is answered by attribution on the raw metabolome, mapped back through each embedding.

```
src/          the installable-free package (metabo/), the CLI pipeline (scripts/, analysis/), the tests (tests/), run_final.sh
notebooks/    one notebook per research question, run against a synthetic dataset in data/
```

`data/` is not part of this repository -- every notebook generates it itself on first run
(purely from chosen statistical distributions, not derived from real data in any way; see
"Notebooks" below).

## Install

```bash
pip install -r requirements.txt    # uncomment the last line in it for the VAE (PyTorch)
```

No build step and nothing gets installed as a package: `scripts/`, `analysis/` and the notebooks all find `metabo` by putting `src/` on `PYTHONPATH` (`run_final.sh` and every notebook do this themselves; see "Running things directly" below if you call a script on its own).

The code was run with Python 3.10 and 3.11 and scikit-learn 1.7 to 1.9. `analysis/record_versions.py` records the versions in use.

## Data

`data/` does not need to be present at all: every notebook generates the synthetic dataset
itself on first run if it's missing (same recipe as `00_build_data.ipynb`, see "Notebooks"
below). The code's own expected file layout — for any dataset in this format, pointed to with
`METABO_DATA_DIR` (see `src/metabo/config.py` for all settings) — is:

| File | Content |
|---|---|
| raw intensities file | One row per sample. Semicolon separated, decimal comma, an id column, a timepoint column (`TPn`: 6m, 18mth, 6y, plus any other sample rows), `GENDER`, then one column per metabolite. Empty cells are values below detection. |
| target file | One row per subject (id) and one 0/1 column per outcome in `config.OUTCOMES`. |
| per-timepoint input matrices | log10 input matrices, written by `scripts/build_inputs.py`. |

`build_inputs.py` fills values below detection with one tenth of the metabolite's minimum over all samples (all ages pooled), then takes log10. Scaling happens inside each cross-validation fold. The metabolite set used can be restricted with `--ref` (or `METABO_REF` in `run_final.sh`).

The synthetic `data/` each notebook generates uses distinct filenames (`synthetic_*`) from
`metabo/config.py`'s own defaults, purely so it reads as obviously synthetic at a glance; the
environment variables each notebook sets (`METABO_SOURCE`, `METABO_TARGET_FILE`,
`METABO_METAB_6M`/`_18M`/`_6Y`) point it at the renamed files instead.

## Notebooks

One notebook per research question, plus one for the preprocessing sensitivity analyses, each
calling the real pipeline code inline against a synthetic dataset each notebook generates
itself into `data/` on first run (not part of this repository). No extra data or batch run
needed. Every value in it is drawn from chosen statistical distributions (`numpy.random`,
with hand-picked parameters) — none of it is fit to, sampled from, or otherwise derived from
real data; see `src/scripts/make_synthetic_data.py`'s docstring for exactly what structure is
planted and why.

```bash
pip install -r requirements.txt
cd notebooks && jupyter notebook 01_training_evaluation.ipynb
```

| Notebook | Mirrors | Research question |
|---|---|---|
| `00_build_data.ipynb` | `scripts/make_synthetic_data.py`, `scripts/build_inputs.py`, `tests/` | Where does `data/` come from, and does the test suite pass? |
| `01_training_evaluation.ipynb` | `scripts/run_experiment.py`, `scripts/run_repeated_cv.py` | RQ2: does an embedding keep the predictive signal? |
| `02_permutation_test.ipynb` | `scripts/run_permutation.py` | RQ1: is the raw metabolome predictive at all? |
| `03_attribution.ipynb` | `analysis/explainability_stage1.py`, `_stage2.py`, `_stage2_rf.py`, `_stage2_vae_encoder.py`, `_stage2_fold_seeds.py` | RQ3: do the metabolites behind a prediction survive compression? |
| `04_sensitivity_analyses.ipynb` | `scripts/run_fill_sensitivity.py`, `run_scaling_sensitivity.py`, `run_trainonly_sensitivity.py` | Does the answer depend on the fill rule, the scaler, or when the fill is estimated? |

Each notebook puts `src/` on `sys.path` and imports the real `metabo` package (and, for
attribution, the real `analysis/*.py` modules) directly — nothing here is a reimplementation.
Every iteration count (embedding dimensions, bootstrap resamples, permutations, repeats) is
reduced from the study's own settings purely for speed, so each notebook finishes in well
under a minute; each notebook's intro cell says by how much, and its final cell gives the real
(unreduced) command for a full batch run. Fixed throughout: 6 months, `sex` and `fishoil` (the
two outcomes with planted, known structure) and `vitd` (no planted signal) — see
`scripts/make_synthetic_data.py`'s docstring for exactly what is planted and why.

No environment variables to set by hand: each notebook's first code cell points `metabo` at
`../data/` and writes any results of its own to `notebooks/results/` (not committed; regenerate
by rerunning).

### What they show, on this synthetic data

- **01**: `sex` (planted signal) loses most of its AUROC under every embedding at every
  dimension tried; `vitd` (no planted signal) stays near chance regardless. The repeated-CV
  paired difference for `sex` (raw minus PCA) has a 95% interval well clear of zero.
- **02**: `sex` and `fishoil` reject the permutation null clearly (p well below 0.05); `vitd`
  does not.
- **03**: the planted fish-oil marker (`metabolite001`) ranks 1st of ~90 metabolites on the raw
  data under every Stage 1 attribution method, then drops to a much lower rank after PCA, UMAP
  or VAE back-projection — accuracy and interpretability both degrade under compression. This
  holds up over a couple of extra fold seeds and under the VAE's encoder gradient too (its last
  section, run only if PyTorch is installed).
- **04**: the fill rule and the scaler barely move `sex`'s AUROC; estimating the fill separately
  per training fold (instead of once upfront) does not blow up the raw-vs-PCA gap either — no
  sign of the kind of leakage the sensitivity analyses were designed to catch, on this synthetic
  data.

These are illustrative numbers from synthetic data, not the paper's results.

## Running things directly, outside the notebooks

`run_final.sh` (in `src/`) orchestrates the full pipeline in stages (`inputs`, `perm`, `eval`,
`vae`, `repeated`, `sensitivity`, `attribution`, `summary`, or `all`) on whatever dataset
`METABO_DATA_DIR` points to; see its own header comment for each stage. It puts `src/` on
`PYTHONPATH` itself. A script run on its own needs the same, e.g. from the repo root:
`PYTHONPATH=src python src/scripts/run_experiment.py ...`, or `cd src` first and drop the
`src/` prefix. Model stages skip finished records when run again.

## Where the paper's tables and figures come from

| Paper | Output | Script |
|---|---|---|
| Table 1 (outcomes, n, events) | `runlog_*.jsonl`, `facts_for_draft.md` | `run_experiment.py`, `build_report.py` |
| Tables 2 and 3, Table S1 | `supplement_S1_auroc_all_pairs.csv`, `results_rq1_complete.csv` | `build_table_s1.py`, `plot_figures.py` |
| Table 4 (mean AUROC by k) | `stats/comparison_sets.csv`, `summary_auroc_vs_dim.csv` | `pair_statistics.py`, `summarize.py` |
| Repeated cross-validation | `repeated_cv/summary_*.csv` | `run_repeated_cv.py`, `summarize_repeated_cv.py` |
| BH q-values, Wilcoxon tests, calibration | `stats/*.csv` | `pair_statistics.py` |
| Fill, scaling and training-only arms | `sensitivity_*/summary_*.csv` | `run_*_sensitivity.py`, `summarize_*_sensitivity.py` |
| Table 5, Figure 3 | `explainability/stage2_*.csv`, `figures/fig6_explainability_compression.png` | `explainability_stage2*.py`, `summarize_*seeds.py`, `plot_explainability.py` |
| Table 6 | `explainability/stage1_*.csv` | `explainability_stage1.py`, `plot_attribution_controls.py` |
| Figure 1A, 1B | `figures/fig4_significance.png`, `figures/fig2_rq1_predictiveness.png` | `plot_figures.py` |
| Figure 2 | `figures/fig5_embedding_signal_loss.png` | `plot_embedding_loss.py` |
| Data checks (fill, missingness, PC1) | `stats/data_checks.csv` | `data_checks.py`, `marker_loadings.py` |

## Notes

- **Attribution through PCA.** For the logistic model, the metabolite weights are w = V' b, the exact linear map; every fold checks that they reproduce the classifier output. UMAP and the VAE have no linear inverse and use magnitude maps (absolute correlation, absolute decoder Jacobian, or the encoder gradient in `explainability_stage2_vae_encoder.py`). Permutation importances carry no sign, so the random-forest check sets negative values to zero and uses |V|' for PCA.
- **Checks.** `pytest tests` runs known-answer tests of the mappings. The attribution and data-check scripts stop when a built-in check fails: PCA weights against the decision function, the encoder gradient against a finite step, filled values against the fill rule, and row order before tables are combined.
- **Reproducing UMAP.** UMAP results depend on the machine and library versions even with a fixed seed.
- **macOS.** PyTorch and UMAP in one process can crash. The attribution scripts set `KMP_DUPLICATE_LIB_OK=TRUE` and `NUMBA_THREADING_LAYER=workqueue`.

## License

MIT (see `LICENSE`).


## Development

Parts of this codebase were developed with the assistance of an AI coding assistant (Claude, Anthropic), under the author's direction and review. All analysis logic, statistical methods, and reported results were verified by the author.