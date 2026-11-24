"""Sensitivity of the raw-vs-embedding comparison to the below-detection fill.

Runs the nested-CV evaluation on the positive controls (sex at 6m, 18m and 6y, fish oil
at 6m) with input matrices built from the same source file:

  baseline  fill = metabolite minimum / 10 (the main analysis), rerun so that all arms
            share one software stack; checked against the main input matrices
  minfill   fill = metabolite minimum
  perage    fill = minimum at the sample's own age / 10
  wang      fill = metabolite minimum, and metabolites with more than 30% missing in the
            child samples removed (as in Wang et al. 2026, Nat Commun,
            doi 10.1038/s41467-025-68115-3; their batch normalization is not reproduced)

Except in perage, the minimum is taken over the three child timepoints pooled.
Inputs go to <data dir>/sensitivity_fill/<arm>/, results to results/sensitivity_fill/<arm>/.
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from metabo import config

TP_MAP = {"6m": "6m", "18mth": "18m", "6y": "6y"}
ARMS = ["baseline", "minfill", "wang", "perage"]
RUNS = [("6m", ["sex", "fishoil"]), ("18m", ["sex"]), ("6y", ["sex"])]


def build_inputs(arm, df, ref_cols):
    """Write the three timepoint matrices of one arm; return their directory."""
    outdir = config.DATA_DIR / "sensitivity_fill" / arm
    outdir.mkdir(parents=True, exist_ok=True)
    met = df[ref_cols].apply(pd.to_numeric, errors="coerce")
    if arm == "wang":
        met = met.loc[:, met.isna().mean() <= 0.30]
    if arm == "perage":
        fill = met.groupby(df["TPn"].values).transform("min") / 10.0
        assert not fill.isna().any().any(), "a metabolite is unobserved at one age"
        logd = np.log10(met.fillna(fill))
    else:
        divisor = 10.0 if arm == "baseline" else 1.0
        logd = np.log10(met.fillna(met.min(axis=0) / divisor))
    logd.index = df["ABCno"].values
    logd.index.name = "id"
    for label, token in TP_MAP.items():
        logd[df["TPn"].values == label].to_csv(outdir / f"metabolites_{token}_for_ml.csv")
    print(f"{arm}: {logd.shape[1]} metabolites -> {outdir}")
    return outdir


def check_baseline(outdir):
    """The baseline arm must equal the main input matrices."""
    for tp, path in config.METAB_FILES.items():
        a = pd.read_csv(path, index_col=0)
        b = pd.read_csv(outdir / path.name, index_col=0)
        assert a.index.equals(b.index) and list(a.columns) == list(b.columns), tp
        assert np.allclose(a.values, b.values), tp
    print("baseline inputs match data/metabolites_*_for_ml.csv")


def main(arms, reps, models, dims, jobs):
    ref_cols = list(pd.read_csv(config.METAB_FILES["6m"], index_col=0, nrows=0).columns)
    df = pd.read_csv(config.SOURCE_FILE, sep=";", decimal=",", low_memory=False)
    df = df[df["TPn"].isin(TP_MAP)].copy()
    base_results = config.RESULTS_DIR

    import run_experiment  # reads config at call time, so patching config below redirects it

    for arm in arms:
        indir = build_inputs(arm, df, ref_cols)
        if arm == "baseline":
            check_baseline(indir)
        config.METAB_FILES = {t: indir / f"metabolites_{t}_for_ml.csv"
                              for t in config.TIMEPOINTS}
        config.RESULTS_DIR = base_results / "sensitivity_fill" / arm
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for tp, targets in RUNS:
            run_experiment.main([tp], targets, reps, dims, models, jobs=jobs,
                                run_tag="local")
        config.METAB_FILES = {t: config.DATA_DIR / f"metabolites_{t}_for_ml.csv"
                              for t in config.TIMEPOINTS}
    config.RESULTS_DIR = base_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--reps", nargs="+", default=config.REPRESENTATIONS)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--dims", nargs="+", type=int, default=config.EMBED_DIMS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    main(a.arms, a.reps, a.models, a.dims, a.jobs)
