"""Fill and missingness filter estimated on training rows only, on the positive controls.

In the main analysis the fill (one tenth of the metabolite minimum over all child samples)
is set once, before cross-validation. Here it is estimated inside every training set,
inner and outer, and applied to the held-out rows. Held-out children are excluded at all
ages. Everything else matches the baseline arm of run_fill_sensitivity.py.

Arms:
  fill         training-only minimum, all metabolites
  fill_filter  training-only minimum, and metabolites over 33% missing in the training
               children's samples dropped
  check        minimum over all child samples through the same code; must reproduce
               the main analysis

Results go to results/sensitivity_trainonly/<arm>/.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from metabo import config, data_io

RUNS = [("6m", ["sex", "fishoil"]), ("18m", ["sex"]), ("6y", ["sex"])]
TP_MAP = {"6m": "6m", "18mth": "18m", "6y": "6y"}
OUT = config.RESULTS_DIR / "sensitivity_trainonly"


class TrainOnlyFill(BaseEstimator, TransformerMixin):
    """Fill (and optional filter) fitted on the training children at all ages.

    The first column of X is the child ID; it is dropped on transform.
    """

    def __init__(self, other=None, all_samples=None, use_filter=False):
        self.other = other              # log10 values at the other ages, indexed by ID
        self.all_samples = all_samples  # check arm only
        self.use_filter = use_filter

    def fit(self, X, y=None):
        ids, values = X[:, 0], X[:, 1:]
        if self.all_samples is not None:
            pool = self.all_samples
        else:
            pool = np.vstack([values, self.other[self.other.index.isin(ids)].values])
        self.fill_ = np.nanmin(pool, axis=0) - 1.0  # log10 scale: minimum / 10
        self.keep_ = (np.isnan(pool).mean(axis=0) <= 1 / 3 if self.use_filter
                      else np.ones(values.shape[1], bool))
        return self

    def transform(self, X):
        values = X[:, 1:].copy()
        mask = np.isnan(values)
        values[mask] = np.take(self.fill_, np.where(mask)[1])
        return values[:, self.keep_]


def _eval_one(X, y, other, all_samples, use_filter, tp, tar, rep, dim, model):
    from sklearn.pipeline import Pipeline
    from metabo import config, evaluation

    # workers are reused between records, so wrap the original function only once
    if not hasattr(evaluation, "_unpatched_build"):
        evaluation._unpatched_build = evaluation.build_embed_pipeline

    def patched(representation, d, random_state, n_neighbors=None):
        base = evaluation._unpatched_build(representation, d, random_state, n_neighbors)
        step = TrainOnlyFill(other=other, all_samples=all_samples, use_filter=use_filter)
        return Pipeline([("trainfill", step)] + base.steps)

    evaluation.build_embed_pipeline = patched  # patched inside the worker process
    t0 = time.time()
    key = f"{tp}|{tar}|{rep}|{dim}|{model}"
    try:
        rec, _ = evaluation.evaluate(X, y, rep, dim, model, random_state=config.SEED, clf_n_jobs=1)
        rec.update(key=key, timepoint=tp, target=tar, representation=rep, dim=dim, model=model,
                   seconds=round(time.time() - t0, 1))
        return rec
    except Exception as e:
        return {"key": key, "__error__": f"{type(e).__name__}: {e}"}


def load_nan_matrices():
    """log10 values with missing values kept; checked against the main input matrices."""
    df = pd.read_csv(config.SOURCE_FILE, sep=";", decimal=",", low_memory=False)
    df = df[df["TPn"].isin(TP_MAP)]
    cols = list(pd.read_csv(config.METAB_FILES["6m"], index_col=0, nrows=0).columns)
    logd = np.log10(df[cols].apply(pd.to_numeric, errors="coerce"))
    logd.index = df["ABCno"].values
    out = {t: logd[df["TPn"].values == s] for s, t in TP_MAP.items()}
    fill = logd.min(axis=0) - 1.0
    for t, frame in out.items():
        ref = pd.read_csv(config.METAB_FILES[t], index_col=0)
        assert frame.index.equals(ref.index) and list(frame.columns) == list(ref.columns), t
        assert np.allclose(frame.fillna(fill).values, ref.values), t
    return out, logd.values


def main(arm, reps, models, dims, runs, jobs):
    n_failed = 0
    path_dir = OUT / arm
    path_dir.mkdir(parents=True, exist_ok=True)
    nan_by_tp, all_values = load_nan_matrices()
    T = data_io.load_targets()
    t0 = time.time()
    for tp, targets in runs:
        path = path_dir / f"eval_{tp}.json"
        done = {r["key"]: r for r in data_io.load_results(path)} if path.exists() else {}
        frame = nan_by_tp[tp]
        other = pd.concat([f for t, f in nan_by_tp.items() if t != tp])
        for tar in targets:
            Xf, y = data_io.align(frame, T[tar])
            X = Xf.copy()
            X.insert(0, "__id__", X.index.values.astype(float))
            pending = [(rep, dim, m) for rep in reps for m in models
                       for dim in ([0] if rep == "raw" else dims)
                       if f"{tp}|{tar}|{rep}|{dim}|{m}" not in done]
            res = Parallel(n_jobs=jobs)(
                delayed(_eval_one)(X, y, other, all_values if arm == "check" else None,
                                   arm == "fill_filter", tp, tar, *p) for p in pending)
            failed = [r for r in res if "__error__" in r]
            n_failed += len(failed)
            done.update({r["key"]: r for r in res if "__error__" not in r})
            data_io.save_results(list(done.values()), path)
            print(f"{arm} {tp}/{tar}: {len(done)} records, {len(failed)} failed, {round(time.time() - t0)}s", flush=True)
            for r in failed:
                print("FAIL", r["key"], r["__error__"], flush=True)
    if n_failed:
        raise SystemExit(f"{arm}: {n_failed} records failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["fill", "fill_filter", "check"])
    ap.add_argument("--reps", nargs="+", default=config.REPRESENTATIONS)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--dims", nargs="+", type=int, default=config.EMBED_DIMS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--one-pair", action="store_true", help="only sex at 6m")
    a = ap.parse_args()
    main(a.arm, a.reps, a.models, a.dims, [("6m", ["sex"])] if a.one_pair else RUNS, a.jobs)
