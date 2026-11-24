"""Sensitivity of the raw-vs-embedding comparison to scaling: StandardScaler instead of
RobustScaler, on the positive controls. Everything else matches the baseline arm of
run_fill_sensitivity.py, which is the comparison. Results go to
results/sensitivity_scaling/standard/.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

from joblib import Parallel, delayed

from metabo import config, data_io

RUNS = [("6m", ["sex", "fishoil"]), ("18m", ["sex"]), ("6y", ["sex"])]
OUT = config.RESULTS_DIR / "sensitivity_scaling" / "standard"


def _eval_one(X, y, tp, tar, rep, dim, model):
    from sklearn.preprocessing import StandardScaler
    from metabo import config, evaluation

    evaluation.RobustScaler = StandardScaler  # patched inside the worker process
    t0 = time.time()
    key = f"{tp}|{tar}|{rep}|{dim}|{model}"
    try:
        scaler = type(evaluation.build_embed_pipeline("raw", 0, 0).named_steps["scale"]).__name__
        assert scaler == "StandardScaler", scaler
        rec, _ = evaluation.evaluate(X, y, rep, dim, model, random_state=config.SEED, clf_n_jobs=1)
        rec.update(key=key, timepoint=tp, target=tar, representation=rep, dim=dim, model=model,
                   scaler=scaler, seconds=round(time.time() - t0, 1))
        return rec
    except Exception as e:
        return {"key": key, "__error__": f"{type(e).__name__}: {e}"}


def main(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = data_io.load_targets()
    for tp, targets in RUNS:
        path = OUT / f"eval_{tp}.json"
        done = {r["key"]: r for r in data_io.load_results(path)} if path.exists() else {}
        X_all = data_io.load_metabolites(tp)
        for tar in targets:
            X, y = data_io.align(X_all, T[tar])
            pending = [(rep, dim, m) for rep in config.REPRESENTATIONS for m in config.MODELS
                       for dim in ([0] if rep == "raw" else config.EMBED_DIMS)
                       if f"{tp}|{tar}|{rep}|{dim}|{m}" not in done]
            res = Parallel(n_jobs=jobs)(delayed(_eval_one)(X, y, tp, tar, *p) for p in pending)
            failed = [r for r in res if "__error__" in r]
            done.update({r["key"]: r for r in res if "__error__" not in r})
            data_io.save_results(list(done.values()), path)
            print(f"{tp}/{tar}: {len(done)} records, {len(failed)} failed, {round(time.time() - t0)}s", flush=True)
            for r in failed:
                print("FAIL", r["key"], r["__error__"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    main(ap.parse_args().jobs)
