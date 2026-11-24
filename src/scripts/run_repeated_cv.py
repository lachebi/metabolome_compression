"""Repeated nested CV, logistic regression, raw against each embedding.

The classifier is fixed, so nothing is selected on the outer folds. Repeat r uses seed
SEED + r for the outer and inner splits, the embedding and the classifier. Within a
repeat all representations share the outer folds, so differences are paired. Repeat 0
uses the seed of the main run and reproduces its LR values.

Output in results/repeated_cv/: repeated_<tp>__<tag>.json (pooled and per-fold AUROC)
and oof/<tp>/<target>/<rep>_<dim>_r<repeat>.npy (pooled out-of-fold probabilities).

    python scripts/run_repeated_cv.py --reps raw pca umap --jobs 24 --run-tag cpu
    python scripts/run_repeated_cv.py --reps vae --jobs 1 --run-tag vae
"""
import argparse
import os
import re
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from metabo import config, data_io, evaluation, runlog

MODEL = "lr"
OUT_DIR = config.RESULTS_DIR / "repeated_cv"


def record_key(tp, tar, rep, dim, repeat):
    return f"{tp}|{tar}|{rep}|{dim}|r{repeat}"


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def _eval_one(X, y, tp, tar, rep, dim, repeat):
    """One record; on failure return the error instead of raising."""
    t0 = time.time()
    key = record_key(tp, tar, rep, dim, repeat)
    seed = config.SEED + repeat
    base = dict(key=key, timepoint=tp, target=tar, representation=rep, dim=dim,
                model=MODEL, repeat=repeat, seed=seed)
    try:
        yv, oof = evaluation.nested_cv_oof(X, y, rep, dim, MODEL, random_state=seed,
                                           clf_n_jobs=1)
        # same splitter and seed as the outer loop of nested_cv_oof
        outer = StratifiedKFold(config.OUTER_K, shuffle=True, random_state=seed)
        folds = [float(roc_auc_score(yv[te], oof[te])) for _, te in outer.split(X.values, yv)]
        return {**base, "auroc": float(roc_auc_score(yv, oof)), "fold_auroc": folds,
                "n": int(len(yv)), "n_events": int(yv.sum()),
                "seconds": round(time.time() - t0, 1), "__oof__": oof.astype("float32")}
    except Exception as e:
        return {**base, "__error__": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}


def main(timepoints, targets, reps, dims, repeats, jobs, run_tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rl = runlog.init_run(f"repeated_cv__{run_tag}",
                         {"timepoints": timepoints, "targets": targets, "reps": reps, "dims": dims,
                          "repeats": repeats, "model": MODEL, "jobs": jobs, "run_tag": run_tag})
    t_run = time.time()
    n_done = n_fail = 0
    for tp in timepoints:
        X_all = data_io.load_metabolites(tp)
        T = data_io.load_targets()
        path = OUT_DIR / f"repeated_{tp}__{run_tag}.json"
        done = {r["key"]: r for r in data_io.load_results(path)} if path.exists() else {}
        for tar in targets:
            X, y = data_io.align(X_all, T[tar])
            n_events = int(y.sum())
            if y.nunique() < 2 or n_events < config.MIN_EVENTS:
                rl.event("skip", msg=f"skip {tp}/{tar}: {n_events} events",
                         timepoint=tp, target=tar, reason="too_few_events",
                         n=int(len(y)), n_events=n_events)
                continue
            pending = [(rep, dim, r) for r in repeats for rep in reps
                       for dim in ([0] if rep == "raw" else dims)
                       if record_key(tp, tar, rep, dim, r) not in done]
            if not pending:
                continue
            results = Parallel(n_jobs=jobs)(
                delayed(_eval_one)(X, y, tp, tar, rep, dim, r) for rep, dim, r in pending)
            for rec in results:
                if "__error__" in rec:
                    rl.event("record_fail", msg=f"FAIL {rec['key']}: {rec['__error__']}",
                             key=rec["key"], error=rec["__error__"])
                    n_fail += 1
                    continue
                oof_dir = OUT_DIR / "oof" / tp / _slug(tar)
                oof_dir.mkdir(parents=True, exist_ok=True)
                np.save(oof_dir / f"{rec['representation']}_{rec['dim']}_r{rec['repeat']}.npy",
                        rec.pop("__oof__"))
                done[rec["key"]] = rec
                n_done += 1
            tmp = path.with_suffix(".tmp")
            data_io.save_results(list(done.values()), tmp)
            os.replace(tmp, path)
            rl.event("target_done", msg=f"{tp}/{tar}: {n_done} done, {n_fail} failed, "
                     f"{round(time.time() - t_run)}s", timepoint=tp, target=tar,
                     n_done=n_done, n_fail=n_fail)
    rl.event("run_end", msg=f"end repeated_cv: {n_done} done, {n_fail} failed, "
             f"{round(time.time() - t_run)}s", n_done=n_done, n_fail=n_fail,
             runtime_s=round(time.time() - t_run, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoints", nargs="+", default=config.TIMEPOINTS)
    ap.add_argument("--targets", nargs="+", default=config.OUTCOMES)
    ap.add_argument("--reps", nargs="+", default=config.REPRESENTATIONS)
    ap.add_argument("--dims", nargs="+", type=int, default=config.EMBED_DIMS)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--run-tag", required=True)
    a = ap.parse_args()
    main(a.timepoints, a.targets, a.reps, a.dims, a.repeats, a.jobs, a.run_tag)
