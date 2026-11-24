"""Nested-CV evaluation of every timepoint x outcome x representation x dimension x classifier.

Records go to results/eval_<tp>[__<tag>].json. Records run in parallel with joblib
(--jobs); finished records are skipped when the script is run again.

    python scripts/run_experiment.py --timepoints 6m --targets sex --reps raw pca --jobs 8
"""
import argparse
import os
import time
from itertools import product

from joblib import Parallel, delayed

from metabo import config, data_io, evaluation, runlog


def record_key(tp, tar, rep, dim, model):
    return f"{tp}|{tar}|{rep}|{dim}|{model}"


def _eval_one(X, y, tp, tar, rep, dim, model, seed):
    """Evaluate one record; on failure return the error instead of raising."""
    t0 = time.time()
    key = record_key(tp, tar, rep, dim, model)
    try:
        rec, _ = evaluation.evaluate(X, y, rep, dim, model, random_state=seed,
                                     clf_n_jobs=1)
        rec.update(key=key, timepoint=tp, target=tar, representation=rep,
                   dim=dim, model=model, seconds=round(time.time() - t0, 1))
        return rec
    except Exception as e:
        return {"key": key, "timepoint": tp, "target": tar, "representation": rep,
                "dim": dim, "model": model, "__error__": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}


def main(timepoints, targets, reps, dims, models, smoke=False, run_tag=None,
         jobs=None):
    jobs = 1 if smoke else (jobs or os.cpu_count() or 1)
    suffix = f"__{run_tag}" if run_tag else ""
    rl = runlog.init_run(f"experiment{suffix}",
                         {"timepoints": timepoints, "reps": reps, "dims": dims,
                          "models": models, "jobs": jobs, "run_tag": run_tag})
    rl.log.info(f"parallel jobs = {jobs}")
    t_run = time.time()
    n_done = n_skip = n_fail = 0

    for tp in timepoints:
        X_all = data_io.load_metabolites(tp)
        T = data_io.load_targets()
        rl.event("timepoint_loaded", msg=f"[{tp}] metabolites {X_all.shape}",
                 timepoint=tp, n_subjects=int(X_all.shape[0]),
                 n_metabolites=int(X_all.shape[1]))
        out_path = config.RESULTS_DIR / f"eval_{tp}{suffix}.json"
        done = {r["key"]: r for r in data_io.load_results(out_path)} if out_path.exists() else {}

        for tar in (targets or list(T.columns)):
            if tar not in T.columns:
                rl.event("skip", timepoint=tp, target=tar, reason="not_a_target")
                continue
            X, y = data_io.align(X_all, T[tar])
            n_events = int(y.sum())
            if y.nunique() < 2 or n_events < config.MIN_EVENTS:
                rl.event("skip", msg=f"skip {tp}/{tar}: {n_events} events",
                         timepoint=tp, target=tar, reason="too_few_events",
                         n=int(len(y)), n_events=n_events)
                n_skip += 1
                continue
            rl.event("target", timepoint=tp, target=tar, n=int(len(y)),
                     n_events=n_events, prevalence=round(float(y.mean()), 4),
                     epv=round(n_events / X.shape[1], 4))

            pending = [(rep, dim, model)
                       for rep, model in product(reps, models)
                       for dim in ([0] if rep == "raw" else dims)
                       if record_key(tp, tar, rep, dim, model) not in done]
            if not pending:
                continue
            if smoke:
                pending = pending[:1]

            results = Parallel(n_jobs=jobs)(
                delayed(_eval_one)(X, y, tp, tar, rep, dim, model, config.SEED)
                for rep, dim, model in pending)

            for rec in results:
                if "__error__" in rec:
                    rl.event("record_fail", msg=f"FAIL {rec['key']}: {rec['__error__']}",
                             key=rec["key"], error=rec["__error__"])
                    n_fail += 1
                    continue
                done[rec["key"]] = rec
                n_done += 1
                rl.event("record_done",
                         msg=(f"done {rec['key']}  auroc={rec['auroc']:.3f} "
                              f"[{rec['auroc_lo']:.3f},{rec['auroc_hi']:.3f}]  "
                              f"{rec['seconds']}s"),
                         key=rec["key"], timepoint=tp, target=tar,
                         representation=rec["representation"], dim=rec["dim"],
                         model=rec["model"], auroc=rec["auroc"],
                         auroc_lo=rec["auroc_lo"], auroc_hi=rec["auroc_hi"],
                         auprc=rec["auprc"], mcc=rec["mcc"], brier=rec["brier"],
                         cal_slope=rec["cal_slope"], seconds=rec["seconds"])
            data_io.save_results(list(done.values()), out_path)
            if smoke:
                rl.event("run_end", msg="smoke stop", smoke=True)
                return

    rl.event("run_end", msg=f"end experiment: {n_done} done, {n_skip} skipped, "
             f"{n_fail} failed, {round(time.time()-t_run,1)}s",
             n_done=n_done, n_skip=n_skip, n_fail=n_fail,
             runtime_s=round(time.time() - t_run, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoints", nargs="+", default=config.TIMEPOINTS)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--reps", nargs="+", default=config.REPRESENTATIONS)
    ap.add_argument("--dims", nargs="+", type=int, default=config.EMBED_DIMS)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel record workers (default: all cores)")
    ap.add_argument("--run-tag", default=None,
                    help="output suffix so parallel machines don't clobber files")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.timepoints, a.targets, a.reps, a.dims, a.models, a.smoke, a.run_tag, a.jobs)
