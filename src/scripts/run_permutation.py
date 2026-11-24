"""Label-permutation test on the raw metabolome, per timepoint and outcome.

Writes results/permutation_<tp>_n<n_perm>[__<tag>].json. Outcomes run in parallel
(--jobs); finished outcomes are skipped when the script is run again.

    python scripts/run_permutation.py --timepoints 6m --n_perm 1000 --jobs 8
"""
import argparse
import os
import time

from joblib import Parallel, delayed

from metabo import config, data_io, evaluation, runlog


def _perm_one(X, y, tp, tar, model, n_perm, seed):
    """Permutation test for one outcome; on failure return the error instead of raising."""
    t0 = time.time()
    key = f"{tp}|{tar}|raw|{model}"
    try:
        res = evaluation.permutation_null(X, y, "raw", 0, model, n_perm=n_perm,
                                          random_state=seed, clf_n_jobs=1)
        res.update(key=key, timepoint=tp, target=tar, model=model,
                   n=int(len(y)), n_events=int(y.sum()),
                   seconds=round(time.time() - t0, 1))
        return res
    except Exception as e:
        return {"key": key, "timepoint": tp, "target": tar,
                "__error__": f"{type(e).__name__}: {e}"}


def main(timepoints, targets, model="lr", n_perm=None, run_tag=None, jobs=None):
    jobs = jobs or os.cpu_count() or 1
    n_perm = n_perm or config.N_PERMUTATION
    suffix = f"__{run_tag}" if run_tag else ""
    rl = runlog.init_run(f"permutation{suffix}",
                         {"timepoints": timepoints, "model": model,
                          "n_perm": n_perm,
                          "jobs": jobs, "run_tag": run_tag})
    rl.log.info(f"parallel jobs = {jobs}")
    t_run = time.time()
    n_done = n_skip = 0

    for tp in timepoints:
        X_all = data_io.load_metabolites(tp)
        T = data_io.load_targets()
        rl.event("timepoint_loaded", msg=f"[{tp}] metabolites {X_all.shape}",
                 timepoint=tp, n_subjects=int(X_all.shape[0]),
                 n_metabolites=int(X_all.shape[1]))
        out_path = config.RESULTS_DIR / f"permutation_{tp}_n{n_perm}{suffix}.json"
        done = {r["key"]: r for r in data_io.load_results(out_path)} if out_path.exists() else {}

        pending = []
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
            if f"{tp}|{tar}|raw|{model}" in done:
                continue
            pending.append((tar, X, y))
        if not pending:
            continue

        results = Parallel(n_jobs=jobs)(
            delayed(_perm_one)(X, y, tp, tar, model, n_perm, config.SEED)
            for tar, X, y in pending)

        for res in results:
            if "__error__" in res:
                rl.event("record_fail", msg=f"FAIL {res['key']}: {res['__error__']}",
                         key=res["key"], error=res["__error__"])
                continue
            done[res["key"]] = res
            n_done += 1
            rl.event("permutation_done",
                     msg=(f"{res['key']}: obs_auroc={res['obs_auroc']:.3f}  "
                          f"p={res['p_value']:.3f}  {res['seconds']}s"),
                     key=res["key"], timepoint=res["timepoint"], target=res["target"],
                     n=res["n"], n_events=res["n_events"], obs_auroc=res["obs_auroc"],
                     null_mean=res["null_mean"], p_value=res["p_value"],
                     predictive=bool(res["p_value"] < 0.05), seconds=res["seconds"])
        data_io.save_results(list(done.values()), out_path)

    rl.event("run_end", msg=f"end permutation: {n_done} done, {n_skip} skipped, "
             f"{round(time.time()-t_run,1)}s",
             n_done=n_done, n_skip=n_skip, runtime_s=round(time.time() - t_run, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoints", nargs="+", default=config.TIMEPOINTS)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--model", default="lr")
    ap.add_argument("--n_perm", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel target workers (default: all cores)")
    ap.add_argument("--run-tag", default=None,
                    help="output suffix so parallel machines don't clobber files")
    a = ap.parse_args()
    main(a.timepoints, a.targets, a.model, a.n_perm, a.run_tag, a.jobs)
