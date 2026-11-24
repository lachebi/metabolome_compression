"""Main runs on the 14 outcomes (config.OUTCOMES) at the three timepoints.

  perm  permutation test on the raw metabolome, config.N_PERMUTATION permutations
        -> results/permutation_<tp>_n<N>__cpu.json
  eval  nested CV for raw, PCA and UMAP with all three classifiers
        -> results/eval_<tp>__cpu.json
  vae   nested CV for the VAE (needs torch; uses a GPU if available)
        -> results/eval_<tp>__gpu.json

Finished records are skipped when a stage is run again.
"""
import argparse

from metabo import config


def main(stage, timepoints, jobs):
    if stage == "perm":
        import run_permutation
        run_permutation.main(timepoints, config.OUTCOMES, model="lr",
                             n_perm=config.N_PERMUTATION, run_tag="cpu", jobs=jobs)
    elif stage == "eval":
        import run_experiment
        run_experiment.main(timepoints, config.OUTCOMES, ["raw", "pca", "umap"],
                            config.EMBED_DIMS, config.MODELS, run_tag="cpu", jobs=jobs)
    elif stage == "vae":
        import run_experiment
        # One process by default: several processes would share one GPU.
        run_experiment.main(timepoints, config.OUTCOMES, ["vae"], config.EMBED_DIMS,
                            config.MODELS, run_tag="gpu", jobs=jobs or 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["perm", "eval", "vae"])
    ap.add_argument("--timepoints", nargs="+", default=config.TIMEPOINTS)
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers (default: all cores; vae: 1)")
    a = ap.parse_args()
    main(a.stage, a.timepoints, a.jobs)
