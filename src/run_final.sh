#!/usr/bin/env bash
# Runs the analysis in stages:  bash run_final.sh <stage> [jobs]
#
#   inputs       build the input matrices from the source file (METABO_REF = metabolite set)
#   perm         permutation test, raw metabolome
#   eval         nested CV, raw, PCA and UMAP
#   vae          nested CV, VAE (needs torch)
#   repeated     repeated nested CV, logistic regression
#   sensitivity  fill, scaling and training-only arms on the positive controls
#   attribution  Stage 1 and 2 attribution, seed and shuffled-label runs
#   summary      tables and figures from the results
#   all          everything above, in order
#
# Optional: REPEATS (default 10), SEEDS (default "42 ... 51"), NULL_RUNS (default 100).
# All model stages skip finished records when run again.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${1:-}"
JOBS="${2:-}"
JOBS_ARG=""; [ -n "$JOBS" ] && JOBS_ARG="--jobs $JOBS"
REPEATS="${REPEATS:-10}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
NULL_RUNS="${NULL_RUNS:-100}"

run_stage() {
  case "$1" in
    inputs)
      python scripts/build_inputs.py ${METABO_REF:+--ref "$METABO_REF"} ;;
    perm)
      python scripts/run_final.py --stage perm $JOBS_ARG ;;
    eval)
      python scripts/run_final.py --stage eval $JOBS_ARG ;;
    vae)
      python scripts/run_final.py --stage vae $JOBS_ARG ;;
    repeated)
      python scripts/run_repeated_cv.py --reps raw pca umap vae --repeats $(seq 0 $((REPEATS - 1))) \
        --run-tag main $JOBS_ARG ;;
    sensitivity)
      python scripts/run_fill_sensitivity.py $JOBS_ARG
      python scripts/run_scaling_sensitivity.py $JOBS_ARG
      python scripts/run_trainonly_sensitivity.py --arm fill $JOBS_ARG
      python scripts/run_trainonly_sensitivity.py --arm fill_filter $JOBS_ARG ;;
    attribution)
      python analysis/explainability_stage1.py
      python analysis/explainability_stage2.py
      python analysis/explainability_stage2_rf.py
      python analysis/explainability_stage2_pca_signed.py
      python analysis/explainability_stage2_vae_encoder.py
      python analysis/explainability_stage2_vae_encoder.py --seeds $SEEDS --vary vae
      python analysis/explainability_stage2_vae_encoder.py --seeds $SEEDS --vary both
      python analysis/explainability_stage2_vae_encoder.py --null "$NULL_RUNS"
      python analysis/explainability_stage2_fold_seeds.py --seeds $SEEDS
      python analysis/explainability_stage2_fold_seeds.py --null "$NULL_RUNS" ;;
    summary)
      python analysis/record_versions.py
      python analysis/summarize.py
      python analysis/build_report.py
      python analysis/plot_figures.py
      python analysis/plot_embedding_loss.py
      python analysis/pair_statistics.py
      python analysis/data_checks.py
      python analysis/build_table_s1.py
      python analysis/summarize_best3_singlesplit.py
      python analysis/summarize_repeated_cv.py
      python analysis/summarize_repeated_cv_sets_by_dim.py
      python analysis/summarize_fill_sensitivity.py
      python analysis/summarize_scaling_sensitivity.py
      python analysis/summarize_trainonly_sensitivity.py
      python analysis/marker_loadings.py
      python analysis/summarize_encoder_seeds.py
      python analysis/summarize_fold_seeds.py
      python analysis/plot_explainability.py
      python analysis/plot_attribution_controls.py ;;
    *)
      echo "usage: bash run_final.sh {inputs|perm|eval|vae|repeated|sensitivity|attribution|summary|all} [jobs]" >&2
      exit 2 ;;
  esac
}

if [ "$STAGE" = "all" ]; then
  for s in inputs perm eval vae repeated sensitivity attribution summary; do
    echo ">>> $s"; run_stage "$s"
  done
else
  run_stage "$STAGE"
fi
