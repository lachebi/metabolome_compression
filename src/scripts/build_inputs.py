"""Build the per-timepoint input matrices from the raw metabolome file.

Steps: keep the child samples (6m, 18mth, 6y), keep the metabolite columns of the
reference set, fill values below detection with one tenth of the metabolite's minimum
over all child samples (all ages pooled), take log10 and write one file per timepoint.
Scaling is done later, inside each CV fold.

The source is semicolon-separated with decimal commas and has the columns ABCno and TPn.
--ref is a CSV whose header defines the metabolite set (in the paper, the 679 metabolites
of Lovric et al. 2024). Without --ref, all metabolite columns of the source are used.

    python scripts/build_inputs.py [--source FILE] [--ref FILE] [--outdir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metabo import config

META_COLS = ["PARENT_ID", "ABCno", "TPn", "AGE", "GENDER", "TIME POINT"]
TP_MAP = {"6m": "6m", "18mth": "18m", "6y": "6y"}


def metabolite_columns(df, ref):
    if ref:
        cols = list(pd.read_csv(ref, index_col=0, nrows=0).columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"{len(missing)} reference metabolites not in source, e.g. {missing[:3]}")
        return cols
    return [c for c in df.columns if c not in META_COLS]


def main(source, outdir, ref):
    df = pd.read_csv(source, sep=";", decimal=",", low_memory=False)
    df = df[df["TPn"].isin(TP_MAP)].copy()
    cols = metabolite_columns(df, ref)
    met = df[cols].apply(pd.to_numeric, errors="coerce")
    print(f"{len(df)} child samples, {met.shape[1]} metabolites, "
          f"{met.isna().mean().mean() * 100:.1f}% below detection")

    logd = np.log10(met.fillna(met.min(axis=0) / 10.0))
    logd.index = df["ABCno"].values
    logd.index.name = "id"
    outdir.mkdir(parents=True, exist_ok=True)
    for label, token in TP_MAP.items():
        sub = logd[df["TPn"].values == label]
        sub.to_csv(outdir / f"metabolites_{token}_for_ml.csv")
        print(f"metabolites_{token}_for_ml.csv: {sub.shape[0]} children")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(config.SOURCE_FILE))
    ap.add_argument("--ref", default=None, help="CSV whose header lists the metabolites to keep")
    ap.add_argument("--outdir", default=str(config.DATA_DIR))
    a = ap.parse_args()
    main(a.source, Path(a.outdir), a.ref)
