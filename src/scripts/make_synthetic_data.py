"""Synthetic data with the pipeline's expected layout, for testing the code without needing any outside data.

Every value is drawn from chosen statistical distributions (numpy.random, with hand-picked
parameters below) -- none of it is fit to, sampled from, or otherwise derived from real data
of any kind.

Writes two files into --outdir (default: config.DATA_DIR); existing files are only
overwritten with --force:
  datacorr_6m6y18mMw1w24_final.csv  raw intensities in the source layout (semicolon
                                    separated, decimal comma, columns ABCno, TPn, GENDER,
                                    ...), child samples at 6m, 18mth and 6y plus maternal
                                    samples, values below detection left empty
  target_for_metabolites_ml.csv     the 14 outcomes of config.OUTCOMES, one row per child
Then build the input matrices with:  python scripts/build_inputs.py

Planted structure (log10 scale); metabolites are named generically (metabolite001, ...),
not with real chemical nomenclature (deliberately with no "_", unlike the real naming
convention "name_pathway_superclass", so nothing truncates it looking for that pattern):
  - 30 nuisance factors carry most of the variance, so the leading components of an
    embedding mostly describe variation unrelated to the outcomes
  - sex: a panel of six metabolites, at every age
  - fishoil: one metabolite with a strong effect, two more with weaker effects, at 6
    months only (mirrors a real single-marker fish-oil footprint with a weaker parent
    compound and two still-weaker related metabolites)
  - mother_smoking: two metabolites at 6 months
  - a latent atopy trait drives the allergy outcomes and three metabolites at 6 years
  - fetal-Delivery: two metabolites at 6 months
  - vitd and the antibiotic outcomes: no effect
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metabo import config

SEED = 2026
N_CHILDREN = 260
N_AT = {"6m": 240, "18mth": 220, "6y": 200}
N_OTHER = 80
N_NUISANCE = 30

# Keys are used only to look up each marker's column position below; values are the
# generic, sequential names actually written to the data (metabolite_001, metabolite_002,
# ...) -- not real chemical nomenclature.
MARKERS = list(dict.fromkeys([
    "hydroxy_cmpf", "cmpf", "epa", "dha",
    "pregnanediol", "pregnenetriol", "androstenediol", "pregnendiol", "dhea", "androsterone",
    "cotinine", "hydroxycotinine",
]))


def make(seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.arange(9001, 9001 + N_CHILDREN)
    p = len(MARKERS) + N_OTHER
    names = [f"metabolite{i + 1:03d}" for i in range(p)]
    col = {k: i for i, k in enumerate(MARKERS)}

    # child-level traits and outcomes
    male = rng.random(N_CHILDREN) < 0.5
    fishoil = (rng.random(N_CHILDREN) < 0.5).astype(float)
    vitd = (rng.random(N_CHILDREN) < 0.5).astype(float)
    smoking = (rng.random(N_CHILDREN) < 0.09).astype(float)
    atopy = rng.normal(size=N_CHILDREN)
    ige = (atopy + rng.normal(0, 0.8, N_CHILDREN) > 0.6).astype(float)
    food = ige * (rng.normal(size=N_CHILDREN) + atopy > 0.8)
    inhal = ige * (rng.normal(size=N_CHILDREN) + atopy > 0.3)
    delivery = (rng.random(N_CHILDREN) < 0.22).astype(float)
    targets = pd.DataFrame({
        "AD 6 yr ever": (0.4 * atopy + rng.normal(size=N_CHILDREN) > 0.9).astype(float),
        "Asthma 6 yr ever": (0.4 * atopy + rng.normal(size=N_CHILDREN) > 1.1).astype(float),
        "atopy_temp": ((ige + (atopy > 1.0)) > 0).astype(float),
        "sens_s_ig_e_any": ige,
        "sens_inhal": inhal.astype(float),
        "sens_food": food.astype(float),
        "ar_6y": (inhal * (rng.random(N_CHILDREN) < 0.5)).astype(float),
        "fetal-Antibiotics during pregnancy": (rng.random(N_CHILDREN) < 0.3).astype(float),
        "fetal-Antibiotics during birth": np.where(delivery == 1, rng.random(N_CHILDREN) < 0.8,
                                                   rng.random(N_CHILDREN) < 0.1).astype(float),
        "fetal-Delivery": delivery,
        "mother_smoking": smoking,
        "fishoil": fishoil,
        "vitd": vitd,
        "sex": male.astype(float),
    }, index=pd.Index(ids, name="id"))
    clinical = list(config.ROSTER[:7])
    for c in clinical:  # some children lack the 6-year assessment
        targets.loc[rng.random(N_CHILDREN) < 0.08, c] = np.nan

    base = rng.uniform(5.0, 8.5, p)
    loadings = rng.normal(0, 1, (N_NUISANCE, p)) * np.linspace(0.16, 0.05, N_NUISANCE)[:, None]
    child_nuisance = rng.normal(size=(N_CHILDREN, N_NUISANCE))

    rows, tps, sample_ids, gender = [], [], [], []
    for tp, n in N_AT.items():
        idx = np.sort(rng.choice(N_CHILDREN, n, replace=False))
        # nuisance: partly child-specific, partly new at each age
        z = 0.4 * child_nuisance[idx] + 0.9 * rng.normal(size=(n, N_NUISANCE))
        x = base + z @ loadings + rng.normal(0, 0.25, (n, p)) + rng.normal(0, 0.3, p)
        steroid = [col[k] for k in ("pregnanediol", "pregnenetriol", "androstenediol",
                                    "pregnendiol", "dhea", "androsterone")]
        x[:, steroid] += 0.25 * male[idx, None]
        if tp == "6m":
            x[:, col["hydroxy_cmpf"]] += 0.45 * fishoil[idx]
            x[:, col["cmpf"]] += 0.15 * fishoil[idx]
            x[:, [col["epa"], col["dha"]]] += 0.1 * fishoil[idx, None]
            x[:, [col["cotinine"], col["hydroxycotinine"]]] += 0.45 * smoking[idx, None]
            x[:, [N_OTHER // 2 + len(MARKERS), N_OTHER // 3 + len(MARKERS)]] += 0.15 * delivery[idx, None]
        if tp == "6y":
            x[:, len(MARKERS):len(MARKERS) + 3] += 0.15 * atopy[idx, None]
        rows.append(x)
        tps += [tp] * n
        sample_ids += list(ids[idx])
        gender += list(np.where(male[idx], "Male", "Female"))

    for tp in ("Mor w24", "Mor w1"):  # maternal samples, not analyzed
        x = base + rng.normal(size=(N_CHILDREN, N_NUISANCE)) @ loadings + rng.normal(0, 0.25, (N_CHILDREN, p)) + 0.3
        rows.append(x)
        tps += [tp] * N_CHILDREN
        sample_ids += list(ids)
        gender += [np.nan] * N_CHILDREN

    logv = np.vstack(rows)
    # values below a metabolite-specific detection limit are missing
    q = rng.uniform(0.0, 0.12, p)
    q[rng.choice(p, 10, replace=False)] = rng.uniform(0.15, 0.3, 10)
    lod = np.array([np.quantile(logv[:, j], q[j]) for j in range(p)])
    raw = np.where(logv < lod, np.nan, 10 ** logv)
    source = pd.DataFrame(raw, columns=names)
    source.insert(0, "TIME POINT", "")
    source.insert(0, "GENDER", gender)
    source.insert(0, "AGE", "")
    source.insert(0, "TPn", tps)
    source.insert(0, "ABCno", sample_ids)
    source.insert(0, "PARENT_ID", np.arange(len(source)))
    return source, targets


def main(outdir, force=False):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / "datacorr_6m6y18mMw1w24_final.csv", outdir / "target_for_metabolites_ml.csv"]
    if not force and any(p.exists() for p in paths):
        raise SystemExit(f"{outdir} already has data files; use another --outdir or --force")
    source, targets = make()
    source.to_csv(outdir / "datacorr_6m6y18mMw1w24_final.csv", sep=";", decimal=",", index=False)
    targets.to_csv(outdir / "target_for_metabolites_ml.csv")
    print(f"wrote {len(source)} samples x {source.shape[1] - 6} metabolites and "
          f"{targets.shape[1]} outcomes to {outdir}")
    print("events:", {c: int(targets[c].sum()) for c in targets.columns})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(config.DATA_DIR))
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()
    main(Path(a.outdir), a.force)
