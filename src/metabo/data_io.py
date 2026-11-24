"""Loading inputs and reading/writing result records (JSON lists of dicts)."""
import json

import pandas as pd

from metabo import config


def load_metabolites(timepoint):
    """Metabolite matrix (log10) for one timepoint, indexed by child id."""
    return pd.read_csv(config.METAB_FILES[timepoint], index_col=0)


def load_targets():
    """Binary outcome columns, indexed by child id."""
    return pd.read_csv(config.TARGET_FILE, index_col=0)


def align(X, y):
    """Drop missing labels, keep children present in both, labels as int."""
    y = y.dropna()
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx].astype(int)


def save_results(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def load_results(path):
    with open(path) as f:
        return json.load(f)
