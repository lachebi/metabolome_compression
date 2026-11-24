"""Run log: settings, software versions, dataset sizes and per-record results,
written as one JSON object per line to results/runlog_<name>.jsonl.
"""
import datetime
import json
import logging
import platform
import socket
import subprocess
import sys

from metabo import config

_PROV_CONFIG_KEYS = ["SEED", "OUTER_K", "INNER_K", "N_BOOTSTRAP", "N_PERMUTATION",
                     "MIN_EVENTS", "EMBED_DIMS", "REPRESENTATIONS", "MODELS",
                     "UMAP_N_NEIGHBORS", "UMAP_MIN_DIST",
                     "VAE_MAX_EPOCHS", "VAE_PATIENCE", "VAE_VAL_FRACTION"]


def _versions():
    vers = {}
    for p in ["sklearn", "numpy", "pandas", "scipy", "umap"]:
        try:
            vers[p] = getattr(__import__(p), "__version__", "?")
        except Exception:
            vers[p] = None
    try:
        import torch
        vers["torch"] = torch.__version__
        vers["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        vers["torch"] = None
    return vers


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(config.ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class RunLogger:
    def __init__(self, name):
        self.name = name
        self.path = config.RESULTS_DIR / f"runlog_{name}.jsonl"
        self.log = logging.getLogger(f"runlog.{name}")
        if not self.log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
            self.log.addHandler(h)
            self.log.setLevel(logging.INFO)

    def event(self, kind, msg=None, **fields):
        """Append one event; print msg if given."""
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "kind": kind, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if msg:
            self.log.info(msg)
        return rec


def init_run(name, params):
    """Open a run log and record settings and software versions."""
    rl = RunLogger(name)
    rl.event("run_start",
             msg=f"start {name}",
             params=params,
             provenance={"git": _git_sha(), "host": socket.gethostname(),
                         "python": sys.version.split()[0],
                         "platform": platform.platform(),
                         "versions": _versions()},
             settings={k: getattr(config, k) for k in _PROV_CONFIG_KEYS})
    return rl
