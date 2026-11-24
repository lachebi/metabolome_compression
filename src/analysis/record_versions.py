"""Software versions of this machine, written to results/stats/software_versions.json."""
import importlib
import json
import platform

from metabo import config

out = {"python": platform.python_version(), "machine": platform.platform()}
for name in ["numpy", "scipy", "pandas", "sklearn", "umap", "torch", "joblib"]:
    try:
        out[name] = getattr(importlib.import_module(name), "__version__", "unknown")
    except ImportError:
        out[name] = None
(config.RESULTS_DIR / "stats").mkdir(exist_ok=True)
(config.RESULTS_DIR / "stats" / "software_versions.json").write_text(json.dumps(out, indent=1) + "\n")
print(out)
