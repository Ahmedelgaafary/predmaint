"""Model lifecycle on REAL CWRU data: train on light loads, monitor on heavy loads, gated retraining.

Run: python examples/cwru_lifecycle.py CWRU_DIR [state_dir]
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predmaint.cwru import load_cwru  # noqa: E402
from predmaint.pipeline import monitor, run_training  # noqa: E402
from predmaint.registry import Store  # noqa: E402

root = sys.argv[1]
store = Store(sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="predmaint_cwru_"))
LIGHT, HEAVY = (0, 1), (2, 3)


def line(title, d):
    print(f"\n== {title}")
    for k in ("version", "feature_set", "promoted", "reason", "drift", "drift_share", "live_macro_f1", "action"):
        if k in d:
            v = d[k]
            print(f"   {k}: {round(v, 3) if isinstance(v, float) else v}")


train_light = load_cwru(root, loads=LIGHT, segment=(0.0, 0.7))
hold_light = load_cwru(root, loads=LIGHT, segment=(0.7, 1.0), hop_s=1.0)
r1 = run_training(store, train_light, hold_light, "physics_orders")
line("1. v1 trained on 0-1 hp (first 70% of each recording)", {**r1, "live_macro_f1": r1["metrics"]["macro_f1"]})

heavy_now = load_cwru(root, loads=HEAVY, hop_s=1.0)
line("2. production moves to 2-3 hp: monitor v1 with delayed labels", monitor(store, heavy_now, with_labels=True))

train_all = load_cwru(root, loads=(0, 1, 2, 3), segment=(0.0, 0.7))
hold_heavy = load_cwru(root, loads=HEAVY, segment=(0.7, 1.0), hop_s=1.0)
r2 = run_training(store, train_all, hold_heavy, "physics_orders")
line("3. retrain on all loads; gate on the heavy-load holdout (last 30%, unseen)", r2)
m = r2["metrics"]
print(f"   champion v1 on that holdout: {m['champion_macro_f1_on_holdout']:.3f} | challenger v2: {m['macro_f1']:.3f}")
