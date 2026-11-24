"""End-to-end lifecycle: train -> monitor -> drift -> retrain -> gated promotion -> rollback.

Run: python examples/lifecycle_demo.py [state_dir]   (synthetic data, no network needed)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predmaint.pipeline import monitor, run_training  # noqa: E402
from predmaint.registry import Store, champion_version, rollback  # noqa: E402
from predmaint.simulate import SimConfig, make_dataset  # noqa: E402

store = Store(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="predmaint_"))
regime_a = SimConfig(rpm_range=(1500, 1800))  # commissioning conditions
regime_b = SimConfig(rpm_range=(2200, 2600))  # plant moves to a higher operating speed


def show(title, d):
    print(f"\n== {title}")
    for k, v in d.items():
        if k in ("promoted", "reason", "version", "feature_set", "action", "drift", "live_macro_f1", "drift_share"):
            print(f"   {k}: {round(v, 3) if isinstance(v, float) else v}")


r1 = run_training(
    store, make_dataset(100, regime_a, 1), make_dataset(50, regime_a, 2), "generic_envelope"
)
show("1. v1 (generic envelope bins) trained on regime A", {**r1, "live_macro_f1": r1["metrics"]["macro_f1"]})

mon = monitor(store, make_dataset(50, regime_b, 3), with_labels=True)
show("2. plant runs at higher speed: monitor v1 with delayed labels", mon)

r2 = run_training(store, make_dataset(100, regime_b, 4), make_dataset(50, regime_b, 5), "physics_orders")
show("3. retrain with physics-order features; gate on regime-B holdout", r2)
m2 = r2["metrics"]
print(f"   champion macro-F1 on that holdout: {m2['champion_macro_f1_on_holdout']:.3f} | challenger: {m2['macro_f1']:.3f}")

r3 = run_training(store, make_dataset(100, regime_b, 4), make_dataset(50, regime_b, 5), "physics_orders")
show("4. same model retrained again: gate must reject (no improvement)", r3)

rollback(store, "1")
print(f"\n== 5. rollback -> champion is v{champion_version(store.activate())}")
rollback(store, r2["version"])
print(f"   restored     -> champion is v{champion_version(store.activate())}")
print(f"\nMLflow UI:  mlflow ui --backend-store-uri {store.tracking_uri}")
