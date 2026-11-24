"""Real-data benchmark on CWRU (12 kHz drive end, 4 classes).

  python benchmarks/run_cwru.py --root DIR --protocol temporal [--seeds 3]
  python benchmarks/run_cwru.py --report            # aggregate saved runs into benchmarks/results_cwru.md

One protocol per invocation (the CNN is slow on a single CPU core); results are saved as JSON.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

from predmaint.cwru import PROTOCOLS, class_counts, protocol  # noqa: E402
from predmaint.features import FEATURE_SETS, build_matrix  # noqa: E402

OUT = ROOT / "benchmarks"
NAMES = {
    "generic_spectrum": "HGB + generic spectrum",
    "generic_envelope": "HGB + generic envelope (Hz bins)",
    "physics_orders": "HGB + physics orders",
    "cnn": "1D-CNN on raw windows",
}


def score(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ball_recall": float(recall_score(y, pred, labels=[3], average=None)[0]),
    }


def run(root: str, proto: str, seeds: int) -> dict:
    train, test = protocol(proto, root)
    out = {"protocol": proto, "n_train": len(train), "n_test": len(test), "test_counts": class_counts(test), "models": {}}
    print(f"{proto}: train {len(train)} windows, test {len(test)} {class_counts(test)}", flush=True)
    for kind in FEATURE_SETS:
        Xtr, Xte = build_matrix(train, kind), build_matrix(test, kind)
        out["models"][kind] = [
            score(test.labels, HistGradientBoostingClassifier(max_iter=150, random_state=s).fit(Xtr, train.labels).predict(Xte))
            for s in range(seeds)
        ]
        print(f"  {kind}: macro-F1 {np.mean([r['macro_f1'] for r in out['models'][kind]]):.3f}", flush=True)
    try:
        from predmaint.cnn import CNNClassifier
    except ImportError:
        print("  torch not installed: skipping CNN")
        return out
    runs = []
    for s in range(seeds):
        t0 = time.time()
        runs.append(score(test.labels, CNNClassifier(seed=s).fit(train.signals, train.labels).predict(test.signals)))
        print(f"  cnn seed {s}: macro-F1 {runs[-1]['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    out["models"]["cnn"] = runs
    return out


def report() -> str:
    lines = ["Macro-F1 / recall of the hard 'ball' class, mean ± std over seeds. REAL CWRU data (12 kHz drive end).", ""]
    lines += ["| model | " + " | ".join(PROTOCOLS) + " |", "|---|" + "---|" * len(PROTOCOLS)]
    data = {p: json.loads((OUT / f"cwru_{p}.json").read_text()) for p in PROTOCOLS if (OUT / f"cwru_{p}.json").exists()}
    for key, name in NAMES.items():
        cells = []
        for p in PROTOCOLS:
            runs = data.get(p, {}).get("models", {}).get(key)
            if not runs:
                cells.append("n/a")
                continue
            f1, br = [r["macro_f1"] for r in runs], [r["ball_recall"] for r in runs]
            cells.append(f"{np.mean(f1):.3f} ± {np.std(f1):.3f} (ball {np.mean(br):.2f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for p, d in data.items():
        lines.append(f"- {p}: {d['n_train']} train / {d['n_test']} test windows, test classes {d['test_counts']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--protocol", choices=PROTOCOLS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        text = report()
        (OUT / "results_cwru.md").write_text(text + "\n")
        print(text)
    else:
        res = run(a.root, a.protocol, a.seeds)
        (OUT / f"cwru_{a.protocol}.json").write_text(json.dumps(res, indent=1))
