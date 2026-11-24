"""Feature-set comparison across seeds. Writes benchmarks/results.md.  SYNTHETIC data (see README)."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from predmaint.features import FEATURE_SETS, build_matrix  # noqa: E402
from predmaint.simulate import SimConfig, make_dataset  # noqa: E402

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
A, B = SimConfig(rpm_range=(1500, 1800)), SimConfig(rpm_range=(2200, 2600))
res = {k: {"id": [], "shift": []} for k in FEATURE_SETS}
t0 = time.time()
for seed in SEEDS:
    tr, te, sh = make_dataset(100, A, 10 * seed + 1), make_dataset(40, A, 10 * seed + 2), make_dataset(40, B, 10 * seed + 3)
    for k in FEATURE_SETS:
        m = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(build_matrix(tr, k), tr.labels)
        res[k]["id"].append(f1_score(te.labels, m.predict(build_matrix(te, k)), average="macro"))
        res[k]["shift"].append(f1_score(sh.labels, m.predict(build_matrix(sh, k)), average="macro"))

def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

lines = [
    f"Macro-F1 over {len(SEEDS)} seeds (mean ± std). Train: 1500-1800 rpm, 400 windows. SYNTHETIC data.",
    "",
    "| feature set | in-distribution (1500-1800 rpm) | speed shift (2200-2600 rpm) |",
    "|---|---|---|",
] + [f"| {k} | {ms(res[k]['id'])} | {ms(res[k]['shift'])} |" for k in FEATURE_SETS]
out = "\n".join(lines)
print(out, f"\n({time.time() - t0:.0f}s)")
(ROOT / "benchmarks" / "results.md").write_text(out + "\n")
