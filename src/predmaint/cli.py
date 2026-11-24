"""python -m predmaint.cli {train,monitor,status,rollback} --state DIR ..."""
from __future__ import annotations

import argparse
import json

from .pipeline import monitor, run_training
from .registry import MODEL_NAME, Store, champion_version, rollback
from .simulate import SimConfig, make_dataset


def _cfg(a) -> SimConfig:
    return SimConfig(rpm_range=(a.rpm_min, a.rpm_max), gain=a.gain)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="predmaint")
    p.add_argument("--state", default="mlops_state")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "monitor"):
        s = sub.add_parser(name)
        s.add_argument("--rpm-min", type=float, default=1500)
        s.add_argument("--rpm-max", type=float, default=1800)
        s.add_argument("--gain", type=float, default=1.0)
        s.add_argument("--n", type=int, default=100, help="windows per class")
        s.add_argument("--seed", type=int, default=0)
        if name == "train":
            s.add_argument("--feature-set", default="physics_orders")
    rb = sub.add_parser("rollback")
    rb.add_argument("version")
    sub.add_parser("status")
    a = p.parse_args(argv)
    store = Store(a.state)

    if a.cmd == "train":
        train = make_dataset(a.n, _cfg(a), seed=a.seed)
        holdout = make_dataset(max(a.n // 2, 20), _cfg(a), seed=a.seed + 1000)
        print(json.dumps(run_training(store, train, holdout, a.feature_set, seed=a.seed), indent=2))
    elif a.cmd == "monitor":
        current = make_dataset(max(a.n // 2, 20), _cfg(a), seed=a.seed + 2000)
        print(json.dumps(monitor(store, current, with_labels=True), indent=2, default=str))
    elif a.cmd == "rollback":
        rollback(store, a.version)
        print(f"{MODEL_NAME}: champion -> v{a.version}")
    else:
        print(f"champion: v{champion_version(store.activate())}")


if __name__ == "__main__":
    main()
