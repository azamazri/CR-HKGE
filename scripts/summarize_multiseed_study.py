"""Aggregate subset-eval logs across multiple seeds into mean +/- std tables.

Each run script writes one subset-eval log per (target, seed), named:

    <target>_seed<seed>_subset_eval.log

This script groups logs by ``<target>`` (ignoring the seed), and for every
scope (overall / enriched / standard), metric (recall / precision / hit / ndcg)
and K, reports ``mean +/- std`` across the seeds so significance can be assessed.

Usage:
    python summarize_multiseed_study.py LOG_DIR/*_subset_eval.log
    python summarize_multiseed_study.py --csv out.csv LOG_DIR/*_subset_eval.log
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


SCOPE_RE = re.compile(r"^(overall|enriched|standard):")
METRIC_RE = re.compile(r"(recall|precision|hit|ndcg)=\[([^\]]+)\]")
# <target>_seed<seed>_subset_eval.log  -> capture target and seed.
NAME_RE = re.compile(r"^(?P<target>.+?)_seed(?P<seed>\d+)_subset_eval$")

KS = [3, 5, 10]
# (column label, metric, k-index)
COLUMNS = [
    ("Recall@3", "recall", 0),
    ("Precision@3", "precision", 0),
    ("Hit@3", "hit", 0),
    ("NDCG@3", "ndcg", 0),
    ("Recall@5", "recall", 1),
    ("NDCG@5", "ndcg", 1),
    ("Recall@10", "recall", 2),
    ("NDCG@10", "ndcg", 2),
]


def parse_values(raw: str) -> list[float]:
    return [float(value) for value in raw.replace("\t", " ").split()]


def parse_log(path: Path) -> dict[str, dict[str, list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result: dict[str, dict[str, list[float]]] = {}
    for idx, line in enumerate(lines):
        match = SCOPE_RE.match(line.strip())
        if not match:
            continue
        if idx + 1 >= len(lines):
            continue
        metrics = {
            name: parse_values(values)
            for name, values in METRIC_RE.findall(lines[idx + 1])
        }
        if metrics:
            result[match.group(1)] = metrics
    return result


def split_target_seed(stem: str) -> tuple[str, str]:
    match = NAME_RE.match(stem)
    if match:
        return match.group("target"), match.group("seed")
    # Fall back: no seed encoded in the filename -> treat as a single unnamed seed.
    return stem.replace("_subset_eval", ""), "single"


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
    return mean, math.sqrt(var)


def collect(logs: list[str]):
    # target -> scope -> metric -> k-index -> list of values (one per seed)
    store: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    seeds_per_target: dict[str, set] = defaultdict(set)
    target_order: list[str] = []

    for raw in logs:
        path = Path(raw)
        target, seed = split_target_seed(path.stem)
        if target not in target_order:
            target_order.append(target)
        seeds_per_target[target].add(seed)
        parsed = parse_log(path)
        for scope, metrics in parsed.items():
            for metric, values in metrics.items():
                for k_idx, value in enumerate(values):
                    store[target][scope][metric][k_idx].append(value)
    return store, seeds_per_target, target_order


def fmt_cell(values: list[float]) -> str:
    if not values:
        return "-"
    mean, std = mean_std(values)
    return "%.5f +/- %.5f" % (mean, std)


def print_table(scope: str, store, target_order, seeds_per_target) -> None:
    print("\n## %s" % scope.capitalize())
    header = ["Model", "seeds"] + [label for label, _m, _k in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for target in target_order:
        scope_data = store[target].get(scope)
        if not scope_data:
            continue
        n_seeds = len(seeds_per_target[target])
        cells = [target, str(n_seeds)]
        for _label, metric, k_idx in COLUMNS:
            cells.append(fmt_cell(scope_data.get(metric, {}).get(k_idx, [])))
        print("| " + " | ".join(cells) + " |")


def write_csv(path: Path, store, target_order, seeds_per_target) -> None:
    rows = ["scope,model,n_seeds,metric,k,mean,std,n,values"]
    k_for_idx = {0: 3, 1: 5, 2: 10}
    for scope in ["overall", "enriched", "standard"]:
        for target in target_order:
            scope_data = store[target].get(scope)
            if not scope_data:
                continue
            n_seeds = len(seeds_per_target[target])
            for metric in ["recall", "precision", "hit", "ndcg"]:
                for k_idx, values in sorted(scope_data.get(metric, {}).items()):
                    mean, std = mean_std(values)
                    rows.append(
                        "%s,%s,%d,%s,%d,%.6f,%.6f,%d,%s"
                        % (
                            scope,
                            target,
                            n_seeds,
                            metric,
                            k_for_idx.get(k_idx, k_idx),
                            mean,
                            std,
                            len(values),
                            " ".join("%.6f" % v for v in values),
                        )
                    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--csv", default="", help="Optional path to also write a tidy CSV.")
    args = parser.parse_args()

    store, seeds_per_target, target_order = collect(args.logs)

    print("# Multi-seed study summary (mean +/- sample std)")
    for scope in ["overall", "enriched", "standard"]:
        print_table(scope, store, target_order, seeds_per_target)

    if args.csv:
        write_csv(Path(args.csv), store, target_order, seeds_per_target)
        print("\nwrote CSV: %s" % args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
