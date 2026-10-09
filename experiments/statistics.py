"""Chapter 3.6: seed statistics and dataset-family-level paired rank tests."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def family_of(dataset):
    return "SMD" if dataset.startswith("SMD-") else dataset.split("@", 1)[0]


def read_results(paths, metric_view):
    latest = {}
    labels = {"native": "Native", "lora": "LoRA-Only", "srf": "SRF-Only", "joint": "SRF+LoRA"}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "ok") != "ok":
                continue
            detail = row.get("detail", row)
            metrics = detail.get("metrics", row)
            condition = detail.get("condition", row.get("condition"))
            if metric_view == "full_strength" and condition in {"lora", "joint"}:
                full = detail.get("lora", {}).get("full_strength_test")
                if not full or "metrics" not in full:
                    raise ValueError("Full-strength LoRA metrics missing; use --metric-view selected explicitly")
                metrics = full["metrics"]
            method = row.get("plugin", labels.get(condition, condition))
            if method == "Native audit":
                method = "Native"
            if not method or not all(k in metrics and metrics[k] is not None for k in ("auprc", "auroc")):
                continue
            key = (row["dataset"], row.get("backbone", "patch_transformer"), method, int(row["seed"]))
            latest[key] = {"dataset": key[0], "family": family_of(key[0]), "backbone": key[1],
                           "method": method, "seed": key[3], "auprc": float(metrics["auprc"]),
                           "auroc": float(metrics["auroc"])}
    if not latest:
        raise ValueError("No valid result rows")
    return pd.DataFrame(latest.values())


def analyze(frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for family, rows in frame.groupby("family"):
        expected = set(product(rows.dataset.unique(), rows.backbone.unique(),
                               frame.method.unique(), frame.seed.unique()))
        actual = set(rows[["dataset", "backbone", "method", "seed"]].itertuples(index=False, name=None))
        if expected != actual:
            raise ValueError(f"{family}: incomplete/unbalanced entity/model/method/seed cells; finish all requested runs first")
    # Machines/entities are nested within five dataset families, not independent blocks.
    family_seed = frame.groupby(["family", "backbone", "method", "seed"], as_index=False)[["auprc", "auroc"]].mean()
    family_seed.to_csv(output_dir / "family_seed.csv", index=False)
    summary = family_seed.groupby(["family", "backbone", "method"])[["auprc", "auroc"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output_dir / "mean_std.csv", index=False)
    table = family_seed.groupby(["family", "method"])["auprc"].mean().unstack("method")
    incomplete = table.index[table.isna().any(axis=1)].tolist()
    table = table.dropna()
    table.to_csv(output_dir / "friedman_input.csv")
    if table.shape[0] < 3 or table.shape[1] < 3:
        raise ValueError("Friedman/Nemenyi require >=3 complete families and >=3 methods")
    values = table.to_numpy()
    ranks = stats.rankdata(-values, axis=1, method="average")
    mean_rank = ranks.mean(axis=0)
    n, k = values.shape
    if np.all(np.ptp(values, axis=1) == 0):
        friedman_stat, friedman_p = 0.0, 1.0
    else:
        friedman_stat, friedman_p = stats.friedmanchisquare(*values.T)
    se = np.sqrt(k * (k + 1) / (6 * n))
    critical_difference = float(stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * se)
    pair_rows = []
    for left, right in combinations(range(k), 2):
        diff = values[:, left] - values[:, right]
        wilcoxon_p = 1.0 if np.all(diff == 0) else float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
        gap = abs(mean_rank[left] - mean_rank[right])
        pair_rows.append({"left": table.columns[left], "right": table.columns[right],
                          "nemenyi_p": float(stats.studentized_range.sf(gap / se * np.sqrt(2), k, np.inf)),
                          "wilcoxon_p": wilcoxon_p, "rank_difference": float(gap),
                          "exceeds_cd": bool(gap > critical_difference)})
    # Holm correction over all requested Wilcoxon pairs.
    order = np.argsort([r["wilcoxon_p"] for r in pair_rows])
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, pair_rows[index]["wilcoxon_p"] * (len(order) - position)))
        pair_rows[index]["wilcoxon_holm_p"] = running
    pd.DataFrame(pair_rows).to_csv(output_dir / "pairwise_tests.csv", index=False)
    report = {"metric": "raw AUPRC, no PA", "independent_blocks": "dataset families",
              "n_families": n, "n_methods": k, "incomplete_families_excluded": incomplete,
              "friedman_statistic": float(friedman_stat), "friedman_p": float(friedman_p),
              "nemenyi_cd_alpha_0_05": critical_difference,
              "average_ranks": dict(zip(table.columns, map(float, mean_rank))),
              "seed_sd_ddof": 1, "caution": "Few families limit power; CD is descriptive if omnibus p>=0.05."}
    (output_dir / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_cd(table.columns.tolist(), mean_rank, critical_difference, output_dir)
    return report


def plot_cd(methods, ranks, cd, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(ranks)
    fig, ax = plt.subplots(figsize=(8, max(3, len(methods) * 0.38)))
    ax.set_xlim(0.7, len(methods) + 0.3)
    for y, index in enumerate(order):
        ax.scatter(ranks[index], y, color="#336699")
        ax.text(ranks[index] + 0.08, y, f"{methods[index]} ({ranks[index]:.2f})", va="center", fontsize=9)
    # Maximal contiguous nonsignificant groups by average-rank distance.
    y = len(methods) + 0.2
    intervals = []
    ordered = ranks[order]
    for left in range(len(methods)):
        right = left
        while right + 1 < len(methods) and ordered[right + 1] - ordered[left] <= cd:
            right += 1
        if right > left and not any(a <= left and b >= right for a, b in intervals):
            intervals.append((left, right))
    for left, right in intervals:
        ax.plot([ordered[left], ordered[right]], [y, y], color="black", linewidth=3)
        y += 0.35
    ax.set_ylim(-0.7, y + 0.8)
    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(f"Nemenyi critical difference: {cd:.3f} (alpha=0.05)")
    fig.tight_layout()
    fig.savefig(output_dir / "cd.svg")
    fig.savefig(output_dir / "cd.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/statistics"))
    parser.add_argument("--metric-view", choices=("selected", "full_strength"), default="full_strength")
    args = parser.parse_args()
    print(json.dumps(analyze(read_results(args.results, args.metric_view), args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
