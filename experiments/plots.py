"""Chapter 3.7: plot freshly computed sensitivity CSVs, with no embedded results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sensitivity_figures"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"suite", "factor", "level", "family", "backbone", "auprc_mean", "auroc_mean"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    frame = frame[(frame.suite == "sensitivity") & ~frame.family.str.startswith("MACRO")].copy()
    if frame.empty:
        raise ValueError("No sensitivity rows; complete the sensitivity run first")
    frame["level"] = pd.to_numeric(frame.level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (factor, backbone), subset in frame.groupby(["factor", "backbone"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for family, rows in subset.groupby("family"):
            rows = rows.sort_values("level")
            for ax, metric in zip(axes, ("auprc", "auroc")):
                error = rows.get(metric + "_std")
                ax.errorbar(rows.level, rows[metric + "_mean"],
                            yerr=None if error is None else error.fillna(0), marker="o", label=family, capsize=3)
                ax.set_xlabel(factor)
                ax.set_ylabel("raw " + metric.upper())
                ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(backbone + "; error bars: SD across seeds")
        fig.tight_layout()
        for extension in ("svg", "png"):
            fig.savefig(args.output_dir / f"{factor}_{backbone}.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
