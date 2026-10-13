#!/usr/bin/env python3
"""
Plot CG residual vs. iteration for each preconditioner method.

Reads:  schwarz_fd_vs_jacobi_triangle_history.csv
        columns: method,iteration,residual

Writes: residual_history.png
"""

import sys
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

CSV_PATH = "./build/schwarz_fd_vs_jacobi_triangle_history.csv"
OUT_PATH_LOG = "./build/residual_history_LOG.png"
OUT_PATH_LINEAR = "./build/residual_history_LINEAR.png"


def load_history(path):
    data = defaultdict(lambda: {"iter": [], "res": []})
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            data[method]["iter"].append(int(row["iteration"]))
            data[method]["res"].append(float(row["residual"]))
    return data


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    data = load_history(path)

    if not data:
        print(f"No data found in {path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, series in data.items():
        ax.semilogy(series["iter"], series["res"], markersize=3, label=method)

    ax.set_xlabel("CG iteration")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title("CG residual convergence: Jacobi vs Schwarz-FD")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PATH_LOG, dpi=150)
    print(f"Saved plot to {OUT_PATH_LOG}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for method, series in data.items():
        ax2.plot(series["iter"], series["res"], markersize=3, label=method)

    ax2.set_xlabel("CG iteration")
    ax2.set_ylabel("Residual")
    ax2.set_title("CG residual convergence: Jacobi vs Schwarz-FD")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_PATH_LINEAR, dpi=150)
    print(f"Saved linear scale plot to {OUT_PATH_LINEAR}")


if __name__ == "__main__":
    main()