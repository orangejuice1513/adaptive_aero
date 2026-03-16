"""analyze_high_g.py: Compare filter performance at high-G vs all timesteps.

Saves a CSV table and comparison plots to logs/sim_tii/high_g_analysis/

Usage
-----
    python -m scripts.analyze_high_g
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLIGHTS = [
    "flight-11a-lemniscate",
    "flight-01a-ellipse",
    "flight-05a-ellipse",
]

FLIGHT_LABELS = {
    "flight-11a-lemniscate": "Lemniscate",
    "flight-01a-ellipse":    "Ellipse-01",
    "flight-05a-ellipse":    "Ellipse-05",
}

G_THRESHOLD = 2.0

COLORS = {
    "Hover":    "#e04040",
    "Race":     "#e09000",
    "Adaptive": "#2080d0",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(est, truth, mask):
    if not np.any(mask):
        return float("nan")
    e = est[mask] - truth[mask]
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def _nees_mean(est, truth, cov, mask):
    vals = []
    for k in np.where(mask)[0]:
        e = truth[k] - est[k]
        P3 = cov[k, :3, :3]
        try:
            vals.append(float(e @ np.linalg.solve(P3, e)))
        except np.linalg.LinAlgError:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def _nees_series(est, truth, cov, mask):
    out = np.full(len(est), np.nan)
    for k in np.where(mask)[0]:
        e = truth[k] - est[k]
        P3 = cov[k, :3, :3]
        try:
            out[k] = float(e @ np.linalg.solve(P3, e))
        except np.linalg.LinAlgError:
            pass
    return out


# ---------------------------------------------------------------------------
# Load and compute
# ---------------------------------------------------------------------------

def load_metrics(sim_dir: Path) -> list[dict]:
    rows = []
    for fname in FLIGHTS:
        npz = np.load(sim_dir / f"{fname}_filter_results.npz", allow_pickle=False)

        truth_pos  = npz["truth_pos"]
        imu_sf_mag = npz["imu_sf_mag"]
        g_load     = imu_sf_mag / 9.81
        high_g     = g_load > G_THRESHOLD
        valid_t    = np.all(np.isfinite(truth_pos), axis=1)

        for filter_name, pos_key, cov_key in [
            ("Hover",    "hover_pos",  "hover_cov"),
            ("Race",     "race_pos",   "race_cov"),
            ("Adaptive", "adapt_pos",  "adapt_cov"),
        ]:
            pos = npz[pos_key]
            cov = npz[cov_key]
            valid_e  = np.all(np.isfinite(pos), axis=1)
            mask_all = valid_t & valid_e
            mask_hig = valid_t & valid_e & high_g

            rows.append({
                "flight":       fname,
                "label":        FLIGHT_LABELS[fname],
                "filter":       filter_name,
                "rmse_all":     _rmse(pos, truth_pos, mask_all),
                "nees_all":     _nees_mean(pos, truth_pos, cov, mask_all),
                "rmse_hig":     _rmse(pos, truth_pos, mask_hig),
                "nees_hig":     _nees_mean(pos, truth_pos, cov, mask_hig),
                "hig_steps":    int(np.sum(mask_hig)),
                "total_steps":  int(np.sum(mask_all)),
                "hig_pct":      100.0 * np.sum(mask_hig) / max(1, np.sum(mask_all)),
            })
    return rows


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / "high_g_analysis.csv"
    fields = [
        "flight", "filter",
        "rmse_all", "nees_all",
        "rmse_hig", "nees_hig",
        "hig_steps", "total_steps", "hig_pct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _grouped_bar(ax, flight_labels, all_vals, hig_vals, filter_names, ylabel, title, show_target=None):
    n_flights  = len(flight_labels)
    n_filters  = len(filter_names)
    x          = np.arange(n_flights)
    bar_w      = 0.13
    offsets    = np.linspace(-(n_filters - 1) / 2, (n_filters - 1) / 2, n_filters) * bar_w * 2

    for i, fname in enumerate(filter_names):
        col = COLORS[fname]
        ax.bar(x + offsets[i] - bar_w / 2, all_vals[i], bar_w,
               color=col, alpha=0.4, label=f"{fname} (all)" if i == 0 else "_")
        ax.bar(x + offsets[i] + bar_w / 2, hig_vals[i], bar_w,
               color=col, alpha=1.0, label=f"{fname} (>2G)" if i == 0 else "_")

    # Rebuild legend properly
    from matplotlib.patches import Patch
    legend_items = []
    for fname in filter_names:
        col = COLORS[fname]
        legend_items.append(Patch(facecolor=col, alpha=0.4, label=f"{fname} — all steps"))
        legend_items.append(Patch(facecolor=col, alpha=1.0, label=f"{fname} — >2G only"))
    ax.legend(handles=legend_items, fontsize=7, ncol=2, loc="upper left")

    if show_target is not None:
        ax.axhline(show_target, color="black", ls="--", lw=1.0, label=f"Target={show_target}")

    ax.set_xticks(x)
    ax.set_xticklabels(flight_labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)


def make_plots(rows: list[dict], out_dir: Path) -> None:
    filter_names  = ["Hover", "Race", "Adaptive"]
    flight_labels = [FLIGHT_LABELS[f] for f in FLIGHTS]

    def _extract(metric):
        # returns list (per filter) of list (per flight)
        result = []
        for fn in filter_names:
            vals = [r[metric] for f in FLIGHTS for r in rows if r["flight"] == f and r["filter"] == fn]
            result.append(vals)
        return result

    rmse_all = _extract("rmse_all")
    rmse_hig = _extract("rmse_hig")
    nees_all = _extract("nees_all")
    nees_hig = _extract("nees_hig")

    # ------------------------------------------------------------------
    # 1. NEES comparison (all vs high-G), one panel per flight
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("NEES: All Steps vs High-G (>2G) — target = 3.0", fontsize=13)

    for col, fname in enumerate(FLIGHTS):
        ax = axes[col]
        flight_rows = [r for r in rows if r["flight"] == fname]
        filters = [r["filter"] for r in flight_rows]
        n_all  = [r["nees_all"] for r in flight_rows]
        n_hig  = [r["nees_hig"] for r in flight_rows]
        x      = np.arange(len(filters))
        w      = 0.3

        bars_all = ax.bar(x - w / 2, n_all, w, color=[COLORS[f] for f in filters],
                          alpha=0.35, label="All steps")
        bars_hig = ax.bar(x + w / 2, n_hig, w, color=[COLORS[f] for f in filters],
                          alpha=1.00, label=">2G only")

        for bar, v in list(zip(bars_all, n_all)) + list(zip(bars_hig, n_hig)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(3.0, color="black", ls="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(filters, fontsize=9)
        ax.set_title(FLIGHT_LABELS[fname], fontsize=10)
        ax.set_ylabel("NEES" if col == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)
        if col == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor="gray", alpha=0.35, label="All steps"),
                Patch(facecolor="gray", alpha=1.00, label=">2G only"),
            ], fontsize=8)

    plt.tight_layout()
    p = out_dir / "high_g_nees_bars.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  Saved: {p.name}", flush=True)

    # ------------------------------------------------------------------
    # 2. RMSE comparison (all vs high-G)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("RMSE: All Steps vs High-G (>2G)", fontsize=13)

    for col, fname in enumerate(FLIGHTS):
        ax = axes[col]
        flight_rows = [r for r in rows if r["flight"] == fname]
        filters = [r["filter"] for r in flight_rows]
        r_all  = [r["rmse_all"] for r in flight_rows]
        r_hig  = [r["rmse_hig"] for r in flight_rows]
        x      = np.arange(len(filters))
        w      = 0.3

        bars_all = ax.bar(x - w / 2, r_all, w, color=[COLORS[f] for f in filters], alpha=0.35)
        bars_hig = ax.bar(x + w / 2, r_hig, w, color=[COLORS[f] for f in filters], alpha=1.00)

        for bar, v in list(zip(bars_all, r_all)) + list(zip(bars_hig, r_hig)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(filters, fontsize=9)
        ax.set_title(FLIGHT_LABELS[fname], fontsize=10)
        ax.set_ylabel("RMSE (m)" if col == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)
        if col == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor="gray", alpha=0.35, label="All steps"),
                Patch(facecolor="gray", alpha=1.00, label=">2G only"),
            ], fontsize=8)

    plt.tight_layout()
    p = out_dir / "high_g_rmse_bars.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  Saved: {p.name}", flush=True)

    # ------------------------------------------------------------------
    # 3. NEES degradation factor (high-G NEES / all NEES)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("NEES Degradation at High-G  (high-G NEES / all-steps NEES)\nLower = filter stays calibrated under stress", fontsize=11)

    x         = np.arange(len(FLIGHTS))
    bar_w     = 0.22
    offsets   = [-bar_w, 0, bar_w]

    for i, fname in enumerate(filter_names):
        factors = []
        for f in FLIGHTS:
            r = next(r for r in rows if r["flight"] == f and r["filter"] == fname)
            factors.append(r["nees_hig"] / r["nees_all"] if r["nees_all"] > 0 else float("nan"))
        bars = ax.bar(x + offsets[i], factors, bar_w, color=COLORS[fname], label=fname)
        for bar, v in zip(bars, factors):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.1f}x", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", ls="--", lw=1.0, label="No degradation (1x)")
    ax.set_xticks(x)
    ax.set_xticklabels(flight_labels, fontsize=10)
    ax.set_ylabel("NEES degradation factor")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "high_g_nees_degradation.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  Saved: {p.name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim_dir   = repo_root / "logs" / "sim_tii"
    out_dir   = sim_dir / "high_g_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sim results ...", flush=True)
    rows = load_metrics(sim_dir)

    # Print table
    print(f"\n{'Flight':<22}  {'Filter':<10}  {'RMSE-all':>9}  {'NEES-all':>9}  {'RMSE->2G':>9}  {'NEES->2G':>9}  {'>2G steps':>10}")
    print("-" * 100)
    prev = None
    for r in rows:
        if r["flight"] != prev:
            if prev is not None:
                print()
            prev = r["flight"]
        print(f"  {r['label']:<20}  {r['filter']:<10}  {r['rmse_all']:>9.4f}  {r['nees_all']:>9.2f}  "
              f"{r['rmse_hig']:>9.4f}  {r['nees_hig']:>9.2f}  {r['hig_steps']:>10} ({r['hig_pct']:.1f}%)")

    csv_path = save_csv(rows, out_dir)
    print(f"\nSaved CSV: {csv_path}", flush=True)

    print("\nGenerating plots ...", flush=True)
    make_plots(rows, out_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
