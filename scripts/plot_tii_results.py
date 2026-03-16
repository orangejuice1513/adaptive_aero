"""plot_tii_results.py: Plot filter results from run_filters_on_tii.py.

Generates (per flight):
  1. 3D trajectory comparison
  2. Position error vs time with G-load overlay
  3. Position error vs G-load (scatter + binned mean)
  4. Adaptive process noise evolution (Q diagonal)
  5. RMSE bar chart
  6. NEES vs time with expected value reference line

Usage
-----
    python -m scripts.plot_tii_results --flight flight-01a-ellipse
    python -m scripts.plot_tii_results                          # all flights in logs/tii/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def _pos_error(est: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    err = np.full(len(truth), np.nan)
    if np.any(valid):
        diff = est[valid] - truth[valid]
        err[valid] = np.linalg.norm(diff, axis=1)
    return err


def _binned_mean(x: np.ndarray, y: np.ndarray, nbins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1] if i < nbins - 1 else edges[i + 1] + 1e-9
        m = (x >= lo) & (x < hi)
        if np.any(m):
            means[i] = float(np.mean(y[m]))
    keep = np.isfinite(means)
    return centers[keep], means[keep]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_trajectory_3d(t, truth_pos, hover_pos, race_pos, adaptive_pos,
                       hover_valid, race_valid, adaptive_valid, out_dir, prefix):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2],
            linewidth=2.2, color="black", label="Ground truth (mocap)")
    if np.any(hover_valid):
        ax.plot(hover_pos[hover_valid, 0], hover_pos[hover_valid, 1], hover_pos[hover_valid, 2],
                linewidth=1.4, label="Hover UKF")
    if np.any(race_valid):
        ax.plot(race_pos[race_valid, 0], race_pos[race_valid, 1], race_pos[race_valid, 2],
                linewidth=1.4, label="Race UKF")
    if np.any(adaptive_valid):
        ax.plot(adaptive_pos[adaptive_valid, 0], adaptive_pos[adaptive_valid, 1], adaptive_pos[adaptive_valid, 2],
                linewidth=1.8, label="Adaptive UKF")

    ax.set_title(f"3D Trajectory — {prefix}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    _save(fig, out_dir / f"{prefix}_trajectory_3d.png")


def plot_error_vs_time(t, hover_err, race_err, adaptive_err, g_load, out_dir, prefix):
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(t, hover_err, linewidth=1.3, label="Hover UKF")
    ax1.plot(t, race_err, linewidth=1.3, label="Race UKF")
    ax1.plot(t, adaptive_err, linewidth=1.5, label="Adaptive UKF")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position error (m)")
    ax1.set_title(f"Position Error vs Time — {prefix}")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, g_load, linestyle="--", linewidth=0.9, alpha=0.6, color="gray", label="G-load")
    ax2.set_ylabel("G-load")
    ax2.legend(loc="upper right")

    _save(fig, out_dir / f"{prefix}_error_vs_time.png")


def plot_error_vs_gload(g_load, hover_err, race_err, adaptive_err, out_dir, prefix):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    configs = [
        ("Hover UKF", hover_err, "tab:blue"),
        ("Race UKF", race_err, "tab:orange"),
        ("Adaptive UKF", adaptive_err, "tab:green"),
    ]
    for ax, (title, err, color) in zip(axs, configs):
        mask = np.isfinite(g_load) & np.isfinite(err)
        ax.scatter(g_load[mask], err[mask], s=4, alpha=0.15, color=color)
        cx, cy = _binned_mean(g_load, err, nbins=20)
        if len(cx) > 0:
            ax.plot(cx, cy, linewidth=2.2, color=color)
        ax.set_title(title)
        ax.set_xlabel("G-load")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("Position error (m)")
    fig.suptitle(f"Position Error vs G-load — {prefix}", y=1.01)
    _save(fig, out_dir / f"{prefix}_error_vs_gload.png")


def plot_adaptive_noise(t, pos_var, vel_var, att_var, out_dir, prefix):
    if pos_var is None:
        return
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axs[0].plot(t, pos_var, linewidth=1.4)
    axs[0].set_ylabel("Pos Q diag")
    axs[0].set_title(f"Adaptive Process Noise Evolution — {prefix}")
    axs[1].plot(t, vel_var, linewidth=1.4)
    axs[1].set_ylabel("Vel Q diag")
    axs[2].plot(t, att_var, linewidth=1.4)
    axs[2].set_ylabel("Att Q diag")
    axs[2].set_xlabel("Time (s)")
    _save(fig, out_dir / f"{prefix}_adaptive_q_evolution.png")


def plot_rmse_bar(summary: dict, out_dir: Path, prefix: str) -> None:
    labels = ["Hover UKF", "Race UKF", "Adaptive UKF"]
    vals = [
        summary.get("hover_rmse_m", float("nan")),
        summary.get("race_rmse_m", float("nan")),
        summary.get("adaptive_rmse_m", float("nan")),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green"])
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.4f} m",
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"Filter Position RMSE — {prefix}")
    _save(fig, out_dir / f"{prefix}_rmse_bar.png")


def plot_nees(t, hover_nees, race_nees, adaptive_nees, out_dir, prefix):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(t, hover_nees, linewidth=0.8, alpha=0.7, label="Hover UKF")
    ax.plot(t, race_nees, linewidth=0.8, alpha=0.7, label="Race UKF")
    ax.plot(t, adaptive_nees, linewidth=0.8, alpha=0.7, label="Adaptive UKF")
    ax.axhline(3.0, linestyle="--", linewidth=1.5, color="black", label="Expected NEES = 3")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES (3-DOF position)")
    ax.set_title(f"NEES vs Time — {prefix}")
    ax.set_ylim(bottom=0)
    ax.legend()
    _save(fig, out_dir / f"{prefix}_nees_vs_time.png")


def plot_nees_summary(summaries: list[dict], out_dir: Path) -> None:
    """Bar chart of mean NEES across all flights."""
    names = [s["flight_name"] for s in summaries]
    hover_nees = [s["hover_mean_nees"] for s in summaries]
    race_nees = [s["race_mean_nees"] for s in summaries]
    adapt_nees = [s["adaptive_mean_nees"] for s in summaries]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    ax.bar(x - width, hover_nees, width, label="Hover UKF", color="tab:blue")
    ax.bar(x, race_nees, width, label="Race UKF", color="tab:orange")
    ax.bar(x + width, adapt_nees, width, label="Adaptive UKF", color="tab:green")
    ax.axhline(3.0, linestyle="--", linewidth=1.5, color="black", label="Expected = 3")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean NEES")
    ax.set_title("Mean NEES Across All Flights")
    ax.legend()

    _save(fig, out_dir / "all_flights_nees.png")


def plot_rmse_summary(summaries: list[dict], out_dir: Path) -> None:
    """Bar chart of RMSE across all flights."""
    names = [s["flight_name"] for s in summaries]
    hover_rmse = [s["hover_rmse_m"] for s in summaries]
    race_rmse = [s["race_rmse_m"] for s in summaries]
    adapt_rmse = [s["adaptive_rmse_m"] for s in summaries]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    ax.bar(x - width, hover_rmse, width, label="Hover UKF", color="tab:blue")
    ax.bar(x, race_rmse, width, label="Race UKF", color="tab:orange")
    ax.bar(x + width, adapt_rmse, width, label="Adaptive UKF", color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Position RMSE Across All Flights")
    ax.legend()

    _save(fig, out_dir / "all_flights_rmse.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_result_file(npz_path: Path, summary_path: Path, out_dir: Path) -> dict | None:
    prefix = npz_path.stem.replace("_filter_results", "")

    if not npz_path.exists():
        print(f"[skip] {npz_path} not found")
        return None

    print(f"Plotting {prefix} ...")

    D = _load_npz(npz_path)

    t = D["t"]
    truth_pos = D["truth_pos"]
    hover_pos = D["hover_pos"]
    race_pos = D["race_pos"]
    adaptive_pos = D["adaptive_pos"]

    hover_valid = D["hover_valid"].astype(bool)
    race_valid = D["race_valid"].astype(bool)
    adaptive_valid = D["adaptive_valid"].astype(bool)

    hover_err = _pos_error(hover_pos, truth_pos, hover_valid)
    race_err = _pos_error(race_pos, truth_pos, race_valid)
    adaptive_err = _pos_error(adaptive_pos, truth_pos, adaptive_valid)

    # G-load as specific force magnitude / 9.81
    imu_sf_mag = D.get("imu_sf_mag", np.full(len(t), np.nan))
    g_load = imu_sf_mag / 9.81

    hover_nees = D.get("hover_nees", np.full(len(t), np.nan))
    race_nees = D.get("race_nees", np.full(len(t), np.nan))
    adaptive_nees = D.get("adaptive_nees", np.full(len(t), np.nan))

    adaptive_pos_var = D.get("adaptive_pos_var", None)
    adaptive_vel_var = D.get("adaptive_vel_var", None)
    adaptive_att_var = D.get("adaptive_att_var", None)

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    plot_trajectory_3d(t, truth_pos, hover_pos, race_pos, adaptive_pos,
                       hover_valid, race_valid, adaptive_valid, out_dir, prefix)
    plot_error_vs_time(t, hover_err, race_err, adaptive_err, g_load, out_dir, prefix)
    plot_error_vs_gload(g_load, hover_err, race_err, adaptive_err, out_dir, prefix)
    plot_adaptive_noise(t, adaptive_pos_var, adaptive_vel_var, adaptive_att_var, out_dir, prefix)
    if summary:
        plot_rmse_bar(summary, out_dir, prefix)
    plot_nees(t, hover_nees, race_nees, adaptive_nees, out_dir, prefix)

    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot TII filter results.")
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing *_filter_results.npz files. Default: <repo>/logs/tii/",
    )
    p.add_argument(
        "--flight",
        type=str,
        default=None,
        help="Plot only this flight (e.g. flight-01a-ellipse). Default: all available.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else repo_root / "logs" / "tii"

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    if args.flight:
        npz_files = [log_dir / f"{args.flight}_filter_results.npz"]
    else:
        npz_files = sorted(log_dir.glob("*_filter_results.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No *_filter_results.npz found in {log_dir}")

    summaries = []
    for npz_path in npz_files:
        summary_path = npz_path.with_name(
            npz_path.stem.replace("_filter_results", "_summary") + ".json"
        )
        summary = _process_result_file(npz_path, summary_path, log_dir)
        if summary:
            summaries.append(summary)

    if len(summaries) > 1:
        print("\nGenerating cross-flight summary plots ...")
        plot_rmse_summary(summaries, log_dir)
        plot_nees_summary(summaries, log_dir)


if __name__ == "__main__":
    main()
