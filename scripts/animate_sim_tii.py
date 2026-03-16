"""animate_sim_tii.py: Render a pre-saved TII sim log as an animated GIF/MP4.

Usage
-----
    python -m scripts.animate_sim_tii
    python -m scripts.animate_sim_tii --flight flight-11a-lemniscate --fmt gif
    python -m scripts.animate_sim_tii --flight flight-11a-lemniscate --fmt mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--flight", type=str, default="flight-11a-lemniscate")
    p.add_argument("--fmt", type=str, default="gif", choices=["gif", "mp4"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sim_dir = repo_root / "logs" / "sim_tii"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sim_dir

    npz_path = sim_dir / f"{args.flight}_filter_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}\n  Run scripts.run_sim_tii first.")

    print(f"Loading {npz_path.name} ...", flush=True)
    raw = np.load(npz_path, allow_pickle=False)
    t         = raw["t"]
    truth_pos = raw["truth_pos"]
    ref_pos   = raw["ref_pos"]
    hover_pos = raw["hover_pos"]
    race_pos  = raw["race_pos"]
    adapt_pos = raw["adapt_pos"]

    # Downsample to target fps
    sim_hz = 1.0 / float(np.median(np.diff(t)))
    stride = max(1, int(round(sim_hz / args.fps)))
    sl = slice(0, len(t), stride)

    t_s         = t[sl]
    truth_s     = truth_pos[sl]
    ref_s       = ref_pos[sl]
    hover_s     = hover_pos[sl]
    race_s      = race_s  = race_pos[sl]
    adapt_s     = adapt_pos[sl]

    n_frames = len(t_s)
    print(f"  {len(t)} steps -> {n_frames} frames @ {args.fps} fps", flush=True)

    # Axis limits with padding
    all_pts = np.concatenate([truth_s, ref_s, hover_s, race_s, adapt_s], axis=0)
    all_pts = all_pts[np.all(np.isfinite(all_pts), axis=1)]
    pad = 0.3
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    colors = {
        "truth":    "#20a020",
        "ref":      "#aaaaaa",
        "hover":    "#e04040",
        "race":     "#e09000",
        "adaptive": "#2080d0",
    }

    # Trail length in frames
    trail = min(n_frames, int(args.fps * 2))  # 2-second trail

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # Ghost full path (faint)
    ax.plot(truth_s[:, 0], truth_s[:, 1], truth_s[:, 2],
            color=colors["truth"], lw=0.4, alpha=0.15)
    ax.plot(ref_s[:, 0], ref_s[:, 1], ref_s[:, 2],
            color=colors["ref"], lw=0.4, alpha=0.10, ls="--")

    # Lines that will be updated each frame (trail)
    line_truth, = ax.plot([], [], [], color=colors["truth"],    lw=1.5, label="Truth (PyBullet)")
    line_ref,   = ax.plot([], [], [], color=colors["ref"],      lw=0.8, ls="--", alpha=0.5, label="Ref (TII mocap)")
    line_hover, = ax.plot([], [], [], color=colors["hover"],    lw=1.0, label="Hover UKF")
    line_race,  = ax.plot([], [], [], color=colors["race"],     lw=1.0, label="Race UKF")
    line_adapt, = ax.plot([], [], [], color=colors["adaptive"], lw=1.0, label="Adaptive UKF*")

    # Dot markers at current position
    dot_truth, = ax.plot([], [], [], "o", color=colors["truth"],    ms=6, zorder=10)
    dot_hover, = ax.plot([], [], [], "o", color=colors["hover"],    ms=5)
    dot_race,  = ax.plot([], [], [], "o", color=colors["race"],     ms=5)
    dot_adapt, = ax.plot([], [], [], "o", color=colors["adaptive"], ms=5)

    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    def _set(line, dot, data, k):
        lo = max(0, k - trail)
        seg = data[lo:k+1]
        fin = np.all(np.isfinite(seg), axis=1)
        seg = seg[fin]
        if len(seg) > 0:
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])
            dot.set_data([seg[-1, 0]], [seg[-1, 1]])
            dot.set_3d_properties([seg[-1, 2]])
        else:
            line.set_data([], [])
            line.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])

    def update(k):
        _set(line_truth, dot_truth, truth_s, k)
        lo = max(0, k - trail)
        ref_seg = ref_s[lo:k+1]
        line_ref.set_data(ref_seg[:, 0], ref_seg[:, 1])
        line_ref.set_3d_properties(ref_seg[:, 2])
        _set(line_hover, dot_hover, hover_s, k)
        _set(line_race,  dot_race,  race_s,  k)
        _set(line_adapt, dot_adapt, adapt_s, k)
        title.set_text(f"{args.flight}  t={t_s[k]:.1f}s")
        return line_truth, line_ref, line_hover, line_race, line_adapt, \
               dot_truth, dot_hover, dot_race, dot_adapt, title

    interval_ms = 1000 / args.fps
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms, blit=False
    )

    out_path = out_dir / f"{args.flight}_animation.{args.fmt}"
    print(f"Rendering {n_frames} frames to {out_path.name} ...", flush=True)

    if args.fmt == "gif":
        writer = animation.PillowWriter(fps=args.fps)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)

    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
