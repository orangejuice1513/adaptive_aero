"""run_sim_tii.py: Fly a real TII trajectory in PyBullet, then compare Hover/Race/Adaptive UKF.

Workflow
--------
1. Load a pre-parsed TII .npz (mocap ground truth positions).
2. Build a TIIReplayTrajectory and run the PyBullet GeometricController sim.
3. Collect synthetic IMU + VIO sensor data.
4. Replay all three filters on the sensor log.
5. Save results to logs/sim_tii/ and generate comparison plots.

Usage
-----
    python -m scripts.run_sim_tii
    python -m scripts.run_sim_tii --flight flight-11a-lemniscate --gui
    python -m scripts.run_sim_tii --flight flight-01a-ellipse --duration 22
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.env import DroneEnv, SimConfig, DroneConfig
from sim.controller import GeometricController, ControllerConfig
from sim.trajectory import TIIReplayTrajectory
from sim.sensors import SensorSuite, SensorSuiteConfig

from src.filters.hover_ukf import HoverUKF
from src.filters.race_ukf import RaceUKF
from src.filters.adaptive_ukf import AdaptiveUKF, AdaptiveUKFConfig


# ---------------------------------------------------------------------------
# Tuned Adaptive UKF config (beta_scale=0.01x, subtract_gravity=True)
# from hyperparam_search — best NEES=2.98 on 3 test flights.
# ---------------------------------------------------------------------------
TUNED_ADAPTIVE_CFG = AdaptiveUKFConfig(
    beta_pos_var=1e-4 * 0.01,   # 1e-6
    beta_vel_var=2e-2 * 0.01,   # 2e-4
    beta_att_var=1e-4 * 0.01,   # 1e-6
    subtract_gravity=True,
    motor_max_rpm=1.0,           # thrust_sum ∈ [0,4], same as TII pipeline
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate TII flight in PyBullet and compare UKFs.")
    p.add_argument("--flight", type=str, default="flight-11a-lemniscate",
                   help="Flight name (NPZ must exist in logs/tii/)")
    p.add_argument("--duration", type=float, default=None,
                   help="Simulation duration in seconds. Default: flight duration.")
    p.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics timestep")
    p.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Default: logs/sim_tii/")
    return p


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _make_x0(pos: np.ndarray, vel: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float)
    return np.array([
        pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2],
        q[0], q[1], q[2], q[3],
    ], dtype=float)


def _make_P0() -> np.ndarray:
    return np.diag([
        0.10, 0.10, 0.10,
        0.25, 0.25, 0.25,
        0.05, 0.05, 0.05,
    ]).astype(float)


def _compute_rmse(est: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    e = est[mask] - truth[mask]
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def _compute_nees(
    est: np.ndarray,
    truth: np.ndarray,
    cov: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
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
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    npz_path: Path,
    duration: float | None,
    dt: float,
    gui: bool,
    seed: int,
    repo_root: Path,
) -> dict:
    """Run PyBullet sim following the TII mocap path. Returns raw log arrays."""

    traj = TIIReplayTrajectory(npz_path=str(npz_path))
    ref0 = traj.sample(0.0)
    start_pos = ref0.pos_w.copy()

    sim_duration = duration if duration is not None else traj.duration

    env = DroneEnv(
        sim_cfg=SimConfig(dt=dt, gui=gui, seed=seed),
        drone_cfg=DroneConfig(
            urdf_path=str(repo_root / "assets" / "quad.urdf"),
            expected_mass_kg=1.35,
            start_pos_w=start_pos,
        ),
    )

    ctrl_cfg = ControllerConfig(
        kp_pos=np.array([10.0, 10.0, 14.0]),
        kd_vel=np.array([6.5, 6.5, 8.5]),
        ki_pos=np.array([0.0, 0.0, 0.8]),
        kp_att=np.array([1.8, 1.8, 0.8]),
        kd_rate=np.array([0.18, 0.18, 0.10]),
        max_acc_xy=30.0,
        max_acc_up=30.0,
        max_acc_down=18.0,
        max_tilt_rad=np.radians(70.0),
        max_yaw_rate=np.radians(360.0),
        max_torque_nm=np.array([1.5, 1.5, 0.6]),
    )

    sensors = SensorSuite(SensorSuiteConfig(
        motor_max_rpm=env.drone_cfg.motor.max_rpm,
        seed=seed,
    ))

    truth_state = env.reset()
    controller = GeometricController.from_env(env, cfg=ctrl_cfg)
    controller.reset()
    sensors.reset()

    n_steps = int(np.ceil(sim_duration / dt))
    print(f"  Sim: {n_steps} steps ({sim_duration:.1f}s @ {1/dt:.0f} Hz)", flush=True)

    # Pre-allocate log arrays
    t_log        = np.zeros(n_steps, dtype=float)
    truth_pos    = np.zeros((n_steps, 3), dtype=float)
    truth_vel    = np.zeros((n_steps, 3), dtype=float)
    truth_quat   = np.zeros((n_steps, 4), dtype=float)   # xyzw
    ref_pos      = np.zeros((n_steps, 3), dtype=float)

    imu_valid    = np.zeros(n_steps, dtype=np.int32)
    imu_accel    = np.full((n_steps, 3), np.nan, dtype=float)
    imu_gyro     = np.full((n_steps, 3), np.nan, dtype=float)
    imu_sf_mag   = np.full(n_steps, np.nan, dtype=float)

    vio_valid    = np.zeros(n_steps, dtype=np.int32)
    vio_pos      = np.full((n_steps, 3), np.nan, dtype=float)

    rpm_sq_sum   = np.zeros(n_steps, dtype=float)

    try:
        for k in range(n_steps):
            t_now = env.get_truth_state().t
            ref = traj.sample(t_now)
            ctrl = controller.compute(env.get_truth_state(), ref, dt=dt)
            truth = env.step(ctrl.rpm_cmd)
            sens = sensors.update(truth)

            t_log[k]      = truth.t
            truth_pos[k]  = truth.pos_w
            truth_vel[k]  = truth.vel_w
            truth_quat[k] = truth.quat_wb_xyzw
            ref_pos[k]    = ref.pos_w

            if sens.imu is not None:
                imu_valid[k]  = 1
                imu_accel[k]  = sens.imu.accel_mps2
                imu_gyro[k]   = sens.imu.gyro_radps
                imu_sf_mag[k] = sens.imu.specific_force_mag_mps2

            if sens.vio is not None:
                vio_valid[k] = 1
                vio_pos[k]   = sens.vio.pos_w_m

            rpm_sq_sum[k] = float(sens.telemetry.rpm_sq_sum)

            if k % 10000 == 0 and k > 0:
                pct = 100 * k / n_steps
                print(f"    {pct:.0f}%  t={truth.t:.1f}s", flush=True)
    finally:
        env.close()

    return {
        "t": t_log,
        "truth_pos": truth_pos,
        "truth_vel": truth_vel,
        "truth_quat": truth_quat,
        "ref_pos": ref_pos,
        "imu_valid": imu_valid,
        "imu_accel": imu_accel,
        "imu_gyro": imu_gyro,
        "imu_sf_mag": imu_sf_mag,
        "vio_valid": vio_valid,
        "vio_pos": vio_pos,
        "rpm_sq_sum": rpm_sq_sum,
    }


# ---------------------------------------------------------------------------
# Filter replay
# ---------------------------------------------------------------------------

def run_filters(log: dict) -> dict:
    """Run Hover, Race, and tuned Adaptive UKF on sim sensor log."""
    t          = log["t"]
    imu_valid  = log["imu_valid"].astype(bool)
    imu_accel  = log["imu_accel"]
    imu_gyro   = log["imu_gyro"]
    imu_sf_mag = log["imu_sf_mag"]
    vio_valid  = log["vio_valid"].astype(bool)
    vio_pos    = log["vio_pos"]
    rpm_sq_sum = log["rpm_sq_sum"]
    n = len(t)

    x0 = _make_x0(log["truth_pos"][0], log["truth_vel"][0], log["truth_quat"][0])
    P0 = _make_P0()
    cov_dim = P0.shape[0]

    hover    = HoverUKF()
    race     = RaceUKF()
    adaptive = AdaptiveUKF(TUNED_ADAPTIVE_CFG)

    hover.initialize(x0.copy(), P0.copy())
    race.initialize(x0.copy(), P0.copy())
    adaptive.initialize(x0.copy(), P0.copy())

    hover_pos  = np.full((n, 3), np.nan)
    race_pos   = np.full((n, 3), np.nan)
    adapt_pos  = np.full((n, 3), np.nan)
    hover_cov  = np.full((n, cov_dim, cov_dim), np.nan)
    race_cov   = np.full((n, cov_dim, cov_dim), np.nan)
    adapt_cov  = np.full((n, cov_dim, cov_dim), np.nan)

    # Step 0
    hover_pos[0]  = hover.state()[:3]
    race_pos[0]   = race.state()[:3]
    adapt_pos[0]  = adaptive.state()[:3]
    hover_cov[0]  = hover.covariance()
    race_cov[0]   = race.covariance()
    adapt_cov[0]  = adaptive.covariance()

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if dt <= 0.0:
            continue

        if imu_valid[k]:
            hover.predict(imu_accel[k], imu_gyro[k], dt)
            race.predict(imu_accel[k], imu_gyro[k], dt)
            adaptive.predict(
                imu_accel_mps2=imu_accel[k],
                imu_gyro_radps=imu_gyro[k],
                dt=dt,
                rpm_sq_sum=float(rpm_sq_sum[k]),
                specific_force_mag_mps2=float(imu_sf_mag[k]),
            )

        if vio_valid[k] and np.all(np.isfinite(vio_pos[k])):
            hover.update_vio(vio_pos[k])
            race.update_vio(vio_pos[k])
            adaptive.update_vio(vio_pos[k])

        hover_pos[k]  = hover.state()[:3]
        race_pos[k]   = race.state()[:3]
        adapt_pos[k]  = adaptive.state()[:3]
        hover_cov[k]  = hover.covariance()
        race_cov[k]   = race.covariance()
        adapt_cov[k]  = adaptive.covariance()

    return {
        "hover_pos": hover_pos,
        "race_pos": race_pos,
        "adapt_pos": adapt_pos,
        "hover_cov": hover_cov,
        "race_cov": race_cov,
        "adapt_cov": adapt_cov,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(log: dict, filters: dict, out_dir: Path, flight_name: str) -> None:
    t         = log["t"]
    truth_pos = log["truth_pos"]
    ref_pos   = log["ref_pos"]

    hover_pos = filters["hover_pos"]
    race_pos  = filters["race_pos"]
    adapt_pos = filters["adapt_pos"]
    hover_cov = filters["hover_cov"]
    race_cov  = filters["race_cov"]
    adapt_cov = filters["adapt_cov"]

    valid = np.all(np.isfinite(truth_pos), axis=1)
    mask  = valid

    # RMSE
    h_rmse = _compute_rmse(hover_pos, truth_pos, mask & np.all(np.isfinite(hover_pos), axis=1))
    r_rmse = _compute_rmse(race_pos,  truth_pos, mask & np.all(np.isfinite(race_pos),  axis=1))
    a_rmse = _compute_rmse(adapt_pos, truth_pos, mask & np.all(np.isfinite(adapt_pos), axis=1))

    # NEES (mean)
    h_nees_arr = _compute_nees(hover_pos, truth_pos, hover_cov, mask & np.all(np.isfinite(hover_pos), axis=1))
    r_nees_arr = _compute_nees(race_pos,  truth_pos, race_cov,  mask & np.all(np.isfinite(race_pos),  axis=1))
    a_nees_arr = _compute_nees(adapt_pos, truth_pos, adapt_cov, mask & np.all(np.isfinite(adapt_pos), axis=1))
    h_nees = float(np.nanmean(h_nees_arr))
    r_nees = float(np.nanmean(r_nees_arr))
    a_nees = float(np.nanmean(a_nees_arr))

    print(f"\n  {'Filter':<12}  {'RMSE (m)':>9}  {'NEES':>7}")
    print(f"  {'-'*32}")
    print(f"  {'Hover':<12}  {h_rmse:>9.4f}  {h_nees:>7.2f}")
    print(f"  {'Race':<12}  {r_rmse:>9.4f}  {r_nees:>7.2f}")
    print(f"  {'Adaptive*':<12}  {a_rmse:>9.4f}  {a_nees:>7.2f}  (* tuned beta=0.01x, sub_g=True)")
    print(f"  (NEES target = 3.0)\n", flush=True)

    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = {"hover": "#e04040", "race": "#e09000", "adaptive": "#2080d0", "truth": "#20a020"}

    # ------------------------------------------------------------------
    # 1. Position time series (3 subplots: X, Y, Z)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Position vs Ground Truth — {flight_name} (PyBullet sim)", fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(t, truth_pos[:, i], color=colors["truth"], lw=1.5, label="Truth (PyBullet)", zorder=5)
        ax.plot(t, ref_pos[:, i],   color="gray",          lw=1.0, ls="--", label="Reference (TII mocap)", alpha=0.6)
        ax.plot(t, hover_pos[:, i], color=colors["hover"],    lw=0.9, label=f"Hover (RMSE={h_rmse:.3f}m)")
        ax.plot(t, race_pos[:, i],  color=colors["race"],     lw=0.9, label=f"Race  (RMSE={r_rmse:.3f}m)")
        ax.plot(t, adapt_pos[:, i], color=colors["adaptive"], lw=0.9, label=f"Adaptive* (RMSE={a_rmse:.3f}m)")
        ax.set_ylabel(axes_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    path = out_dir / f"{flight_name}_sim_position.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)

    # ------------------------------------------------------------------
    # 2. 3-D trajectory
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_title(f"3-D Trajectory — {flight_name}", fontsize=12)

    ax3.plot(truth_pos[:, 0], truth_pos[:, 1], truth_pos[:, 2],
             color=colors["truth"], lw=1.5, label="Truth")
    ax3.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
             color="gray", lw=0.8, ls="--", alpha=0.5, label="Reference (TII)")
    ax3.plot(hover_pos[:, 0], hover_pos[:, 1], hover_pos[:, 2],
             color=colors["hover"], lw=0.9, label="Hover", alpha=0.8)
    ax3.plot(race_pos[:, 0], race_pos[:, 1], race_pos[:, 2],
             color=colors["race"], lw=0.9, label="Race", alpha=0.8)
    ax3.plot(adapt_pos[:, 0], adapt_pos[:, 1], adapt_pos[:, 2],
             color=colors["adaptive"], lw=0.9, label="Adaptive*", alpha=0.8)

    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / f"{flight_name}_sim_3d.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)

    # ------------------------------------------------------------------
    # 3. Position error magnitude over time
    # ------------------------------------------------------------------
    h_err = np.linalg.norm(hover_pos - truth_pos, axis=1)
    r_err = np.linalg.norm(race_pos  - truth_pos, axis=1)
    a_err = np.linalg.norm(adapt_pos - truth_pos, axis=1)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(f"Position Error vs Time — {flight_name}", fontsize=12)
    ax.plot(t, h_err, color=colors["hover"],    lw=0.8, label=f"Hover  RMSE={h_rmse:.3f}m")
    ax.plot(t, r_err, color=colors["race"],     lw=0.8, label=f"Race   RMSE={r_rmse:.3f}m")
    ax.plot(t, a_err, color=colors["adaptive"], lw=0.8, label=f"Adaptive* RMSE={a_rmse:.3f}m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||error|| (m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{flight_name}_sim_error.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)

    # ------------------------------------------------------------------
    # 4. NEES over time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(f"NEES vs Time (target = 3) — {flight_name}", fontsize=12)
    win = max(1, len(t) // 200)
    def smooth(x):
        return np.convolve(np.where(np.isfinite(x), x, 0), np.ones(win)/win, mode="same")
    ax.plot(t, smooth(h_nees_arr), color=colors["hover"],    lw=0.9, label=f"Hover  μNEES={h_nees:.2f}")
    ax.plot(t, smooth(r_nees_arr), color=colors["race"],     lw=0.9, label=f"Race   μNEES={r_nees:.2f}")
    ax.plot(t, smooth(a_nees_arr), color=colors["adaptive"], lw=0.9, label=f"Adaptive* μNEES={a_nees:.2f}")
    ax.axhline(3.0, color="black", ls="--", lw=1.0, label="Ideal NEES=3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_ylim(0, min(50, np.nanpercentile(np.concatenate([h_nees_arr, r_nees_arr, a_nees_arr]), 99) * 1.2))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{flight_name}_sim_nees.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)

    # ------------------------------------------------------------------
    # 5. RMSE bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"RMSE Comparison — {flight_name}", fontsize=12)
    names  = ["Hover", "Race", "Adaptive*"]
    values = [h_rmse, r_rmse, a_rmse]
    bar_colors = [colors["hover"], colors["race"], colors["adaptive"]]
    bars = ax.bar(names, values, color=bar_colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}m",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{flight_name}_sim_rmse_bar.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {path.name}", flush=True)

    return {
        "hover_rmse": h_rmse, "race_rmse": r_rmse, "adaptive_rmse": a_rmse,
        "hover_nees": h_nees, "race_nees": r_nees, "adaptive_nees": a_nees,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    npz_path = repo_root / "logs" / "tii" / f"{args.flight}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}\n  Run: python -m scripts.parse_data first.")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else repo_root / "logs" / "sim_tii"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== PyBullet TII Sim: {args.flight} ===", flush=True)
    print(f"  NPZ:      {npz_path}", flush=True)
    print(f"  Output:   {out_dir}", flush=True)

    # Step 1: Simulation
    print("\n[1/3] Running PyBullet simulation ...", flush=True)
    sim_log = run_simulation(
        npz_path=npz_path,
        duration=args.duration,
        dt=args.dt,
        gui=args.gui,
        seed=args.seed,
        repo_root=repo_root,
    )
    n = len(sim_log["t"])
    vio_count = int(np.sum(sim_log["vio_valid"]))
    imu_count = int(np.sum(sim_log["imu_valid"]))
    print(f"  Done: {n} steps, {imu_count} IMU, {vio_count} VIO", flush=True)

    # Save sim log
    sim_npz_path = out_dir / f"{args.flight}_sim_log.npz"
    np.savez_compressed(sim_npz_path, **{k: v for k, v in sim_log.items() if isinstance(v, np.ndarray)})
    print(f"  Sim log saved to {sim_npz_path.name}", flush=True)

    # Step 2: Filter replay
    print("\n[2/3] Running filters (Hover / Race / Adaptive) ...", flush=True)
    filter_results = run_filters(sim_log)
    print("  Done.", flush=True)

    # Save filter results
    filt_npz_path = out_dir / f"{args.flight}_filter_results.npz"
    save_dict = {}
    for k, v in {**sim_log, **filter_results}.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
    np.savez_compressed(filt_npz_path, **save_dict)

    # Step 3: Plots
    print("\n[3/3] Generating plots ...", flush=True)
    metrics = make_plots(sim_log, filter_results, out_dir, args.flight)

    # Save summary JSON
    summary = {
        "flight": args.flight,
        "n_steps": n,
        "imu_samples": imu_count,
        "vio_samples": vio_count,
        "adaptive_config": {
            "beta_pos_var": TUNED_ADAPTIVE_CFG.beta_pos_var,
            "beta_vel_var": TUNED_ADAPTIVE_CFG.beta_vel_var,
            "beta_att_var": TUNED_ADAPTIVE_CFG.beta_att_var,
            "subtract_gravity": TUNED_ADAPTIVE_CFG.subtract_gravity,
        },
        **metrics,
    }
    summary_path = out_dir / f"{args.flight}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to {summary_path.name}", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
