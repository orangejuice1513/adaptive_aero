from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate one synthetic sim log.")
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to sim_run.npz. Default: <repo>/logs/sim_run.npz",
    )
    return p


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def maybe_load_metadata(npz_path: Path) -> dict:
    meta_path = npz_path.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def finite_mask(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return np.isfinite(x)
    return np.all(np.isfinite(x), axis=1)


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_path = Path(args.log).expanduser().resolve() if args.log else repo_root / "logs" / "sim_run.npz"

    if not log_path.exists():
        raise FileNotFoundError(f"Could not find log file: {log_path}")

    D = load_npz(log_path)
    meta = maybe_load_metadata(log_path)

    t = D["t"]

    truth_pos = D["truth_pos_w"]
    truth_vel = D["truth_vel_w"]
    truth_sf = D["truth_specific_force_b"]
    truth_omega = D["truth_omega_b"]
    truth_g = D["truth_g_load"]

    ref_pos = D["ref_pos_w"]

    imu_valid = D["imu_valid"].astype(bool)
    imu_acc = D["imu_accel_mps2"]
    imu_gyro = D["imu_gyro_radps"]
    imu_acc_cov = D["imu_accel_cov_diag"]
    imu_gyro_cov = D["imu_gyro_cov_diag"]
    imu_rpm_sq = D["imu_rpm_sq_sum"]
    imu_sf_mag = D["imu_specific_force_mag_mps2"]
    imu_g = D["imu_g_load"]

    vio_valid = D["vio_valid"].astype(bool)
    vio_pos = D["vio_pos_w_m"]

    ctrl_pos_err = D["ctrl_pos_error_w"]
    rpm_cmd = D["rpm_cmd"]
    rpm_actual = D["truth_motor_rpm_actual"]

    # Residuals only where IMU/VIO valid
    imu_acc_res = np.full_like(imu_acc, np.nan)
    imu_gyro_res = np.full_like(imu_gyro, np.nan)
    imu_acc_res[imu_valid] = imu_acc[imu_valid] - truth_sf[imu_valid]
    imu_gyro_res[imu_valid] = imu_gyro[imu_valid] - truth_omega[imu_valid]

    vio_pos_res = np.full_like(vio_pos, np.nan)
    vio_pos_res[vio_valid] = vio_pos[vio_valid] - truth_pos[vio_valid]

    # Scalar summaries
    pos_err_norm = np.linalg.norm(ctrl_pos_err, axis=1)
    rpm_cmd_max = np.max(rpm_cmd, axis=1)
    rpm_actual_max = np.max(rpm_actual, axis=1)

    # Means for covariance diagnostics
    imu_acc_cov_mean = np.full(len(t), np.nan)
    imu_gyro_cov_mean = np.full(len(t), np.nan)
    imu_acc_cov_mean[imu_valid] = np.mean(imu_acc_cov[imu_valid], axis=1)
    imu_gyro_cov_mean[imu_valid] = np.mean(imu_gyro_cov[imu_valid], axis=1)

    # -------------
    # Figure 1
    # -------------
    fig1, axs = plt.subplots(2, 2, figsize=(13, 9))

    axs[0, 0].plot(ref_pos[:, 0], ref_pos[:, 1], "--", linewidth=1.2, label="Reference XY")
    axs[0, 0].plot(truth_pos[:, 0], truth_pos[:, 1], linewidth=1.8, label="Actual XY")
    axs[0, 0].set_title("Top-Down Path")
    axs[0, 0].set_xlabel("X (m)")
    axs[0, 0].set_ylabel("Y (m)")
    axs[0, 0].axis("equal")
    axs[0, 0].legend()

    axs[0, 1].plot(t, truth_g, linewidth=1.3)
    axs[0, 1].set_title("Truth G-Load vs Time")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("G")

    axs[1, 0].plot(t, pos_err_norm, linewidth=1.3)
    axs[1, 0].set_title("Controller Position Error")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("||p - p_ref|| (m)")

    axs[1, 1].plot(t, rpm_cmd_max, linewidth=1.2, label="Max RPM cmd")
    axs[1, 1].plot(t, rpm_actual_max, linewidth=1.2, label="Max RPM actual")
    axs[1, 1].set_title("Motor Demand vs Actual")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("RPM")
    axs[1, 1].legend()

    fig1.tight_layout()
    out1 = log_path.with_name(log_path.stem + "_overview.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight")

    # -------------
    # Figure 2: IMU truth vs measurement
    # -------------
    fig2, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    for i, lbl in enumerate(["x", "y", "z"]):
        axs[0, 0].plot(t, truth_sf[:, i], linewidth=1.0, label=f"truth {lbl}")
    axs[0, 0].set_title("Truth Specific Force (body)")
    axs[0, 0].set_ylabel("m/s²")
    axs[0, 0].legend(ncol=3, fontsize=8)

    valid_t = t[imu_valid]
    if np.any(imu_valid):
        for i, lbl in enumerate(["x", "y", "z"]):
            axs[0, 1].plot(valid_t, imu_acc[imu_valid, i], linewidth=0.9, label=f"meas {lbl}")
        axs[0, 1].set_title("IMU Accel Measurement")
        axs[0, 1].set_ylabel("m/s²")
        axs[0, 1].legend(ncol=3, fontsize=8)

        for i, lbl in enumerate(["x", "y", "z"]):
            axs[1, 0].plot(valid_t, imu_acc_res[imu_valid, i], linewidth=0.9, label=f"res {lbl}")
        axs[1, 0].set_title("IMU Accel Residual = meas - truth")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("m/s²")
        axs[1, 0].legend(ncol=3, fontsize=8)

        for i, lbl in enumerate(["x", "y", "z"]):
            axs[1, 1].plot(valid_t, imu_acc_cov[imu_valid, i], linewidth=0.9, label=f"cov {lbl}")
        axs[1, 1].set_title("IMU Accel Covariance Diagonal")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Variance")
        axs[1, 1].legend(ncol=3, fontsize=8)

    fig2.tight_layout()
    out2 = log_path.with_name(log_path.stem + "_imu_accel.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")

    # -------------
    # Figure 3: Gyro truth / residual / covariance
    # -------------
    fig3, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    for i, lbl in enumerate(["x", "y", "z"]):
        axs[0, 0].plot(t, truth_omega[:, i], linewidth=1.0, label=f"truth {lbl}")
    axs[0, 0].set_title("Truth Angular Rate (body)")
    axs[0, 0].set_ylabel("rad/s")
    axs[0, 0].legend(ncol=3, fontsize=8)

    if np.any(imu_valid):
        for i, lbl in enumerate(["x", "y", "z"]):
            axs[0, 1].plot(valid_t, imu_gyro[imu_valid, i], linewidth=0.9, label=f"meas {lbl}")
        axs[0, 1].set_title("IMU Gyro Measurement")
        axs[0, 1].set_ylabel("rad/s")
        axs[0, 1].legend(ncol=3, fontsize=8)

        for i, lbl in enumerate(["x", "y", "z"]):
            axs[1, 0].plot(valid_t, imu_gyro_res[imu_valid, i], linewidth=0.9, label=f"res {lbl}")
        axs[1, 0].set_title("IMU Gyro Residual = meas - truth")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("rad/s")
        axs[1, 0].legend(ncol=3, fontsize=8)

        for i, lbl in enumerate(["x", "y", "z"]):
            axs[1, 1].plot(valid_t, imu_gyro_cov[imu_valid, i], linewidth=0.9, label=f"cov {lbl}")
        axs[1, 1].set_title("IMU Gyro Covariance Diagonal")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Variance")
        axs[1, 1].legend(ncol=3, fontsize=8)

    fig3.tight_layout()
    out3 = log_path.with_name(log_path.stem + "_imu_gyro.png")
    fig3.savefig(out3, dpi=200, bbox_inches="tight")

    # -------------
    # Figure 4: Covariance vs RPM^2 and G-load
    # -------------
    fig4, axs = plt.subplots(2, 2, figsize=(13, 9))

    mask_imu = imu_valid & np.isfinite(imu_acc_cov_mean) & np.isfinite(imu_rpm_sq) & np.isfinite(imu_g)
    if np.any(mask_imu):
        axs[0, 0].scatter(imu_rpm_sq[mask_imu], imu_acc_cov_mean[mask_imu], s=6, alpha=0.5)
        axs[0, 0].set_title("Accel Covariance vs ΣRPM²")
        axs[0, 0].set_xlabel("Σ RPM²")
        axs[0, 0].set_ylabel("Mean accel variance")

        axs[0, 1].scatter(imu_g[mask_imu], imu_acc_cov_mean[mask_imu], s=6, alpha=0.5)
        axs[0, 1].set_title("Accel Covariance vs G-load")
        axs[0, 1].set_xlabel("G-load")
        axs[0, 1].set_ylabel("Mean accel variance")

        axs[1, 0].scatter(imu_rpm_sq[mask_imu], imu_gyro_cov_mean[mask_imu], s=6, alpha=0.5)
        axs[1, 0].set_title("Gyro Covariance vs ΣRPM²")
        axs[1, 0].set_xlabel("Σ RPM²")
        axs[1, 0].set_ylabel("Mean gyro variance")

        axs[1, 1].scatter(imu_g[mask_imu], imu_gyro_cov_mean[mask_imu], s=6, alpha=0.5)
        axs[1, 1].set_title("Gyro Covariance vs G-load")
        axs[1, 1].set_xlabel("G-load")
        axs[1, 1].set_ylabel("Mean gyro variance")

    fig4.tight_layout()
    out4 = log_path.with_name(log_path.stem + "_covariance_relationships.png")
    fig4.savefig(out4, dpi=200, bbox_inches="tight")

    # -------------
    # Figure 5: VIO timing / residuals
    # -------------
    fig5, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    if np.any(vio_valid):
        vio_t = t[vio_valid]
        axs[0].plot(t, truth_pos[:, 0], linewidth=1.0, label="truth x")
        axs[0].plot(vio_t, vio_pos[vio_valid, 0], ".", markersize=3, label="vio x")
        axs[0].plot(t, truth_pos[:, 1], linewidth=1.0, label="truth y")
        axs[0].plot(vio_t, vio_pos[vio_valid, 1], ".", markersize=3, label="vio y")
        axs[0].plot(t, truth_pos[:, 2], linewidth=1.0, label="truth z")
        axs[0].plot(vio_t, vio_pos[vio_valid, 2], ".", markersize=3, label="vio z")
        axs[0].set_title("VIO Samples vs Truth")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend(ncol=3, fontsize=8)

        for i, lbl in enumerate(["x", "y", "z"]):
            axs[1].plot(vio_t, vio_pos_res[vio_valid, i], ".", markersize=3, label=f"{lbl} residual")
        axs[1].set_title("VIO Residual = meas - truth")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Residual (m)")
        axs[1].legend(ncol=3, fontsize=8)

    fig5.tight_layout()
    out5 = log_path.with_name(log_path.stem + "_vio.png")
    fig5.savefig(out5, dpi=200, bbox_inches="tight")

    # -------------
    # Text summary
    # -------------
    print(f"Loaded log: {log_path}")
    if meta:
        print(f"Metadata: {json.dumps(meta, indent=2)}")

    print("\n--- Validation Summary ---")
    print(f"Total steps:               {len(t)}")
    print(f"Duration:                  {t[-1] - t[0]:.3f} s")
    print(f"IMU valid samples:         {int(np.sum(imu_valid))}")
    print(f"VIO valid samples:         {int(np.sum(vio_valid))}")
    print(f"Mean truth G-load:         {float(np.mean(truth_g)):.3f}")
    print(f"Max truth G-load:          {float(np.max(truth_g)):.3f}")
    print(f"Mean controller pos error: {float(np.mean(pos_err_norm)):.3f} m")
    print(f"Max controller pos error:  {float(np.max(pos_err_norm)):.3f} m")
    if np.any(imu_valid):
        print(f"Mean accel cov diag:       {float(np.nanmean(imu_acc_cov_mean)):.4f}")
        print(f"Mean gyro cov diag:        {float(np.nanmean(imu_gyro_cov_mean)):.4f}")
    if np.any(vio_valid):
        vio_err = np.linalg.norm(vio_pos_res[vio_valid], axis=1)
        print(f"Mean VIO position error:   {float(np.mean(vio_err)):.4f} m")

    print("\nSaved:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")
    print(f"  {out4}")
    print(f"  {out5}")


if __name__ == "__main__":
    main()