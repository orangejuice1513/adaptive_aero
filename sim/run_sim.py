from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from sim.env import DroneEnv, SimConfig, DroneConfig
from sim.controller import GeometricController, ControllerConfig
from sim.trajectory import StressTestTrajectory
from sim.sensors import SensorSuite, SensorSuiteConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one synthetic drone sim and log truth + sensors.")
    p.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds")
    p.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics timestep")
    p.add_argument("--gui", action="store_true", help="Run PyBullet with GUI")
    p.add_argument("--speed-scale", type=float, default=2.0, help="Trajectory speed multiplier")
    p.add_argument("--seed", type=int, default=1, help="Random seed for sensor noise")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Default: <repo>/logs/sim_run.npz",
    )
    return p


def make_output_path(repo_root: Path, user_path: str | None) -> Path:
    if user_path is not None:
        out = Path(user_path).expanduser().resolve()
    else:
        out = repo_root / "logs" / "sim_run.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def nan_vec(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def main() -> None:
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = make_output_path(repo_root, args.out)

    env = DroneEnv(
        sim_cfg=SimConfig(
            dt=args.dt,
            gui=args.gui,
            seed=args.seed,
        ),
        drone_cfg=DroneConfig(
            urdf_path=str(repo_root / "assets" / "quad.urdf"),
            expected_mass_kg=1.35,
            start_pos_w=np.array([0.0, 0.0, 1.0], dtype=float),
        ),
    )

    # Use your currently tuned controller config here if you want.
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

    truth = env.reset()
    controller = GeometricController.from_env(env, cfg=ctrl_cfg)
    controller.reset()

    traj = StressTestTrajectory(speed_scale=args.speed_scale)

    sensors = SensorSuite(
        SensorSuiteConfig(
            motor_max_rpm=env.drone_cfg.motor.max_rpm,
            seed=args.seed,
        )
    )
    sensors.reset()

    n_steps = int(np.ceil(args.duration / args.dt))

    # Fixed-length logs
    log = {
        # Time
        "t": [],
        # Truth
        "truth_pos_w": [],
        "truth_vel_w": [],
        "truth_acc_w": [],
        "truth_quat_wb": [],
        "truth_omega_b": [],
        "truth_specific_force_b": [],
        "truth_g_load": [],
        "truth_motor_rpm_actual": [],
        "truth_motor_thrusts_n": [],
        # Reference
        "ref_pos_w": [],
        "ref_vel_w": [],
        "ref_acc_w": [],
        "ref_yaw": [],
        "ref_yaw_rate": [],
        # Controller output
        "rpm_cmd": [],
        "ctrl_total_thrust_cmd_n": [],
        "ctrl_body_torque_cmd_nm": [],
        "ctrl_pos_error_w": [],
        "ctrl_vel_error_w": [],
        "ctrl_att_error_b": [],
        "ctrl_rate_error_b": [],
        # IMU
        "imu_valid": [],
        "imu_accel_mps2": [],
        "imu_gyro_radps": [],
        "imu_accel_bias_mps2": [],
        "imu_gyro_bias_radps": [],
        "imu_accel_cov_diag": [],
        "imu_gyro_cov_diag": [],
        "imu_rpm_sq_sum": [],
        "imu_specific_force_mag_mps2": [],
        "imu_g_load": [],
        # VIO
        "vio_valid": [],
        "vio_pos_w_m": [],
        "vio_sigma_m": [],
        "vio_latency_s": [],
        # Telemetry
        "telemetry_rpm_sq_sum": [],
        "telemetry_specific_force_mag_mps2": [],
        "telemetry_g_load": [],
        "telemetry_motor_rpm_actual": [],
        "telemetry_motor_thrusts_n": [],
    }

    try:
        for _ in range(n_steps):
            # Current truth for control
            truth_now = env.get_truth_state()
            ref = traj.sample(truth_now.t)
            ctrl = controller.compute(truth_now, ref, dt=env.sim_cfg.dt)

            # Advance physics
            truth = env.step(ctrl.rpm_cmd)

            # Sample sensors on post-step truth
            sens = sensors.update(truth)

            # Log truth
            log["t"].append(truth.t)
            log["truth_pos_w"].append(truth.pos_w.copy())
            log["truth_vel_w"].append(truth.vel_w.copy())
            log["truth_acc_w"].append(truth.acc_w.copy())
            log["truth_quat_wb"].append(truth.quat_wb_xyzw.copy())
            log["truth_omega_b"].append(truth.omega_b.copy())
            log["truth_specific_force_b"].append(truth.specific_force_b.copy())
            log["truth_g_load"].append(float(truth.g_load))
            log["truth_motor_rpm_actual"].append(truth.motor_rpm_actual.copy())
            log["truth_motor_thrusts_n"].append(truth.motor_thrusts_n.copy())

            # Log reference at same sim time for easier analysis
            ref_post = traj.sample(truth.t)
            log["ref_pos_w"].append(ref_post.pos_w.copy())
            log["ref_vel_w"].append(ref_post.vel_w.copy())
            log["ref_acc_w"].append(ref_post.acc_w.copy())
            log["ref_yaw"].append(float(ref_post.yaw))
            log["ref_yaw_rate"].append(float(ref_post.yaw_rate))

            # Log controller
            log["rpm_cmd"].append(ctrl.rpm_cmd.copy())
            log["ctrl_total_thrust_cmd_n"].append(float(ctrl.total_thrust_cmd_n))
            log["ctrl_body_torque_cmd_nm"].append(ctrl.body_torque_cmd_nm.copy())
            log["ctrl_pos_error_w"].append(ctrl.pos_error_w.copy())
            log["ctrl_vel_error_w"].append(ctrl.vel_error_w.copy())
            log["ctrl_att_error_b"].append(ctrl.att_error_b.copy())
            log["ctrl_rate_error_b"].append(ctrl.rate_error_b.copy())

            # Log telemetry
            log["telemetry_rpm_sq_sum"].append(float(sens.telemetry.rpm_sq_sum))
            log["telemetry_specific_force_mag_mps2"].append(float(sens.telemetry.specific_force_mag_mps2))
            log["telemetry_g_load"].append(float(sens.telemetry.g_load))
            log["telemetry_motor_rpm_actual"].append(sens.telemetry.motor_rpm_actual.copy())
            log["telemetry_motor_thrusts_n"].append(sens.telemetry.motor_thrusts_n.copy())

            # Log IMU
            if sens.imu is not None:
                log["imu_valid"].append(1)
                log["imu_accel_mps2"].append(sens.imu.accel_mps2.copy())
                log["imu_gyro_radps"].append(sens.imu.gyro_radps.copy())
                log["imu_accel_bias_mps2"].append(sens.imu.accel_bias_mps2.copy())
                log["imu_gyro_bias_radps"].append(sens.imu.gyro_bias_radps.copy())
                log["imu_accel_cov_diag"].append(sens.imu.accel_cov_diag.copy())
                log["imu_gyro_cov_diag"].append(sens.imu.gyro_cov_diag.copy())
                log["imu_rpm_sq_sum"].append(float(sens.imu.rpm_sq_sum))
                log["imu_specific_force_mag_mps2"].append(float(sens.imu.specific_force_mag_mps2))
                log["imu_g_load"].append(float(sens.imu.g_load))
            else:
                log["imu_valid"].append(0)
                log["imu_accel_mps2"].append(np.full(3, np.nan))
                log["imu_gyro_radps"].append(np.full(3, np.nan))
                log["imu_accel_bias_mps2"].append(np.full(3, np.nan))
                log["imu_gyro_bias_radps"].append(np.full(3, np.nan))
                log["imu_accel_cov_diag"].append(np.full(3, np.nan))
                log["imu_gyro_cov_diag"].append(np.full(3, np.nan))
                log["imu_rpm_sq_sum"].append(np.nan)
                log["imu_specific_force_mag_mps2"].append(np.nan)
                log["imu_g_load"].append(np.nan)

            # Log VIO
            if sens.vio is not None:
                log["vio_valid"].append(1)
                log["vio_pos_w_m"].append(sens.vio.pos_w_m.copy())
                log["vio_sigma_m"].append(float(sens.vio.sigma_m))
                log["vio_latency_s"].append(float(sens.vio.latency_s))
            else:
                log["vio_valid"].append(0)
                log["vio_pos_w_m"].append(np.full(3, np.nan))
                log["vio_sigma_m"].append(np.nan)
                log["vio_latency_s"].append(np.nan)

    finally:
        env.close()

    # Convert to arrays
    arr = {k: np.asarray(v) for k, v in log.items()}

    metadata = {
        "duration_s": float(args.duration),
        "dt_s": float(args.dt),
        "physics_rate_hz": float(1.0 / args.dt),
        "trajectory": "StressTestTrajectory",
        "trajectory_speed_scale": float(args.speed_scale),
        "imu_rate_hz": float(sensors.cfg.imu.rate_hz),
        "vio_rate_hz": float(sensors.cfg.vio.rate_hz),
        "vio_latency_s": float(sensors.cfg.vio.latency_s),
        "vio_sigma_m": float(sensors.cfg.vio.pos_noise_std_m),
        "motor_max_rpm": float(sensors.cfg.motor_max_rpm),
        "seed": int(args.seed),
    }

    np.savez_compressed(out_path, **arr)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved sim log to: {out_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Steps: {len(arr['t'])}")
    print(f"IMU samples: {int(np.sum(arr['imu_valid']))}")
    print(f"VIO samples: {int(np.sum(arr['vio_valid']))}")
    print(f"Mean truth g-load: {float(np.mean(arr['truth_g_load'])):.3f}")
    print(f"Max truth g-load:  {float(np.max(arr['truth_g_load'])):.3f}")


if __name__ == "__main__":
    main()