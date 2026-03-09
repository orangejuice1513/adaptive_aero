from pathlib import Path
import numpy as np
import matplotlib

# Non-interactive backend: avoids macOS Tk / GUI backend crashes
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.env import DroneEnv, SimConfig, DroneConfig
from sim.controller import GeometricController
from sim.trajectory import StressTestTrajectory

ROOT = Path(__file__).resolve().parents[1]

env = DroneEnv(
    sim_cfg=SimConfig(dt=1 / 240, gui=True),
    drone_cfg=DroneConfig(
        urdf_path=str(ROOT / "assets" / "quad.urdf"),
        expected_mass_kg=1.35,
        start_pos_w=np.array([0.0, 0.0, 1.0]),
    ),
)

truth = env.reset()
controller = GeometricController.from_env(env)
traj = StressTestTrajectory(speed_scale=2.0)

# Actual trajectory logs
log_t = []
log_x = []
log_y = []
log_z = []
log_g = []

# Reference trajectory logs
log_ref_x = []
log_ref_y = []
log_ref_z = []

# Tracking / actuator logs
log_pos_err = []
log_rpm_cmd_max = []
log_rpm_actual_max = []

try:
    for k in range(10000):
        truth = env.get_truth_state()
        ref = traj.sample(truth.t)
        rpm_cmd = controller.compute_rpm(truth, ref, dt=env.sim_cfg.dt)
        truth = env.step(rpm_cmd)

        # Time
        log_t.append(truth.t)

        # Actual position
        log_x.append(truth.pos_w[0])
        log_y.append(truth.pos_w[1])
        log_z.append(truth.pos_w[2])

        # Reference position
        log_ref_x.append(ref.pos_w[0])
        log_ref_y.append(ref.pos_w[1])
        log_ref_z.append(ref.pos_w[2])

        # IMU-style g-load from specific force
        g_load = np.linalg.norm(truth.specific_force_b) / 9.81
        log_g.append(min(g_load, 8.0))  # clip huge spikes for readability

        # Position tracking error
        log_pos_err.append(np.linalg.norm(truth.pos_w - ref.pos_w))

        # Motor demand vs actual
        log_rpm_cmd_max.append(np.max(rpm_cmd))
        log_rpm_actual_max.append(np.max(truth.motor_rpm_actual))

finally:
    env.close()

# Convert to arrays
log_t = np.asarray(log_t, dtype=float)

log_x = np.asarray(log_x, dtype=float)
log_y = np.asarray(log_y, dtype=float)
log_z = np.asarray(log_z, dtype=float)

log_ref_x = np.asarray(log_ref_x, dtype=float)
log_ref_y = np.asarray(log_ref_y, dtype=float)
log_ref_z = np.asarray(log_ref_z, dtype=float)

log_g = np.asarray(log_g, dtype=float)
log_pos_err = np.asarray(log_pos_err, dtype=float)

log_rpm_cmd_max = np.asarray(log_rpm_cmd_max, dtype=float)
log_rpm_actual_max = np.asarray(log_rpm_actual_max, dtype=float)

# ----------------------------
# Figure 1: trajectory + G-load
# ----------------------------
fig1 = plt.figure(figsize=(12, 5))

ax1 = fig1.add_subplot(1, 2, 1, projection="3d")
ax1.plot(log_ref_x, log_ref_y, log_ref_z, linestyle="--", linewidth=1.2, label="Reference")
ax1.plot(log_x, log_y, log_z, linewidth=2.0, label="Actual")
ax1.set_title("Drone Trajectory")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()

ax2 = fig1.add_subplot(1, 2, 2)
ax2.plot(log_t, log_g, linewidth=1.5)
ax2.set_title("G-Load vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("G")

plt.tight_layout()

out_path1 = ROOT / "trajectory_and_gload.png"
fig1.savefig(out_path1, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {out_path1}")

# ---------------------------------------
# Figure 2: controller-limiter diagnostics
# ---------------------------------------
fig2, axs = plt.subplots(2, 2, figsize=(12, 8))

# XY top-down reference vs actual
axs[0, 0].plot(log_ref_x, log_ref_y, linestyle="--", linewidth=1.2, label="Reference")
axs[0, 0].plot(log_x, log_y, linewidth=1.8, label="Actual")
axs[0, 0].set_title("Top-Down Path: Reference vs Actual")
axs[0, 0].set_xlabel("X (m)")
axs[0, 0].set_ylabel("Y (m)")
axs[0, 0].axis("equal")
axs[0, 0].legend()

# Z tracking
axs[0, 1].plot(log_t, log_ref_z, linestyle="--", linewidth=1.2, label="Ref Z")
axs[0, 1].plot(log_t, log_z, linewidth=1.6, label="Actual Z")
axs[0, 1].set_title("Altitude Tracking")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Z (m)")
axs[0, 1].legend()

# Position error
axs[1, 0].plot(log_t, log_pos_err, linewidth=1.5)
axs[1, 0].set_title("Position Tracking Error")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("||p - p_ref|| (m)")

# RPM command vs actual
axs[1, 1].plot(log_t, log_rpm_cmd_max, linewidth=1.2, label="Max RPM Command")
axs[1, 1].plot(log_t, log_rpm_actual_max, linewidth=1.2, label="Max RPM Actual")
axs[1, 1].set_title("Motor Demand vs Actual")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("RPM")
axs[1, 1].legend()

plt.tight_layout()

out_path2 = ROOT / "controller_diagnostics.png"
fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {out_path2}")

# ----------------------------
# Text summary
# ----------------------------
print("\n--- Controller Diagnostic Summary ---")
print(f"Mean position error: {np.mean(log_pos_err):.3f} m")
print(f"Max position error:  {np.max(log_pos_err):.3f} m")
print(f"Mean G-load:         {np.mean(log_g):.3f} G")
print(f"Max G-load:          {np.max(log_g):.3f} G")
print(f"Max RPM command:     {np.max(log_rpm_cmd_max):.1f}")
print(f"Max RPM actual:      {np.max(log_rpm_actual_max):.1f}")