"""measurement.py: Sensor measurement representations.

Two sensor modalities are used:

IMU (MEMS Inertial Measurement Unit) — sampled at ~240 Hz in simulation, ~500 Hz on real hardware:
    - Accelerometer: specific force in body frame, m/s²
      (raw reading includes structural vibration and G-loading effects;
       does NOT include gravity — hovering reads ≈ +9.81 on z-axis)
    - Gyroscope: angular rate in body frame, rad/s

VIO (Visual-Inertial Odometry) — sampled at 30 Hz (sim) / ~277 Hz (TII dataset):
    - Position estimate in world frame, m
    - Modelled as noisy position measurement:  z_vio = p_true + η,  η ~ N(0, R_vio)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class IMUMeasurement:
    """Single IMU sample (accelerometer + gyroscope)."""

    accel_b: np.ndarray   # (3,) specific force in body frame, m/s²
    gyro_b: np.ndarray    # (3,) angular rate in body frame, rad/s
    t: float = 0.0        # timestamp, s

    # Telemetry fields used by the AdaptiveUKF noise model
    rpm_sq_sum: float = 0.0          # Σᵢ ωᵢ² (four motors), RPM²
    specific_force_mag: float = field(init=False)

    def __post_init__(self) -> None:
        self.specific_force_mag = float(np.linalg.norm(self.accel_b))

    @property
    def g_load(self) -> float:
        """Scalar G-load (specific force magnitude in units of g)."""
        return self.specific_force_mag / 9.81

    def __repr__(self) -> str:
        a = np.round(self.accel_b, 3)
        w = np.round(self.gyro_b, 4)
        return (
            f"IMUMeasurement(t={self.t:.3f}s, accel={a}, gyro={w}, "
            f"G={self.g_load:.2f})"
        )


@dataclass
class VIOMeasurement:
    """Single VIO position measurement."""

    pos_w: np.ndarray   # (3,) estimated position in world frame, m
    t: float = 0.0      # timestamp, s
    valid: bool = True  # False if the VIO pipeline dropped a frame

    def __repr__(self) -> str:
        p = np.round(self.pos_w, 4)
        return f"VIOMeasurement(t={self.t:.3f}s, pos={p}, valid={self.valid})"


@dataclass
class SensorBundle:
    """All sensor data available at one filter timestep."""

    imu: IMUMeasurement | None = None
    vio: VIOMeasurement | None = None
    t: float = 0.0

    @property
    def has_imu(self) -> bool:
        return self.imu is not None and self.imu.valid if hasattr(self.imu, "valid") else self.imu is not None

    @property
    def has_vio(self) -> bool:
        return self.vio is not None and self.vio.valid
