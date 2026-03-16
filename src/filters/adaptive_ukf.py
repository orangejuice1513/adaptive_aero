from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.filters.ukf_core import (
    UnscentedKalmanFilter,
    DroneStateSpaceModel,
    make_process_noise,
    make_position_measurement_noise,
)


@dataclass(slots=True)
class AdaptiveUKFConfig:
    # Base process noise floor
    pos_process_var: float = 3e-5
    vel_process_var: float = 2e-2
    att_process_var: float = 3e-5

    # Keep VIO measurement noise fixed
    vio_pos_var: float = 0.05 ** 2

    # Adaptive scaling for PROCESS noise Q
    # Q_k = Q_floor + alpha * normalized_sum_rpm_sq + beta * ||specific_force||
    alpha_pos_var: float = 2e-4
    alpha_vel_var: float = 5e-2
    alpha_att_var: float = 2e-4

    beta_pos_var: float = 1e-4
    beta_vel_var: float = 2e-2
    beta_att_var: float = 1e-4

    motor_max_rpm: float = 12000.0

    # Caps so Q does not explode
    max_pos_var: float = 5e-3
    max_vel_var: float = 5e-1
    max_att_var: float = 5e-3

    # If True, subtract 9.81 m/s² from specific force before applying beta,
    # so Q only grows for G-load *above* hover (excess Gs from manoeuvres).
    subtract_gravity: bool = False

    def rpm_sq_norm(self, rpm_sq_sum: float) -> float:
        return float(rpm_sq_sum / (4.0 * self.motor_max_rpm ** 2))

    def effective_sf(self, specific_force_mag_mps2: float) -> float:
        if self.subtract_gravity:
            return max(0.0, specific_force_mag_mps2 - 9.81)
        return float(specific_force_mag_mps2)


class AdaptiveUKF:
    def __init__(self, cfg: AdaptiveUKFConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveUKFConfig()
        self.model = DroneStateSpaceModel()
        self.ukf = UnscentedKalmanFilter(self.model)

        # Fixed VIO measurement covariance
        self.R = make_position_measurement_noise(self.cfg.vio_pos_var)

        # For logging/debugging
        self.last_pos_var: float = self.cfg.pos_process_var
        self.last_vel_var: float = self.cfg.vel_process_var
        self.last_att_var: float = self.cfg.att_process_var

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.ukf.initialize(x0, P0)

    def compute_Q(
        self,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> np.ndarray:
        rpm_term = self.cfg.rpm_sq_norm(rpm_sq_sum)
        g_term = self.cfg.effective_sf(specific_force_mag_mps2)

        pos_var = (
            self.cfg.pos_process_var
            + self.cfg.alpha_pos_var * rpm_term
            + self.cfg.beta_pos_var * g_term
        )
        vel_var = (
            self.cfg.vel_process_var
            + self.cfg.alpha_vel_var * rpm_term
            + self.cfg.beta_vel_var * g_term
        )
        att_var = (
            self.cfg.att_process_var
            + self.cfg.alpha_att_var * rpm_term
            + self.cfg.beta_att_var * g_term
        )

        pos_var = float(np.clip(pos_var, self.cfg.pos_process_var, self.cfg.max_pos_var))
        vel_var = float(np.clip(vel_var, self.cfg.vel_process_var, self.cfg.max_vel_var))
        att_var = float(np.clip(att_var, self.cfg.att_process_var, self.cfg.max_att_var))

        self.last_pos_var = pos_var
        self.last_vel_var = vel_var
        self.last_att_var = att_var

        return make_process_noise(
            pos_var=pos_var,
            vel_var=vel_var,
            att_var=att_var,
        )

    def predict(
        self,
        imu_accel_mps2: np.ndarray,
        imu_gyro_radps: np.ndarray,
        dt: float,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> None:
        u = np.concatenate([imu_accel_mps2, imu_gyro_radps]).astype(float)
        Q = self.compute_Q(
            rpm_sq_sum=rpm_sq_sum,
            specific_force_mag_mps2=specific_force_mag_mps2,
        )
        self.ukf.predict(u=u, dt=dt, Q=Q)

    def update_vio(self, vio_pos_w_m: np.ndarray) -> None:
        self.ukf.update(z=np.asarray(vio_pos_w_m, dtype=float), R=self.R)

    def state(self) -> np.ndarray:
        return self.ukf.state.x.copy()

    def covariance(self) -> np.ndarray:
        return self.ukf.state.P.copy()