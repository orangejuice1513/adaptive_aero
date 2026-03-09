from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Optional
import math

import numpy as np

from sim.env import TruthState


Array3 = np.ndarray
Array4 = np.ndarray


def _as_vec3(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_vec4(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _gauss_markov_step(
    x: np.ndarray,
    dt: float,
    tau_s: float,
    sigma_ss: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    First-order Gauss-Markov process with steady-state std sigma_ss.
    """
    x = np.asarray(x, dtype=float)
    sigma_ss = np.asarray(sigma_ss, dtype=float)

    if tau_s <= 1e-9:
        return rng.normal(0.0, sigma_ss, size=x.shape)

    phi = math.exp(-dt / tau_s)
    q = sigma_ss * math.sqrt(max(0.0, 1.0 - phi * phi))
    return phi * x + rng.normal(0.0, q, size=x.shape)


@dataclass(slots=True)
class ImuSample:
    t: float
    accel_mps2: Array3
    gyro_radps: Array3

    accel_bias_mps2: Array3
    gyro_bias_radps: Array3

    accel_cov_diag: Array3
    gyro_cov_diag: Array3

    rpm_sq_sum: float
    specific_force_mag_mps2: float
    g_load: float


@dataclass(slots=True)
class VioSample:
    t: float
    pos_w_m: Array3
    sigma_m: float
    latency_s: float


@dataclass(slots=True)
class SensorTelemetry:
    t: float
    motor_rpm_actual: Array4
    motor_thrusts_n: Array4
    rpm_sq_sum: float
    specific_force_mag_mps2: float
    g_load: float


@dataclass(slots=True)
class ImuConfig:
    rate_hz: float = 240.0

    # Floor covariance diagonals (variance, not std).
    # These are the "R_floor" terms.
    accel_floor_var_mps2_sq: Array3 = field(
        default_factory=lambda: np.array([0.08**2, 0.08**2, 0.10**2], dtype=float)
    )
    gyro_floor_var_radps_sq: Array3 = field(
        default_factory=lambda: np.array([0.015**2, 0.015**2, 0.018**2], dtype=float)
    )

    # Paper-inspired heteroscedastic scaling:
    # R_vibe = alpha * (sum_i RPM_i^2) * I
    # R_load = beta  * ||a|| * I
    #
    # Because RPM^2 is huge numerically, we normalize by motor_max_rpm^2 later.
    # alpha / beta still control the covariance growth.
    alpha_accel_var: float = 0.80
    alpha_gyro_var: float = 0.030

    beta_accel_var: float = 0.020
    beta_gyro_var: float = 0.0015

    # Slowly varying bias terms
    accel_bias_std_mps2: Array3 = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.04], dtype=float)
    )
    gyro_bias_std_radps: Array3 = field(
        default_factory=lambda: np.array([0.004, 0.004, 0.004], dtype=float)
    )
    accel_bias_tau_s: float = 20.0
    gyro_bias_tau_s: float = 20.0

    # Optional caps so the sim does not explode numerically
    max_accel_var_mps2_sq: float = 20.0
    max_gyro_var_radps_sq: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("IMU rate_hz must be > 0")

        self.accel_floor_var_mps2_sq = _as_vec3(
            self.accel_floor_var_mps2_sq, "accel_floor_var_mps2_sq"
        )
        self.gyro_floor_var_radps_sq = _as_vec3(
            self.gyro_floor_var_radps_sq, "gyro_floor_var_radps_sq"
        )
        self.accel_bias_std_mps2 = _as_vec3(
            self.accel_bias_std_mps2, "accel_bias_std_mps2"
        )
        self.gyro_bias_std_radps = _as_vec3(
            self.gyro_bias_std_radps, "gyro_bias_std_radps"
        )

        if np.any(self.accel_floor_var_mps2_sq < 0.0) or np.any(self.gyro_floor_var_radps_sq < 0.0):
            raise ValueError("floor variances must be >= 0")
        if np.any(self.accel_bias_std_mps2 < 0.0) or np.any(self.gyro_bias_std_radps < 0.0):
            raise ValueError("bias std must be >= 0")
        if self.accel_bias_tau_s <= 0.0 or self.gyro_bias_tau_s <= 0.0:
            raise ValueError("bias tau must be > 0")
        if self.alpha_accel_var < 0.0 or self.alpha_gyro_var < 0.0:
            raise ValueError("alpha terms must be >= 0")
        if self.beta_accel_var < 0.0 or self.beta_gyro_var < 0.0:
            raise ValueError("beta terms must be >= 0")


@dataclass(slots=True)
class VioConfig:
    rate_hz: float = 30.0
    pos_noise_std_m: float = 0.05
    latency_s: float = 0.03

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("VIO rate_hz must be > 0")
        if self.pos_noise_std_m < 0.0:
            raise ValueError("VIO pos_noise_std_m must be >= 0")
        if self.latency_s < 0.0:
            raise ValueError("VIO latency_s must be >= 0")


@dataclass(slots=True)
class SensorSuiteConfig:
    imu: ImuConfig = field(default_factory=ImuConfig)
    vio: VioConfig = field(default_factory=VioConfig)

    # Needed only to normalize sum(RPM^2) into a sane scale.
    motor_max_rpm: float = 12000.0

    seed: int = 0

    def __post_init__(self) -> None:
        if self.motor_max_rpm <= 0.0:
            raise ValueError("motor_max_rpm must be > 0")


class SensorSuite:
    """
    TruthState -> synthetic IMU / VIO measurements.

    Paper-aligned IMU heteroscedastic covariance:
        R_k = R_floor + R_vibe(Omega_k) + R_load(a_k)

    where
        R_vibe ~ alpha * sum_i(RPM_i^2) * I
        R_load ~ beta * ||specific_force_b|| * I
    """

    @dataclass(slots=True)
    class Output:
        imu: Optional[ImuSample]
        vio: Optional[VioSample]
        telemetry: SensorTelemetry

    def __init__(self, cfg: Optional[SensorSuiteConfig] = None) -> None:
        self.cfg = cfg or SensorSuiteConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self._last_imu_t: Optional[float] = None
        self._last_vio_t: Optional[float] = None

        self._accel_bias = np.zeros(3, dtype=float)
        self._gyro_bias = np.zeros(3, dtype=float)

        self._vio_delay_q: Deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._last_imu_t = None
        self._last_vio_t = None
        self._accel_bias[:] = 0.0
        self._gyro_bias[:] = 0.0
        self._vio_delay_q.clear()

    def update(self, truth: TruthState) -> Output:
        telemetry = self._make_telemetry(truth)
        imu = self._maybe_sample_imu(truth, telemetry)
        vio = self._maybe_sample_vio(truth)
        return SensorSuite.Output(imu=imu, vio=vio, telemetry=telemetry)

    # ------------------------------------------------------------------
    # Telemetry / proposal quantities
    # ------------------------------------------------------------------

    def rpm_sq_sum(self, truth: TruthState) -> float:
        rpm = _as_vec4(truth.motor_rpm_actual, "motor_rpm_actual")
        return float(np.sum(rpm ** 2))

    def specific_force_mag(self, truth: TruthState) -> float:
        return float(np.linalg.norm(np.asarray(truth.specific_force_b, dtype=float)))

    def _make_telemetry(self, truth: TruthState) -> SensorTelemetry:
        sf_mag = self.specific_force_mag(truth)
        return SensorTelemetry(
            t=float(truth.t),
            motor_rpm_actual=np.asarray(truth.motor_rpm_actual, dtype=float).copy(),
            motor_thrusts_n=np.asarray(truth.motor_thrusts_n, dtype=float).copy(),
            rpm_sq_sum=self.rpm_sq_sum(truth),
            specific_force_mag_mps2=sf_mag,
            g_load=sf_mag / 9.81,
        )

    # ------------------------------------------------------------------
    # IMU
    # ------------------------------------------------------------------

    def _maybe_sample_imu(self, truth: TruthState, telemetry: SensorTelemetry) -> Optional[ImuSample]:
        t = float(truth.t)
        imu_dt = 1.0 / self.cfg.imu.rate_hz

        if self._last_imu_t is not None and (t - self._last_imu_t) < (imu_dt - 1e-9):
            return None

        dt = imu_dt if self._last_imu_t is None else max(imu_dt, t - self._last_imu_t)
        self._last_imu_t = t

        # Bias random walk / Gauss-Markov
        self._accel_bias = _gauss_markov_step(
            self._accel_bias,
            dt=dt,
            tau_s=self.cfg.imu.accel_bias_tau_s,
            sigma_ss=self.cfg.imu.accel_bias_std_mps2,
            rng=self.rng,
        )
        self._gyro_bias = _gauss_markov_step(
            self._gyro_bias,
            dt=dt,
            tau_s=self.cfg.imu.gyro_bias_tau_s,
            sigma_ss=self.cfg.imu.gyro_bias_std_radps,
            rng=self.rng,
        )

        rpm_sq_sum = telemetry.rpm_sq_sum
        sf_mag = telemetry.specific_force_mag_mps2
        g_load = telemetry.g_load

        # Normalize RPM^2 so alpha coefficients stay human-sized.
        rpm_sq_norm = rpm_sq_sum / (4.0 * (self.cfg.motor_max_rpm ** 2))

        # Proposal-style covariance law
        accel_cov_diag = (
            self.cfg.imu.accel_floor_var_mps2_sq
            + self.cfg.imu.alpha_accel_var * rpm_sq_norm * np.ones(3, dtype=float)
            + self.cfg.imu.beta_accel_var * sf_mag * np.ones(3, dtype=float)
        )
        gyro_cov_diag = (
            self.cfg.imu.gyro_floor_var_radps_sq
            + self.cfg.imu.alpha_gyro_var * rpm_sq_norm * np.ones(3, dtype=float)
            + self.cfg.imu.beta_gyro_var * sf_mag * np.ones(3, dtype=float)
        )

        accel_cov_diag = np.clip(
            accel_cov_diag, 0.0, self.cfg.imu.max_accel_var_mps2_sq
        )
        gyro_cov_diag = np.clip(
            gyro_cov_diag, 0.0, self.cfg.imu.max_gyro_var_radps_sq
        )

        accel_sigma = np.sqrt(accel_cov_diag)
        gyro_sigma = np.sqrt(gyro_cov_diag)

        accel_true = np.asarray(truth.specific_force_b, dtype=float)
        gyro_true = np.asarray(truth.omega_b, dtype=float)

        accel_noise = self.rng.normal(0.0, accel_sigma, size=3)
        gyro_noise = self.rng.normal(0.0, gyro_sigma, size=3)

        accel_meas = accel_true + self._accel_bias + accel_noise
        gyro_meas = gyro_true + self._gyro_bias + gyro_noise

        return ImuSample(
            t=t,
            accel_mps2=accel_meas,
            gyro_radps=gyro_meas,
            accel_bias_mps2=self._accel_bias.copy(),
            gyro_bias_radps=self._gyro_bias.copy(),
            accel_cov_diag=accel_cov_diag.copy(),
            gyro_cov_diag=gyro_cov_diag.copy(),
            rpm_sq_sum=rpm_sq_sum,
            specific_force_mag_mps2=sf_mag,
            g_load=g_load,
        )

    # ------------------------------------------------------------------
    # VIO
    # ------------------------------------------------------------------

    def _maybe_sample_vio(self, truth: TruthState) -> Optional[VioSample]:
        t = float(truth.t)
        vio_dt = 1.0 / self.cfg.vio.rate_hz

        sampled_now = False
        if self._last_vio_t is None or (t - self._last_vio_t) >= (vio_dt - 1e-9):
            self._last_vio_t = t
            sampled_now = True

        if sampled_now:
            pos_true = np.asarray(truth.pos_w, dtype=float)
            pos_meas = pos_true + self.rng.normal(
                0.0, self.cfg.vio.pos_noise_std_m, size=3
            )
            ready_t = t + self.cfg.vio.latency_s
            self._vio_delay_q.append((ready_t, pos_meas))

        if len(self._vio_delay_q) == 0:
            return None

        if self._vio_delay_q[0][0] <= t + 1e-12:
            _, pos_meas = self._vio_delay_q.popleft()
            return VioSample(
                t=t,
                pos_w_m=pos_meas.copy(),
                sigma_m=self.cfg.vio.pos_noise_std_m,
                latency_s=self.cfg.vio.latency_s,
            )

        return None


__all__ = [
    "ImuConfig",
    "VioConfig",
    "SensorSuiteConfig",
    "ImuSample",
    "VioSample",
    "SensorTelemetry",
    "SensorSuite",
]