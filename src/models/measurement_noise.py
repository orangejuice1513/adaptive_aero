"""measurement_noise.py: Heteroscedastic measurement noise model.

The measurement noise covariance R_k is modelled as a non-stationary
Gaussian process whose instantaneous value is a function of the current
motor RPM (vibration) and the specific force magnitude (G-loading):

    R_k = R_floor + R_vibe(Ω_k) + R_load(a_k)

where

    R_floor = r_floor · I₃                          (sensor noise floor)

    R_vibe(Ω_k) = α · (Σᵢ Ωᵢ²) / (4 · Ω_max²) · I₃
                                                     (RPM-induced vibration)
    R_load(a_k) = β · ‖a_k‖ · I₃                   (G-loading sensitivity)

Physical interpretation
-----------------------
α  (alpha) — structural resonance / damping coefficient.  Higher α means
    the airframe transmits more motor vibration to the IMU.

β  (beta) — empirical G-sensitivity coefficient of the MEMS accelerometer
    and gyroscope axes.  Higher β means the sensors saturate/distort more
    during high-G manoeuvres.

Note on implementation
----------------------
In the filter implementations (adaptive_ukf.py) this same mathematical
structure is used to adapt the PROCESS noise Q_k rather than the
measurement noise R_k.  Adapting Q captures the same physical effects
(IMU readings become less trustworthy under vibration/G-loading) while
keeping the VIO measurement noise R fixed, which is well-motivated
because camera-based position estimates are not affected by motor RPM.
This module provides the standalone noise model for analysis and can be
used to compute either R_k or Q_k.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HeteroscedasticNoiseConfig:
    """Parameters for the composite noise model R_k = R_floor + R_vibe + R_load."""

    # Noise floor (constant baseline variance per axis)
    r_floor: float = 1e-4        # m² (or rad²/s² for gyro)

    # RPM-vibration coefficient α
    alpha: float = 2e-3

    # G-loading coefficient β
    beta: float = 1e-3

    # Maximum motor RPM (for normalising RPM² sum)
    motor_max_rpm: float = 12000.0

    def __post_init__(self) -> None:
        for name, val in [("r_floor", self.r_floor),
                          ("alpha", self.alpha),
                          ("beta", self.beta),
                          ("motor_max_rpm", self.motor_max_rpm)]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.motor_max_rpm == 0.0:
            raise ValueError("motor_max_rpm must be > 0")

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def rpm_sq_norm(self, rpm_sq_sum: float) -> float:
        """Normalised motor kinetic energy: Σᵢ Ωᵢ² / (4 · Ω_max²) ∈ [0, 1]."""
        return float(rpm_sq_sum) / (4.0 * self.motor_max_rpm ** 2)

    def r_vibe(self, rpm_sq_sum: float) -> float:
        """Scalar vibration-noise variance contribution per axis."""
        return self.alpha * self.rpm_sq_norm(rpm_sq_sum)

    def r_load(self, specific_force_mag_mps2: float) -> float:
        """Scalar G-loading noise variance contribution per axis."""
        return self.beta * float(specific_force_mag_mps2)

    def scalar_variance(self, rpm_sq_sum: float, specific_force_mag_mps2: float) -> float:
        """
        Total scalar noise variance per axis:
            σ²(Ω, a) = r_floor + α · Σrpm² / (4 Ω_max²) + β · ‖a‖
        """
        return (
            self.r_floor
            + self.r_vibe(rpm_sq_sum)
            + self.r_load(specific_force_mag_mps2)
        )

    def matrix_3x3(self, rpm_sq_sum: float, specific_force_mag_mps2: float) -> np.ndarray:
        """
        Full 3×3 isotropic noise covariance matrix:
            R_k = σ²(Ω, a) · I₃
        """
        var = self.scalar_variance(rpm_sq_sum, specific_force_mag_mps2)
        return np.eye(3, dtype=float) * var

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def breakdown(self, rpm_sq_sum: float, specific_force_mag_mps2: float) -> dict:
        """Return a dict showing each noise component (useful for debugging / plots)."""
        floor = self.r_floor
        vibe = self.r_vibe(rpm_sq_sum)
        load = self.r_load(specific_force_mag_mps2)
        return {
            "r_floor": floor,
            "r_vibe": vibe,
            "r_load": load,
            "total": floor + vibe + load,
            "rpm_sq_norm": self.rpm_sq_norm(rpm_sq_sum),
            "g_load": specific_force_mag_mps2 / 9.81,
        }


# Default configuration matching the coefficients used in adaptive_ukf.py
DEFAULT_NOISE_CONFIG = HeteroscedasticNoiseConfig(
    r_floor=1e-4,
    alpha=2e-3,
    beta=1e-3,
    motor_max_rpm=12000.0,
)


def compute_R_k(
    rpm_sq_sum: float,
    specific_force_mag_mps2: float,
    cfg: HeteroscedasticNoiseConfig | None = None,
) -> np.ndarray:
    """
    Compute the instantaneous 3×3 noise covariance R_k.

    Parameters
    ----------
    rpm_sq_sum : float
        Sum of the squared RPMs of all four motors: Σᵢ Ωᵢ²  (RPM²)
    specific_force_mag_mps2 : float
        Magnitude of the IMU specific force vector ‖a_k‖  (m/s²)
    cfg : HeteroscedasticNoiseConfig, optional
        Model parameters. Uses DEFAULT_NOISE_CONFIG if not provided.

    Returns
    -------
    np.ndarray
        3×3 diagonal covariance matrix R_k.
    """
    cfg = cfg or DEFAULT_NOISE_CONFIG
    return cfg.matrix_3x3(rpm_sq_sum, specific_force_mag_mps2)
