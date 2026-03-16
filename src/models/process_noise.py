"""process_noise.py: Process noise model for the quadrotor UKF.

The process noise matrix Q represents unmodelled dynamics and disturbances
in the quadrotor kinematics.  For the standard (non-adaptive) UKF it is
constant; for the Adaptive UKF (see adaptive_ukf.py) Q_k is recomputed at
every step using the heteroscedastic model.

State error-space dimension is 9:  [δp(3), δv(3), δθ(3)]

    Q = diag(σ²_p · I₃,  σ²_v · I₃,  σ²_att · I₃)

Typical values
--------------
Hover (low dynamic):
    σ²_p   ≈ 1e-5  m²
    σ²_v   ≈ 5e-3  (m/s)²
    σ²_att ≈ 1e-5  rad²

Racing (high dynamic):
    σ²_p   ≈ 5e-5  m²
    σ²_v   ≈ 5e-2  (m/s)²
    σ²_att ≈ 5e-5  rad²

For the Adaptive UKF these are lower bounds; the actual Q_k is inflated
by the RPM-vibration and G-loading terms (see measurement_noise.py for
the physical model and adaptive_ukf.py for the implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProcessNoiseConfig:
    """Parameters for the static process noise model Q."""

    pos_var: float = 1e-5    # position variance, m²
    vel_var: float = 5e-3    # velocity variance, (m/s)²
    att_var: float = 1e-5    # attitude error variance, rad²

    def __post_init__(self) -> None:
        for name, val in [("pos_var", self.pos_var),
                          ("vel_var", self.vel_var),
                          ("att_var", self.att_var)]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    def matrix(self) -> np.ndarray:
        """Return the 9×9 diagonal process noise matrix Q."""
        return np.diag(
            [self.pos_var] * 3 +
            [self.vel_var] * 3 +
            [self.att_var] * 3
        ).astype(float)


# Preset configurations matching hover_ukf.py and race_ukf.py
HOVER_PROCESS_NOISE = ProcessNoiseConfig(
    pos_var=1e-5,
    vel_var=5e-3,
    att_var=1e-5,
)

RACE_PROCESS_NOISE = ProcessNoiseConfig(
    pos_var=5e-5,
    vel_var=5e-2,
    att_var=5e-5,
)


def make_Q(pos_var: float, vel_var: float, att_var: float) -> np.ndarray:
    """Convenience function: build a 9×9 diagonal Q matrix."""
    return ProcessNoiseConfig(pos_var, vel_var, att_var).matrix()
