"""state.py: Quadrotor state representation.

State vector x ∈ ℝ¹⁰:
    x = [p(3), v(3), q(4)]

where
    p : position in world frame (m)
    v : velocity in world frame (m/s)
    q : unit quaternion representing rotation from body to world frame,
        stored as [qx, qy, qz, qw]  (xyzw / Hamilton convention)

Attitude is represented with a unit quaternion to avoid gimbal lock.
The covariance matrix P lives in the 9-dimensional error-state (tangent) space:
    [δp(3), δv(3), δθ(3)]
where δθ is the local rotation-vector error.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DroneState:
    """Full quadrotor state at a single timestep."""

    pos_w: np.ndarray        # (3,) position in world frame, m
    vel_w: np.ndarray        # (3,) velocity in world frame, m/s
    quat_xyzw: np.ndarray    # (4,) unit quaternion [qx, qy, qz, qw]
    t: float = 0.0           # timestamp, s

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vector(cls, x: np.ndarray, t: float = 0.0) -> "DroneState":
        """Unpack a 10-D state vector [p, v, q_xyzw] into a DroneState."""
        x = np.asarray(x, dtype=float).reshape(10)
        return cls(
            pos_w=x[0:3].copy(),
            vel_w=x[3:6].copy(),
            quat_xyzw=x[6:10].copy(),
            t=t,
        )

    def to_vector(self) -> np.ndarray:
        """Pack into a 10-D state vector [p, v, q_xyzw]."""
        return np.concatenate([self.pos_w, self.vel_w, self.quat_xyzw])

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def rotation_matrix(self) -> np.ndarray:
        """3×3 rotation matrix R_wb (body-to-world) from the stored quaternion."""
        x, y, z, w = self.quat_xyzw
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),  2.0 * (xz + wy)],
            [2.0 * (xy + wz),  1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),  2.0 * (yz + wx),  1.0 - 2.0 * (xx + yy)],
        ], dtype=float)

    def speed_mps(self) -> float:
        """Scalar speed in m/s."""
        return float(np.linalg.norm(self.vel_w))

    def __repr__(self) -> str:
        p = np.round(self.pos_w, 3)
        v = np.round(self.vel_w, 3)
        q = np.round(self.quat_xyzw, 4)
        return f"DroneState(t={self.t:.3f}s, pos={p}, vel={v}, quat={q})"


@dataclass
class FilteredState:
    """Filter output: estimated state + covariance at a single timestep."""

    mean: DroneState         # point estimate (10-D)
    cov: np.ndarray          # (9, 9) error-state covariance matrix

    @property
    def pos_cov(self) -> np.ndarray:
        """3×3 position covariance sub-block."""
        return self.cov[:3, :3]

    @property
    def vel_cov(self) -> np.ndarray:
        """3×3 velocity covariance sub-block."""
        return self.cov[3:6, 3:6]

    @property
    def att_cov(self) -> np.ndarray:
        """3×3 attitude (error-angle) covariance sub-block."""
        return self.cov[6:9, 6:9]

    def position_3sigma_m(self) -> float:
        """3-sigma position uncertainty radius (metres)."""
        return 3.0 * float(np.sqrt(np.trace(self.pos_cov)))
