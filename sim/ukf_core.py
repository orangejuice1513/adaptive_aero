from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence
import math

import numpy as np


Array = np.ndarray


def _as_vec(x: Array | Sequence[float], n: int, name: str) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _normalize_quat_xyzw(q: Array) -> Array:
    q = _as_vec(q, 4, "q")
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    qn = q / n
    if qn[3] < 0.0:
        qn = -qn
    return qn


def quat_mul_xyzw(q1: Array, q2: Array) -> Array:
    x1, y1, z1, w1 = _normalize_quat_xyzw(q1)
    x2, y2, z2, w2 = _normalize_quat_xyzw(q2)
    return _normalize_quat_xyzw(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
    )


def quat_from_rotvec_xyzw(rv: Array) -> Array:
    rv = _as_vec(rv, 3, "rv")
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        # small-angle approximation
        half = 0.5 * rv
        return _normalize_quat_xyzw(np.array([half[0], half[1], half[2], 1.0], dtype=float))
    axis = rv / theta
    half_theta = 0.5 * theta
    s = math.sin(half_theta)
    return _normalize_quat_xyzw(
        np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half_theta)], dtype=float)
    )


def quat_to_rotmat_xyzw(q: Array) -> Array:
    x, y, z, w = _normalize_quat_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def ensure_psd(P: Array, jitter: float = 1e-9, max_tries: int = 8) -> Array:
    P = 0.5 * (P + P.T)
    eye = np.eye(P.shape[0], dtype=float)
    eps = jitter
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(P)
            return P
        except np.linalg.LinAlgError:
            P = P + eps * eye
            eps *= 10.0
    raise np.linalg.LinAlgError("Failed to make covariance PSD")


@dataclass(slots=True)
class UKFConfig:
    alpha: float = 1e-1
    beta: float = 2.0
    kappa: float = 0.0
    jitter: float = 1e-9

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0")


@dataclass(slots=True)
class UKFState:
    x: Array
    P: Array
    t: Optional[float] = None


class StateSpaceModel(ABC):
    """
    Minimal interface for the unscented filter backbone.

    The filter itself works in an 'error-state' local tangent space of dimension n.
    For ordinary Euclidean state, retract() = x + dx and difference() = x1 - x2.
    For the drone model, quaternion orientation is handled by retract()/difference().
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def retract(self, x: Array, dx: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def difference(self, x: Array, x_ref: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def process_fn(self, x: Array, u: Array, dt: float) -> Array:
        raise NotImplementedError

    @abstractmethod
    def measurement_fn(self, x: Array) -> Array:
        raise NotImplementedError

    def post_process_state(self, x: Array) -> Array:
        return x

    def innovation(self, z: Array, z_pred: Array) -> Array:
        return np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)


class UnscentedKalmanFilter:
    def __init__(self, model: StateSpaceModel, cfg: Optional[UKFConfig] = None) -> None:
        self.model = model
        self.cfg = cfg or UKFConfig()
        self.n = self.model.state_dim
        self.m = self.model.meas_dim

        self._lambda = self.cfg.alpha * self.cfg.alpha * (self.n + self.cfg.kappa) - self.n
        self._gamma = math.sqrt(self.n + self._lambda)

        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self._lambda)), dtype=float)
        self.Wc = self.Wm.copy()
        self.Wm[0] = self._lambda / (self.n + self._lambda)
        self.Wc[0] = self.Wm[0] + (1.0 - self.cfg.alpha * self.cfg.alpha + self.cfg.beta)

        self.state: Optional[UKFState] = None

    def initialize(self, x0: Array, P0: Array, t0: Optional[float] = None) -> None:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (self.n,):
            raise ValueError(f"x0 must have shape ({self.n},), got {x0.shape}")
        P0 = np.asarray(P0, dtype=float)
        if P0.shape != (self.n, self.n):
            raise ValueError(f"P0 must have shape ({self.n}, {self.n}), got {P0.shape}")
        self.state = UKFState(x=self.model.post_process_state(x0.copy()), P=ensure_psd(P0, self.cfg.jitter), t=t0)

    def require_initialized(self) -> UKFState:
        if self.state is None:
            raise RuntimeError("UKF must be initialized before use")
        return self.state

    def sigma_points(self, x: Array, P: Array) -> list[Array]:
        P = ensure_psd(P, self.cfg.jitter)
        S = np.linalg.cholesky(P)
        sigmas = [x.copy()]
        for i in range(self.n):
            d = self._gamma * S[:, i]
            sigmas.append(self.model.post_process_state(self.model.retract(x, d)))
            sigmas.append(self.model.post_process_state(self.model.retract(x, -d)))
        return sigmas

    def weighted_state_mean(self, sigmas: Sequence[Array]) -> Array:
        # Iterative intrinsic mean in the model's local coordinates.
        x_mean = np.asarray(sigmas[0], dtype=float).copy()
        for _ in range(10):
            dx_mean = np.zeros(self.n, dtype=float)
            for w, x_i in zip(self.Wm, sigmas):
                dx_mean += w * self.model.difference(np.asarray(x_i, dtype=float), x_mean)
            if np.linalg.norm(dx_mean) < 1e-10:
                break
            x_mean = self.model.post_process_state(self.model.retract(x_mean, dx_mean))
        return x_mean

    def weighted_covariance(self, sigmas: Sequence[Array], x_mean: Array, Q: Optional[Array] = None) -> Array:
        P = np.zeros((self.n, self.n), dtype=float)
        for w, x_i in zip(self.Wc, sigmas):
            dx = self.model.difference(np.asarray(x_i, dtype=float), x_mean)
            P += w * np.outer(dx, dx)
        if Q is not None:
            P += np.asarray(Q, dtype=float)
        return ensure_psd(P, self.cfg.jitter)

    def predict(self, u: Array, dt: float, Q: Array, t: Optional[float] = None) -> UKFState:
        st = self.require_initialized()
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (self.n, self.n):
            raise ValueError(f"Q must have shape ({self.n}, {self.n}), got {Q.shape}")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        sigmas = self.sigma_points(st.x, st.P)
        pred_sigmas = [self.model.post_process_state(self.model.process_fn(x_i, np.asarray(u, dtype=float), dt)) for x_i in sigmas]
        x_pred = self.weighted_state_mean(pred_sigmas)
        P_pred = self.weighted_covariance(pred_sigmas, x_pred, Q=Q)

        self.state = UKFState(x=x_pred, P=P_pred, t=t)
        return self.state

    def update(self, z: Array, R: Array) -> UKFState:
        st = self.require_initialized()
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (self.m,):
            raise ValueError(f"z must have shape ({self.m},), got {z.shape}")
        R = np.asarray(R, dtype=float)
        if R.shape != (self.m, self.m):
            raise ValueError(f"R must have shape ({self.m}, {self.m}), got {R.shape}")

        sigmas = self.sigma_points(st.x, st.P)
        Zsig = [np.asarray(self.model.measurement_fn(x_i), dtype=float).reshape(self.m) for x_i in sigmas]
        z_pred = np.zeros(self.m, dtype=float)
        for w, z_i in zip(self.Wm, Zsig):
            z_pred += w * z_i

        S = np.zeros((self.m, self.m), dtype=float)
        Pxz = np.zeros((self.n, self.m), dtype=float)
        for w, x_i, z_i in zip(self.Wc, sigmas, Zsig):
            dz = self.model.innovation(z_i, z_pred)
            dx = self.model.difference(x_i, st.x)
            S += w * np.outer(dz, dz)
            Pxz += w * np.outer(dx, dz)
        S = ensure_psd(S + R, self.cfg.jitter)

        K = Pxz @ np.linalg.inv(S)
        innov = self.model.innovation(z, z_pred)
        dx_update = K @ innov

        x_new = self.model.post_process_state(self.model.retract(st.x, dx_update))
        P_new = ensure_psd(st.P - K @ S @ K.T, self.cfg.jitter)

        self.state = UKFState(x=x_new, P=P_new, t=st.t)
        return self.state


@dataclass(slots=True)
class DroneUKFConfig:
    gravity_mps2: float = 9.81
    accel_in_world_frame: bool = False


class DroneStateSpaceModel(StateSpaceModel):
    """
    10-state quadrotor model with state x = [p(3), v(3), q_xyzw(4)].

    Process input u is 6D:
        [ax, ay, az, wx, wy, wz]

    accel is interpreted as *specific force* in body frame by default, matching
    the synthetic IMU produced by sim.sensors. Gravity is added in world frame.

    Measurement model is 3D position only, matching the current VIO simulator.
    """

    def __init__(self, cfg: Optional[DroneUKFConfig] = None) -> None:
        self.cfg = cfg or DroneUKFConfig()

    @property
    def state_dim(self) -> int:
        return 9  # 3 position + 3 velocity + 3 local attitude error

    @property
    def meas_dim(self) -> int:
        return 3

    @property
    def full_state_dim(self) -> int:
        return 10

    def retract(self, x: Array, dx: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.full_state_dim)
        dx = _as_vec(dx, self.state_dim, "dx")
        out = x.copy()
        out[0:3] += dx[0:3]
        out[3:6] += dx[3:6]
        dq = quat_from_rotvec_xyzw(dx[6:9])
        out[6:10] = quat_mul_xyzw(out[6:10], dq)
        return self.post_process_state(out)

    def difference(self, x: Array, x_ref: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.full_state_dim)
        x_ref = np.asarray(x_ref, dtype=float).reshape(self.full_state_dim)
        dp = x[0:3] - x_ref[0:3]
        dv = x[3:6] - x_ref[3:6]

        q = _normalize_quat_xyzw(x[6:10])
        q_ref = _normalize_quat_xyzw(x_ref[6:10])
        # q_err = q_ref^{-1} * q
        q_ref_inv = np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]], dtype=float)
        q_err = quat_mul_xyzw(q_ref_inv, q)
        xyz = q_err[:3]
        w = float(np.clip(q_err[3], -1.0, 1.0))
        norm_xyz = np.linalg.norm(xyz)
        if norm_xyz < 1e-12:
            dtheta = 2.0 * xyz
        else:
            angle = 2.0 * math.atan2(norm_xyz, w)
            dtheta = angle * xyz / norm_xyz
        return np.concatenate([dp, dv, dtheta])

    def process_fn(self, x: Array, u: Array, dt: float) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.full_state_dim)
        u = np.asarray(u, dtype=float).reshape(6)
        p = x[0:3]
        v = x[3:6]
        q = _normalize_quat_xyzw(x[6:10])
        accel = u[0:3]
        omega = u[3:6]

        R_wb = quat_to_rotmat_xyzw(q)
        g_w = np.array([0.0, 0.0, -self.cfg.gravity_mps2], dtype=float)
        if self.cfg.accel_in_world_frame:
            a_w = accel + g_w
        else:
            a_w = R_wb @ accel + g_w

        p_new = p + v * dt + 0.5 * a_w * dt * dt
        v_new = v + a_w * dt
        dq = quat_from_rotvec_xyzw(omega * dt)
        q_new = quat_mul_xyzw(q, dq)
        return self.post_process_state(np.concatenate([p_new, v_new, q_new]))

    def measurement_fn(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.full_state_dim)
        return x[0:3].copy()

    def post_process_state(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.full_state_dim).copy()
        x[6:10] = _normalize_quat_xyzw(x[6:10])
        return x


def make_process_noise(
    pos_var: float,
    vel_var: float,
    att_var: float,
) -> Array:
    if pos_var < 0.0 or vel_var < 0.0 or att_var < 0.0:
        raise ValueError("Noise variances must be >= 0")
    return np.diag([pos_var] * 3 + [vel_var] * 3 + [att_var] * 3).astype(float)


def make_position_measurement_noise(var_xyz: Array | Sequence[float] | float) -> Array:
    if np.isscalar(var_xyz):
        v = float(var_xyz)
        if v < 0.0:
            raise ValueError("variance must be >= 0")
        return np.diag([v, v, v]).astype(float)
    v3 = _as_vec(np.asarray(var_xyz, dtype=float), 3, "var_xyz")
    if np.any(v3 < 0.0):
        raise ValueError("variances must be >= 0")
    return np.diag(v3).astype(float)
