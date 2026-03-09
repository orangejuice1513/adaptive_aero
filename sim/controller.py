from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math

import numpy as np

from sim.env import DroneConfig, DroneEnv, TruthState


Array3 = np.ndarray
Array4 = np.ndarray
Matrix3 = np.ndarray


def _as_vec3(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = _as_vec3(v, "v")
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _vee(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"M must have shape (3, 3), got {M.shape}")
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def _wrap_angle(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


@dataclass(slots=True)
class ReferenceState:
    """
    Time-local tracking target.

    Fields use world-frame quantities.
    `acc_w` is desired translational acceleration excluding gravity.
    """

    pos_w: Array3
    vel_w: Array3 = field(default_factory=lambda: np.zeros(3, dtype=float))
    acc_w: Array3 = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        self.pos_w = _as_vec3(self.pos_w, "pos_w")
        self.vel_w = _as_vec3(self.vel_w, "vel_w")
        self.acc_w = _as_vec3(self.acc_w, "acc_w")
        self.yaw = float(self.yaw)
        self.yaw_rate = float(self.yaw_rate)
        if not np.isfinite(self.yaw) or not np.isfinite(self.yaw_rate):
            raise ValueError("yaw and yaw_rate must be finite")


@dataclass(slots=True)
class ControllerConfig:
    """
    Position + geometric-attitude controller gains and limits.

    More aggressive defaults for high-dynamic trajectory tracking.
    """

    kp_pos: Array3 = field(default_factory=lambda: np.array([16.0, 16.0, 21.0], dtype=float))
    kd_vel: Array3 = field(default_factory=lambda: np.array([10, 12, 13], dtype=float))
    ki_pos: Array3 = field(default_factory=lambda: np.array([0.0, 0.0, 0.8], dtype=float))

    kp_att: Array3 = field(default_factory=lambda: np.array([4, 4, 1.5], dtype=float))
    kd_rate: Array3 = field(default_factory=lambda: np.array([0.3, 0.3, 0.15], dtype=float))

    max_acc_xy: float = 40.0
    max_acc_up: float = 22.0
    max_acc_down: float = 18.0
    max_tilt_rad: float = math.radians(75.0)
    max_yaw_rate: float = math.radians(360.0)

    max_torque_nm: Array3 = field(default_factory=lambda: np.array([2, 2, 0.8], dtype=float))
    min_total_thrust_n: float = 0.0
    hover_thrust_bias_n: float = 0.0

    pos_integrator_limit: Array3 = field(
        default_factory=lambda: np.array([1.5, 1.5, 2.0], dtype=float)
    )
    use_coriolis_comp: bool = True

    def __post_init__(self) -> None:
        self.kp_pos = _as_vec3(self.kp_pos, "kp_pos")
        self.kd_vel = _as_vec3(self.kd_vel, "kd_vel")
        self.ki_pos = _as_vec3(self.ki_pos, "ki_pos")
        self.kp_att = _as_vec3(self.kp_att, "kp_att")
        self.kd_rate = _as_vec3(self.kd_rate, "kd_rate")
        self.max_torque_nm = _as_vec3(self.max_torque_nm, "max_torque_nm")
        self.pos_integrator_limit = _as_vec3(
            self.pos_integrator_limit, "pos_integrator_limit"
        )

        if np.any(self.kp_pos < 0.0) or np.any(self.kd_vel < 0.0) or np.any(self.ki_pos < 0.0):
            raise ValueError("position gains must be >= 0")
        if np.any(self.kp_att < 0.0) or np.any(self.kd_rate < 0.0):
            raise ValueError("attitude gains must be >= 0")
        if self.max_acc_xy <= 0.0 or self.max_acc_up <= 0.0 or self.max_acc_down <= 0.0:
            raise ValueError("acceleration limits must be > 0")
        if self.max_tilt_rad <= 0.0 or self.max_tilt_rad >= math.radians(89.0):
            raise ValueError("max_tilt_rad must be in (0, 89deg)")
        if self.max_yaw_rate <= 0.0:
            raise ValueError("max_yaw_rate must be > 0")
        if np.any(self.max_torque_nm <= 0.0):
            raise ValueError("max_torque_nm must be > 0")
        if self.min_total_thrust_n < 0.0 or self.hover_thrust_bias_n < 0.0:
            raise ValueError("thrust bias / lower bound must be >= 0")
        if np.any(self.pos_integrator_limit < 0.0):
            raise ValueError("pos_integrator_limit must be >= 0")


@dataclass(slots=True)
class ControlOutput:
    rpm_cmd: Array4
    thrust_cmds_n: Array4
    total_thrust_cmd_n: float
    body_torque_cmd_nm: Array3
    desired_rot_wb: Matrix3
    desired_acc_w: Array3
    desired_force_w: Array3
    pos_error_w: Array3
    vel_error_w: Array3
    att_error_b: Array3
    rate_error_b: Array3
    debug: Dict[str, Any] = field(default_factory=dict)


class GeometricController:
    """
    Truth-based outer-loop position controller + geometric attitude controller.

    Public API:
        controller = GeometricController.from_env(env)
        ref = ReferenceState(pos_w=np.array([0, 0, 1.5]), yaw=0.0)
        rpm_cmd = controller.compute_rpm(truth, ref)

    Notes:
    - This controller outputs motor RPM commands for the exact rotor model used
      in `sim.env`.
    - It is intentionally simple and robust rather than hyper-optimized.
    - Desired angular-rate feedforward is kept minimal. For your current phase,
      that is a feature, not a bug.
    """

    def __init__(
        self,
        drone_cfg: DroneConfig,
        mass_kg: float,
        inertia_diag: np.ndarray,
        gravity_mps2: float = 9.81,
        cfg: Optional[ControllerConfig] = None,
    ) -> None:
        if mass_kg <= 0.0:
            raise ValueError("mass_kg must be > 0")
        self.drone_cfg = drone_cfg
        self.mass_kg = float(mass_kg)
        self.inertia_diag = _as_vec3(inertia_diag, "inertia_diag")
        if np.any(self.inertia_diag <= 0.0):
            raise ValueError("inertia_diag entries must be > 0")
        if gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be > 0")
        self.gravity_mps2 = float(gravity_mps2)
        self.cfg = cfg or ControllerConfig()

        self.J = np.diag(self.inertia_diag)
        self.g_w = np.array([0.0, 0.0, -self.gravity_mps2], dtype=float)
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=float)

        motor = self.drone_cfg.motor
        self.kf = float(motor.thrust_coeff_n_per_rpm2)
        self.km = float(motor.yaw_moment_coeff_nm_per_rpm2)
        self.spin_dirs = np.asarray(motor.spin_dirs, dtype=float).reshape(4)
        self.rotor_offsets_b = np.asarray(motor.rotor_offsets_b, dtype=float).reshape(4, 3)
        self.min_rpm = float(motor.min_rpm)
        self.max_rpm = float(motor.max_rpm)
        self.min_rotor_thrust_n = self.kf * self.min_rpm * self.min_rpm
        self.max_rotor_thrust_n = self.kf * self.max_rpm * self.max_rpm
        self.max_total_thrust_n = 4.0 * self.max_rotor_thrust_n
        self.gamma = self.km / self.kf if self.kf > 0.0 else 0.0

        self._allocation = self._build_allocation_matrix()
        self._allocation_pinv = np.linalg.pinv(self._allocation)
        self._pos_int = np.zeros(3, dtype=float)
        self._last_output: Optional[ControlOutput] = None

    @classmethod
    def from_env(
        cls,
        env: DroneEnv,
        cfg: Optional[ControllerConfig] = None,
    ) -> "GeometricController":
        return cls(
            drone_cfg=env.drone_cfg,
            mass_kg=env.get_mass_kg(),
            inertia_diag=env.get_inertia_diag(),
            gravity_mps2=env.sim_cfg.gravity_mps2,
            cfg=cfg,
        )

    def reset(self) -> None:
        self._pos_int[:] = 0.0
        self._last_output = None

    def get_last_output(self) -> Optional[ControlOutput]:
        return self._last_output

    def compute_rpm(self, truth: TruthState, ref: ReferenceState, dt: Optional[float] = None) -> np.ndarray:
        return self.compute(truth=truth, ref=ref, dt=dt).rpm_cmd.copy()

    def compute(
        self,
        truth: TruthState,
        ref: ReferenceState,
        dt: Optional[float] = None,
    ) -> ControlOutput:
        dt_eff = float(dt) if dt is not None else None
        if dt_eff is not None and dt_eff <= 0.0:
            raise ValueError("dt must be > 0 when provided")

        # -------------------------------------------------------------
        # Outer loop: translational control in world frame
        # -------------------------------------------------------------
        pos_error_w = ref.pos_w - truth.pos_w
        vel_error_w = ref.vel_w - truth.vel_w

        if dt_eff is not None:
            self._pos_int += pos_error_w * dt_eff
            lim = self.cfg.pos_integrator_limit
            self._pos_int = np.clip(self._pos_int, -lim, lim)

        a_fb_w = (
            self.cfg.kp_pos * pos_error_w
            + self.cfg.kd_vel * vel_error_w
            + self.cfg.ki_pos * self._pos_int
        )
        desired_acc_w = ref.acc_w + a_fb_w
        desired_acc_w = self._saturate_translational_acc(desired_acc_w)

        # Desired total force in world coordinates.
        desired_force_w = self.mass_kg * (desired_acc_w - self.g_w)
        desired_force_w[2] += self.cfg.hover_thrust_bias_n

        # Guard against pathological commands that would demand near-zero or
        # downward total thrust in a thrust-only vehicle.
        min_fz = max(self.cfg.min_total_thrust_n, 1e-6)
        if desired_force_w[2] < min_fz:
            desired_force_w[2] = min_fz
        desired_force_w = self._limit_tilt_by_force(desired_force_w)

        # -------------------------------------------------------------
        # Desired attitude from force vector + yaw command
        # -------------------------------------------------------------
        desired_rot_wb = self._desired_rotation_from_force_and_yaw(
            force_w=desired_force_w,
            yaw=ref.yaw,
        )

        # -------------------------------------------------------------
        # Inner loop: geometric attitude control
        # -------------------------------------------------------------
        R = truth.rot_wb
        R_des = desired_rot_wb
        omega_b = truth.omega_b
        omega_des_b = np.array([0.0, 0.0, np.clip(ref.yaw_rate, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)], dtype=float)

        att_err_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_error_b = _vee(att_err_matrix)
        rate_error_b = omega_b - omega_des_b

        body_torque_cmd_nm = -self.cfg.kp_att * att_error_b - self.cfg.kd_rate * rate_error_b
        if self.cfg.use_coriolis_comp:
            body_torque_cmd_nm += np.cross(omega_b, self.J @ omega_b)
        body_torque_cmd_nm = np.clip(
            body_torque_cmd_nm,
            -self.cfg.max_torque_nm,
            self.cfg.max_torque_nm,
        )

        # Total thrust command along the CURRENT body z-axis.
        b3_current_w = truth.rot_wb[:, 2]
        total_thrust_cmd_n = float(np.dot(desired_force_w, b3_current_w))
        total_thrust_cmd_n = float(np.clip(
            total_thrust_cmd_n,
            max(self.cfg.min_total_thrust_n, 0.0),
            self.max_total_thrust_n,
        ))

        thrust_cmds_n = self._mix_wrench_to_rotor_thrusts(
            total_thrust_n=total_thrust_cmd_n,
            body_torque_nm=body_torque_cmd_nm,
        )
        rpm_cmd = np.sqrt(np.clip(thrust_cmds_n / self.kf, 0.0, None))
        rpm_cmd = np.clip(rpm_cmd, self.min_rpm, self.max_rpm)

        output = ControlOutput(
            rpm_cmd=rpm_cmd,
            thrust_cmds_n=thrust_cmds_n,
            total_thrust_cmd_n=total_thrust_cmd_n,
            body_torque_cmd_nm=body_torque_cmd_nm,
            desired_rot_wb=desired_rot_wb,
            desired_acc_w=desired_acc_w,
            desired_force_w=desired_force_w,
            pos_error_w=pos_error_w,
            vel_error_w=vel_error_w,
            att_error_b=att_error_b,
            rate_error_b=rate_error_b,
            debug={
                "ref_yaw": float(ref.yaw),
                "ref_yaw_rate": float(ref.yaw_rate),
                "desired_roll": float(math.atan2(desired_rot_wb[2, 1], desired_rot_wb[2, 2])),
                "desired_pitch": float(math.asin(-np.clip(desired_rot_wb[2, 0], -1.0, 1.0))),
                "desired_yaw": float(math.atan2(desired_rot_wb[1, 0], desired_rot_wb[0, 0])),
                "pos_int_x": float(self._pos_int[0]),
                "pos_int_y": float(self._pos_int[1]),
                "pos_int_z": float(self._pos_int[2]),
            },
        )
        self._last_output = output
        return output

    def hover_reference(self, z: float, yaw: float = 0.0) -> ReferenceState:
        return ReferenceState(
            pos_w=np.array([0.0, 0.0, z], dtype=float),
            vel_w=np.zeros(3, dtype=float),
            acc_w=np.zeros(3, dtype=float),
            yaw=yaw,
            yaw_rate=0.0,
        )

    def position_hold_reference(self, pos_w: Any, yaw: float = 0.0) -> ReferenceState:
        return ReferenceState(pos_w=_as_vec3(pos_w, "pos_w"), yaw=yaw)

    def mixer_matrix(self) -> np.ndarray:
        return self._allocation.copy()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_allocation_matrix(self) -> np.ndarray:
        """
        Rotor thrusts -> body wrench.

        Wrench is ordered as [Fz, tau_x, tau_y, tau_z].
        For rotor i at offset r_i = [x_i, y_i, z_i], thrust is +z_body.

        tau_from_thrust = r x [0, 0, f] = [y f, -x f, 0]
        tau_z = spin_dir * gamma * f
        """
        A = np.zeros((4, 4), dtype=float)
        for i in range(4):
            x_i = self.rotor_offsets_b[i, 0]
            y_i = self.rotor_offsets_b[i, 1]
            A[0, i] = 1.0
            A[1, i] = y_i
            A[2, i] = -x_i
            A[3, i] = self.spin_dirs[i] * self.gamma
        return A

    def _saturate_translational_acc(self, acc_w: np.ndarray) -> np.ndarray:
        acc_w = _as_vec3(acc_w, "acc_w")
        acc_xy = _clip_norm(acc_w[:2], self.cfg.max_acc_xy)
        acc_z = float(np.clip(acc_w[2], -self.cfg.max_acc_down, self.cfg.max_acc_up))
        return np.array([acc_xy[0], acc_xy[1], acc_z], dtype=float)

    def _limit_tilt_by_force(self, force_w: np.ndarray) -> np.ndarray:
        force_w = _as_vec3(force_w, "force_w")
        fz = max(force_w[2], 1e-6)
        lateral = force_w[:2]
        lateral_norm = np.linalg.norm(lateral)
        lateral_max = math.tan(self.cfg.max_tilt_rad) * fz
        if lateral_norm > lateral_max and lateral_norm > 1e-12:
            lateral = lateral * (lateral_max / lateral_norm)
        return np.array([lateral[0], lateral[1], fz], dtype=float)

    def _desired_rotation_from_force_and_yaw(self, force_w: np.ndarray, yaw: float) -> np.ndarray:
        b3_des = _normalize(force_w)
        c1 = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)

        # Construct a right-handed frame [b1 b2 b3].
        b2_des = np.cross(b3_des, c1)
        if np.linalg.norm(b2_des) < 1e-8:
            c1 = np.array([math.cos(yaw + 1e-3), math.sin(yaw + 1e-3), 0.0], dtype=float)
            b2_des = np.cross(b3_des, c1)
        b2_des = _normalize(b2_des)
        b1_des = _normalize(np.cross(b2_des, b3_des))

        R_des = np.column_stack((b1_des, b2_des, b3_des))
        return R_des

    def _mix_wrench_to_rotor_thrusts(
        self,
        total_thrust_n: float,
        body_torque_nm: np.ndarray,
    ) -> np.ndarray:
        body_torque_nm = _as_vec3(body_torque_nm, "body_torque_nm")
        wrench = np.array(
            [
                float(total_thrust_n),
                float(body_torque_nm[0]),
                float(body_torque_nm[1]),
                float(body_torque_nm[2]),
            ],
            dtype=float,
        )

        thrusts = self._allocation_pinv @ wrench
        thrusts = np.asarray(thrusts, dtype=float).reshape(4)

        # Clip to feasible per-rotor thrust range.
        thrusts = np.clip(thrusts, self.min_rotor_thrust_n, self.max_rotor_thrust_n)

        # Try to preserve the requested common-mode thrust after clipping by
        # re-distributing any total-thrust residual uniformly over available headroom.
        residual = float(total_thrust_n - np.sum(thrusts))
        if abs(residual) > 1e-9:
            if residual > 0.0:
                headroom = self.max_rotor_thrust_n - thrusts
                denom = float(np.sum(headroom > 1e-12))
                if denom > 0.0:
                    add_each = residual / denom
                    thrusts = np.minimum(thrusts + add_each * (headroom > 1e-12), self.max_rotor_thrust_n)
            else:
                room_down = thrusts - self.min_rotor_thrust_n
                denom = float(np.sum(room_down > 1e-12))
                if denom > 0.0:
                    sub_each = (-residual) / denom
                    thrusts = np.maximum(thrusts - sub_each * (room_down > 1e-12), self.min_rotor_thrust_n)

        return thrusts
