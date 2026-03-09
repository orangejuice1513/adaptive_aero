from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math

import numpy as np
import pybullet as p
import pybullet_data


ArrayLike3 = np.ndarray
ArrayLike4 = np.ndarray


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


def _normalize_quat_xyzw(q_xyzw: Any) -> np.ndarray:
    q = _as_vec4(q_xyzw, "quat_xyzw")
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("quat_xyzw has near-zero norm")
    return q / n


def _quat_xyzw_to_rot_wb(q_xyzw: np.ndarray) -> np.ndarray:
    # PyBullet returns quaternions as [x, y, z, w].
    m = np.asarray(p.getMatrixFromQuaternion(q_xyzw.tolist()), dtype=float)
    return m.reshape(3, 3)


def _copy_truth(truth: "TruthState") -> "TruthState":
    return TruthState(
        t=truth.t,
        step_idx=truth.step_idx,
        pos_w=truth.pos_w.copy(),
        vel_w=truth.vel_w.copy(),
        acc_w=truth.acc_w.copy(),
        quat_wb_xyzw=truth.quat_wb_xyzw.copy(),
        rot_wb=truth.rot_wb.copy(),
        rot_bw=truth.rot_bw.copy(),
        euler_rpy_wb=truth.euler_rpy_wb.copy(),
        omega_w=truth.omega_w.copy(),
        omega_b=truth.omega_b.copy(),
        specific_force_b=truth.specific_force_b.copy(),
        g_load=truth.g_load,
        motor_rpm_cmd=truth.motor_rpm_cmd.copy(),
        motor_rpm_actual=truth.motor_rpm_actual.copy(),
        motor_thrusts_n=truth.motor_thrusts_n.copy(),
        motor_yaw_torques_nm=truth.motor_yaw_torques_nm.copy(),
        mass_kg=truth.mass_kg,
        inertia_diag=truth.inertia_diag.copy(),
        extras=deepcopy(truth.extras),
    )


@dataclass(slots=True)
class MotorConfig:
    """
    Rotor layout and motor model.

    Conventions:
    - rotor_offsets_b are expressed in the body/base frame in meters.
    - positive thrust is along +z_body.
    - spin_dirs should be +1 / -1 for rotor reaction torque sign.
    """

    thrust_coeff_n_per_rpm2: float = 2.50e-7
    yaw_moment_coeff_nm_per_rpm2: float = 2.50e-9
    motor_time_constant_s: float = 0.03
    min_rpm: float = 0.0
    max_rpm: float = 35000.0
    rotor_offsets_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.070710678, 0.070710678, 0.0],   # front-left
                [0.070710678, -0.070710678, 0.0],  # front-right
                [-0.070710678, -0.070710678, 0.0], # rear-right
                [-0.070710678, 0.070710678, 0.0],  # rear-left
            ],
            dtype=float,
        )
    )
    spin_dirs: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    )

    def validate(self) -> None:
        if self.thrust_coeff_n_per_rpm2 <= 0.0:
            raise ValueError("thrust_coeff_n_per_rpm2 must be > 0")
        if self.yaw_moment_coeff_nm_per_rpm2 < 0.0:
            raise ValueError("yaw_moment_coeff_nm_per_rpm2 must be >= 0")
        if self.motor_time_constant_s <= 0.0:
            raise ValueError("motor_time_constant_s must be > 0")
        if self.min_rpm < 0.0 or self.max_rpm <= self.min_rpm:
            raise ValueError("RPM bounds are invalid")
        rotor_offsets_b = np.asarray(self.rotor_offsets_b, dtype=float)
        if rotor_offsets_b.shape != (4, 3):
            raise ValueError(
                f"rotor_offsets_b must have shape (4, 3), got {rotor_offsets_b.shape}"
            )
        if not np.all(np.isfinite(rotor_offsets_b)):
            raise ValueError("rotor_offsets_b must be finite")
        spin_dirs = np.asarray(self.spin_dirs, dtype=float).reshape(-1)
        if spin_dirs.shape != (4,):
            raise ValueError(f"spin_dirs must have shape (4,), got {spin_dirs.shape}")
        if not np.all(np.isfinite(spin_dirs)):
            raise ValueError("spin_dirs must be finite")
        self.rotor_offsets_b = rotor_offsets_b
        self.spin_dirs = spin_dirs


@dataclass(slots=True)
class DroneConfig:
    urdf_path: str
    start_pos_w: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 2.5], dtype=float)
    )
    start_quat_wb_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    )
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    # Quadratic drag in the body frame: F_b = -c * v_b * |v_b|
    drag_coeff_body: np.ndarray = field(
        default_factory=lambda: np.array([0.08, 0.08, 0.12], dtype=float)
    )
    # Quadratic angular drag in the body frame: tau_b = -c * w_b * |w_b|
    angular_drag_coeff_body: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-4, 2.0e-4, 3.0e-4], dtype=float)
    )
    # Optional validation against URDF dynamics.
    expected_mass_kg: Optional[float] = None
    expected_inertia_diag: Optional[np.ndarray] = None
    dynamics_tolerance: float = 1e-5
    motor: MotorConfig = field(default_factory=MotorConfig)

    def validate(self) -> None:
        if not self.urdf_path:
            raise ValueError("urdf_path must be provided")
        self.start_pos_w = _as_vec3(self.start_pos_w, "start_pos_w")
        self.start_quat_wb_xyzw = _normalize_quat_xyzw(self.start_quat_wb_xyzw)
        self.drag_coeff_body = _as_vec3(self.drag_coeff_body, "drag_coeff_body")
        self.angular_drag_coeff_body = _as_vec3(
            self.angular_drag_coeff_body, "angular_drag_coeff_body"
        )
        if np.any(self.drag_coeff_body < 0.0):
            raise ValueError("drag_coeff_body must be >= 0")
        if np.any(self.angular_drag_coeff_body < 0.0):
            raise ValueError("angular_drag_coeff_body must be >= 0")
        if self.linear_damping < 0.0 or self.angular_damping < 0.0:
            raise ValueError("damping coefficients must be >= 0")
        if self.expected_mass_kg is not None and self.expected_mass_kg <= 0.0:
            raise ValueError("expected_mass_kg must be > 0 if provided")
        if self.expected_inertia_diag is not None:
            self.expected_inertia_diag = _as_vec3(
                self.expected_inertia_diag, "expected_inertia_diag"
            )
            if np.any(self.expected_inertia_diag <= 0.0):
                raise ValueError("expected_inertia_diag entries must be > 0")
        if self.dynamics_tolerance < 0.0:
            raise ValueError("dynamics_tolerance must be >= 0")
        self.motor.validate()


@dataclass(slots=True)
class SimConfig:
    dt: float = 1.0 / 240.0
    physics_substeps: int = 1
    gravity_mps2: float = 9.81
    gui: bool = False
    real_time: bool = False
    enable_ground: bool = True
    seed: int = 0
    disable_default_joint_motors: bool = True
    solver_iterations: int = 50

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be >= 1")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be > 0")
        if self.solver_iterations < 1:
            raise ValueError("solver_iterations must be >= 1")


@dataclass(slots=True)
class WorldConfig:
    plane_urdf: str = "plane.urdf"


@dataclass(slots=True)
class MotorState:
    rpm_cmd: np.ndarray
    rpm_actual: np.ndarray
    thrusts_n: np.ndarray
    yaw_torques_nm: np.ndarray


@dataclass(slots=True)
class _RawKinematics:
    pos_w: np.ndarray
    vel_w: np.ndarray
    quat_wb_xyzw: np.ndarray
    rot_wb: np.ndarray
    rot_bw: np.ndarray
    euler_rpy_wb: np.ndarray
    omega_w: np.ndarray
    omega_b: np.ndarray


@dataclass(slots=True)
class TruthState:
    t: float
    step_idx: int

    pos_w: np.ndarray
    vel_w: np.ndarray
    acc_w: np.ndarray

    quat_wb_xyzw: np.ndarray
    rot_wb: np.ndarray
    rot_bw: np.ndarray
    euler_rpy_wb: np.ndarray

    omega_w: np.ndarray
    omega_b: np.ndarray

    specific_force_b: np.ndarray
    g_load: float

    motor_rpm_cmd: np.ndarray
    motor_rpm_actual: np.ndarray
    motor_thrusts_n: np.ndarray
    motor_yaw_torques_nm: np.ndarray

    mass_kg: float
    inertia_diag: np.ndarray

    extras: Dict[str, Any] = field(default_factory=dict)


class DroneEnv:
    """
    Final-style PyBullet environment for the AA273 adaptive UKF sim.

    Design choices:
    - Uses PyBullet quaternion convention everywhere: [x, y, z, w].
    - World frame is z-up, gravity along -z.
    - Forces and torques are applied in WORLD_FRAME after explicit rotation.
      This avoids depending on LINK_FRAME torque behavior.
    - `step()` is an outer simulation step of size `sim_cfg.dt`.
      PyBullet can internally substep at `dt / physics_substeps`.
    - Ground-truth snapshots are typed and stable. The public getter returns the
      last committed snapshot, not an on-the-fly recomputation.
    """

    def __init__(
        self,
        sim_cfg: SimConfig,
        drone_cfg: DroneConfig,
        world_cfg: Optional[WorldConfig] = None,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.drone_cfg = drone_cfg
        self.world_cfg = world_cfg or WorldConfig()

        self.sim_cfg.validate()
        self.drone_cfg.validate()

        self._sub_dt = self.sim_cfg.dt / self.sim_cfg.physics_substeps
        self.rng = np.random.default_rng(self.sim_cfg.seed)

        self.client_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.drone_id: Optional[int] = None

        self.t: float = 0.0
        self.step_idx: int = 0

        self._mass_kg: Optional[float] = None
        self._inertia_diag: Optional[np.ndarray] = None

        self._rpm_cmd = np.zeros(4, dtype=float)
        self._rpm_actual = np.zeros(4, dtype=float)

        self._prev_vel_w = np.zeros(3, dtype=float)
        self._last_truth: Optional[TruthState] = None

    # ---------------------------------------------------------------------
    # Public lifecycle
    # ---------------------------------------------------------------------

    def connect(self) -> None:
        if self.client_id is not None:
            return

        mode = p.GUI if self.sim_cfg.gui else p.DIRECT
        self.client_id = p.connect(mode)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -self.sim_cfg.gravity_mps2, physicsClientId=self.client_id)
        p.setTimeStep(self._sub_dt, physicsClientId=self.client_id)
        p.setRealTimeSimulation(1 if self.sim_cfg.real_time else 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.sim_cfg.solver_iterations,
            physicsClientId=self.client_id,
        )

        if self.sim_cfg.enable_ground:
            self.plane_id = p.loadURDF(
                self.world_cfg.plane_urdf,
                physicsClientId=self.client_id,
            )

        self.drone_id = p.loadURDF(
            self.drone_cfg.urdf_path,
            basePosition=self.drone_cfg.start_pos_w.tolist(),
            baseOrientation=self.drone_cfg.start_quat_wb_xyzw.tolist(),
            useFixedBase=False,
            physicsClientId=self.client_id,
        )

        p.changeDynamics(
            self.drone_id,
            -1,
            linearDamping=self.drone_cfg.linear_damping,
            angularDamping=self.drone_cfg.angular_damping,
            physicsClientId=self.client_id,
        )

        if self.sim_cfg.disable_default_joint_motors:
            self._disable_default_joint_motors()

        self._cache_and_validate_dynamics()
        self._reset_internal_state_only()
        self._commit_truth(acc_w=np.zeros(3, dtype=float))

    def reset(self) -> TruthState:
        self._require_connected()
        assert self.drone_id is not None

        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.drone_cfg.start_pos_w.tolist(),
            self.drone_cfg.start_quat_wb_xyzw.tolist(),
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

        self._reset_internal_state_only()
        self._commit_truth(acc_w=np.zeros(3, dtype=float))
        return self.get_truth_state()

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.plane_id = None
        self.drone_id = None
        self._last_truth = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_state(
        self,
        pos_w: Any,
        quat_wb_xyzw: Any,
        vel_w: Any | None = None,
        omega_w: Any | None = None,
    ) -> TruthState:
        """
        Hard-reset the rigid-body state.

        Parameters:
        - pos_w: world position [x, y, z]
        - quat_wb_xyzw: body orientation in world frame, [x, y, z, w]
        - vel_w: optional world linear velocity
        - omega_w: optional world angular velocity
        """
        self._require_connected()
        assert self.drone_id is not None

        pos_w = _as_vec3(pos_w, "pos_w")
        quat_wb_xyzw = _normalize_quat_xyzw(quat_wb_xyzw)
        vel_w = np.zeros(3, dtype=float) if vel_w is None else _as_vec3(vel_w, "vel_w")
        omega_w = np.zeros(3, dtype=float) if omega_w is None else _as_vec3(omega_w, "omega_w")

        p.resetBasePositionAndOrientation(
            self.drone_id,
            pos_w.tolist(),
            quat_wb_xyzw.tolist(),
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=vel_w.tolist(),
            angularVelocity=omega_w.tolist(),
            physicsClientId=self.client_id,
        )

        self._prev_vel_w = vel_w.copy()
        self._commit_truth(acc_w=np.zeros(3, dtype=float))
        return self.get_truth_state()

    def set_motor_rpm_command(self, rpm_cmd: Any) -> None:
        rpm_cmd = np.asarray(rpm_cmd, dtype=float).reshape(-1)
        if rpm_cmd.shape != (4,):
            raise ValueError(f"rpm_cmd must have shape (4,), got {rpm_cmd.shape}")
        if not np.all(np.isfinite(rpm_cmd)):
            raise ValueError("rpm_cmd must be finite")

        motor = self.drone_cfg.motor
        self._rpm_cmd = np.clip(rpm_cmd, motor.min_rpm, motor.max_rpm)

    def step(self, rpm_cmd: Any | None = None) -> TruthState:
        """
        Advance the simulation by one environment step of duration sim_cfg.dt.
        """
        self._require_connected()
        if rpm_cmd is not None:
            self.set_motor_rpm_command(rpm_cmd)

        # Apply actuation and drag across PyBullet substeps.
        for _ in range(self.sim_cfg.physics_substeps):
            pre = self._read_raw_kinematics()
            self._update_motor_dynamics(dt=self._sub_dt)
            self._apply_motor_forces_and_yaw_torque(pre)
            self._apply_body_drag(pre)
            p.stepSimulation(physicsClientId=self.client_id)

        post = self._read_raw_kinematics()
        acc_w = (post.vel_w - self._prev_vel_w) / self.sim_cfg.dt

        self.t += self.sim_cfg.dt
        self.step_idx += 1
        self._prev_vel_w = post.vel_w.copy()
        self._last_truth = self._build_truth_state(post, acc_w=acc_w)

        return self.get_truth_state()

    def get_truth_state(self) -> TruthState:
        if self._last_truth is None:
            self.connect()
        assert self._last_truth is not None
        return _copy_truth(self._last_truth)

    def get_mass_kg(self) -> float:
        self._require_connected()
        assert self._mass_kg is not None
        return float(self._mass_kg)

    def get_inertia_diag(self) -> np.ndarray:
        self._require_connected()
        assert self._inertia_diag is not None
        return self._inertia_diag.copy()

    def estimate_hover_rpm(self) -> float:
        """
        Hover estimate using total thrust = mass * g and thrust_i = k_f * rpm_i^2.
        """
        mass = self.get_mass_kg()
        kf = self.drone_cfg.motor.thrust_coeff_n_per_rpm2
        return math.sqrt((mass * self.sim_cfg.gravity_mps2 / 4.0) / kf)

    @staticmethod
    def truth_to_log_dict(truth: TruthState) -> Dict[str, float]:
        d: Dict[str, float] = {
            "t": float(truth.t),
            "step_idx": float(truth.step_idx),
            "px": float(truth.pos_w[0]),
            "py": float(truth.pos_w[1]),
            "pz": float(truth.pos_w[2]),
            "vx": float(truth.vel_w[0]),
            "vy": float(truth.vel_w[1]),
            "vz": float(truth.vel_w[2]),
            "ax": float(truth.acc_w[0]),
            "ay": float(truth.acc_w[1]),
            "az": float(truth.acc_w[2]),
            "qx": float(truth.quat_wb_xyzw[0]),
            "qy": float(truth.quat_wb_xyzw[1]),
            "qz": float(truth.quat_wb_xyzw[2]),
            "qw": float(truth.quat_wb_xyzw[3]),
            "roll": float(truth.euler_rpy_wb[0]),
            "pitch": float(truth.euler_rpy_wb[1]),
            "yaw": float(truth.euler_rpy_wb[2]),
            "omega_w_x": float(truth.omega_w[0]),
            "omega_w_y": float(truth.omega_w[1]),
            "omega_w_z": float(truth.omega_w[2]),
            "omega_b_x": float(truth.omega_b[0]),
            "omega_b_y": float(truth.omega_b[1]),
            "omega_b_z": float(truth.omega_b[2]),
            "specific_force_b_x": float(truth.specific_force_b[0]),
            "specific_force_b_y": float(truth.specific_force_b[1]),
            "specific_force_b_z": float(truth.specific_force_b[2]),
            "g_load": float(truth.g_load),
            "mass_kg": float(truth.mass_kg),
            "Ixx": float(truth.inertia_diag[0]),
            "Iyy": float(truth.inertia_diag[1]),
            "Izz": float(truth.inertia_diag[2]),
        }
        for i in range(4):
            d[f"rpm_cmd_{i}"] = float(truth.motor_rpm_cmd[i])
            d[f"rpm_actual_{i}"] = float(truth.motor_rpm_actual[i])
            d[f"thrust_n_{i}"] = float(truth.motor_thrusts_n[i])
            d[f"yaw_torque_nm_{i}"] = float(truth.motor_yaw_torques_nm[i])
        return d

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _require_connected(self) -> None:
        if self.client_id is None or self.drone_id is None:
            self.connect()
        assert self.client_id is not None and self.drone_id is not None

    def _reset_internal_state_only(self) -> None:
        self.t = 0.0
        self.step_idx = 0
        self._rpm_cmd[:] = 0.0
        self._rpm_actual[:] = 0.0
        self._prev_vel_w[:] = 0.0
        self._last_truth = None

    def _disable_default_joint_motors(self) -> None:
        assert self.drone_id is not None
        num_joints = p.getNumJoints(self.drone_id, physicsClientId=self.client_id)
        for joint_idx in range(num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.drone_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client_id,
            )

    def _cache_and_validate_dynamics(self) -> None:
        assert self.drone_id is not None
        dyn = p.getDynamicsInfo(self.drone_id, -1, physicsClientId=self.client_id)
        mass_kg = float(dyn[0])
        inertia_diag = np.asarray(dyn[2], dtype=float)

        if mass_kg <= 0.0:
            raise RuntimeError(f"Loaded drone has non-positive mass: {mass_kg}")
        if inertia_diag.shape != (3,) or np.any(inertia_diag <= 0.0):
            raise RuntimeError(f"Loaded drone has invalid inertia diagonal: {inertia_diag}")

        if self.drone_cfg.expected_mass_kg is not None:
            if abs(mass_kg - self.drone_cfg.expected_mass_kg) > self.drone_cfg.dynamics_tolerance:
                raise RuntimeError(
                    f"URDF mass mismatch: actual={mass_kg}, expected={self.drone_cfg.expected_mass_kg}"
                )

        if self.drone_cfg.expected_inertia_diag is not None:
            if not np.allclose(
                inertia_diag,
                self.drone_cfg.expected_inertia_diag,
                atol=self.drone_cfg.dynamics_tolerance,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "URDF inertia mismatch: "
                    f"actual={inertia_diag}, expected={self.drone_cfg.expected_inertia_diag}"
                )

        self._mass_kg = mass_kg
        self._inertia_diag = inertia_diag

    def _read_raw_kinematics(self) -> _RawKinematics:
        assert self.drone_id is not None
        pos_w, quat_xyzw = p.getBasePositionAndOrientation(
            self.drone_id,
            physicsClientId=self.client_id,
        )
        vel_w, omega_w = p.getBaseVelocity(
            self.drone_id,
            physicsClientId=self.client_id,
        )

        pos_w = np.asarray(pos_w, dtype=float)
        vel_w = np.asarray(vel_w, dtype=float)
        quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
        omega_w = np.asarray(omega_w, dtype=float)

        rot_wb = _quat_xyzw_to_rot_wb(quat_xyzw)
        rot_bw = rot_wb.T
        omega_b = rot_bw @ omega_w
        euler_rpy = np.asarray(p.getEulerFromQuaternion(quat_xyzw.tolist()), dtype=float)

        return _RawKinematics(
            pos_w=pos_w,
            vel_w=vel_w,
            quat_wb_xyzw=quat_xyzw,
            rot_wb=rot_wb,
            rot_bw=rot_bw,
            euler_rpy_wb=euler_rpy,
            omega_w=omega_w,
            omega_b=omega_b,
        )

    def _commit_truth(self, acc_w: np.ndarray) -> None:
        raw = self._read_raw_kinematics()
        self._prev_vel_w = raw.vel_w.copy()
        self._last_truth = self._build_truth_state(raw, acc_w=acc_w)

    def _build_truth_state(self, raw: _RawKinematics, acc_w: np.ndarray) -> TruthState:
        assert self._mass_kg is not None
        assert self._inertia_diag is not None

        acc_w = _as_vec3(acc_w, "acc_w")
        g_w = np.array([0.0, 0.0, -self.sim_cfg.gravity_mps2], dtype=float)
        specific_force_b = raw.rot_bw @ (acc_w - g_w)
        g_load = float(np.linalg.norm(specific_force_b) / self.sim_cfg.gravity_mps2)

        motor = self._current_motor_state()

        return TruthState(
            t=self.t,
            step_idx=self.step_idx,
            pos_w=raw.pos_w.copy(),
            vel_w=raw.vel_w.copy(),
            acc_w=acc_w.copy(),
            quat_wb_xyzw=raw.quat_wb_xyzw.copy(),
            rot_wb=raw.rot_wb.copy(),
            rot_bw=raw.rot_bw.copy(),
            euler_rpy_wb=raw.euler_rpy_wb.copy(),
            omega_w=raw.omega_w.copy(),
            omega_b=raw.omega_b.copy(),
            specific_force_b=specific_force_b,
            g_load=g_load,
            motor_rpm_cmd=motor.rpm_cmd,
            motor_rpm_actual=motor.rpm_actual,
            motor_thrusts_n=motor.thrusts_n,
            motor_yaw_torques_nm=motor.yaw_torques_nm,
            mass_kg=float(self._mass_kg),
            inertia_diag=self._inertia_diag.copy(),
            extras={},
        )

    def _current_motor_state(self) -> MotorState:
        motor = self.drone_cfg.motor
        thrusts_n = motor.thrust_coeff_n_per_rpm2 * np.square(self._rpm_actual)
        yaw_torques_nm = motor.spin_dirs * motor.yaw_moment_coeff_nm_per_rpm2 * np.square(
            self._rpm_actual
        )
        return MotorState(
            rpm_cmd=self._rpm_cmd.copy(),
            rpm_actual=self._rpm_actual.copy(),
            thrusts_n=thrusts_n,
            yaw_torques_nm=yaw_torques_nm,
        )

    def _update_motor_dynamics(self, dt: float) -> None:
        tau = self.drone_cfg.motor.motor_time_constant_s
        alpha = 1.0 - math.exp(-dt / tau)
        self._rpm_actual += alpha * (self._rpm_cmd - self._rpm_actual)
        motor = self.drone_cfg.motor
        np.clip(self._rpm_actual, motor.min_rpm, motor.max_rpm, out=self._rpm_actual)

    def _apply_motor_forces_and_yaw_torque(self, raw: _RawKinematics) -> None:
        assert self.drone_id is not None
        motor_cfg = self.drone_cfg.motor
        motor = self._current_motor_state()

        # Apply each rotor thrust in WORLD_FRAME at its rotor hub position.
        for i in range(4):
            rotor_offset_b = motor_cfg.rotor_offsets_b[i]
            rotor_pos_w = raw.pos_w + raw.rot_wb @ rotor_offset_b
            thrust_b = np.array([0.0, 0.0, motor.thrusts_n[i]], dtype=float)
            thrust_w = raw.rot_wb @ thrust_b

            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=thrust_w.tolist(),
                posObj=rotor_pos_w.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

        # Reaction torques from rotor drag act about the body z-axis.
        total_yaw_torque_b = np.array([0.0, 0.0, np.sum(motor.yaw_torques_nm)], dtype=float)
        total_yaw_torque_w = raw.rot_wb @ total_yaw_torque_b
        p.applyExternalTorque(
            objectUniqueId=self.drone_id,
            linkIndex=-1,
            torqueObj=total_yaw_torque_w.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )

    def _apply_body_drag(self, raw: _RawKinematics) -> None:
        assert self.drone_id is not None

        vel_b = raw.rot_bw @ raw.vel_w
        drag_force_b = -self.drone_cfg.drag_coeff_body * vel_b * np.abs(vel_b)
        drag_force_w = raw.rot_wb @ drag_force_b

        p.applyExternalForce(
            objectUniqueId=self.drone_id,
            linkIndex=-1,
            forceObj=drag_force_w.tolist(),
            posObj=raw.pos_w.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )

        omega_b = raw.omega_b
        drag_torque_b = -self.drone_cfg.angular_drag_coeff_body * omega_b * np.abs(omega_b)
        drag_torque_w = raw.rot_wb @ drag_torque_b

        p.applyExternalTorque(
            objectUniqueId=self.drone_id,
            linkIndex=-1,
            torqueObj=drag_torque_w.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )
