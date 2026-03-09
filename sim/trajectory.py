from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Sequence
import math

import numpy as np

from sim.controller import ReferenceState


Array3 = np.ndarray


def _vec3(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _wrap_angle(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n



def _lerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    return (1.0 - s) * a + s * b


class Trajectory(Protocol):
    def sample(self, t: float) -> ReferenceState:
        ...

@dataclass(slots=True)
class StressTestTrajectory:
    """
    High-G periodic stress course with analytic derivatives.

    Important:
    - Uses integer harmonics of the base frequency so the path is truly periodic.
    - Uses analytic velocity/acceleration, avoiding finite-difference spikes.
    - speed_scale changes traversal speed without changing path shape.
    """

    center_w: Array3 = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    duration: float = 20.0
    speed_scale: float = 1.0

    def __post_init__(self) -> None:
        self.center_w = _vec3(self.center_w, "center_w")
        if self.duration <= 0.0:
            raise ValueError("duration must be > 0")
        if self.speed_scale <= 0.0:
            raise ValueError("speed_scale must be > 0")

    def sample(self, t: float) -> ReferenceState:
        # Wrapped trajectory time
        tau = float((t * self.speed_scale) % self.duration)

        # Base angular frequency
        w = 2.0 * np.pi / self.duration

        # ----------------------------
        # Position
        # ----------------------------
        # X: aggressive horizontal sweep/reversal
        x = (
            10.0 * np.sin(3.0 * w * tau)
            + 4.0 * np.sin(7.0 * w * tau)
            - 2.0 * np.sin(11.0 * w * tau)
        )

        # Y:
        # Original product term replaced using exact trig identity:
        # 8*sin(2wt)*cos(4wt) = 4*sin(6wt) - 4*sin(2wt)
        y = (
            4.0 * np.sin(6.0 * w * tau)
            - 4.0 * np.sin(2.0 * w * tau)
            + 5.0 * np.sin(8.0 * w * tau)
        )

        # Z: strong but smooth vertical envelope, ~1m to ~9m
        z = (
            5.0
            + 3.0 * np.sin(4.0 * w * tau)
            + 1.2 * np.sin(7.0 * w * tau + 0.5)
        )

        pos = np.array([x, y, z], dtype=float) + self.center_w

        # ----------------------------
        # Derivatives w.r.t. tau
        # ----------------------------
        dx_dtau = (
            10.0 * 3.0 * w * np.cos(3.0 * w * tau)
            + 4.0 * 7.0 * w * np.cos(7.0 * w * tau)
            - 2.0 * 11.0 * w * np.cos(11.0 * w * tau)
        )

        dy_dtau = (
            4.0 * 6.0 * w * np.cos(6.0 * w * tau)
            - 4.0 * 2.0 * w * np.cos(2.0 * w * tau)
            + 5.0 * 8.0 * w * np.cos(8.0 * w * tau)
        )

        dz_dtau = (
            3.0 * 4.0 * w * np.cos(4.0 * w * tau)
            + 1.2 * 7.0 * w * np.cos(7.0 * w * tau + 0.5)
        )

        d2x_dtau2 = (
            -10.0 * (3.0 * w) ** 2 * np.sin(3.0 * w * tau)
            - 4.0 * (7.0 * w) ** 2 * np.sin(7.0 * w * tau)
            + 2.0 * (11.0 * w) ** 2 * np.sin(11.0 * w * tau)
        )

        d2y_dtau2 = (
            -4.0 * (6.0 * w) ** 2 * np.sin(6.0 * w * tau)
            + 4.0 * (2.0 * w) ** 2 * np.sin(2.0 * w * tau)
            - 5.0 * (8.0 * w) ** 2 * np.sin(8.0 * w * tau)
        )

        d2z_dtau2 = (
            -3.0 * (4.0 * w) ** 2 * np.sin(4.0 * w * tau)
            - 1.2 * (7.0 * w) ** 2 * np.sin(7.0 * w * tau + 0.5)
        )

        # ----------------------------
        # Chain rule: tau = speed_scale * t
        # ----------------------------
        s = self.speed_scale

        vel = np.array(
            [
                s * dx_dtau,
                s * dy_dtau,
                s * dz_dtau,
            ],
            dtype=float,
        )

        acc = np.array(
            [
                (s ** 2) * d2x_dtau2,
                (s ** 2) * d2y_dtau2,
                (s ** 2) * d2z_dtau2,
            ],
            dtype=float,
        )

        yaw = math.atan2(vel[1], vel[0])
        yaw_rate = _estimate_yaw_rate_from_vel_acc(vel, acc)

        return ReferenceState(
            pos_w=pos,
            vel_w=vel,
            acc_w=acc,
            yaw=yaw,
            yaw_rate=yaw_rate,
        )
@dataclass(slots=True)
class HoverTrajectory:
    pos_w: Array3
    yaw: float = 0.0

    def __post_init__(self) -> None:
        self.pos_w = _vec3(self.pos_w, "pos_w")
        self.yaw = float(self.yaw)

    def sample(self, t: float) -> ReferenceState:
        _ = t
        return ReferenceState(
            pos_w=self.pos_w.copy(),
            vel_w=np.zeros(3, dtype=float),
            acc_w=np.zeros(3, dtype=float),
            yaw=self.yaw,
            yaw_rate=0.0,
        )


@dataclass(slots=True)
class CircleTrajectory:
    center_w: Array3
    radius_m: float
    speed_mps: float
    z_m: float
    clockwise: bool = False
    face_forward: bool = True
    yaw_offset_rad: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        self.center_w = _vec3(self.center_w, "center_w")
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be > 0")
        if self.speed_mps < 0.0:
            raise ValueError("speed_mps must be >= 0")
        self.z_m = float(self.z_m)
        self.phase_rad = float(self.phase_rad)
        self.yaw_offset_rad = float(self.yaw_offset_rad)

    def sample(self, t: float) -> ReferenceState:
        t = float(t)
        sign = -1.0 if self.clockwise else 1.0
        omega = 0.0 if self.radius_m == 0.0 else sign * self.speed_mps / self.radius_m
        theta = self.phase_rad + omega * t

        c, s = math.cos(theta), math.sin(theta)
        pos = np.array([
            self.center_w[0] + self.radius_m * c,
            self.center_w[1] + self.radius_m * s,
            self.z_m,
        ], dtype=float)
        vel = np.array([
            -self.radius_m * omega * s,
            self.radius_m * omega * c,
            0.0,
        ], dtype=float)
        acc = np.array([
            -self.radius_m * omega * omega * c,
            -self.radius_m * omega * omega * s,
            0.0,
        ], dtype=float)

        if self.face_forward and _norm(vel[:2]) > 1e-9:
            yaw = math.atan2(vel[1], vel[0]) + self.yaw_offset_rad
            yaw_rate = omega
        else:
            yaw = self.yaw_offset_rad
            yaw_rate = 0.0

        return ReferenceState(pos_w=pos, vel_w=vel, acc_w=acc, yaw=yaw, yaw_rate=yaw_rate)


@dataclass(slots=True)
class LemniscateTrajectory:
    center_w: Array3
    ax_m: float
    ay_m: float
    z_m: float
    omega_radps: float
    face_forward: bool = True
    yaw_offset_rad: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        self.center_w = _vec3(self.center_w, "center_w")
        if self.ax_m <= 0.0 or self.ay_m <= 0.0:
            raise ValueError("ax_m and ay_m must be > 0")
        if self.omega_radps <= 0.0:
            raise ValueError("omega_radps must be > 0")
        self.z_m = float(self.z_m)
        self.yaw_offset_rad = float(self.yaw_offset_rad)
        self.phase_rad = float(self.phase_rad)

    def sample(self, t: float) -> ReferenceState:
        t = float(t)
        th = self.phase_rad + self.omega_radps * t
        s, c = math.sin(th), math.cos(th)
        s2, c2 = math.sin(2.0 * th), math.cos(2.0 * th)
        w = self.omega_radps

        pos = np.array([
            self.center_w[0] + self.ax_m * s,
            self.center_w[1] + 0.5 * self.ay_m * s2,
            self.z_m,
        ], dtype=float)
        vel = np.array([
            self.ax_m * w * c,
            self.ay_m * w * c2,
            0.0,
        ], dtype=float)
        acc = np.array([
            -self.ax_m * w * w * s,
            -2.0 * self.ay_m * w * w * s2,
            0.0,
        ], dtype=float)

        if self.face_forward and _norm(vel[:2]) > 1e-9:
            yaw = math.atan2(vel[1], vel[0]) + self.yaw_offset_rad
            yaw_rate = _estimate_yaw_rate_from_vel_acc(vel, acc)
        else:
            yaw = self.yaw_offset_rad
            yaw_rate = 0.0

        return ReferenceState(pos_w=pos, vel_w=vel, acc_w=acc, yaw=yaw, yaw_rate=yaw_rate)


@dataclass(slots=True)
class HelixTrajectory:
    center_w: Array3
    radius_m: float
    speed_mps: float
    climb_rate_mps: float
    face_forward: bool = True
    yaw_offset_rad: float = 0.0
    phase_rad: float = 0.0
    z0_m: float = 1.0

    def __post_init__(self) -> None:
        self.center_w = _vec3(self.center_w, "center_w")
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be > 0")
        if self.speed_mps < 0.0:
            raise ValueError("speed_mps must be >= 0")
        self.climb_rate_mps = float(self.climb_rate_mps)
        self.yaw_offset_rad = float(self.yaw_offset_rad)
        self.phase_rad = float(self.phase_rad)
        self.z0_m = float(self.z0_m)

    def sample(self, t: float) -> ReferenceState:
        t = float(t)
        omega = self.speed_mps / self.radius_m if self.radius_m > 0.0 else 0.0
        th = self.phase_rad + omega * t
        s, c = math.sin(th), math.cos(th)

        pos = np.array([
            self.center_w[0] + self.radius_m * c,
            self.center_w[1] + self.radius_m * s,
            self.z0_m + self.climb_rate_mps * t,
        ], dtype=float)
        vel = np.array([
            -self.radius_m * omega * s,
            self.radius_m * omega * c,
            self.climb_rate_mps,
        ], dtype=float)
        acc = np.array([
            -self.radius_m * omega * omega * c,
            -self.radius_m * omega * omega * s,
            0.0,
        ], dtype=float)

        if self.face_forward and _norm(vel[:2]) > 1e-9:
            yaw = math.atan2(vel[1], vel[0]) + self.yaw_offset_rad
            yaw_rate = _estimate_yaw_rate_from_vel_acc(vel, acc)
        else:
            yaw = self.yaw_offset_rad
            yaw_rate = 0.0

        return ReferenceState(pos_w=pos, vel_w=vel, acc_w=acc, yaw=yaw, yaw_rate=yaw_rate)


@dataclass(slots=True)
class StraightLineTrajectory:
    start_w: Array3
    end_w: Array3
    speed_mps: float
    hold_end: bool = True
    yaw_mode: str = "path"  # 'path' or 'fixed'
    fixed_yaw_rad: float = 0.0

    _delta: np.ndarray = field(init=False, repr=False)
    _dist: float = field(init=False, repr=False)
    _dir: np.ndarray = field(init=False, repr=False)
    _duration: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.start_w = _vec3(self.start_w, "start_w")
        self.end_w = _vec3(self.end_w, "end_w")
        if self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be > 0")
        if self.yaw_mode not in {"path", "fixed"}:
            raise ValueError("yaw_mode must be 'path' or 'fixed'")
        self.fixed_yaw_rad = float(self.fixed_yaw_rad)

        self._delta = self.end_w - self.start_w
        self._dist = _norm(self._delta)
        self._dir = _unit(self._delta)
        self._duration = 0.0 if self._dist < 1e-9 else self._dist / self.speed_mps

    def sample(self, t: float) -> ReferenceState:
        t = float(t)
        if self._duration <= 0.0:
            pos = self.start_w.copy()
            vel = np.zeros(3, dtype=float)
            yaw = self.fixed_yaw_rad
            return ReferenceState(
                pos_w=pos,
                vel_w=vel,
                acc_w=np.zeros(3, dtype=float),
                yaw=yaw,
                yaw_rate=0.0,
            )

        if t >= self._duration and self.hold_end:
            pos = self.end_w.copy()
            vel = np.zeros(3, dtype=float)
            yaw = (
                self.fixed_yaw_rad
                if self.yaw_mode == "fixed"
                else math.atan2(self._dir[1], self._dir[0])
            )
            return ReferenceState(
                pos_w=pos,
                vel_w=vel,
                acc_w=np.zeros(3, dtype=float),
                yaw=yaw,
                yaw_rate=0.0,
            )

        s = np.clip(t / self._duration, 0.0, 1.0)
        pos = _lerp(self.start_w, self.end_w, s)
        vel = self._dir * self.speed_mps if t < self._duration else np.zeros(3, dtype=float)
        yaw = (
            self.fixed_yaw_rad
            if self.yaw_mode == "fixed"
            else math.atan2(self._dir[1], self._dir[0])
        )
        return ReferenceState(
            pos_w=pos,
            vel_w=vel,
            acc_w=np.zeros(3, dtype=float),
            yaw=yaw,
            yaw_rate=0.0,
        )

@dataclass(slots=True)
class MinimumJerkSegment:
    start_w: Array3
    end_w: Array3
    duration_s: float
    yaw_start: float | None = None
    yaw_end: float | None = None

    def __post_init__(self) -> None:
        self.start_w = _vec3(self.start_w, "start_w")
        self.end_w = _vec3(self.end_w, "end_w")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")
        if self.yaw_start is not None:
            self.yaw_start = float(self.yaw_start)
        if self.yaw_end is not None:
            self.yaw_end = float(self.yaw_end)

    def sample_local(self, tau: float) -> ReferenceState:
        t = float(np.clip(tau, 0.0, self.duration_s))
        s = t / self.duration_s
        s2, s3, s4, s5 = s * s, s * s * s, s * s * s * s, s * s * s * s * s
        # min-jerk interpolation
        h = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
        hd = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / self.duration_s
        hdd = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (self.duration_s * self.duration_s)

        delta = self.end_w - self.start_w
        pos = self.start_w + h * delta
        vel = hd * delta
        acc = hdd * delta

        if self.yaw_start is not None and self.yaw_end is not None:
            dyaw = _wrap_angle(self.yaw_end - self.yaw_start)
            yaw = _wrap_angle(self.yaw_start + h * dyaw)
            yaw_rate = hd * dyaw
        else:
            if _norm(vel[:2]) > 1e-9:
                yaw = math.atan2(vel[1], vel[0])
                yaw_rate = _estimate_yaw_rate_from_vel_acc(vel, acc)
            else:
                path = delta[:2]
                yaw = math.atan2(path[1], path[0]) if _norm(path) > 1e-9 else 0.0
                yaw_rate = 0.0

        return ReferenceState(pos_w=pos, vel_w=vel, acc_w=acc, yaw=yaw, yaw_rate=yaw_rate)


@dataclass(slots=True)
class PiecewiseTrajectory:
    segments: List[MinimumJerkSegment]
    hold_last: bool = True
    _cum_durations: List[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if len(self.segments) == 0:
            raise ValueError("PiecewiseTrajectory needs at least one segment")
        self._cum_durations = []
        total = 0.0
        for seg in self.segments:
            total += seg.duration_s
            self._cum_durations.append(total)

    @property
    def total_duration_s(self) -> float:
        return self._cum_durations[-1]

    def sample(self, t: float) -> ReferenceState:
        t = float(t)
        if t <= 0.0:
            return self.segments[0].sample_local(0.0)

        prev_end = 0.0
        for seg, end_time in zip(self.segments, self._cum_durations):
            if t <= end_time:
                return seg.sample_local(t - prev_end)
            prev_end = end_time

        if self.hold_last:
            last = self.segments[-1]
            ref = last.sample_local(last.duration_s)
            return ReferenceState(
                pos_w=ref.pos_w.copy(),
                vel_w=np.zeros(3, dtype=float),
                acc_w=np.zeros(3, dtype=float),
                yaw=ref.yaw,
                yaw_rate=0.0,
            )

        return self.segments[-1].sample_local(self.segments[-1].duration_s)

    @classmethod
    def from_waypoints(
        cls,
        waypoints_w: Sequence[Sequence[float]],
        speeds_mps: Sequence[float] | float,
        yaw_mode: str = "path",
        fixed_yaw_rad: float = 0.0,
        hold_last: bool = True,
    ) -> "PiecewiseTrajectory":
        pts = [_vec3(wp, "waypoint") for wp in waypoints_w]
        if len(pts) < 2:
            raise ValueError("Need at least 2 waypoints")
        if isinstance(speeds_mps, (int, float)):
            speeds = [float(speeds_mps)] * (len(pts) - 1)
        else:
            speeds = [float(v) for v in speeds_mps]
        if len(speeds) != len(pts) - 1:
            raise ValueError("speeds_mps length must be len(waypoints)-1")
        if yaw_mode not in {"path", "fixed"}:
            raise ValueError("yaw_mode must be 'path' or 'fixed'")

        segs: List[MinimumJerkSegment] = []
        for i, (a, b, speed) in enumerate(zip(pts[:-1], pts[1:], speeds)):
            if speed <= 0.0:
                raise ValueError("speeds must be > 0")
            dist = _norm(b - a)
            duration = max(dist / speed, 1e-3)
            if yaw_mode == "fixed":
                yaw0 = fixed_yaw_rad
                yaw1 = fixed_yaw_rad
            else:
                path = b[:2] - a[:2]
                yaw_path = math.atan2(path[1], path[0]) if _norm(path) > 1e-9 else 0.0
                yaw0 = yaw_path
                yaw1 = yaw_path
            segs.append(MinimumJerkSegment(start_w=a, end_w=b, duration_s=duration, yaw_start=yaw0, yaw_end=yaw1))
        return cls(segments=segs, hold_last=hold_last)


@dataclass(slots=True)
class RacingTrackTrajectory:
    center_w: Array3 = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    scale: float = 1.0
    nominal_speed_mps: float = 3.5
    hold_last: bool = True

    _traj: "PiecewiseTrajectory" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center_w = _vec3(self.center_w, "center_w")
        if self.scale <= 0.0:
            raise ValueError("scale must be > 0")
        if self.nominal_speed_mps <= 0.0:
            raise ValueError("nominal_speed_mps must be > 0")
        self._traj = self._build_piecewise()

    def _build_piecewise(self) -> PiecewiseTrajectory:
        base = np.array(
            [
                [0.0, 0.0, 1.0],
                [2.2, 0.0, 1.2],
                [4.4, 1.5, 1.6],
                [5.0, 4.0, 2.0],
                [3.0, 6.5, 2.4],
                [0.0, 7.0, 2.1],
                [-2.8, 5.5, 1.6],
                [-4.2, 2.4, 1.1],
                [-2.0, -0.6, 0.8],
                [1.2, -1.8, 1.0],
                [3.6, -0.6, 1.4],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        waypoints = self.center_w + self.scale * base
        return PiecewiseTrajectory.from_waypoints(
            waypoints_w=waypoints,
            speeds_mps=self.nominal_speed_mps,
            yaw_mode="path",
            hold_last=self.hold_last,
        )

    @property
    def total_duration_s(self) -> float:
        return self._traj.total_duration_s

    def sample(self, t: float) -> ReferenceState:
        return self._traj.sample(t)


def _estimate_yaw_rate_from_vel_acc(vel_w: np.ndarray, acc_w: np.ndarray, eps: float = 1e-9) -> float:
    vx, vy = float(vel_w[0]), float(vel_w[1])
    ax, ay = float(acc_w[0]), float(acc_w[1])
    denom = vx * vx + vy * vy
    if denom < eps:
        return 0.0
    return (vx * ay - vy * ax) / denom


def make_default_race_traj() -> RacingTrackTrajectory:
    return RacingTrackTrajectory(center_w=np.array([0.0, 0.0, 0.0], dtype=float), scale=1.0, nominal_speed_mps=3.5)


__all__ = [
    "Trajectory",
    "HoverTrajectory",
    "CircleTrajectory",
    "LemniscateTrajectory",
    "HelixTrajectory",
    "StraightLineTrajectory",
    "MinimumJerkSegment",
    "PiecewiseTrajectory",
    "RacingTrackTrajectory",
    "make_default_race_traj",
]
