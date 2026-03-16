"""parse_data.py: Load and align TII drone racing dataset CSV files into numpy arrays.

Output arrays (keyed in the returned dict / saved .npz):
  t                - (N,)    time in seconds from flight start (IMU timeline)
  imu_accel        - (N, 3)  specific force in body frame, m/s²
  imu_gyro         - (N, 3)  angular rate in body frame, rad/s
  imu_sf_mag       - (N,)    ||imu_accel||, m/s²
  motor_thrust_sum - (N,)    sum of four motor normalized thrusts (Σ thrust_i)
  vio_valid        - (N,)    int32, 1 where a drone_state position is available
  vio_pos          - (N, 3)  drone_state position in world frame, m (NaN where invalid)
  truth_pos        - (N, 3)  mocap ground truth position, m (interpolated to IMU timeline)
  init_pos         - (3,)    initial position from first drone_state sample
  init_vel         - (3,)    initial velocity from first drone_state sample
  init_quat_xyzw   - (4,)    initial quaternion [x, y, z, w] from first drone_state sample
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np


# All flight directory names in the dataset
_ALL_FLIGHTS = [
    "flight-01a-ellipse",
    "flight-02a-ellipse",
    "flight-03a-ellipse",
    "flight-04a-ellipse",
    "flight-05a-ellipse",
    "flight-06a-ellipse",
    "flight-07a-lemniscate",
    "flight-08a-lemniscate",
    "flight-09a-lemniscate",
    "flight-10a-lemniscate",
    "flight-11a-lemniscate",
    "flight-12a-lemniscate",
    "flight-13a-trackRATM",
    "flight-14a-trackRATM",
    "flight-15a-trackRATM",
    "flight-16a-trackRATM",
    "flight-17a-trackRATM",
    "flight-18a-trackRATM",
]


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a CSV file, return (header, float_array)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return header, np.array(rows, dtype=float)


def _load_imu(path: Path) -> dict[str, np.ndarray]:
    """
    Columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    Accelerometer = specific force (body frame), m/s². IMU already subtracts gravity bias
    but the raw reading while hovering shows ~+9.8 on z, consistent with specific force convention.
    Gyro in rad/s.
    """
    _, data = _read_csv(path)
    return {
        "t_us": data[:, 0],
        "accel": data[:, 1:4],
        "gyro": data[:, 4:7],
    }


def _load_drone_state(path: Path) -> dict[str, np.ndarray]:
    """
    Columns: timestamp, pose_position_x/y/z,
             pose_orientation_x/y/z/w, velocity_linear_x/y/z, ...
    Quaternion stored as [x, y, z, w] (Hamilton convention, matches ukf_core).
    """
    _, data = _read_csv(path)
    return {
        "t_us": data[:, 0],
        "pos": data[:, 1:4],
        "quat_xyzw": data[:, 4:8],  # [x, y, z, w]
        "vel": data[:, 8:11],
    }


def _load_motors(path: Path) -> dict[str, np.ndarray]:
    """
    Columns: timestamp, thrust[0], thrust[1], thrust[2], thrust[3]
    Thrusts are normalized (approximately 0-1 per motor).
    """
    _, data = _read_csv(path)
    return {
        "t_us": data[:, 0],
        "thrust_sum": np.sum(data[:, 1:5], axis=1),
    }


def _load_mocap(path: Path) -> dict[str, np.ndarray]:
    """
    Columns: timestamp, frame, drone_x, drone_y, drone_z, ...
    Position in world frame, metres.
    """
    _, data = _read_csv(path)
    return {
        "t_us": data[:, 0],
        "pos": data[:, 2:5],  # drone_x, drone_y, drone_z
    }


def _interp_cols(t_query: np.ndarray, t_src: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Linearly interpolate each column of vals (shape N×C) onto t_query."""
    out = np.empty((len(t_query), vals.shape[1]), dtype=float)
    for i in range(vals.shape[1]):
        out[:, i] = np.interp(t_query, t_src, vals[:, i])
    return out


def _assign_vio(
    imu_t: np.ndarray,
    state_t: np.ndarray,
    state_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each drone_state sample, assign it to the nearest IMU step (by index).
    When multiple state samples map to the same IMU step, the last one wins.
    Returns (vio_valid bool array, vio_pos float array) on the IMU timeline.
    """
    # Subsample drone_state to ~30 Hz (from ~277 Hz) so filters must coast on
    # IMU prediction between updates — this makes Q-adaptation differences visible.
    step = max(1, round(277 / 30))  # ≈ 9
    state_t = state_t[::step]
    state_pos = state_pos[::step]

    n = len(imu_t)
    vio_valid = np.zeros(n, dtype=bool)
    vio_pos = np.full((n, 3), np.nan, dtype=float)

    # For each state sample, find the nearest IMU index
    imu_idx = np.searchsorted(imu_t, state_t, side="right") - 1
    imu_idx = np.clip(imu_idx, 0, n - 1)

    for si, ki in enumerate(imu_idx):
        vio_valid[ki] = True
        vio_pos[ki] = state_pos[si]

    return vio_valid, vio_pos


def parse_flight(flight_dir: Path) -> dict[str, np.ndarray | str]:
    """
    Parse all CSV files for one flight and return aligned numpy arrays.

    Parameters
    ----------
    flight_dir : Path
        Path to a flight directory, e.g., data/flight-01a-ellipse/

    Returns
    -------
    dict with keys documented in the module docstring plus 'flight_name'.
    """
    flight_dir = Path(flight_dir)
    name = flight_dir.name
    csv_raw = flight_dir / "csv_raw"
    ros2bag = csv_raw / "ros2bag_dump"

    imu = _load_imu(ros2bag / f"imu_{name}.csv")
    state = _load_drone_state(ros2bag / f"drone_state_{name}.csv")
    motors = _load_motors(ros2bag / f"motors_thrust_{name}.csv")
    mocap = _load_mocap(csv_raw / f"mocap_{name}.csv")

    # Primary timeline: IMU (highest frequency, ~500 Hz)
    t_us = imu["t_us"]
    t0_us = t_us[0]
    t = (t_us - t0_us) * 1e-6  # seconds from flight start

    # Motor thrust sum interpolated (nearest-neighbour / linear) onto IMU timeline
    motor_t = (motors["t_us"] - t0_us) * 1e-6
    thrust_sum = np.interp(t, motor_t, motors["thrust_sum"])

    # Mocap ground truth interpolated onto IMU timeline
    mocap_t = (mocap["t_us"] - t0_us) * 1e-6
    truth_pos = _interp_cols(t, mocap_t, mocap["pos"])

    # Drone state (used as "VIO" position measurement)
    state_t = (state["t_us"] - t0_us) * 1e-6
    vio_valid, vio_pos = _assign_vio(t, state_t, state["pos"])

    # Initial state from the first drone_state sample
    init_pos = state["pos"][0].copy()
    init_vel = state["vel"][0].copy()
    init_quat_xyzw = state["quat_xyzw"][0].copy()

    return {
        "flight_name": name,
        "t": t,
        "imu_accel": imu["accel"].astype(float),
        "imu_gyro": imu["gyro"].astype(float),
        "imu_sf_mag": np.linalg.norm(imu["accel"], axis=1),
        "motor_thrust_sum": thrust_sum,
        "vio_valid": vio_valid.astype(np.int32),
        "vio_pos": vio_pos,
        "truth_pos": truth_pos,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_quat_xyzw": init_quat_xyzw,
    }


def save_flight_npz(data: dict, out_path: Path) -> None:
    """Save parsed flight data to a .npz file (string fields excluded)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out_path, **arrays)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse TII drone racing dataset CSVs into .npz files."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the data/ directory. Default: <repo>/data/",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for .npz files. Default: <repo>/logs/tii/",
    )
    p.add_argument(
        "--flight",
        type=str,
        default=None,
        help="Parse only this flight (e.g. flight-01a-ellipse). Default: all flights.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else repo_root / "data"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else repo_root / "logs" / "tii"

    flights = [args.flight] if args.flight else _ALL_FLIGHTS

    for flight_name in flights:
        flight_dir = data_dir / flight_name
        if not flight_dir.exists():
            print(f"[skip] {flight_dir} not found")
            continue

        print(f"Parsing {flight_name} ...", end=" ", flush=True)
        try:
            data = parse_flight(flight_dir)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        out_path = out_dir / f"{flight_name}.npz"
        save_flight_npz(data, out_path)

        n = len(data["t"])
        dur = float(data["t"][-1])
        vio_count = int(np.sum(data["vio_valid"]))
        print(f"ok  ({n} IMU steps, {dur:.1f}s, {vio_count} VIO updates) -> {out_path}")


if __name__ == "__main__":
    main()
