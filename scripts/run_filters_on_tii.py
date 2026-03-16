"""run_filters_on_tii.py: Replay one TII flight log through Hover/Race/Adaptive UKFs.

Evaluates against mocap ground truth using RMSE and NEES.
Saves filter trajectories and metrics to <out_dir>/<flight_name>_filter_results.npz
and a JSON summary to <out_dir>/<flight_name>_summary.json.

Usage
-----
    python -m scripts.run_filters_on_tii --flight flight-01a-ellipse
    python -m scripts.run_filters_on_tii                          # all flights
    python -m scripts.run_filters_on_tii --npz logs/tii/flight-01a-ellipse.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.parse_data import parse_flight, _ALL_FLIGHTS
from src.filters.hover_ukf import HoverUKF
from src.filters.race_ukf import RaceUKF
from src.filters.adaptive_ukf import AdaptiveUKF, AdaptiveUKFConfig


# ---------------------------------------------------------------------------
# Initial state helpers
# ---------------------------------------------------------------------------

def _make_x0(init_pos: np.ndarray, init_vel: np.ndarray, init_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Build the 10-D initial state vector [p(3), v(3), q_xyzw(4)].
    The quaternion from drone_state is already [x, y, z, w] — matches ukf_core.
    """
    q = np.asarray(init_quat_xyzw, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q = q / norm
    return np.concatenate([
        np.asarray(init_pos, dtype=float),
        np.asarray(init_vel, dtype=float),
        q,
    ])


def _make_P0() -> np.ndarray:
    return np.diag([
        0.10, 0.10, 0.10,   # position  (m²)
        0.25, 0.25, 0.25,   # velocity  (m²/s²)
        0.05, 0.05, 0.05,   # attitude error (rad²)
    ])


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def _compute_rmse(est: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return float("nan")
    diff = est[valid] - truth[valid]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ---------------------------------------------------------------------------
# NEES  (position-only, 3-DOF)
# ---------------------------------------------------------------------------

def _compute_nees(
    est_pos: np.ndarray,
    truth_pos: np.ndarray,
    cov: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Normalised Estimation Error Squared for position (3-DOF).

    NEES_k = e_k^T  P_pos_k^{-1}  e_k
    where e_k = truth_pos[k] - est_pos[k]  and  P_pos_k = cov[k, :3, :3].

    Expected value is 3 for a consistent filter.
    Returns an array of shape (N,) with NaN for invalid steps.
    """
    n = len(est_pos)
    nees = np.full(n, np.nan, dtype=float)
    for k in range(n):
        if not valid[k]:
            continue
        if not (np.all(np.isfinite(est_pos[k])) and np.all(np.isfinite(cov[k]))):
            continue
        err = truth_pos[k] - est_pos[k]
        P_pos = cov[k, :3, :3]
        try:
            nees[k] = float(err @ np.linalg.solve(P_pos, err))
        except np.linalg.LinAlgError:
            pass
    return nees


# ---------------------------------------------------------------------------
# Single-flight runner
# ---------------------------------------------------------------------------

def run_one_flight(data: dict, motor_max_rpm_equiv: float = 1.0) -> dict:
    """
    Run HoverUKF, RaceUKF, and AdaptiveUKF on one parsed flight.

    For the AdaptiveUKF:
      - rpm_sq_sum  = motor_thrust_sum * motor_max_rpm_equiv^2
      - rpm_sq_norm = motor_thrust_sum / 4   (thrust per motor, normalised)
      - specific_force_mag_mps2 = imu_sf_mag

    Parameters
    ----------
    data : dict
        As returned by parse_flight().
    motor_max_rpm_equiv : float
        Sets the scale factor so that  rpm_sq_norm = thrust_sum / 4.
        Leave at 1.0 — the rpm_sq_norm calculation cancels the max-RPM factor.

    Returns
    -------
    dict of result arrays (ready to pass to np.savez_compressed).
    """
    t = data["t"]
    imu_accel = data["imu_accel"]
    imu_gyro = data["imu_gyro"]
    imu_sf_mag = data["imu_sf_mag"]
    motor_thrust_sum = data["motor_thrust_sum"]
    vio_valid = data["vio_valid"].astype(bool)
    vio_pos = data["vio_pos"]
    truth_pos = data["truth_pos"]

    n = len(t)

    x0 = _make_x0(data["init_pos"], data["init_vel"], data["init_quat_xyzw"])
    P0 = _make_P0()

    # motor_max_rpm_equiv = 1.0 → rpm_sq_sum = thrust_sum → rpm_sq_norm = thrust_sum / 4
    hover = HoverUKF()
    race = RaceUKF()
    adaptive = AdaptiveUKF(AdaptiveUKFConfig(motor_max_rpm=motor_max_rpm_equiv))

    hover.initialize(x0, P0)
    race.initialize(x0, P0)
    adaptive.initialize(x0, P0)

    state_dim = len(x0)     # 10 (full state)
    cov_dim = P0.shape[0]   # 9  (error-state covariance)

    hover_x = np.full((n, state_dim), np.nan)
    race_x = np.full((n, state_dim), np.nan)
    adaptive_x = np.full((n, state_dim), np.nan)

    hover_cov = np.full((n, cov_dim, cov_dim), np.nan)
    race_cov = np.full((n, cov_dim, cov_dim), np.nan)
    adaptive_cov = np.full((n, cov_dim, cov_dim), np.nan)

    adaptive_pos_var = np.full(n, np.nan)
    adaptive_vel_var = np.full(n, np.nan)
    adaptive_att_var = np.full(n, np.nan)

    # Store step 0 (initialisation)
    hover_x[0] = hover.state()
    race_x[0] = race.state()
    adaptive_x[0] = adaptive.state()
    hover_cov[0] = hover.covariance()
    race_cov[0] = race.covariance()
    adaptive_cov[0] = adaptive.covariance()

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if dt <= 0.0:
            continue

        # IMU predict
        hover.predict(imu_accel[k], imu_gyro[k], dt)
        race.predict(imu_accel[k], imu_gyro[k], dt)
        adaptive.predict(
            imu_accel_mps2=imu_accel[k],
            imu_gyro_radps=imu_gyro[k],
            dt=dt,
            rpm_sq_sum=float(motor_thrust_sum[k]) * motor_max_rpm_equiv ** 2,
            specific_force_mag_mps2=float(imu_sf_mag[k]),
        )

        # VIO update (drone_state position)
        if vio_valid[k] and np.all(np.isfinite(vio_pos[k])):
            hover.update_vio(vio_pos[k])
            race.update_vio(vio_pos[k])
            adaptive.update_vio(vio_pos[k])

        hover_x[k] = hover.state()
        race_x[k] = race.state()
        adaptive_x[k] = adaptive.state()

        hover_cov[k] = hover.covariance()
        race_cov[k] = race.covariance()
        adaptive_cov[k] = adaptive.covariance()

        adaptive_pos_var[k] = adaptive.last_pos_var
        adaptive_vel_var[k] = adaptive.last_vel_var
        adaptive_att_var[k] = adaptive.last_att_var

    hover_pos = hover_x[:, :3]
    race_pos = race_x[:, :3]
    adaptive_pos = adaptive_x[:, :3]

    valid_hover = np.all(np.isfinite(hover_pos), axis=1)
    valid_race = np.all(np.isfinite(race_pos), axis=1)
    valid_adapt = np.all(np.isfinite(adaptive_pos), axis=1)

    # Also need valid truth for NEES/RMSE
    valid_truth = np.all(np.isfinite(truth_pos), axis=1)

    hover_rmse = _compute_rmse(hover_pos, truth_pos, valid_hover & valid_truth)
    race_rmse = _compute_rmse(race_pos, truth_pos, valid_race & valid_truth)
    adaptive_rmse = _compute_rmse(adaptive_pos, truth_pos, valid_adapt & valid_truth)

    hover_nees = _compute_nees(hover_pos, truth_pos, hover_cov, valid_hover & valid_truth)
    race_nees = _compute_nees(race_pos, truth_pos, race_cov, valid_race & valid_truth)
    adaptive_nees = _compute_nees(adaptive_pos, truth_pos, adaptive_cov, valid_adapt & valid_truth)

    results = {
        "t": t,
        "truth_pos": truth_pos,
        "hover_state": hover_x,
        "race_state": race_x,
        "adaptive_state": adaptive_x,
        "hover_cov": hover_cov,
        "race_cov": race_cov,
        "adaptive_cov": adaptive_cov,
        "hover_pos": hover_pos,
        "race_pos": race_pos,
        "adaptive_pos": adaptive_pos,
        "hover_valid": valid_hover.astype(np.int32),
        "race_valid": valid_race.astype(np.int32),
        "adaptive_valid": valid_adapt.astype(np.int32),
        "hover_nees": hover_nees,
        "race_nees": race_nees,
        "adaptive_nees": adaptive_nees,
        "adaptive_pos_var": adaptive_pos_var,
        "adaptive_vel_var": adaptive_vel_var,
        "adaptive_att_var": adaptive_att_var,
        "imu_sf_mag": imu_sf_mag,
    }

    summary = {
        "flight_name": data.get("flight_name", "unknown"),
        "n_steps": int(n),
        "duration_s": float(t[-1]),
        "vio_updates": int(np.sum(vio_valid)),
        "hover_rmse_m": hover_rmse,
        "race_rmse_m": race_rmse,
        "adaptive_rmse_m": adaptive_rmse,
        "hover_mean_nees": float(np.nanmean(hover_nees)),
        "race_mean_nees": float(np.nanmean(race_nees)),
        "adaptive_mean_nees": float(np.nanmean(adaptive_nees)),
    }

    return results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Hover/Race/Adaptive UKFs on TII drone racing data."
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
        help="Directory for output files. Default: <repo>/logs/tii/",
    )
    p.add_argument(
        "--flight",
        type=str,
        default=None,
        help="Run only this flight (e.g. flight-01a-ellipse). Default: all flights.",
    )
    p.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Path to a pre-parsed .npz from parse_data.py (overrides --flight / --data-dir).",
    )
    return p


def _load_npz_as_dict(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else repo_root / "data"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else repo_root / "logs" / "tii"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to run
    if args.npz:
        npz_path = Path(args.npz).expanduser().resolve()
        print(f"Loading pre-parsed data from {npz_path}")
        data = _load_npz_as_dict(npz_path)
        flight_name = npz_path.stem
        data["flight_name"] = flight_name
        jobs = [(flight_name, data)]
    else:
        flights = [args.flight] if args.flight else _ALL_FLIGHTS
        jobs = []
        for flight_name in flights:
            flight_dir = data_dir / flight_name
            if not flight_dir.exists():
                print(f"[skip] {flight_dir} not found")
                continue
            jobs.append((flight_name, None))  # lazy-parse

    all_summaries = []

    for flight_name, data in jobs:
        if data is None:
            flight_dir = data_dir / flight_name
            print(f"Parsing {flight_name} ...", end=" ", flush=True)
            try:
                data = parse_flight(flight_dir)
                data["flight_name"] = flight_name
                print(f"({len(data['t'])} IMU steps)", end=" ", flush=True)
            except Exception as exc:
                print(f"ERROR: {exc}")
                continue

        print(f"Running filters on {flight_name} ...", end=" ", flush=True)
        try:
            results, summary = run_one_flight(data)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        # Save arrays
        npz_out = out_dir / f"{flight_name}_filter_results.npz"
        np.savez_compressed(npz_out, **results)

        # Save JSON summary
        json_out = out_dir / f"{flight_name}_summary.json"
        json_out.write_text(json.dumps(summary, indent=2))

        print(
            f"done  |  Hover RMSE={summary['hover_rmse_m']:.4f}m  "
            f"Race={summary['race_rmse_m']:.4f}m  "
            f"Adaptive={summary['adaptive_rmse_m']:.4f}m"
        )
        print(
            f"        NEES (mean/expected=3):  "
            f"Hover={summary['hover_mean_nees']:.2f}  "
            f"Race={summary['race_mean_nees']:.2f}  "
            f"Adaptive={summary['adaptive_mean_nees']:.2f}"
        )
        all_summaries.append(summary)

    # Aggregate summary across all flights
    if len(all_summaries) > 1:
        agg_out = out_dir / "all_flights_summary.json"
        agg_out.write_text(json.dumps(all_summaries, indent=2))
        print(f"\nAggregate summary saved to {agg_out}")


if __name__ == "__main__":
    main()
