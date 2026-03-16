"""hyperparam_search.py: Grid search over AdaptiveUKF beta and subtract_gravity.

Hover and Race are run once per flight as a fixed baseline.
Only AdaptiveUKF is re-run for each config — ~3x faster than re-running all 3.

Usage
-----
    python -m scripts.hyperparam_search
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from scripts.run_filters_on_tii import _make_x0, _make_P0, _compute_rmse, _compute_nees
from src.filters.hover_ukf import HoverUKF
from src.filters.race_ukf import RaceUKF
from src.filters.adaptive_ukf import AdaptiveUKF, AdaptiveUKFConfig


# ---------------------------------------------------------------------------
# Flights to test on (3 shortest by step count)
# ---------------------------------------------------------------------------

TEST_FLIGHTS = [
    "flight-11a-lemniscate",   # 11,216 steps
    "flight-01a-ellipse",      # 11,258 steps
    "flight-05a-ellipse",      # 11,657 steps
]

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------

BETA_SCALES = [0.001, 0.01, 0.1, 1.0, 10.0]   # multiplied against defaults
SUBTRACT_GRAVITY = [False, True]

_DEFAULT_BETA_POS = 1e-4
_DEFAULT_BETA_VEL = 2e-2
_DEFAULT_BETA_ATT = 1e-4


def make_adaptive_config(beta_scale: float, subtract_gravity: bool) -> AdaptiveUKFConfig:
    return AdaptiveUKFConfig(
        beta_pos_var=_DEFAULT_BETA_POS * beta_scale,
        beta_vel_var=_DEFAULT_BETA_VEL * beta_scale,
        beta_att_var=_DEFAULT_BETA_ATT * beta_scale,
        subtract_gravity=subtract_gravity,
        motor_max_rpm=1.0,
    )


def config_label(beta_scale: float, subtract_gravity: bool) -> str:
    sub = "sub_g" if subtract_gravity else "raw_g"
    return f"beta={beta_scale:.0e}x_{sub}"


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_filter_loop(data: dict, filters: dict) -> dict:
    """
    Run one or more filters through a flight.

    filters: dict of name -> filter object (must have .predict / .update_vio /
             .state / .covariance)
    Returns dict of name -> {'pos': (N,3), 'cov': (N,9,9)}
    """
    t = data["t"]
    imu_accel = data["imu_accel"]
    imu_gyro = data["imu_gyro"]
    imu_sf_mag = data["imu_sf_mag"]
    motor_thrust_sum = data["motor_thrust_sum"]
    vio_valid = data["vio_valid"].astype(bool)
    vio_pos = data["vio_pos"]
    n = len(t)

    x0 = _make_x0(data["init_pos"], data["init_vel"], data["init_quat_xyzw"])
    P0 = _make_P0()
    cov_dim = P0.shape[0]

    for filt in filters.values():
        filt.initialize(x0.copy(), P0.copy())

    state_dim = len(x0)
    outputs = {
        name: {
            "pos": np.full((n, 3), np.nan),
            "cov": np.full((n, cov_dim, cov_dim), np.nan),
        }
        for name in filters
    }

    # Step 0
    for name, filt in filters.items():
        outputs[name]["pos"][0] = filt.state()[:3]
        outputs[name]["cov"][0] = filt.covariance()

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if dt <= 0.0:
            continue

        do_update = vio_valid[k] and np.all(np.isfinite(vio_pos[k]))

        for name, filt in filters.items():
            if isinstance(filt, AdaptiveUKF):
                filt.predict(
                    imu_accel_mps2=imu_accel[k],
                    imu_gyro_radps=imu_gyro[k],
                    dt=dt,
                    rpm_sq_sum=float(motor_thrust_sum[k]),
                    specific_force_mag_mps2=float(imu_sf_mag[k]),
                )
            else:
                filt.predict(imu_accel[k], imu_gyro[k], dt)

            if do_update:
                filt.update_vio(vio_pos[k])

            outputs[name]["pos"][k] = filt.state()[:3]
            outputs[name]["cov"][k] = filt.covariance()

    return outputs


def compute_metrics(outputs: dict, truth_pos: np.ndarray) -> dict:
    valid_t = np.all(np.isfinite(truth_pos), axis=1)
    metrics = {}
    for name, out in outputs.items():
        pos = out["pos"]
        cov = out["cov"]
        valid = np.all(np.isfinite(pos), axis=1)
        mask = valid & valid_t
        metrics[name] = {
            "rmse": _compute_rmse(pos, truth_pos, mask),
            "nees": float(np.nanmean(_compute_nees(pos, truth_pos, cov, mask))),
        }
    return metrics


def run_baselines(flight_data: dict, cache_path: Path) -> dict[str, dict]:
    """Run Hover and Race once per flight. Loads from cache if available."""
    if cache_path.exists():
        print(f"Loading cached baselines from {cache_path.name} ...", flush=True)
        return json.loads(cache_path.read_text())

    print("Running Hover + Race baselines ...", flush=True)
    baseline_results = {}
    for fname, data in flight_data.items():
        filters = {"hover": HoverUKF(), "race": RaceUKF()}
        outputs = _run_filter_loop(data, filters)
        baseline_results[fname] = compute_metrics(outputs, data["truth_pos"])
        h = baseline_results[fname]["hover"]
        r = baseline_results[fname]["race"]
        print(f"  {fname:30s}  Hover RMSE={h['rmse']:.4f} NEES={h['nees']:.2f}  "
              f"Race RMSE={r['rmse']:.4f} NEES={r['nees']:.2f}", flush=True)

    cache_path.write_text(json.dumps(baseline_results, indent=2))
    print(f"  Baselines saved to {cache_path.name}\n", flush=True)
    return baseline_results


def run_adaptive_config(
    flight_data: dict,
    cfg: AdaptiveUKFConfig,
    label: str,
) -> dict[str, dict]:
    """Run AdaptiveUKF with one config on all test flights."""
    results = {}
    for fname, data in flight_data.items():
        filters = {"adaptive": AdaptiveUKF(cfg)}
        outputs = _run_filter_loop(data, filters)
        results[fname] = compute_metrics(outputs, data["truth_pos"])["adaptive"]
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    npz_dir = repo_root / "logs" / "tii"
    print("Loading pre-parsed NPZ files ...")
    flight_data = {}
    for name in TEST_FLIGHTS:
        npz_path = npz_dir / f"{name}.npz"
        print(f"  Loading {npz_path.name} ...", end=" ", flush=True)
        raw = np.load(npz_path, allow_pickle=False)
        d = {k: raw[k] for k in raw.files}
        d["flight_name"] = name
        flight_data[name] = d
        print(f"{len(d['t'])} steps, {int(np.sum(d['vio_valid']))} VIO updates")
    print()

    # Baselines (run once, cached)
    baseline_cache = npz_dir / "hyperparam_baselines.json"
    baseline = run_baselines(flight_data, baseline_cache)

    # Grid search (adaptive only)
    configs = list(itertools.product(BETA_SCALES, SUBTRACT_GRAVITY))
    results_cache = npz_dir / "hyperparam_grid_results.json"

    # Load any previously completed configs
    completed = {}
    if results_cache.exists():
        completed = json.loads(results_cache.read_text())
        print(f"Resuming: {len(completed)}/{len(configs)} configs already done.\n", flush=True)

    print(f"Running {len(configs)} AdaptiveUKF configs x {len(TEST_FLIGHTS)} flights ...\n", flush=True)

    grid_results = []

    for i, (beta_scale, sub_g) in enumerate(configs):
        label = config_label(beta_scale, sub_g)
        cfg = make_adaptive_config(beta_scale, sub_g)

        if label in completed:
            r = completed[label]
            print(f"[{i+1}/{len(configs)}] {label}  (cached: RMSE={r['avg_rmse']:.4f} NEES={r['avg_nees']:.2f})", flush=True)
            grid_results.append((beta_scale, sub_g, r["avg_rmse"], r["avg_nees"]))
            continue

        print(f"[{i+1}/{len(configs)}] {label}", flush=True)
        per_flight = run_adaptive_config(flight_data, cfg, label)

        rmses, neeses = [], []
        for fname in TEST_FLIGHTS:
            m = per_flight[fname]
            bh = baseline[fname]["hover"]
            br = baseline[fname]["race"]
            print(f"  {fname:30s}  "
                  f"A RMSE={m['rmse']:.4f}  H={bh['rmse']:.4f}  R={br['rmse']:.4f}  |  "
                  f"A NEES={m['nees']:.2f}  H={bh['nees']:.2f}  R={br['nees']:.2f}", flush=True)
            rmses.append(m["rmse"])
            neeses.append(m["nees"])

        avg_rmse = float(np.mean(rmses))
        avg_nees = float(np.mean(neeses))
        print(f"  AVG  A RMSE={avg_rmse:.4f}  NEES={avg_nees:.2f}\n", flush=True)
        grid_results.append((beta_scale, sub_g, avg_rmse, avg_nees))

        # Save this config's result immediately
        completed[label] = {"avg_rmse": avg_rmse, "avg_nees": avg_nees}
        results_cache.write_text(json.dumps(completed, indent=2))

    # ---------------------------------------------------------------------------
    # Summary table — ranked by |adaptive_nees - 3|
    # ---------------------------------------------------------------------------
    avg_h_rmse = float(np.mean([baseline[f]["hover"]["rmse"] for f in TEST_FLIGHTS]))
    avg_r_rmse = float(np.mean([baseline[f]["race"]["rmse"]  for f in TEST_FLIGHTS]))
    avg_h_nees = float(np.mean([baseline[f]["hover"]["nees"] for f in TEST_FLIGHTS]))
    avg_r_nees = float(np.mean([baseline[f]["race"]["nees"]  for f in TEST_FLIGHTS]))

    print("=" * 80)
    print("SUMMARY — ranked by |Adaptive NEES - 3| (closest to 3 = best calibrated)")
    print("=" * 80)
    print(f"  Baselines:  Hover RMSE={avg_h_rmse:.4f} NEES={avg_h_nees:.2f}  |  "
          f"Race RMSE={avg_r_rmse:.4f} NEES={avg_r_nees:.2f}")
    print()
    print(f"  {'Config':<22}  {'A_RMSE':>7}  {'A_NEES':>7}  {'|A-3|':>6}  {'vs Race RMSE':>12}")
    print("  " + "-" * 62)

    grid_results.sort(key=lambda x: abs(x[3] - 3.0))
    for beta_scale, sub_g, avg_rmse, avg_nees in grid_results:
        label = config_label(beta_scale, sub_g)
        nees_err = abs(avg_nees - 3.0)
        rmse_diff = avg_rmse - avg_r_rmse
        marker = " <-- best" if (beta_scale, sub_g) == (grid_results[0][0], grid_results[0][1]) else ""
        print(f"  {label:<22}  {avg_rmse:>7.4f}  {avg_nees:>7.2f}  {nees_err:>6.2f}  "
              f"{rmse_diff:>+12.4f}{marker}")

    print()
    best = grid_results[0]
    print(f"Best config: beta_scale={best[0]:.0e}x, subtract_gravity={best[1]}")
    print(f"  Adaptive NEES={best[3]:.2f} (target 3.0), RMSE={best[2]:.4f}m "
          f"(Race={avg_r_rmse:.4f}m, diff={best[2]-avg_r_rmse:+.4f}m)")


if __name__ == "__main__":
    main()
