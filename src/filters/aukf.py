# aukf.py: Adaptive UKF that extends ukf.py with a heteroscedastic noise model.
#
# The noise model follows the physics-informed formulation in the docs/README:
#
#     Q_k = Q_floor + Q_vibe(Ω_k) + Q_load(a_k)
#
# where Ω_k = sum of squared motor RPMs (proxy for vibration energy)
# and a_k = IMU specific force magnitude (proxy for G-loading).
#
# NOTE: The production pipeline uses AdaptiveUKF in src/filters/adaptive_ukf.py,
# which integrates with ukf_core.py and the simulation logging system.
# This class provides a self-contained prototype that inherits from ukf.py.

import numpy as np

from src.filters.ukf import UKF
from src.models.measurement_noise import HeteroscedasticNoiseConfig


class AUKF(UKF):
    """
    Adaptive UKF with physics-informed process noise adaptation.

    Inherits the full UKF prediction/update machinery from ukf.py.
    At each prediction step the process noise matrix Q_k is recomputed
    as a function of motor RPM and G-loading instead of using the fixed Q.

    Additional constructor parameter
    ---------------------------------
    noise_cfg : HeteroscedasticNoiseConfig
        Coefficients α, β, and r_floor for the noise model.
        Defaults to the standard configuration from measurement_noise.py.

    Adaptive Q model
    ----------------
    Matches the structure in src/models/measurement_noise.py:

        σ²_pos(Ω, a) = pos_floor + α_pos · norm_rpm² + β_pos · ‖a‖
        σ²_vel(Ω, a) = vel_floor + α_vel · norm_rpm² + β_vel · ‖a‖
        σ²_att(Ω, a) = att_floor + α_att · norm_rpm² + β_att · ‖a‖

        Q_k = diag(σ²_pos·I₃, σ²_vel·I₃, σ²_att·I₃)

    where  norm_rpm² = Σᵢ Ωᵢ² / (4 · Ω_max²) ∈ [0, 1].
    """

    def __init__(
        self,
        P, Q, R, dt,
        ukf_params, quad_params, measurements,
        noise_cfg: HeteroscedasticNoiseConfig | None = None,
        # Per-block adaptive coefficients (fall back to noise_cfg if not given)
        alpha_pos: float | None = None,
        alpha_vel: float | None = None,
        alpha_att: float | None = None,
        beta_pos:  float | None = None,
        beta_vel:  float | None = None,
        beta_att:  float | None = None,
        max_pos_var: float = 5e-3,
        max_vel_var: float = 5e-1,
        max_att_var: float = 5e-3,
    ) -> None:
        super().__init__(P, Q, R, dt, ukf_params, quad_params, measurements)

        from src.models.measurement_noise import DEFAULT_NOISE_CONFIG
        cfg = noise_cfg or DEFAULT_NOISE_CONFIG

        # Store Q_floor diagonal variances (parsed from the provided Q matrix)
        diag = np.diag(Q)
        if len(diag) >= 9:
            self._pos_floor = float(np.mean(diag[0:3]))
            self._vel_floor = float(np.mean(diag[3:6]))
            self._att_floor = float(np.mean(diag[6:9]))
        else:
            # Fallback for smaller state spaces
            self._pos_floor = cfg.r_floor
            self._vel_floor = cfg.r_floor * 10
            self._att_floor = cfg.r_floor

        # Adaptive coefficients (per block, with fallback to isotropic cfg values)
        self._alpha_pos = alpha_pos if alpha_pos is not None else cfg.alpha
        self._alpha_vel = alpha_vel if alpha_vel is not None else cfg.alpha * 10.0
        self._alpha_att = alpha_att if alpha_att is not None else cfg.alpha

        self._beta_pos  = beta_pos  if beta_pos  is not None else cfg.beta
        self._beta_vel  = beta_vel  if beta_vel  is not None else cfg.beta * 10.0
        self._beta_att  = beta_att  if beta_att  is not None else cfg.beta

        self._motor_max_rpm = cfg.motor_max_rpm
        self._max_pos_var   = max_pos_var
        self._max_vel_var   = max_vel_var
        self._max_att_var   = max_att_var

        # Logging
        self.last_pos_var: float = self._pos_floor
        self.last_vel_var: float = self._vel_floor
        self.last_att_var: float = self._att_floor

    # ------------------------------------------------------------------
    # Adaptive Q computation
    # ------------------------------------------------------------------

    def _rpm_sq_norm(self, rpm_sq_sum: float) -> float:
        return float(rpm_sq_sum) / (4.0 * self._motor_max_rpm ** 2)

    def compute_Q_adaptive(
        self,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> np.ndarray:
        """
        Compute the instantaneous process noise matrix Q_k.

        Parameters
        ----------
        rpm_sq_sum : float
            Σᵢ ωᵢ²  (four motors), RPM².
        specific_force_mag_mps2 : float
            ‖a_k‖  (m/s²) from the IMU.

        Returns
        -------
        Q_k : (n, n) diagonal matrix
        """
        norm_rpm2 = self._rpm_sq_norm(rpm_sq_sum)
        sf_mag    = float(specific_force_mag_mps2)

        pos_var = self._pos_floor + self._alpha_pos * norm_rpm2 + self._beta_pos * sf_mag
        vel_var = self._vel_floor + self._alpha_vel * norm_rpm2 + self._beta_vel * sf_mag
        att_var = self._att_floor + self._alpha_att * norm_rpm2 + self._beta_att * sf_mag

        pos_var = float(np.clip(pos_var, self._pos_floor, self._max_pos_var))
        vel_var = float(np.clip(vel_var, self._vel_floor, self._max_vel_var))
        att_var = float(np.clip(att_var, self._att_floor, self._max_att_var))

        self.last_pos_var = pos_var
        self.last_vel_var = vel_var
        self.last_att_var = att_var

        n = self.n
        Q_k = np.zeros((n, n), dtype=float)
        # Fill the first 9 diagonal entries (pos, vel, att blocks)
        if n >= 9:
            Q_k[0, 0] = Q_k[1, 1] = Q_k[2, 2] = pos_var
            Q_k[3, 3] = Q_k[4, 4] = Q_k[5, 5] = vel_var
            Q_k[6, 6] = Q_k[7, 7] = Q_k[8, 8] = att_var
            # Any remaining entries keep the base Q values
            for i in range(9, n):
                Q_k[i, i] = float(self.Q[i, i])
        else:
            # Fallback: scale the original Q uniformly
            scale = 1.0 + norm_rpm2 + sf_mag / 9.81
            Q_k = self.Q * scale

        return Q_k

    # ------------------------------------------------------------------
    # Override prediction step to inject adaptive Q
    # ------------------------------------------------------------------

    def adaptive_prediction_step(
        self,
        mu_t: np.ndarray,
        sigma_t: np.ndarray,
        u_t: np.ndarray,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ):
        """
        Prediction step with adaptive process noise.

        Parameters
        ----------
        mu_t    : (n,) current state mean
        sigma_t : (n, n) current covariance
        u_t     : (4,) motor speed commands
        rpm_sq_sum : float  Σᵢ ωᵢ²
        specific_force_mag_mps2 : float  ‖a_k‖

        Returns
        -------
        pred_mu  : (n,) predicted mean
        pred_cov : (n, n) predicted covariance
        """
        Q_k = self.compute_Q_adaptive(rpm_sq_sum, specific_force_mag_mps2)

        prior_sp = self.unscented_transform(mu_t, sigma_t)
        pred_sp  = self.get_pred_sigma_points(prior_sp, u_t)
        pred_mu, pred_cov = self.inv_unscented_transform(pred_sp)
        pred_cov += Q_k
        return pred_mu, pred_cov

    # ------------------------------------------------------------------
    # Adaptive simulation loop
    # ------------------------------------------------------------------

    def simulate_adaptive(
        self,
        pose0: np.ndarray,
        N: int,
        u=None,
        rpm_sq_sums=None,
        sf_mags=None,
    ) -> np.ndarray:
        """
        Run the AUKF for N steps with adaptive process noise.

        Parameters
        ----------
        pose0        : (n,) initial state
        N            : int number of steps
        u            : (4,) fixed motor command or None (uses self.control)
        rpm_sq_sums  : (N,) array of Σrpm² values; zeros if None
        sf_mags      : (N,) array of ‖a_k‖ values; zeros if None

        Returns
        -------
        poses_history : (N+1, n) array of state estimates
        """
        rpm_sq_sums = np.zeros(N) if rpm_sq_sums is None else np.asarray(rpm_sq_sums)
        sf_mags     = np.zeros(N) if sf_mags is None     else np.asarray(sf_mags)

        poses_history = [np.asarray(pose0, dtype=float)]
        mu_t  = np.asarray(pose0, dtype=float)
        cov_t = self.P.copy()

        for i in range(N):
            u_t = np.asarray(u, dtype=float) if u is not None else self.control(i)

            pred_mu, pred_cov = self.adaptive_prediction_step(
                mu_t, cov_t, u_t,
                rpm_sq_sum=float(rpm_sq_sums[i]),
                specific_force_mag_mps2=float(sf_mags[i]),
            )

            if i < len(self.measurements):
                z_t = self.measurements[i]
            else:
                z_t = self.g_meas(pred_mu)

            mu_t, cov_t = self.update_step(pred_mu, pred_cov, z_t)
            poses_history.append(mu_t.copy())

        return np.array(poses_history)
