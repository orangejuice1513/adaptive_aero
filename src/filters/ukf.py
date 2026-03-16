# ukf.py: standard UKF with constant noise covariances.
#
# NOTE: This is an early prototype.  The production implementation
# that integrates with the simulation pipeline lives in ukf_core.py.
# Use HoverUKF / RaceUKF / AdaptiveUKF from that module instead.

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Quaternion utilities  (w-first convention: q = [qw, qx, qy, qz])
# ---------------------------------------------------------------------------

def _quat_to_rot_mat(q):
    """3×3 rotation matrix from quaternion q = [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1.0 - 2*(yy + zz), 2*(xy - wz),       2*(xz + wy)      ],
        [2*(xy + wz),        1.0 - 2*(xx + zz),  2*(yz - wx)      ],
        [2*(xz - wy),        2*(yz + wx),         1.0 - 2*(xx + yy)],
    ], dtype=float)


def _quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2, both in w-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


# ---------------------------------------------------------------------------
# UKF class
# ---------------------------------------------------------------------------

class UKF():
    def __init__(self, P, Q, R, dt, ukf_params, quad_params, measurements):
        """
        Parameters
        ----------
        P            : (n, n) initial state covariance
        Q            : (n, n) process noise covariance (constant)
        R            : (m, m) measurement noise covariance (constant)
        dt           : float  timestep, s
        ukf_params   : dict with keys 'alpha', 'beta', 'n', 'lambda'
        quad_params  : dict with keys 'mass', 'ct', 'cd', 'L_arm'
        measurements : array-like, shape (T, m)  — sensor measurements
        """
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt

        self.alpha = ukf_params['alpha']  # structural resonance / damping
        self.beta  = ukf_params['beta']   # G-sensitivity coefficient
        self.n     = ukf_params['n']      # state dimension
        self.lambd = ukf_params['lambda']

        self.Wm = np.full(2*self.n + 1, 0.5 / (self.n + self.lambd))
        self.Wm[0] = self.lambd / (self.n + self.lambd)
        self.Wc = np.full(2*self.n + 1, 0.5 / (self.n + self.lambd))
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        self.mass  = quad_params['mass']
        self.ct    = quad_params['ct']      # thrust coefficient  T_i = ct * ω_i²
        self.cd    = quad_params['cd']      # drag coefficient
        self.L_arm = quad_params['L_arm']   # arm length, m
        self.g     = 9.81

        self.measurements = np.asarray(measurements)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def control(self, t):
        """
        Return motor prop speeds (rad/s or RPM, matching ct units) at
        timestep t.  The default implementation returns the hover RPM
        that exactly counters gravity for this quadrotor:

            4 · ct · ω_hover² = m · g   →   ω_hover = sqrt(m·g / (4·ct))

        Override or subclass for trajectory-following.
        """
        omega_hover = np.sqrt(self.mass * self.g / (4.0 * self.ct))
        return np.full(4, omega_hover, dtype=float)

    # ------------------------------------------------------------------
    # Unscented transform
    # ------------------------------------------------------------------

    def unscented_transform(self, mu, cov):
        """
        Performs the unscented transform on a mean and covariance.
        Returns the 2n+1 sigma points as an array of shape (2n+1, n).
        """
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)   # guard against tiny negatives
        U = vecs @ np.diag(np.sqrt((self.n + self.lambd) * vals))

        sigma_points = np.zeros((2*self.n + 1, self.n))
        sigma_points[0] = mu
        for i in range(self.n):
            sigma_points[i + 1]          = mu + U[:, i]
            sigma_points[self.n + i + 1] = mu - U[:, i]
        return sigma_points

    def inv_unscented_transform(self, sigma_points):
        """
        Given sigma points, recover the weighted mean and covariance.
        No noise added here — add Q or R at the call site.
        """
        mu  = np.dot(self.Wm, sigma_points)
        cov = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            diff = sigma_points[i] - mu
            cov += self.Wc[i] * np.outer(diff, diff)
        return mu, cov

    # ------------------------------------------------------------------
    # Dynamics propagation
    # ------------------------------------------------------------------

    def get_pred_sigma_points(self, prior_sigma_points, u):
        """
        Propagate prior sigma points through the nonlinear dynamics f.

        State layout: [px, py, pz, vx, vy, vz, qw, qx, qy, qz]  (n=10)
        Control u   : motor speeds [ω₀, ω₁, ω₂, ω₃]

        Motor layout (top-down, + configuration):
            0 → front (+x), 1 → right (+y), 2 → back, 3 → left
        """
        pred = np.zeros_like(prior_sigma_points)

        T     = self.ct * np.sum(u**2)
        tau_p = self.L_arm * self.ct * (u[1]**2 - u[3]**2)
        tau_q = self.L_arm * self.ct * (u[2]**2 - u[0]**2)
        tau_r = self.cd    * (u[0]**2 - u[1]**2 + u[2]**2 - u[3]**2)
        w_body = np.array([tau_p, tau_q, tau_r])

        for i in range(2*self.n + 1):
            p_prev = prior_sigma_points[i, 0:3]
            v_prev = prior_sigma_points[i, 3:6]
            q_prev = prior_sigma_points[i, 6:10]   # [qw, qx, qy, qz]

            # Position
            p_k = p_prev + v_prev * self.dt

            # Velocity (specific force in body frame → world)
            R_mat     = _quat_to_rot_mat(q_prev)
            a_world   = R_mat @ np.array([0.0, 0.0, T / self.mass]) - np.array([0.0, 0.0, self.g])
            v_k       = v_prev + a_world * self.dt

            # Quaternion integration (first-order)
            q_dot = 0.5 * _quat_multiply(q_prev, np.array([0.0, w_body[0], w_body[1], w_body[2]]))
            q_k   = q_prev + q_dot * self.dt
            q_k  /= np.linalg.norm(q_k)

            pred[i] = np.concatenate([p_k, v_k, q_k])

        return pred

    # ------------------------------------------------------------------
    # Measurement function
    # ------------------------------------------------------------------

    def g_meas(self, sigma_point):
        """
        Map one sigma point → measurement space.

        Measurement z = [px, py, pz, ax_b, ay_b, az_b]
            position from VIO, acceleration from IMU (body frame)
        """
        p = sigma_point[0:3]
        q = sigma_point[6:10]   # [qw, qx, qy, qz]

        R_mat         = _quat_to_rot_mat(q)
        gravity_world = np.array([0.0, 0.0, self.g])
        a_body        = R_mat.T @ gravity_world     # rotate gravity into body frame

        return np.concatenate([p, a_body])

    def get_measured_sigma_points(self, pred_sigma_points):
        """Pass predicted sigma points through the measurement function."""
        meas_dim = 6
        meas_sp  = np.zeros((2*self.n + 1, meas_dim))
        for i in range(2*self.n + 1):
            meas_sp[i] = self.g_meas(pred_sigma_points[i])
        return meas_sp

    # ------------------------------------------------------------------
    # Predict / Update
    # ------------------------------------------------------------------

    def prediction_step(self, mu_t, sigma_t, u_t):
        """
        UKF prediction step.

        Parameters
        ----------
        mu_t    : (n,) current state mean
        sigma_t : (n, n) current state covariance
        u_t     : (4,) motor speeds at this step

        Returns
        -------
        pred_mu  : (n,) predicted mean
        pred_cov : (n, n) predicted covariance
        """
        prior_sp  = self.unscented_transform(mu_t, sigma_t)
        pred_sp   = self.get_pred_sigma_points(prior_sp, u_t)
        pred_mu, pred_cov = self.inv_unscented_transform(pred_sp)
        pred_cov += self.Q
        return pred_mu, pred_cov

    def update_step(self, pred_mu, pred_cov, z_t):
        """
        UKF update step.

        Parameters
        ----------
        pred_mu  : (n,) predicted state mean
        pred_cov : (n, n) predicted state covariance
        z_t      : (m,) actual measurement at this step

        Returns
        -------
        updated_mu  : (n,) posterior mean
        updated_cov : (n, n) posterior covariance
        """
        meas_dim = len(z_t)
        pred_sp  = self.unscented_transform(pred_mu, pred_cov)
        meas_sp  = self.get_measured_sigma_points(pred_sp)

        meas_y, meas_cov_y = self.inv_unscented_transform(meas_sp)
        meas_cov_y += self.R

        meas_cov_xy = np.zeros((self.n, meas_dim))
        for i in range(2*self.n + 1):
            state_diff = pred_sp[i] - pred_mu
            meas_diff  = meas_sp[i] - meas_y
            meas_cov_xy += self.Wc[i] * np.outer(state_diff, meas_diff)

        K           = meas_cov_xy @ np.linalg.inv(meas_cov_y)
        updated_mu  = pred_mu  + K @ (z_t - meas_y)
        updated_cov = pred_cov - K @ meas_cov_y @ K.T

        return updated_mu, updated_cov

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self, pose0, N, u=None):
        """
        Run the UKF for N timesteps from initial state pose0.

        Parameters
        ----------
        pose0 : (n,) initial state
        N     : int  number of steps
        u     : (4,) fixed motor command, or None to use self.control(t)

        Returns
        -------
        poses_history : (N+1, n) array of state estimates
        """
        poses_history = [pose0]
        mu_t  = np.asarray(pose0, dtype=float)
        cov_t = self.P.copy()

        for i in range(N):
            u_t = np.asarray(u, dtype=float) if u is not None else self.control(i)

            # Use stored measurement if available, otherwise use predicted y
            pred_mu, pred_cov = self.prediction_step(mu_t, cov_t, u_t)

            if i < len(self.measurements):
                z_t = self.measurements[i]
            else:
                z_t = self.g_meas(pred_mu)   # fall back to noiseless prediction

            mu_t, cov_t = self.update_step(pred_mu, pred_cov, z_t)
            poses_history.append(mu_t.copy())

        return np.array(poses_history)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_position_history(self, poses, show_plot=True):
        """Plot the XY position history of the quadrotor."""
        assert poses.ndim == 2, (
            "Data contains multiple runs. Must have 2 dimensions, "
            f"but got {poses.ndim} dimensions."
        )
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Position History")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.scatter(poses[0, 0], poses[0, 1],
                    color='black', marker='*', s=100, zorder=3,
                    label="Initial pose")
        plt.plot(poses[:, 0], poses[:, 1], label="XY path")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig

    def plot_quaternion_history(self, poses, show_plot=True):
        """
        Plot quaternion components [qw, qx, qy, qz] over time.

        Parameters
        ----------
        poses : (N, n) state history array — state layout [p, v, qw, qx, qy, qz]
        """
        assert poses.ndim == 2, (
            "Data contains multiple runs. Must have 2 dimensions, "
            f"but got {poses.ndim} dimensions."
        )
        t     = np.arange(len(poses)) * self.dt
        quats = poses[:, 6:10]   # [qw, qx, qy, qz]
        labels = ["qw", "qx", "qy", "qz"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            ax.plot(t, quats[:, i], label=lbl, color=col, linewidth=1.4)

        ax.axhline(0.0, color='black', linewidth=0.6, linestyle='--')
        ax.set_title("Quaternion History")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Component value")
        ax.legend(ncol=4)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig

    def plot_control_history(self, controls, show_plot=True):
        """Plot motor speed history."""
        assert controls.ndim == 2, (
            "Data contains multiple runs. Must have 2 dimensions, "
            f"but got {controls.ndim} dimensions."
        )
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Control History")
        plt.xlabel("Timestep")
        plt.ylabel("Motor speed")
        for i in range(controls.shape[1]):
            plt.plot(controls[:, i], label=f"Motor {i}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig

    def plot_measurement_history(self, measurements, show_plot=True):
        """Plot measurement history."""
        assert measurements.ndim == 2, (
            "Data contains multiple runs. Must have 2 dimensions, "
            f"but got {measurements.ndim} dimensions."
        )
        fig, _ = plt.subplots(figsize=(8, 6))
        plt.title("Measurement History")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        labels = ["px", "py", "pz", "ax", "ay", "az"]
        for i in range(measurements.shape[1]):
            lbl = labels[i] if i < len(labels) else f"z[{i}]"
            plt.plot(measurements[:, i], label=lbl)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig
