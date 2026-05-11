#!/usr/bin/env python3
"""
LMPC Controller untuk USV — Linear MPC 3-DOF
=============================================
Referensi: Kufoalor et al. (2020), Journal of Field Robotics
           "Autonomous maritime collision avoidance"

Model Prediksi (Linearisasi di sekitar u0=1.5 m/s, v=0, r=0):
─────────────────────────────────────────────────────────────
State  : ξ = [x, y, ψ, u, r]ᵀ  (5 state)
Input  : μ = [TX, TN]ᵀ         (2 input)

Persamaan kontinu (linearisasi):
  dx/dt  = u·cos(ψ) - v·sin(ψ)    ≈ u·cos(ψ)   (kinematika)
  dy/dt  = u·sin(ψ) + v·cos(ψ)    ≈ u·sin(ψ)
  dψ/dt  = r
  du/dt  = A2·u + A18·TX          (surge dynamics)
  dr/dt  = A12_eff·r + A19·TN     (yaw dynamics, A12_eff = A12 + A16·|u0|)

Diskretisasi Euler: ξ(k+1) = Ad·ξ(k) + Bd·μ(k) + fd(k)
"""

import numpy as np
import math
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# ──────────────────────────────────────────────────────────────────────────────
#  LMPC Controller Class
# ──────────────────────────────────────────────────────────────────────────────

class LMPCController:
    """
    Linear MPC untuk obstacle avoidance USV.

    Filosofi (sesuai Kufoalor et al., 2020):
    - Prioritas 1: Hindari obstacle ke sisi aman (starboard preference)
    - Prioritas 2: Jika tidak bisa hindari → kurangi kecepatan hingga aman
    - Prioritas 3: Setelah aman → kembali ke jalur global (RRT*)
    - CTE diminimalkan sepanjang waktu
    """

    def __init__(self, params: dict):
        """
        params : dict dari params.yaml bagian /lmpc
        """
        # ── Parameter Model USV (dari params.yaml /usv) ───────────────────
        A2   = params.get('A2',   -0.7405)
        A12  = params.get('A12',  -1.0495)
        A16  = params.get('A16',   0.9671)
        A18  = params.get('A18',   0.0178)
        A19  = params.get('A19',   0.0010)
        self.A2   = A2
        self.A18  = A18
        self.A19  = A19
        self.A12_eff = A12 + A16 * 1.5  # linearisasi di u0=1.5

        # ── Sampling Time ─────────────────────────────────────────────────
        self.dt = float(params.get('dt', 0.05))

        # ── Horizon ───────────────────────────────────────────────────────
        self.N  = int(params.get('N',  15))    # prediction horizon
        self.Nc = int(params.get('Nc',  8))    # control horizon (Nc ≤ N)

        # ── Weight Matrices ───────────────────────────────────────────────
        # Q: state tracking weights [x, y, psi, u, r]
        Q_diag = params.get('Q', [8.0, 8.0, 3.0, 5.0, 0.5])
        self.Q  = np.diag(Q_diag)
        # R: input effort weights [TX, TN]
        R_diag = params.get('R', [0.001, 0.05])
        self.R  = np.diag(R_diag)
        # Qf: terminal cost (lebih besar dari Q untuk stabilitas)
        Qf_diag = params.get('Qf', [15.0, 15.0, 6.0, 8.0, 1.0])
        self.Qf = np.diag(Qf_diag)
        # W_obs: obstacle avoidance penalty weight
        self.W_obs  = float(params.get('W_obs',  500.0))
        # W_cte: cross-track error penalty
        self.W_cte  = float(params.get('W_cte',  12.0))
        # W_du: delta input smoothness weight
        self.W_du   = float(params.get('W_du',   0.01))

        # ── Input Constraints ─────────────────────────────────────────────
        self.TX_min = float(params.get('TX_min',    0.0))
        self.TX_max = float(params.get('TX_max',  200.0))
        self.TN_min = float(params.get('TN_min', -1750.0))
        self.TN_max = float(params.get('TN_max',  1750.0))
        # Rate limits (per step)
        self.dTX_max = float(params.get('dTX_max', 200.0)) * self.dt
        self.dTN_max = float(params.get('dTN_max', 7500.0)) * self.dt

        # ── State Constraints ─────────────────────────────────────────────
        self.u_min = float(params.get('u_min', 0.0))
        self.u_max = float(params.get('u_max', 1.6))
        self.r_max = float(params.get('r_max_deg', 20.0)) * math.pi / 180.0

        # ── Obstacle Parameters ───────────────────────────────────────────
        self.d_safe  = float(params.get('d_safe',  1.5))   # clearance dari tepi obstacle [m]
        self.d_warn  = float(params.get('d_warn',  3.5))   # jarak mulai aktif LMPC [m]
        self.d_exit  = float(params.get('d_exit',  5.0))   # jarak LMPC off [m]

        # ── Speed Reduction ───────────────────────────────────────────────
        self.u_slow    = float(params.get('u_slow',   1.5))   # kecepatan saat terpaksa berhenti
        self.u_normal  = float(params.get('u_normal', 1.5))   # kecepatan normal

        # ── Internal State ────────────────────────────────────────────────
        self.u_prev  = np.array([100.0, 0.0])   # [TX_prev, TN_prev] untuk rate limit
        self.active  = False                      # apakah LMPC sedang aktif
        self.mode    = 'GLOBAL'                   # 'GLOBAL' atau 'LMPC_AVOID' atau 'LMPC_SLOW'
        self._opt_u  = None                       # hasil optimasi sebelumnya (warm start)

    # ──────────────────────────────────────────────────────────────────────
    #  Model Prediksi (Linearisasi)
    # ──────────────────────────────────────────────────────────────────────

    def _predict_one_step(self, xi, mu):
        """
        Prediksi satu langkah: ξ(k+1) = f(ξ(k), μ(k))
        State: [x, y, psi, u, r]
        Input: [TX, TN]
        """
        x, y, psi, u, r = xi
        TX, TN = mu

        dt = self.dt

        # Kinematika (nonlinear, lebih akurat untuk horison pendek)
        dx   = (u * math.cos(psi)) * dt
        dy   = (u * math.sin(psi)) * dt
        dpsi = r * dt

        # Dinamika surge (linearisasi)
        du = (self.A2 * u + self.A18 * TX) * dt

        # Dinamika yaw (linearisasi di u0=1.5)
        dr = (self.A12_eff * r + self.A19 * TN) * dt

        x_new   = x + dx
        y_new   = y + dy
        psi_new = (psi + dpsi + math.pi) % (2 * math.pi) - math.pi
        u_new   = float(np.clip(u + du, self.u_min, self.u_max))
        r_new   = float(np.clip(r + dr, -self.r_max, self.r_max))

        return np.array([x_new, y_new, psi_new, u_new, r_new])

    def _predict_trajectory(self, xi0, U_flat):
        """
        Prediksi seluruh trajektori N langkah ke depan.
        U_flat: array panjang 2*Nc (flatten [TX0,TN0, TX1,TN1, ...])
        Returns: array (N+1) x 5
        """
        U = U_flat.reshape(self.Nc, 2)
        traj = np.zeros((self.N + 1, 5))
        traj[0] = xi0

        for k in range(self.N):
            # Control horizon: setelah Nc, tahan input terakhir
            uk = min(k, self.Nc - 1)
            mu = U[uk]
            traj[k + 1] = self._predict_one_step(traj[k], mu)

        return traj

    # ──────────────────────────────────────────────────────────────────────
    #  Cost Function
    # ──────────────────────────────────────────────────────────────────────

    def _compute_cost(self, U_flat, xi0, ref_traj, obstacles, alpha_path):
        U = U_flat.reshape(self.Nc, 2)
        traj = self._predict_trajectory(xi0, U_flat)

        cost = 0.0

        for k in range(self.N):
            xi  = traj[k]
            ref = ref_traj[k] if k < len(ref_traj) else ref_traj[-1]

            # State tracking cost
            e_state    = xi - ref
            e_state[2] = self._wrap_angle(e_state[2])
            cost += alpha_path * float(e_state @ self.Q @ e_state)

            # Input cost
            uk = min(k, self.Nc - 1)
            mu = U[uk]
            cost += float(mu @ self.R @ mu)

            # Input rate smoothness
            if uk > 0:
                du = U[uk] - U[uk - 1]
            else:
                du = U[uk] - self.u_prev
            cost += self.W_du * float(du @ du)

            # ── TAMBAHKAN: Penalti heading deviation dari path ────────────
            # Cegah LMPC belok terlalu jauh dari heading referensi
            psi_current = xi[2]
            psi_ref     = ref[2]
            psi_dev     = abs(self._wrap_angle(psi_current - psi_ref))
            # Penalti quadratic jika deviasi > 45 derajat
            if psi_dev > math.radians(45.0):
                cost += 50.0 * (psi_dev - math.radians(45.0)) ** 2

            # Obstacle avoidance cost
            obs_cost = 0.0
            for obs in obstacles:
                ox_k, oy_k, obs_r = self._predict_obstacle(obs, k)
                d_min  = obs_r + self.d_safe
                dx_obs = xi[0] - ox_k
                dy_obs = xi[1] - oy_k
                dist   = math.sqrt(dx_obs**2 + dy_obs**2 + 1e-6)
                if dist < d_min * 2.0:
                    penetration = max(0.0, d_min - dist)
                    obs_cost += (penetration / d_min) ** 2 + 10.0 * penetration
            cost += self.W_obs * obs_cost

        # Terminal cost
        xi_N   = traj[self.N]
        ref_N  = ref_traj[-1]
        e_N    = xi_N - ref_N
        e_N[2] = self._wrap_angle(e_N[2])
        cost  += alpha_path * float(e_N @ self.Qf @ e_N)

        return cost

    # ──────────────────────────────────────────────────────────────────────
    #  Obstacle Prediction (Constant Velocity)
    # ──────────────────────────────────────────────────────────────────────

    def _predict_obstacle(self, obs, k):
        """
        Prediksi posisi obstacle pada step k.
        obs = (ox, oy, obs_r, vx, vy)
        """
        ox, oy, obs_r, vx, vy = obs
        t = k * self.dt
        return ox + vx * t, oy + vy * t, obs_r

    # ──────────────────────────────────────────────────────────────────────
    #  Reference Trajectory Builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_ref_traj(self, state, wp_prev, wp_next, target_u, full_path, wp_idx):
        """
        Buat reference trajectory N langkah ke depan dari global path.
        Returns: array (N+1) x 5 [x, y, psi, u, r]
        """
        ref = np.zeros((self.N + 1, 5))
        x0, y0, psi0, _, u0, _, r0, _ = state  # unpack 8-DOF state
        ref[0] = [x0, y0, psi0, u0, r0]

        # Hitung alpha (heading path)
        dx_path = float(wp_next[0]) - float(wp_prev[0])
        dy_path = float(wp_next[1]) - float(wp_prev[1])
        if math.hypot(dx_path, dy_path) < 1e-6:
            alpha = psi0
        else:
            alpha = math.atan2(dy_path, dx_path)

        # Lookahead distance untuk ref trajectory
        lookahead_step = max(1, int(1.5 / (target_u * self.dt + 1e-6)))

        for k in range(1, self.N + 1):
            # Cari titik ref di global path (lookahead k steps)
            idx = min(wp_idx + k * lookahead_step, len(full_path) - 1)
            ref_pt = full_path[idx]

            # Heading ke titik referensi berikutnya
            if idx + 1 < len(full_path):
                next_pt = full_path[min(idx + 1, len(full_path) - 1)]
                psi_ref = math.atan2(next_pt[1] - ref_pt[1], next_pt[0] - ref_pt[0])
            else:
                psi_ref = alpha

            ref[k] = [ref_pt[0], ref_pt[1], psi_ref, target_u, 0.0]

        return ref

    # ──────────────────────────────────────────────────────────────────────
    #  Collision Check & Mode Decision
    # ──────────────────────────────────────────────────────────────────────

    def check_obstacles(self, state, obstacles):
        """
        Cek apakah ada obstacle dalam jangkauan LMPC.
        Returns:
          'CLEAR'       - tidak ada obstacle
          'WARN'        - obstacle terdeteksi, LMPC aktif
          'CRITICAL'    - obstacle sangat dekat, kurangi kecepatan
        """
        x, y = state[0], state[1]

        min_dist = float('inf')
        for obs in obstacles:
            ox, oy, obs_r, vx, vy = obs
            d = math.hypot(x - ox, y - oy) - obs_r
            if d < min_dist:
                min_dist = d

        if min_dist < self.d_safe * 1.2:
            return 'CRITICAL', min_dist
        elif min_dist < self.d_warn:
            return 'WARN', min_dist
        elif min_dist < self.d_exit:
            return 'MONITOR', min_dist
        else:
            return 'CLEAR', min_dist

    def _starboard_safe(self, state, obstacles):
        """
        Cek apakah sisi starboard (kanan) aman untuk menghindar.
        Sesuai referensi Kufoalor et al. — preferensi hindari ke kanan.
        Returns True jika starboard aman.
        """
        x, y, psi = state[0], state[1], state[2]
        # Titik 2m ke starboard
        x_star = x + 2.0 * math.cos(psi - math.pi / 2)
        y_star = y + 2.0 * math.sin(psi - math.pi / 2)

        for obs in obstacles:
            ox, oy, obs_r, vx, vy = obs
            d = math.hypot(x_star - ox, y_star - oy) - obs_r
            if d < self.d_safe:
                return False
        return True

    # ──────────────────────────────────────────────────────────────────────
    #  Main Solve
    # ──────────────────────────────────────────────────────────────────────

    def solve(self, state, wp_prev, wp_next, obstacles, target_u,
              full_path, wp_idx, dist_goal):

        x, y, psi, phi, u, v, r, p = state
        xi0 = np.array([x, y, psi, u, r])

        # ── Cek status obstacle ───────────────────────────────────────────
        obs_status, min_dist = self.check_obstacles(state, obstacles)

        if obs_status == 'CLEAR':
            self.mode   = 'GLOBAL'
            self.active = False
            return None, 'GLOBAL', {'obs_status': obs_status, 'min_dist': min_dist}

        self.active = True

        # ── SELALU LMPC_AVOID — tidak ada SLOW ────────────────────────────
        # Filosofi: belok keluar jalur, pertahankan kecepatan, kembali ke jalur
        # Ini meminimalkan waktu tempuh (tujuan TA)
        self.mode  = 'LMPC_AVOID'

        # Alpha path: seberapa kuat tracking vs avoidance
        # Saat sangat dekat → prioritas avoidance
        # Saat agak jauh   → seimbang tracking + avoidance
        if obs_status == 'CRITICAL':
            alpha_path = 0.2    # lebih prioritas avoidance
        else:
            alpha_path = 0.6    # seimbang

        # Pertahankan kecepatan target — TIDAK dikurangi
        u_target = target_u

        # ── Build Reference Trajectory ────────────────────────────────────
        ref_traj = self._build_ref_traj(state, wp_prev, wp_next,
                                         u_target, full_path, wp_idx)

        # ── Initial Guess ─────────────────────────────────────────────────
        if self._opt_u is not None and len(self._opt_u) == 2 * self.Nc:
            u0_guess = self._opt_u.copy()
        else:
            TX_ff = self._compute_TX_ff(u_target)
            u0_guess = np.tile([TX_ff, 0.0], self.Nc)

        # ── Bounds ────────────────────────────────────────────────────────
        bounds = []
        for k in range(self.Nc):
            bounds.append((self.TX_min, self.TX_max))
            bounds.append((self.TN_min, self.TN_max))

        # ── Optimization ──────────────────────────────────────────────────
        result = minimize(
            fun=self._compute_cost,
            x0=u0_guess,
            args=(xi0, ref_traj, obstacles, alpha_path),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4, 'disp': False}
        )

        if result.success or result.fun < 1e6:
            U_opt = result.x.reshape(self.Nc, 2)
            self._opt_u = result.x.copy()
        else:
            U_opt = u0_guess.reshape(self.Nc, 2)
            self._opt_u = None

        TX_opt, TN_opt = U_opt[0]

        # Rate Limiter
        TX_opt = float(np.clip(
            self.u_prev[0] + np.clip(TX_opt - self.u_prev[0],
                                     -self.dTX_max, self.dTX_max),
            self.TX_min, self.TX_max))
        TN_opt = float(np.clip(
            self.u_prev[1] + np.clip(TN_opt - self.u_prev[1],
                                     -self.dTN_max, self.dTN_max),
            self.TN_min, self.TN_max))

        # ── Sanity check ─────────────────────────────────────────────────
        psi_now = xi0[2]
        heading_err = abs(self._wrap_angle(
            math.atan2(wp_next[1]-wp_prev[1], wp_next[0]-wp_prev[0]) - psi_now
        ))
        if heading_err > math.radians(60.0):
            TN_opt *= 0.3

        # ── Assignment final (SATU kali saja) ────────────────────────────
        self.u_prev = np.array([TX_opt, TN_opt])
        Tcmd = np.array([TX_opt, 0.0, TN_opt, 0.0])

        info = {
            'obs_status' : obs_status,
            'min_dist'   : min_dist,
            'opt_cost'   : result.fun,
            'opt_success': result.success,
            'u_target'   : u_target,
            'alpha_path' : alpha_path,
        }

        return Tcmd, self.mode, info

    # ──────────────────────────────────────────────────────────────────────
    #  Helper Methods
    # ──────────────────────────────────────────────────────────────────────

    def _compute_TX_ff(self, u_ref):
        """Feedforward TX untuk mempertahankan kecepatan u_ref."""
        # Dari A2*u + A18*TX = 0 (steady state du=0)
        # TX = -A2*u / A18
        TX_ff = (-self.A2 * u_ref) / self.A18
        return float(np.clip(TX_ff, self.TX_min, self.TX_max))

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def reset(self):
        """Reset LMPC state (dipanggil saat kembali ke mode GLOBAL)."""
        self._opt_u = None
        self.u_prev = np.array([100.0, 0.0])
        self.active = False
        self.mode   = 'GLOBAL'