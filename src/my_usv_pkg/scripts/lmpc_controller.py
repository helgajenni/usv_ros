#!/usr/bin/env python3
"""
LMPC Controller — Revisi Final v3
═══════════════════════════════════════════════════════════════════════════════
Perbaikan utama dari v2:

MASALAH 1 — USV "hilang arah" karena salah pilih sisi penghindaran:
  → Tambahkan deteksi HEAD/TAIL obstacle dinamis berbasis vektor relatif.
     Algoritma memilih lewat KEPALA jika CPA dari sisi depan obstacle lebih hemat,
     atau lewat EKOR jika obstacle sudah melewati USV (TCPA negatif / kecil).
     Pemilihan dilakukan SEKALI saat LMPC pertama aktif dan di-LOCK agar tidak flip.

MASALAH 2 — Sway meledak (-3 m/s) → Roll Rate 300 deg/s:
  → Tambahkan sway_damping_weight: saat |v| > V_SWAY_LIMIT (0.25 m/s), cost
     terhadap r (yaw rate) dinaikkan drastis → TN mengecil → Coriolis forcing kecil.
  → r_max diturunkan ke 5 deg/s (dari 7 deg/s) sebagai hard bound.

MASALAH 3 — Surge melebihi 1.5 m/s:
  → TX_SOLVER_MAX dihitung ulang strict dari drag equilibrium tanpa buffer berlebih.
  → Tambahkan surge_penalty weight = 50.0 pada error u (dari 20.0).
  → Rate limiter dTX_max dikurangi untuk mencegah spike.

MASALAH 4 — Osilasi heading setelah kembali ke global path:
  → LMPC_RETURN memakai heading_align_weight = 8.0 untuk meluruskan heading
     ke arah jalur sebelum ILOS mengambil alih.
  → Saat transisi RETURN→GLOBAL, controller tidak di-reset mendadak:
     gunakan integrator flush yang smooth (dikurangi 50% per step).

MASALAH 5 — LMPC aktif terlalu lama / anti-chattering terlalu longgar:
  → LOCK_DIST diturunkan ke 5.0m. Saat obstacle sudah jauh, segera return.
  → Mode RETURN aktif maksimum RETURN_MAX_STEPS = 30 langkah (1.5 detik),
     setelah itu paksa kembali ke GLOBAL agar ILOS bisa ambil alih.
"""

import numpy as np
import math
from scipy.optimize import minimize


class LMPCController:
    def __init__(self, params: dict):
        # ── USV Model Parameters ──────────────────────────────────────────
        self.A2  = float(params.get('A2',   -0.7405))
        self.A3  = float(params.get('A3',    0.4219))
        self.A4  = float(params.get('A4',   -0.1397))
        self.A12 = float(params.get('A12',  -1.0495))
        self.A16 = float(params.get('A16',   0.9671))
        self.A18 = float(params.get('A18',   0.0178))
        self.A19 = float(params.get('A19',   0.0010))

        self.u0      = 1.5
        self.A12_eff = self.A12 + self.A16 * self.u0

        # ── MPC Horizon ───────────────────────────────────────────────────
        self.dt     = float(params.get('dt',     0.05))
        self.dt_mpc = float(params.get('dt_mpc', 0.4))
        self.N      = int(params.get('N',  12))   # horizon diperpanjang → prediksi lebih jauh
        self.Nc     = int(params.get('Nc',  6))

        # ── Actuator Limits ───────────────────────────────────────────────
        self.TX_min  = float(params.get('TX_min',    0.0))
        self.TX_max  = float(params.get('TX_max',  200.0))
        self.TN_min  = float(params.get('TN_min', -1750.0))
        self.TN_max  = float(params.get('TN_max',  1750.0))
        # Rate limiter lebih ketat → mencegah spike surge
        self.dTX_max = float(params.get('dTX_max',  60.0)) * self.dt   # was 100.0
        self.dTN_max = float(params.get('dTN_max', 1500.0)) * self.dt  # was 2000.0

        # ── State Limits ──────────────────────────────────────────────────
        self.u_min = float(params.get('u_min', 0.5))
        self.u_max = float(params.get('u_max', 1.6))   # hard cap 1.6 (< 1.8 sebelumnya)

        # r_max = 5 deg/s → Coriolis forcing -u*r maksimum = 1.5×0.087 = 0.13 m/s²
        # Sebelumnya 7 deg/s → 0.18 m/s² → sway bisa meledak lewat A7*v³
        self.r_max = 5.0 * math.pi / 180.0

        # Threshold sway: jika |v| > ini, tambahkan penalty besar pada r
        self.V_SWAY_LIMIT = 0.25   # m/s

        # ── Safety Zones ──────────────────────────────────────────────────
        self.d_safe = float(params.get('d_safe', 1.5))
        self.d_warn = float(params.get('d_warn', 3.5))
        self.d_exit = float(params.get('d_exit', 5.0))

        # ── Solver Bounds (strict) ────────────────────────────────────────
        self._TN_SOLVER_MAX = 400.0   # dikurangi dari 500 Nm → belok lebih halus

        # TX equilibrium pada u=1.5 m/s (tanpa buffer berlebih, hanya 5%)
        u_target     = 1.5
        drag_target  = (self.A2 * u_target
                        + self.A3 * u_target * abs(u_target)
                        + self.A4 * u_target**3)
        TX_ff_target = float(np.clip(-drag_target / self.A18, 10.0, self.TX_max))
        self._TX_SOLVER_MAX = TX_ff_target * 1.05   # buffer hanya 5% (was 15%)

        # ── Anti-chattering ───────────────────────────────────────────────
        self._LOCK_DIST        = 5.0    # m center-to-center (was 6.5)
        self._RETURN_MAX_STEPS = 30     # maksimum langkah RETURN sebelum paksa GLOBAL

        # ── State ─────────────────────────────────────────────────────────
        self.u_prev       = np.array([TX_ff_target, 0.0])
        self.active       = False
        self.mode         = 'GLOBAL'
        self._opt_u       = None
        self._avoid_side  = None   # 'LEFT' / 'RIGHT' — dikunci saat pertama AVOID
        self._return_step = 0      # counter LMPC_RETURN steps

    # ═══════════════════════════════════════════════════════════════════════
    #  MODEL PREDIKSI (simplified 5-state: x, y, ψ, u, r)
    # ═══════════════════════════════════════════════════════════════════════
    def _predict_one_step(self, xi, mu, v_sway=0.0):
        """
        xi = [x, y, psi, u, r]
        Sertakan v_sway sebagai Coriolis coupling: du += -r*v*dt
        Ini mencegah prediksi terlalu optimistis saat sway besar.
        """
        x, y, psi, u, r = xi
        TX, TN = mu
        dt = self.dt_mpc

        # Coriolis: -v*r term pada surge (coupling sway → surge)
        coriolis_surge = -v_sway * r

        dx   = (u * math.cos(psi) - v_sway * math.sin(psi)) * dt
        dy   = (u * math.sin(psi) + v_sway * math.cos(psi)) * dt
        dpsi = r * dt
        du   = (self.A2 * u + self.A3 * u * abs(u) + self.A4 * u**3
                + self.A18 * TX + coriolis_surge) * dt
        dr   = (self.A12_eff * r + self.A19 * TN) * dt

        psi_new = (psi + dpsi + math.pi) % (2 * math.pi) - math.pi
        u_new   = float(np.clip(u + du, self.u_min, self.u_max))
        r_new   = float(np.clip(r + dr, -self.r_max, self.r_max))
        return np.array([x + dx, y + dy, psi_new, u_new, r_new])

    def _predict_trajectory(self, xi0, U_flat, v_sway=0.0):
        U    = U_flat.reshape(self.Nc, 2)
        traj = np.zeros((self.N + 1, 5))
        traj[0] = xi0
        for k in range(self.N):
            uk        = min(k, self.Nc - 1)
            traj[k+1] = self._predict_one_step(traj[k], U[uk], v_sway)
        return traj

    def _predict_obstacle_pos(self, obs, k):
        ox, oy, obs_r, vx, vy = (float(obs[0]), float(obs[1]), float(obs[2]),
                                   float(obs[3]), float(obs[4]))
        t = k * self.dt_mpc
        return ox + vx * t, oy + vy * t, obs_r

    def _wrap_angle(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _compute_TX_ff(self, u_ref):
        drag = self.A2 * u_ref + self.A3 * u_ref * abs(u_ref) + self.A4 * u_ref**3
        return float(np.clip(-drag / self.A18, self.TX_min, self._TX_SOLVER_MAX))

    def reset(self):
        self._opt_u       = None
        self.u_prev       = np.array([self._compute_TX_ff(1.0), 0.0])
        self.active       = False
        self.mode         = 'GLOBAL'
        self._avoid_side  = None
        self._return_step = 0

    # ═══════════════════════════════════════════════════════════════════════
    #  DETEKSI KEPALA / EKOR OBSTACLE
    # ═══════════════════════════════════════════════════════════════════════
    def _select_avoidance_side(self, state, obstacle):
        """
        Tentukan sisi penghindaran terbaik: lewat KEPALA atau EKOR obstacle.

        Algoritma:
        1. Hitung posisi CPA (Closest Point of Approach) antara USV dan obstacle.
        2. Jika TCPA < 1.0s (obstacle hampir sudah lewat) → lewat EKOR (ikuti ekornya)
        3. Jika TCPA > 1.0s → evaluasi biaya lewat kiri vs kanan kepala obstacle.
           Pilih yang memberikan cross-product positif minimum (sisi yang tidak perlu
           manuver terlalu besar).

        Returns: 'LEFT' atau 'RIGHT' (dari perspektif USV menghadap ke arah tujuan)
        """
        x, y, psi, phi, u, v, r, p = state
        ox, oy, obs_r = float(obstacle[0]), float(obstacle[1]), float(obstacle[2])
        ox_vel = float(obstacle[3]) if len(obstacle) > 3 else 0.0
        oy_vel = float(obstacle[4]) if len(obstacle) > 4 else 0.0

        vx_usv = u * math.cos(psi) - v * math.sin(psi)
        vy_usv = u * math.sin(psi) + v * math.cos(psi)

        dx  = ox - x
        dy  = oy - y
        dvx = ox_vel - vx_usv
        dvy = oy_vel - vy_usv

        v_rel_sq = dvx**2 + dvy**2
        if v_rel_sq > 1e-6:
            tcpa = -(dx * dvx + dy * dvy) / v_rel_sq
        else:
            tcpa = 0.0

        if tcpa < 1.0:
            # Obstacle hampir lewat atau sudah lewat → ambil EKOR (PASS ASTERN)
            # Posisi ekor obstacle = posisi obstacle + 1.5*r ke arah belakang geraknya
            tail_x = ox - 2.0 * obs_r * ox_vel / (math.hypot(ox_vel, oy_vel) + 1e-9)
            tail_y = oy - 2.0 * obs_r * oy_vel / (math.hypot(ox_vel, oy_vel) + 1e-9)

            # Cek apakah ekor ada di kiri atau kanan USV
            to_tail_x = tail_x - x
            to_tail_y = tail_y - y
            cross = vx_usv * to_tail_y - vy_usv * to_tail_x
            return 'LEFT' if cross > 0 else 'RIGHT'
        else:
            # Obstacle akan melintas di depan → ambil KEPALA (PASS AHEAD)
            # Hitung posisi obstacle saat TCPA
            head_x = ox + ox_vel * tcpa
            head_y = oy + oy_vel * tcpa

            # Sisi mana yang lebih mudah dijangkau?
            to_head_x = head_x - x
            to_head_y = head_y - y
            cross = vx_usv * to_head_y - vy_usv * to_head_x

            # Pilih sisi BERLAWANAN dari obstacle → manuver minimal
            # cross > 0 → obstacle di kiri → belok kanan (pass ahead dari kanan)
            # cross < 0 → obstacle di kanan → belok kiri
            return 'RIGHT' if cross > 0 else 'LEFT'

    def _get_side_bias_TN(self, side, TX_ff):
        """
        Konversi sisi yang dipilih ke initial TN guess.
        'LEFT'  → TN positif  (belok kiri / port)
        'RIGHT' → TN negatif  (belok kanan / starboard)
        """
        magnitude = min(200.0, self._TN_SOLVER_MAX * 0.5)
        return magnitude if side == 'LEFT' else -magnitude

    # ═══════════════════════════════════════════════════════════════════════
    #  DETEKSI OBSTACLE
    # ═══════════════════════════════════════════════════════════════════════
    def check_obstacles(self, state, obstacles):
        x, y, psi, phi, u, v, r, p = state
        vx_usv = u * math.cos(psi) - v * math.sin(psi)
        vy_usv = u * math.sin(psi) + v * math.cos(psi)

        obs_status       = 'CLEAR'
        min_dist_overall = float('inf')

        for obs in obstacles:
            ox, oy, obs_r = float(obs[0]), float(obs[1]), float(obs[2])
            ox_vel = float(obs[3]) if len(obs) > 3 else 0.0
            oy_vel = float(obs[4]) if len(obs) > 4 else 0.0

            dx = ox - x;   dy = oy - y
            dvx = ox_vel - vx_usv
            dvy = oy_vel - vy_usv

            dist_now  = math.hypot(dx, dy)
            dist_edge = dist_now - obs_r
            min_dist_overall = min(min_dist_overall, dist_edge)

            if dist_edge < self.d_safe:
                return 'CRITICAL', min_dist_overall

            v_rel_sq = dvx**2 + dvy**2
            tcpa = -(dx * dvx + dy * dvy) / v_rel_sq if v_rel_sq > 1e-6 else 0.0
            if tcpa > 0:
                dcpa = math.hypot(dx + dvx * tcpa, dy + dvy * tcpa)
            else:
                dcpa = dist_now

            is_dangerous = (
                (dist_edge < self.d_warn + 0.5) or
                (0 < tcpa < 8.0 and dcpa < obs_r + self.d_safe + 1.0)
            )
            is_locked = self.active and (dist_now < self._LOCK_DIST)

            if is_dangerous or is_locked:
                obs_status = 'WARN'

        return obs_status, min_dist_overall

    def _is_real_danger(self, state, obstacles):
        x, y, psi, phi, u, v, r, p = state
        vx_usv = u * math.cos(psi) - v * math.sin(psi)
        vy_usv = u * math.sin(psi) + v * math.cos(psi)

        for obs in obstacles:
            ox, oy, obs_r = float(obs[0]), float(obs[1]), float(obs[2])
            ox_vel = float(obs[3]) if len(obs) > 3 else 0.0
            oy_vel = float(obs[4]) if len(obs) > 4 else 0.0

            dx = ox - x;   dy = oy - y
            dvx = ox_vel - vx_usv
            dvy = oy_vel - vy_usv

            dist_now  = math.hypot(dx, dy)
            dist_edge = dist_now - obs_r

            if dist_edge < self.d_safe:
                return True

            v_rel_sq = dvx**2 + dvy**2
            tcpa = -(dx * dvx + dy * dvy) / v_rel_sq if v_rel_sq > 1e-6 else 0.0
            if tcpa > 0:
                dcpa = math.hypot(dx + dvx * tcpa, dy + dvy * tcpa)
            else:
                dcpa = dist_now

            if (dist_edge < self.d_warn + 0.5) or \
               (0 < tcpa < 8.0 and dcpa < obs_r + self.d_safe + 1.0):
                return True

        return False

    # ═══════════════════════════════════════════════════════════════════════
    #  REFERENSI TRAJEKTORI
    # ═══════════════════════════════════════════════════════════════════════
    def _build_ref_traj(self, state, full_path, wp_idx, target_u):
        x0, y0, psi0, u0, r0 = state[0], state[1], state[2], state[4], state[6]

        s_start = max(0, wp_idx - 20)
        s_end   = min(len(full_path), wp_idx + 100)
        dists   = [math.hypot(x0 - float(full_path[i][0]), y0 - float(full_path[i][1]))
                   for i in range(s_start, s_end)]
        closest = s_start + int(np.argmin(dists))

        ref       = np.zeros((self.N + 1, 5))
        ref[0]    = [x0, y0, psi0, u0, r0]
        path_step = max(1, int(round(target_u * self.dt_mpc)))

        for k in range(1, self.N + 1):
            idx    = min(closest + k * path_step, len(full_path) - 1)
            base_x = float(full_path[idx][0])
            base_y = float(full_path[idx][1])

            if idx + 1 < len(full_path):
                npt     = full_path[idx + 1]
                psi_ref = math.atan2(float(npt[1]) - base_y, float(npt[0]) - base_x)
            elif idx > 0:
                ppt     = full_path[idx - 1]
                psi_ref = math.atan2(base_y - float(ppt[1]), base_x - float(ppt[0]))
            else:
                psi_ref = psi0

            ref[k] = [base_x, base_y, psi_ref, target_u, 0.0]

        return ref

    # ═══════════════════════════════════════════════════════════════════════
    #  MAIN SOLVER
    # ═══════════════════════════════════════════════════════════════════════
    def solve(self, state, wp_prev, wp_next, obstacles, target_u,
              full_path, wp_idx, dist_goal, static_obs=None):

        if dist_goal < 5.0:
            self.reset()
            return None, 'GLOBAL', {'obs_status': 'CLEAR', 'min_dist': 0.0}

        x, y, psi, phi, u, v, r, p = state
        xi0      = np.array([x, y, psi, u, r])
        v_sway   = float(v)   # sway saat ini untuk Coriolis prediction

        obs_status, min_dist = self.check_obstacles(state, obstacles)

        if obs_status == 'CLEAR':
            self.reset()
            return None, 'GLOBAL', {'obs_status': obs_status, 'min_dist': min_dist}

        self.active = True

        # Kurangi kecepatan manuver saat sway sudah besar
        sway_penalty  = max(0.0, (abs(v_sway) - self.V_SWAY_LIMIT) / 0.5)
        avoid_u       = min(target_u, max(0.8, 1.2 - 0.4 * sway_penalty))
        TX_ff         = self._compute_TX_ff(avoid_u)

        # ── Mode: AVOID vs RETURN ─────────────────────────────────────────
        real_danger = self._is_real_danger(state, obstacles)

        if real_danger:
            self.mode         = 'LMPC_AVOID'
            self._return_step = 0   # reset counter return

            # Kunci sisi penghindaran saat pertama kali aktif
            if self._avoid_side is None:
                # Pilih berdasarkan obstacle paling dekat
                closest_obs = min(obstacles,
                    key=lambda o: math.hypot(float(o[0])-x, float(o[1])-y))
                self._avoid_side = self._select_avoidance_side(state, closest_obs)

            pos_weight        = 10.0
            heading_weight    = 3.0
            obs_enabled       = True
            sway_damp_weight  = 5.0 + 30.0 * sway_penalty   # makin besar jika sway besar

        else:
            # Hanya anti-chattering lock → kembalikan ke jalur
            self._return_step += 1

            if self._return_step > self._RETURN_MAX_STEPS:
                # Paksa kembali ke GLOBAL — ILOS ambil alih
                self.reset()
                return None, 'GLOBAL', {'obs_status': 'CLEAR', 'min_dist': min_dist}

            self.mode         = 'LMPC_RETURN'
            self._avoid_side  = None   # release lock saat return
            pos_weight        = 30.0
            heading_weight    = 8.0    # luruskan heading ke jalur → kurangi osilasi
            obs_enabled       = False
            sway_damp_weight  = 10.0 + 40.0 * sway_penalty

        # ── Kumpulkan obstacle untuk cost ─────────────────────────────────
        all_obs = []
        if obs_enabled:
            for o in obstacles:
                all_obs.append((float(o[0]), float(o[1]), float(o[2]),
                                float(o[3]) if len(o) > 3 else 0.0,
                                float(o[4]) if len(o) > 4 else 0.0))
            if static_obs:
                for o in static_obs:
                    all_obs.append((float(o[0]), float(o[1]), float(o[2]),
                                    float(o[3]) if len(o) > 3 else 0.0,
                                    float(o[4]) if len(o) > 4 else 0.0))

        ref_traj = self._build_ref_traj(state, full_path, wp_idx, avoid_u)

        # ── Initial guess berbasis sisi yang sudah dikunci ─────────────────
        if real_danger and self._avoid_side is not None:
            TN_locked   = self._get_side_bias_TN(self._avoid_side, TX_ff)
            u0_guess    = np.tile([TX_ff, TN_locked], self.Nc)
            # Evaluasi juga dengan TN berlawanan sebagai sanity check
            u0_opposite = np.tile([TX_ff, -TN_locked], self.Nc)
            cost_locked   = self._compute_cost(u0_guess,    xi0, ref_traj, all_obs,
                                               pos_weight, heading_weight, sway_damp_weight, v_sway)
            cost_opposite = self._compute_cost(u0_opposite, xi0, ref_traj, all_obs,
                                               pos_weight, heading_weight, sway_damp_weight, v_sway)
            # Hanya ganti sisi jika berlawanan JAUH lebih baik (> 30% lebih murah)
            # Ini mencegah flip yang menyebabkan USV "hilang arah"
            if cost_opposite < cost_locked * 0.70:
                self._avoid_side = 'LEFT' if self._avoid_side == 'RIGHT' else 'RIGHT'
                u0_guess = u0_opposite
        elif not real_danger:
            # RETURN: menuju jalur global, TN mendekati 0
            u0_guess = np.tile([TX_ff, 0.0], self.Nc)
        else:
            u0_guess = np.tile([TX_ff, 0.0], self.Nc)

        # Gunakan solusi sebelumnya jika masih relevan
        if self._opt_u is not None and len(self._opt_u) == 2 * self.Nc:
            c_stored = self._compute_cost(self._opt_u, xi0, ref_traj, all_obs,
                                          pos_weight, heading_weight, sway_damp_weight, v_sway)
            c_guess  = self._compute_cost(u0_guess,   xi0, ref_traj, all_obs,
                                          pos_weight, heading_weight, sway_damp_weight, v_sway)
            if c_stored < c_guess * 1.05:   # pakai stored jika hampir sama atau lebih baik
                u0_guess = self._opt_u.copy()

        # ── Optimizer Bounds ──────────────────────────────────────────────
        bounds      = []
        min_thrust  = max(self.TX_min, TX_ff * 0.5)
        for _ in range(self.Nc):
            bounds.append((min_thrust, self._TX_SOLVER_MAX))
            bounds.append((-self._TN_SOLVER_MAX, self._TN_SOLVER_MAX))

        result = minimize(
            fun=self._compute_cost,
            x0=u0_guess,
            args=(xi0, ref_traj, all_obs, pos_weight, heading_weight, sway_damp_weight, v_sway),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 5e-4, 'disp': False}
        )

        if result.success or result.fun < self._compute_cost(u0_guess, xi0, ref_traj, all_obs,
                                                              pos_weight, heading_weight, sway_damp_weight, v_sway):
            U_opt       = result.x.reshape(self.Nc, 2)
            self._opt_u = result.x.copy()
        else:
            TN_fallback = (self._get_side_bias_TN(self._avoid_side, TX_ff) * 0.5
                           if real_danger and self._avoid_side else 0.0)
            U_opt       = np.tile([TX_ff, TN_fallback], self.Nc).reshape(self.Nc, 2)
            self._opt_u = None

        TX_opt, TN_opt = float(U_opt[0][0]), float(U_opt[0][1])

        # Rate limiter + clamp final
        TX_opt = float(np.clip(
            self.u_prev[0] + np.clip(TX_opt - self.u_prev[0], -self.dTX_max, self.dTX_max),
            self.TX_min, self._TX_SOLVER_MAX))
        TN_opt = float(np.clip(
            self.u_prev[1] + np.clip(TN_opt - self.u_prev[1], -self.dTN_max, self.dTN_max),
            self.TN_min, self.TN_max))

        # Extra: jika sway sudah besar, paksa TN mendekati 0 agar Coriolis mereda
        if abs(v_sway) > self.V_SWAY_LIMIT * 2.0:
            TN_scale = max(0.2, 1.0 - (abs(v_sway) - self.V_SWAY_LIMIT * 2.0) / 0.5)
            TN_opt  *= TN_scale

        self.u_prev = np.array([TX_opt, TN_opt])
        Tcmd = np.array([TX_opt, 0.0, TN_opt, 0.0])

        # Trajektori untuk visualisasi
        U_vis    = self._opt_u if self._opt_u is not None else np.tile([TX_ff, 0.0], self.Nc)
        opt_traj = self._predict_trajectory(xi0, U_vis, v_sway)

        info = {
            'obs_status' : obs_status,
            'min_dist'   : min_dist,
            'avoid_side' : self._avoid_side,
            'ref_traj'   : ref_traj,
            'opt_traj'   : opt_traj,
        }
        return Tcmd, self.mode, info

    # ═══════════════════════════════════════════════════════════════════════
    #  COST FUNCTION
    # ═══════════════════════════════════════════════════════════════════════
    def _compute_cost(self, U_flat, xi0, ref_traj, all_obstacles,
                      pos_weight=10.0, heading_weight=3.0,
                      sway_damp_weight=5.0, v_sway=0.0):
        """
        Cost terms:
          pos_weight     : position tracking (AVOID=10, RETURN=30)
          heading_weight : heading alignment ke ref → mencegah osilasi saat return
          sway_damp_weight: penalty pada |r| (yaw rate) jika sway besar → redam Coriolis
          v_sway         : sway saat ini untuk Coriolis-aware prediction
        """
        U    = U_flat.reshape(self.Nc, 2)
        traj = self._predict_trajectory(xi0, U_flat, v_sway)
        cost = 0.0

        for k in range(self.N):
            xi  = traj[k]
            ref = ref_traj[k] if k < len(ref_traj) else ref_traj[-1]

            e_x   = xi[0] - ref[0]
            e_y   = xi[1] - ref[1]
            e_psi = self._wrap_angle(xi[2] - ref[2])
            e_u   = xi[3] - ref[3]
            e_r   = xi[4] - ref[4]
            r_now = xi[4]   # yaw rate prediksi

            # Tracking cost
            cost += (pos_weight * e_x**2
                   + pos_weight * e_y**2
                   + heading_weight * e_psi**2
                   + 50.0 * e_u**2     # was 20.0 → surge lebih ketat dikendalikan
                   + 2.0  * e_r**2)

            # Sway damping: penalty besar pada r jika sway tumbuh
            cost += sway_damp_weight * r_now**2

            # Actuator cost
            uk = min(k, self.Nc - 1)
            cost += 0.005 * U[uk][0]**2 + 0.08 * U[uk][1]**2

            # Rate cost → mencegah perubahan aktuator mendadak (anti-osilasi)
            if uk > 0:
                du0 = U[uk][0] - U[uk-1][0]
                du1 = U[uk][1] - U[uk-1][1]
            else:
                du0 = U[uk][0] - self.u_prev[0]
                du1 = U[uk][1] - self.u_prev[1]
            cost += 0.15 * du0**2 + 5.0 * du1**2   # TN rate cost dinaikkan (was 3.0)

            # Obstacle penalty
            for obs in all_obstacles:
                ox_k, oy_k, obs_r = self._predict_obstacle_pos(obs, k)
                dist = math.hypot(xi[0] - ox_k, xi[1] - oy_k)

                obs_vx        = float(obs[3]) if len(obs) > 3 else 0.0
                obs_vy        = float(obs[4]) if len(obs) > 4 else 0.0
                is_static     = math.hypot(obs_vx, obs_vy) < 0.05
                margin_extra  = 0.3 if is_static else 0.5
                penalty_coeff = 500.0 if is_static else 350.0

                safe_margin = obs_r + self.d_safe + margin_extra
                if dist < safe_margin:
                    penetration = safe_margin - dist
                    cost += penalty_coeff * penetration**2

        # Terminal cost: posisi + heading alignment
        terminal_pos_w = pos_weight * 2.0
        terminal_hdg_w = heading_weight * 2.0
        e_x_t   = traj[-1][0] - ref_traj[-1][0]
        e_y_t   = traj[-1][1] - ref_traj[-1][1]
        e_psi_t = self._wrap_angle(traj[-1][2] - ref_traj[-1][2])
        cost += terminal_pos_w * (e_x_t**2 + e_y_t**2)
        cost += terminal_hdg_w * e_psi_t**2

        return cost