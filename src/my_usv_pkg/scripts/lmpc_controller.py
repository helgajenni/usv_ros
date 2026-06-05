#!/usr/bin/env python3
"""
LMPC Controller — v11  (Clean Rewrite, Dual-Obstacle Aware)
═══════════════════════════════════════════════════════════════════════════════
Referensi utama:
  - Kufoalor et al. (2020), J. Field Robotics — MPC collision avoidance USV
  - Eriksen & Breivik (2017), IEEE CCTA — MPC path following + obstacle avoidance
  - Fossen (2011), Handbook of Marine Craft Hydrodynamics — model prediksi

Konfigurasi obstacle:
  DO1: x0=13.75, y0=5.5,  vy=+0.62  → naik dari bawah, hindari ke KIRI  (TN>0)
  DO2: x0=28.5,  y0=28.0, vy=-0.27  → turun dari atas, hindari ke KANAN (TN<0)

Bug yang diperbaiki dari v10:
  [B1] _predict_clearance_side: TN_sim=80 terlalu kecil (dr≈0.024 rad/step)
       → diganti dengan TN_sim yang proporsional ke A19 agar simulasi realistis
  [B2] _active_obs_has_passed: reset terlalu agresif untuk obs lambat (vy=-0.27)
       → kondisi "passed" diperketat: TCPA < -3.0 DAN dist_edge > 6.0 DAN
         obs sudah benar-benar di belakang heading USV
  [B3] RETURN mode tidak memasukkan dynamic obs ke cost function
       → obs dinamis yang masih dalam radius 12m tetap masuk all_obs saat RETURN
  [B4] _chosen_side tidak pernah di-invalidate setelah obs pertama selesai
       → reset _chosen_side saat _active_obs_idx berubah (pindah ke obs baru)
  [B5] force_sign tidak konsisten: kadang True kadang False untuk obs yang sama
       → force_sign=True SELALU saat LMPC_AVOID agar solver tidak drift ke TN=0

Prinsip desain v11:
  - Ref trajectory SELALU jalur global (avoidance murni via obstacle penalty)
  - Deteksi: d_warn=2.5m dari tepi (bukan 1m) → cukup waktu reaksi
  - Arc avoidance: margin 1.2m di cost function → lateral deviasi ~1.5-2m
  - Return ke jalur: cepat via pos_weight tinggi + dyn obs tetap di cost
  - _chosen_side: satu obs → satu sisi, tidak flip-flop, reset bersih antar obs
"""

import numpy as np
import math
from scipy.optimize import minimize


class LMPCController:
    def __init__(self, params: dict):
        # ── Model koefisien (Taylor 4-DOF) ───────────────────────────────
        self.A1  = float(params.get('A1',   1.5066))
        self.A2  = float(params.get('A2',  -0.7405))
        self.A3  = float(params.get('A3',   0.4219))
        self.A4  = float(params.get('A4',  -0.1397))
        self.A5  = float(params.get('A5',  -0.1464))
        self.A6  = float(params.get('A6',  -3.1952))
        self.A7  = float(params.get('A7',   4.1189))
        self.A12 = float(params.get('A12', -1.0495))
        self.A13 = float(params.get('A13',  1.4038))
        self.A14 = float(params.get('A14', -2.0764))
        self.A16 = float(params.get('A16',  0.9671))
        self.A18 = float(params.get('A18',  0.0178))
        self.A19 = float(params.get('A19',  0.0010))

        # ── Horizon & sampling ────────────────────────────────────────────
        self.dt     = float(params.get('dt',     0.05))
        self.dt_mpc = float(params.get('dt_mpc', 0.3))
        self.N      = int(params.get('N',  12))
        self.Nc     = int(params.get('Nc',  6))

        # ── Aktuator constraints ──────────────────────────────────────────
        self.TX_min  = float(params.get('TX_min',    0.0))
        self.TX_max  = float(params.get('TX_max',  200.0))
        self.TN_min  = float(params.get('TN_min', -1750.0))
        self.TN_max  = float(params.get('TN_max',  1750.0))
        # Rate limiter dalam satuan per-step (bukan per-detik)
        self.dTX_max = float(params.get('dTX_max',  60.0))  * self.dt
        self.dTN_max = float(params.get('dTN_max', 5000.0)) * self.dt  # dinaikkan: ramp TN lebih cepat

        # ── State constraints ─────────────────────────────────────────────
        self.u_min = float(params.get('u_min', 0.8))
        self.u_max = float(params.get('u_max', 1.8))
        self.r_max = 0.3  # rad/s — dibatasi agar roll tidak meledak

        # ── Obstacle parameters ───────────────────────────────────────────
        self.d_safe  = float(params.get('d_safe', 1.2))   # clearance minimum dari tepi
        self.d_warn  = float(params.get('d_warn', 3.5))   # jarak aktivasi LMPC (direct) — 1-2m dari tepi
        self.d_exit  = float(params.get('d_exit', 3.5))   # jarak deaktivasi LMPC
        self._d_gate = float(params.get('d_gate', 5.0))   # radius TCPA check — dikecilkan dari 5.5

        # ── Feed-forward TX nominal (berdasarkan drag di u=1.5 m/s) ──────
        u_target = 1.5
        drag_ff  = (self.A2*u_target + self.A3*u_target*abs(u_target)
                    + self.A4*u_target**3)
        self._TX_FF_NOMINAL = float(np.clip(-drag_ff / self.A18, 10.0, self.TX_max))
        # Solver hanya diberi ruang sedikit di atas FF agar tidak overspeed
        self._TX_SOLVER_MAX = self._TX_FF_NOMINAL * 1.05

        # [FIX B1] TN_sim untuk simulasi clearance — proporsional ke A19
        # dr = A19 * TN_sim * dt_mpc → target dr ≈ 0.15 rad/s (belok nyata)
        # TN_sim = 0.15 / (A19 * dt_mpc) = 0.15 / (0.001 * 0.3) = 500
        # Tapi clip ke TN_SOLVER_MAX agar konsisten dengan bounds solver
        self._TN_SOLVER_MAX = 400.0  # batas solver (lebih kecil dari TN_max fisik)
        _dr_target  = 0.12           # rad/s belok nyata dalam simulasi clearance
        self._TN_SIM = float(np.clip(
            _dr_target / max(self.A19 * self.dt_mpc, 1e-6),
            50.0, self._TN_SOLVER_MAX))

        # ── Batas LMPC aktif ──────────────────────────────────────────────
        self._RETURN_MAX_STEPS = 120   # ~6s di dt=0.05
        self._MAX_TOTAL_STEPS  = 350   # hard timeout

        # ── State internal ────────────────────────────────────────────────
        self.u_prev          = np.array([self._TX_FF_NOMINAL, 0.0])
        self.active          = False
        self.mode            = 'GLOBAL'
        self._opt_u          = None
        self._return_step    = 0
        self._total_steps    = 0
        self._active_obs_idx = None
        self._chosen_side    = None   # +1=kiri(port,TN>0), -1=kanan(stbd,TN<0)
        self._prev_obs_idx   = None   # [FIX B4] deteksi pergantian obs

    # ════════════════════════════════════════════════════════════════════════
    #  MODEL PREDIKSI
    # ════════════════════════════════════════════════════════════════════════

    def _predict_one_step(self, xi, mu, v_sway=0.0):
        """Satu langkah Euler untuk [x, y, psi, u, r] + v_sway terpisah."""
        x, y, psi, u, r = xi
        TX, TN = mu
        dt = self.dt_mpc

        dx  = (u * math.cos(psi) - v_sway * math.sin(psi)) * dt
        dy  = (u * math.sin(psi) + v_sway * math.cos(psi)) * dt
        dpsi = r * dt

        du = (self.A2*u + self.A3*u*abs(u) + self.A4*u**3
              + self.A18*TX - v_sway*r) * dt

        A12_eff = self.A12 + self.A16 * abs(u)
        dr = (A12_eff*r + self.A13*r*abs(r) + self.A14*r**3
              + self.A19*TN) * dt

        dv = (self.A5*v_sway + self.A6*v_sway*abs(v_sway) + self.A7*v_sway**3
              - (1.0/self.A1)*u*r) * dt

        v_new   = float(np.clip(v_sway + dv, -2.5, 2.5))
        psi_new = (psi + dpsi + math.pi) % (2*math.pi) - math.pi
        u_new   = float(np.clip(u + du, self.u_min, self.u_max))
        r_new   = float(np.clip(r + dr, -self.r_max, self.r_max))

        return np.array([x+dx, y+dy, psi_new, u_new, r_new]), v_new

    def _predict_trajectory(self, xi0, U_flat, v_sway=0.0):
        """Rollout N langkah dari xi0 dengan input sequence U_flat."""
        traj = np.zeros((self.N+1, 6))
        traj[0, :5] = xi0
        traj[0,  5] = v_sway
        for k in range(self.N):
            uk = min(k, self.Nc-1)
            mu = U_flat.reshape(self.Nc, 2)[uk]
            xi_new, v_k = self._predict_one_step(traj[k, :5], mu, traj[k, 5])
            traj[k+1, :5] = xi_new
            traj[k+1,  5] = v_k
        return traj

    def _predict_obs_pos(self, obs, k):
        """Posisi obstacle di langkah prediksi ke-k."""
        ox  = float(obs[0]) + (float(obs[3]) if len(obs) > 3 else 0.0) * k * self.dt_mpc
        oy  = float(obs[1]) + (float(obs[4]) if len(obs) > 4 else 0.0) * k * self.dt_mpc
        return ox, oy, float(obs[2])

    def _wrap(self, a):
        return (a + math.pi) % (2*math.pi) - math.pi

    def _compute_TX_ff(self, u_ref):
        drag = self.A2*u_ref + self.A3*u_ref*abs(u_ref) + self.A4*u_ref**3
        return float(np.clip(-drag / self.A18, self.TX_min, self._TX_SOLVER_MAX))

    # ════════════════════════════════════════════════════════════════════════
    #  RESET
    # ════════════════════════════════════════════════════════════════════════

    def reset(self):
        self._opt_u          = None
        self.u_prev          = np.array([self._TX_FF_NOMINAL, 0.0])
        self.active          = False
        self.mode            = 'GLOBAL'
        self._return_step    = 0
        self._total_steps    = 0
        self._active_obs_idx = None
        self._chosen_side    = None
        self._prev_obs_idx   = None

    # ════════════════════════════════════════════════════════════════════════
    #  CHECK OBSTACLES
    # ════════════════════════════════════════════════════════════════════════

    def check_obstacles(self, state, obstacles):
        """
        Return (obs_status, min_dist_edge).
        obs_status: 'CLEAR' | 'WARN' | 'CRITICAL'
        """
        x, y, psi, phi, u, v, r, p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)

        obs_status       = 'CLEAR'
        min_dist_overall = float('inf')
        closest_threat   = None
        closest_dist     = float('inf')

        for idx, obs in enumerate(obstacles):
            ox    = float(obs[0]); oy    = float(obs[1]); obs_r = float(obs[2])
            ov_x  = float(obs[3]) if len(obs) > 3 else 0.0
            ov_y  = float(obs[4]) if len(obs) > 4 else 0.0

            dx = ox - x; dy = oy - y
            dvx = ov_x - vx_usv; dvy = ov_y - vy_usv

            dist_now  = math.hypot(dx, dy)
            dist_edge = dist_now - obs_r
            min_dist_overall = min(min_dist_overall, dist_edge)

            # Collision sudah terjadi
            if dist_edge < self.d_safe:
                return 'CRITICAL', min_dist_overall

            # TCPA / DCPA
            v_rel_sq = dvx**2 + dvy**2
            if v_rel_sq > 1e-6:
                tcpa = -(dx*dvx + dy*dvy) / v_rel_sq
                dcpa = math.hypot(dx + dvx*tcpa, dy + dvy*tcpa) - obs_r if tcpa > 0 else dist_edge
            else:
                tcpa = 0.0
                dcpa = dist_edge

            # Tiga kondisi aktivasi:
            is_direct = dist_edge < self.d_warn
            is_tcpa   = (dist_edge < self._d_gate
                         and 0 < tcpa < 5.0
                         and dcpa < self.d_safe + 0.8)
            # Lock: jaga LMPC aktif selama obs masih dekat & belum benar-benar lewat
            is_locked = (self.active
                         and self._active_obs_idx == idx
                         and dist_now < 6.0
                         and tcpa > -10.0)

            if is_direct or is_tcpa or is_locked:
                obs_status = 'WARN'
                if dist_edge < closest_dist:
                    closest_dist   = dist_edge
                    closest_threat = idx

        # Set active obs hanya jika belum ada (tidak ganti di tengah avoidance)
        if obs_status == 'WARN' and self._active_obs_idx is None:
            self._active_obs_idx = closest_threat

        return obs_status, min_dist_overall

    def _is_real_danger(self, state, obstacles):
        """True jika ada obs yang benar-benar mengancam (bukan hanya dalam gate)."""
        x, y, psi, phi, u, v, r, p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)

        for obs in obstacles:
            ox = float(obs[0]); oy = float(obs[1]); obs_r = float(obs[2])
            ov_x = float(obs[3]) if len(obs) > 3 else 0.0
            ov_y = float(obs[4]) if len(obs) > 4 else 0.0

            dx = ox - x; dy = oy - y
            dvx = ov_x - vx_usv; dvy = ov_y - vy_usv
            dist_edge = math.hypot(dx, dy) - obs_r

            if dist_edge < self.d_safe:
                return True
            if dist_edge < self.d_warn:
                return True

            v_rel_sq = dvx**2 + dvy**2
            if v_rel_sq > 1e-6:
                tcpa = -(dx*dvx + dy*dvy) / v_rel_sq
                dcpa = math.hypot(dx+dvx*tcpa, dy+dvy*tcpa) - obs_r if tcpa > 0 else dist_edge
                # Threshold SAMA dengan check_obstacles agar mode AVOID/RETURN konsisten
                if dist_edge < self._d_gate and 0 < tcpa < 5.0 and dcpa < self.d_safe + 0.8:
                    return True

        return False

    # ════════════════════════════════════════════════════════════════════════
    #  OBS SUDAH LEWAT — [FIX B2]
    # ════════════════════════════════════════════════════════════════════════

    def _active_obs_has_passed(self, state, obstacles):
        """
        [FIX B2] Kondisi lebih ketat: obs dianggap lewat hanya jika:
          1. TCPA < -3.0 (obs sudah benar-benar di belakang dalam waktu relatif)
          2. dist_edge > 6.0 (sudah cukup jauh)
          3. dot(heading_usv, arah_ke_obs) < -0.3 (obs di belakang heading)
        Untuk obs lambat (vy=-0.27), kondisi ini tidak terpenuhi prematur.
        """
        if self._active_obs_idx is None:
            return False
        if self._active_obs_idx >= len(obstacles):
            return True

        x, y, psi, phi, u, v, r, p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)

        obs   = obstacles[self._active_obs_idx]
        ox    = float(obs[0]); oy    = float(obs[1]); obs_r = float(obs[2])
        ov_x  = float(obs[3]) if len(obs) > 3 else 0.0
        ov_y  = float(obs[4]) if len(obs) > 4 else 0.0

        dx = ox - x; dy = oy - y
        dvx = ov_x - vx_usv; dvy = ov_y - vy_usv

        dist_edge = math.hypot(dx, dy) - obs_r
        v_rel_sq  = dvx**2 + dvy**2
        tcpa = -(dx*dvx + dy*dvy) / v_rel_sq if v_rel_sq > 1e-6 else -999.0

        # Cek apakah obs di belakang heading USV
        speed_usv = math.hypot(vx_usv, vy_usv)
        if speed_usv > 1e-6:
            dot_fwd = (dx*vx_usv + dy*vy_usv) / (math.hypot(dx, dy) * speed_usv + 1e-9)
        else:
            dot_fwd = 0.0

        # [FIX B2]: semua tiga kondisi harus terpenuhi
        return (tcpa < -3.0 and dist_edge > 6.0 and dot_fwd < -0.3)

    # ════════════════════════════════════════════════════════════════════════
    #  SIDE SELECTION — [FIX B1 + B4 + B5]
    # ════════════════════════════════════════════════════════════════════════

    def _predict_clearance_side(self, state, obstacles, side_sign, n_steps=20):
        """
        [FIX B1] Simulasi clearance dengan TN_sim yang realistis.
        TN_sim dihitung agar dr ≈ 0.12 rad/s per step (belok nyata).
        """
        x, y, psi, phi, u, v, r, p = state
        TN_sim = side_sign * self._TN_SIM
        TX_sim = self._TX_FF_NOMINAL

        xi   = np.array([x, y, psi, u, r])
        v_sw = float(v)
        min_cl = float('inf')

        for k in range(n_steps):
            xi, v_sw = self._predict_one_step(xi, np.array([TX_sim, TN_sim]), v_sw)
            px, py = xi[0], xi[1]
            for obs in obstacles:
                ox_k, oy_k, obs_r = self._predict_obs_pos(obs, k+1)
                dist_edge = math.hypot(px - ox_k, py - oy_k) - obs_r
                if dist_edge < min_cl:
                    min_cl = dist_edge

        return min_cl

    def _select_side(self, state, obstacles):
        """
        Pilih sisi avoidance berdasarkan KECEPATAN obs (lateral_vel),
        bukan posisi obs. Ini mencegah flip saat obs melewati ketinggian USV.

        Prinsip:
          lateral_vel = proyeksi kecepatan obs ke sumbu lateral heading USV
                      = -sin(psi)*vx_obs + cos(psi)*vy_obs

          lateral_vel > 0 → obs bergerak ke KIRI heading USV
                          → USV harus naik/ke kiri juga agar obs lewat di bawah
                          → chosen = +1 (TN>0, belok kiri/port)

          lateral_vel < 0 → obs bergerak ke KANAN heading USV
                          → USV harus turun/ke kanan agar obs lewat di atas
                          → chosen = -1 (TN<0, belok kanan/stbd)

        Verifikasi untuk DO1 dan DO2:
          DO1: vx=0, vy=+0.62, psi≈0.26
               lateral_vel = cos(0.26)*0.62 ≈ +0.60 > 0 → chosen=+1 (kiri) ✓
          DO2: vx=0, vy=-0.27, psi≈-0.17
               lateral_vel = cos(-0.17)*(-0.27) ≈ -0.27 < 0 → chosen=-1 (kanan) ✓

        Keunggulan vs posisi: kecepatan obs tidak berubah tanda sepanjang simulasi,
        sehingga chosen_side tidak pernah flip meski obs sudah melewati USV.
        """
        # Reset jika obs aktif berganti
        if self._active_obs_idx != self._prev_obs_idx:
            self._chosen_side  = None
            self._prev_obs_idx = self._active_obs_idx

        # Jika sisi sudah dipilih, PERTAHANKAN — tidak re-evaluasi setiap step
        # Ini kunci utama: sekali dipilih, tidak berubah sampai obs selesai
        if self._chosen_side is not None:
            return self._chosen_side * self._TN_SOLVER_MAX * 0.8, True

        # Pilih sisi SATU KALI berdasarkan kecepatan obs aktif
        x, y, psi = state[0], state[1], state[2]

        if self._active_obs_idx is None or self._active_obs_idx >= len(obstacles):
            self._chosen_side = +1
            return self._TN_SOLVER_MAX * 0.5, True

        obs  = obstacles[self._active_obs_idx]
        ov_x = float(obs[3]) if len(obs) > 3 else 0.0
        ov_y = float(obs[4]) if len(obs) > 4 else 0.0

        # Proyeksi kecepatan obs ke sumbu lateral heading USV
        lateral_vel = -math.sin(psi)*ov_x + math.cos(psi)*ov_y

        if abs(lateral_vel) > 0.02:
            # Obs bergerak signifikan secara lateral → pakai kecepatan
            chosen = +1 if lateral_vel > 0 else -1
        else:
            # Obs hampir tidak bergerak lateral (vx≈0, vy≈0 atau heading tegak lurus)
            # Fallback: pakai posisi awal obs relatif ke USV
            ox = float(obs[0]); oy = float(obs[1])
            lateral_pos = -math.sin(psi)*(ox - x) + math.cos(psi)*(oy - y)
            chosen = -1 if lateral_pos >= 0 else +1

        self._chosen_side = chosen
        return chosen * self._TN_SOLVER_MAX * 0.8, True  # 0.5→0.8: hint lebih agresif

    # ════════════════════════════════════════════════════════════════════════
    #  REF TRAJECTORY (selalu jalur global)
    # ════════════════════════════════════════════════════════════════════════

    def _build_ref_traj(self, state, full_path, wp_idx, target_u):
        x0, y0, psi0 = state[0], state[1], state[2]
        u0, r0       = state[4], state[6]

        # Cari titik terdekat di jalur
        s_start = max(0, wp_idx - 20)
        s_end   = min(len(full_path), wp_idx + 100)
        dists   = [math.hypot(x0 - float(full_path[i][0]),
                              y0 - float(full_path[i][1]))
                   for i in range(s_start, s_end)]
        closest = s_start + int(np.argmin(dists))

        ref       = np.zeros((self.N+1, 5))
        ref[0]    = [x0, y0, psi0, u0, r0]
        path_step = max(1, int(round(target_u * self.dt_mpc)))

        for k in range(1, self.N+1):
            idx = min(closest + k*path_step, len(full_path)-1)
            bx  = float(full_path[idx][0])
            by  = float(full_path[idx][1])
            if idx+1 < len(full_path):
                npt     = full_path[idx+1]
                psi_ref = math.atan2(float(npt[1])-by, float(npt[0])-bx)
            elif idx > 0:
                ppt     = full_path[idx-1]
                psi_ref = math.atan2(by-float(ppt[1]), bx-float(ppt[0]))
            else:
                psi_ref = psi0
            ref[k] = [bx, by, psi_ref, target_u, 0.0]

        return ref

    # ════════════════════════════════════════════════════════════════════════
    #  COST FUNCTION
    # ════════════════════════════════════════════════════════════════════════

    def _compute_cost(self, U_flat, xi0, ref_traj, all_obs,
                      pos_w=8.0, heading_w=3.0, u_w=10.0,
                      sway_damp_w=10.0, obs_pen=400.0,
                      v_sway=0.0, sway_v_w=8.0):

        traj = self._predict_trajectory(xi0, U_flat, v_sway)
        U    = U_flat.reshape(self.Nc, 2)
        cost = 0.0

        for k in range(self.N):
            xi  = traj[k, :5]
            ref = ref_traj[k] if k < len(ref_traj) else ref_traj[-1]

            # Tracking error
            e_x   = xi[0] - ref[0]
            e_y   = xi[1] - ref[1]
            e_psi = self._wrap(xi[2] - ref[2])
            e_u   = xi[3] - ref[3]

            cost += (pos_w     * e_x**2
                   + pos_w     * e_y**2
                   + heading_w * e_psi**2
                   + u_w       * e_u**2
                   + 0.3       * xi[4]**2)   # yaw rate damping

            # Sway velocity damping
            v_sw = traj[k, 5]
            cost += sway_v_w * v_sw**2
            sway_ex = max(0.0, abs(v_sw) - 0.5)
            cost += 300.0 * sway_ex**2

            # Yaw rate damping tambahan
            cost += sway_damp_w * xi[4]**2

            # Input penalty
            uk = min(k, self.Nc-1)
            cost += 0.005*U[uk][0]**2 + 0.008*U[uk][1]**2

            # Input smoothness
            du0 = U[uk][0] - (U[uk-1][0] if uk > 0 else self.u_prev[0])
            du1 = U[uk][1] - (U[uk-1][1] if uk > 0 else self.u_prev[1])
            cost += 0.05*du0**2 + 0.3*du1**2

            # ── Obstacle penalty ──────────────────────────────────────────
            if obs_pen > 0 and all_obs:
                for obs in all_obs:
                    ox_k, oy_k, obs_r = self._predict_obs_pos(obs, k)
                    dist = math.hypot(xi[0] - ox_k, xi[1] - oy_k)

                    ov_x = float(obs[3]) if len(obs) > 3 else 0.0
                    ov_y = float(obs[4]) if len(obs) > 4 else 0.0
                    is_static = math.hypot(ov_x, ov_y) < 0.05

                    # Margin: dynamic 1.2m, static 0.3m
                    # Arc lateral ≈ margin + d_safe + obs_r = 1.2+1.0+0.25 = 2.45m
                    # Ini sesuai target "avoid 1.5-2m dari jalur"
                    margin = 0.8 if is_static else 2.3
                    pen    = obs_pen * (1.2 if is_static else 1.0)
                    safe_r = obs_r + self.d_safe + margin

                    if dist < safe_r:
                        penetration = safe_r - dist

                        # Boost penalti berdasarkan DCPA (semakin kecil DCPA → boost lebih besar)
                        dcpa_boost = 1.0
                        if not is_static:
                            vx_xi = xi[3]*math.cos(xi[2])
                            vy_xi = xi[3]*math.sin(xi[2])
                            ddx = ox_k - xi[0]; ddy = oy_k - xi[1]
                            ddvx = ov_x - vx_xi; ddvy = ov_y - vy_xi
                            vr2 = ddvx**2 + ddvy**2
                            if vr2 > 1e-6:
                                t_cpa_k = -(ddx*ddvx + ddy*ddvy) / vr2
                                if t_cpa_k > 0:
                                    dcpa_k = math.hypot(
                                        ddx + ddvx*t_cpa_k,
                                        ddy + ddvy*t_cpa_k) - obs_r
                                    dcpa_boost = 1.0 + 2.0 / (max(dcpa_k, 0.1) + 0.5)

                        cost += pen * dcpa_boost * penetration**2 + pen*0.1*penetration

        # Terminal cost (2× bobot tracking agar horizon berakhir dekat jalur)
        e_xt = traj[-1, 0] - ref_traj[-1][0]
        e_yt = traj[-1, 1] - ref_traj[-1][1]
        e_pt = self._wrap(traj[-1, 2] - ref_traj[-1][2])
        cost += pos_w*2.0*(e_xt**2 + e_yt**2)
        cost += heading_w*2.0*e_pt**2

        return cost

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN SOLVE
    # ════════════════════════════════════════════════════════════════════════

    def solve(self, state, wp_prev, wp_next, obstacles, target_u,
              full_path, wp_idx, dist_goal, static_obs=None):
        """
        Return: (Tcmd, mode, info)
          Tcmd = [TX, 0, TN, 0] atau None (→ gunakan ILOS)
          mode = 'GLOBAL' | 'LMPC_AVOID' | 'LMPC_RETURN'
          info = dict dengan obs_status, min_dist, avoid_side, psi_opt, ...
        """

        # Hard stop dekat goal
        if dist_goal < 5.0:
            self.reset()
            return None, 'GLOBAL', {'obs_status': 'CLEAR', 'min_dist': 0.0}

        # Hard timeout
        if self._total_steps > self._MAX_TOTAL_STEPS:
            self.reset()
            return None, 'GLOBAL', {'obs_status': 'HARD_TIMEOUT', 'min_dist': 0.0}

        x, y, psi, phi, u, v, r, p = state
        xi0    = np.array([x, y, psi, u, r])
        v_sway = float(v)

        # Cek apakah obs aktif sudah lewat
        if self.active and self._active_obs_has_passed(state, obstacles):
            self.reset()
            return None, 'GLOBAL', {'obs_status': 'PASSED', 'min_dist': 999.0}

        obs_status, min_dist = self.check_obstacles(state, obstacles)

        if obs_status == 'CLEAR':
            self.reset()
            return None, 'GLOBAL', {'obs_status': obs_status, 'min_dist': min_dist}

        # LMPC aktif
        self.active         = True
        self._total_steps  += 1

        real_danger = self._is_real_danger(state, obstacles)
        avoid_u     = min(target_u, 1.5)   # sedikit kurangi speed saat avoidance
        TX_ff       = self._compute_TX_ff(avoid_u)

        ref_traj = self._build_ref_traj(state, full_path, wp_idx, avoid_u)

        TN_lo = -self._TN_SOLVER_MAX
        TN_hi =  self._TN_SOLVER_MAX
        TN_hint = 0.0

        # ── LMPC_AVOID ───────────────────────────────────────────────────
        if real_danger:
            self.mode = 'LMPC_AVOID'

            # Semua obs masuk cost (dyn + static)
            all_obs = []
            for obs in obstacles:
                all_obs.append((float(obs[0]), float(obs[1]), float(obs[2]),
                                float(obs[3]) if len(obs) > 3 else 0.0,
                                float(obs[4]) if len(obs) > 4 else 0.0))
            if static_obs:
                for obs in static_obs:
                    all_obs.append((float(obs[0]), float(obs[1]), float(obs[2]), 0.0, 0.0))

            # Pilih sisi (dengan fix B1, B4, B5)
            TN_hint, force_sign = self._select_side(state, obstacles)
            # DEBUG log: cetak lateral info untuk verifikasi sisi
            import sys
            _obs = obstacles[self._active_obs_idx] if self._active_obs_idx is not None and self._active_obs_idx < len(obstacles) else None
            if _obs is not None:
                _dy = float(_obs[1]) - y
                _lateral_dbg = -math.sin(psi)*( float(_obs[0])-x ) + math.cos(psi)*_dy
                print(f"[LMPC_DBG] obs_idx={self._active_obs_idx} dy={_dy:.2f} lateral={_lateral_dbg:.2f} chosen_side={self._chosen_side} TN_hint={TN_hint:.1f}", file=sys.stderr)

            # [FIX B5] force_sign selalu True saat AVOID → bounds dipaksa satu sisi
            if force_sign:
                # Paksa solver minimum TN_hint*0.5 agar tidak turun ke ~0
                # Ini mencegah pos_weight menarik TN kembali ke nol
                tn_min_enforce = TN_hint * 0.5  # minimum 50% dari hint
                if TN_hint > 0:
                    TN_lo = max(0.0, tn_min_enforce)
                    TN_hi = self._TN_SOLVER_MAX
                else:
                    TN_lo = -self._TN_SOLVER_MAX
                    TN_hi = min(0.0, tn_min_enforce)
                self._opt_u    = None
                self.u_prev[1] = float(TN_hint)

            pos_w     = 0.5    # sangat kecil: biarkan USV menyimpang dari jalur saat AVOID
            heading_w = 0.5
            u_w       = 10.0
            obs_pen   = 900.0  # besar: paksa USV menjauh dari obs
            sway_damp = 12.0 + 25.0 * max(0.0, abs(v_sway) - 0.3)
            sway_v_w  = 8.0

            u0_guess = np.tile([TX_ff, TN_hint], self.Nc)

        # ── LMPC_RETURN ──────────────────────────────────────────────────
        else:
            self._return_step += 1
            self.mode = 'LMPC_RETURN'

            if self._return_step > self._RETURN_MAX_STEPS:
                self.reset()
                return None, 'GLOBAL', {'obs_status': 'RETURN_TIMEOUT', 'min_dist': min_dist}

            # [FIX B3] Saat RETURN, masukkan dyn obs yang masih dalam radius 10m
            all_obs = []
            for obs in obstacles:
                ox_now = float(obs[0]); oy_now = float(obs[1]); obs_r = float(obs[2])
                if math.hypot(x - ox_now, y - oy_now) - obs_r < 10.0:
                    all_obs.append((ox_now, oy_now, obs_r,
                                    float(obs[3]) if len(obs) > 3 else 0.0,
                                    float(obs[4]) if len(obs) > 4 else 0.0))
            if static_obs:
                for obs in static_obs:
                    all_obs.append((float(obs[0]), float(obs[1]), float(obs[2]), 0.0, 0.0))

            pos_w     = 30.0   # tarik tegas ke jalur
            heading_w = 10.0
            u_w       = 10.0
            obs_pen   = 400.0  # tetap waspada static obs
            sway_damp = 40.0 + 20.0 * max(0.0, abs(v_sway) - 0.2)
            sway_v_w  = 10.0

            u0_guess = np.tile([TX_ff, 0.0], self.Nc)

        # ── Warm-start ───────────────────────────────────────────────────
        if self._opt_u is not None and len(self._opt_u) == 2*self.Nc:
            cs = self._compute_cost(self._opt_u, xi0, ref_traj, all_obs,
                                    pos_w, heading_w, u_w, sway_damp, obs_pen,
                                    v_sway, sway_v_w)
            cg = self._compute_cost(u0_guess, xi0, ref_traj, all_obs,
                                    pos_w, heading_w, u_w, sway_damp, obs_pen,
                                    v_sway, sway_v_w)
            if cs < cg * 1.05:
                u0_guess = self._opt_u.copy()

        # ── Bounds ───────────────────────────────────────────────────────
        min_TX = max(self.TX_min, TX_ff * 0.85)
        bounds = []
        for _ in range(self.Nc):
            bounds.append((min_TX, self._TX_SOLVER_MAX))
            bounds.append((TN_lo,  TN_hi))

        # ── Optimisasi ───────────────────────────────────────────────────
        result = minimize(
            fun=self._compute_cost,
            x0=u0_guess,
            args=(xi0, ref_traj, all_obs, pos_w, heading_w, u_w,
                  sway_damp, obs_pen, v_sway, sway_v_w),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 5e-4, 'disp': False}
        )

        cg_val = self._compute_cost(u0_guess, xi0, ref_traj, all_obs,
                                    pos_w, heading_w, u_w, sway_damp, obs_pen,
                                    v_sway, sway_v_w)
        if result.success or result.fun < cg_val:
            U_opt       = result.x.reshape(self.Nc, 2)
            self._opt_u = result.x.copy()
        else:
            U_opt       = np.tile([TX_ff, TN_hint if real_danger else 0.0],
                                  self.Nc).reshape(self.Nc, 2)
            self._opt_u = None

        TX_opt = float(U_opt[0][0])
        TN_opt = float(U_opt[0][1])

        # Rate limiter output
        TX_opt = float(np.clip(
            self.u_prev[0] + np.clip(TX_opt - self.u_prev[0], -self.dTX_max, self.dTX_max),
            min_TX, self._TX_SOLVER_MAX))
        TN_opt = float(np.clip(
            self.u_prev[1] + np.clip(TN_opt - self.u_prev[1], -self.dTN_max, self.dTN_max),
            self.TN_min, self.TN_max))

        self.u_prev = np.array([TX_opt, TN_opt])
        Tcmd = np.array([TX_opt, 0.0, TN_opt, 0.0])

        # Prediksi traj optimal untuk visualisasi & psi_opt
        U_vis    = self._opt_u if self._opt_u is not None else np.tile([TX_ff, 0.0], self.Nc)
        opt_traj = self._predict_trajectory(xi0, U_vis, v_sway)
        psi_opt  = float(opt_traj[min(2, self.N), 2])

        info = {
            'obs_status': obs_status,
            'min_dist':   min_dist,
            'avoid_side': 'LEFT' if TN_opt > 0 else 'RIGHT',
            'psi_opt':    psi_opt,
            'ref_traj':   ref_traj,
            'opt_traj':   opt_traj,
        }
        return Tcmd, self.mode, info