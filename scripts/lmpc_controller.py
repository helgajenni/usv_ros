#!/usr/bin/env python3
"""
LMPC Controller — v10  (Multi-Obstacle Aware)
═══════════════════════════════════════════════════════════════════════════════
Perbaikan dari v9:

MASALAH v9:
  - _get_TN_hint hanya mempertimbangkan 1 obs paling mengancam (worst_dcpa)
    → saat menghindari obs1, tidak melihat obs2 ada di jalur penghindaran
  - margin dinamis 2.5m tidak memperhitungkan posisi obs DI MASA DEPAN
  - check_obstacles tcpa window hanya 6.0 detik → obs2 yang lebih jauh tidak terdeteksi
  - Tidak ada "cost penghindaran" yang mempertimbangkan semua obs sekaligus saat
    memilih sisi (kiri vs kanan)

SOLUSI v10:
  1. _get_TN_hint_multi(): evaluasi SEMUA obs, pilih sisi yang memberikan
     clearance terkecil lebih besar dari threshold (bukan hanya 1 obs)
  2. _predict_clearance_side(): simulasikan jalur kasar kiri/kanan dan hitung
     minimum clearance ke SEMUA obs (termasuk posisi obs di masa depan)
  3. check_obstacles: perluas tcpa window ke 12.0 detik untuk deteksi lebih awal
  4. margin obs dinamis: dikurangi ke 1.5m (dari 2.5m) agar arc lebih kecil
     dan tidak masuk ke zona obs lain → combined dengan multi-obs hint
  5. _LOCK_DIST diperluas ke 7.0m: lock lebih lama agar return tepat sasaran
  6. RETURN mode: pos_weight sedikit dikurangi agar tidak agresif spike TN

Prinsip: pilih sisi yang PALING AMAN untuk SEMUA obs, bukan hanya satu obs.
"""

import numpy as np
import math
from scipy.optimize import minimize


class LMPCController:
    def __init__(self, params: dict):
        self.A2  = float(params.get('A2',   -0.7405))
        self.A3  = float(params.get('A3',    0.4219))
        self.A4  = float(params.get('A4',   -0.1397))
        self.A12 = float(params.get('A12',  -1.0495))
        self.A13 = float(params.get('A13',   1.4038))
        self.A14 = float(params.get('A14',  -2.0764))
        self.A16 = float(params.get('A16',   0.9671))
        self.A18 = float(params.get('A18',   0.0178))
        self.A19 = float(params.get('A19',   0.0010))
        self.A5  = float(params.get('A5',  -0.1464))
        self.A6  = float(params.get('A6',  -3.1952))
        self.A7  = float(params.get('A7',   4.1189))
        self.A1  = float(params.get('A1',   1.5066))

        self.u0      = 1.5
        self.A12_eff = self.A12 + self.A16 * self.u0

        self.dt     = float(params.get('dt',     0.05))
        self.dt_mpc = float(params.get('dt_mpc', 0.4))
        self.N      = int(params.get('N',  12))
        self.Nc     = int(params.get('Nc',  6))

        self.TX_min  = float(params.get('TX_min',    0.0))
        self.TX_max  = float(params.get('TX_max',  200.0))
        self.TN_min  = float(params.get('TN_min', -1750.0))
        self.TN_max  = float(params.get('TN_max',  1750.0))
        self.dTX_max = float(params.get('dTX_max',  60.0))  * self.dt
        self.dTN_max = float(params.get('dTN_max', 1200.0)) * self.dt

        self.u_min = float(params.get('u_min', 1.1))
        self.u_max = float(params.get('u_max', 1.5))
        self.r_max = 0.8  # rad/s

        self.d_safe  = float(params.get('d_safe', 1.0))
        self.d_warn  = float(params.get('d_warn', 2.5))
        self.d_exit  = float(params.get('d_exit', 4.0))
        self._d_gate = float(params.get('d_gate', 4.0))

        self._TN_SOLVER_MAX = 200.0

        u_target    = 1.5
        drag_target = (self.A2*u_target + self.A3*u_target*abs(u_target)
                       + self.A4*u_target**3)
        TX_ff_target        = float(np.clip(-drag_target/self.A18, 10.0, self.TX_max))
        self._TX_SOLVER_MAX = TX_ff_target * 1.05
        self._TX_FF_NOMINAL = TX_ff_target

        # v11: LOCK_DIST dikembalikan 7→5m (is_locked di check_obstacles pakai 5.0)
        self._LOCK_DIST        = 5.0
        self._RETURN_MAX_STEPS = 150
        self._MAX_TOTAL_STEPS  = 400

        self.u_prev          = np.array([TX_ff_target, 0.0])
        self.active          = False
        self.mode            = 'GLOBAL'
        self._opt_u          = None
        self._return_step    = 0
        self._total_steps    = 0
        self._active_obs_idx = None
        # v10: cache sisi yang dipilih agar konsisten selama avoidance
        self._chosen_side    = None   # +1=kiri(TN>0), -1=kanan(TN<0)

    # ── Model ────────────────────────────────────────────────────────────
    def _predict_one_step(self, xi, mu, v_sway=0.0):
        x, y, psi, u, r = xi
        TX, TN = mu
        dt = self.dt_mpc
        dx   = (u*math.cos(psi) - v_sway*math.sin(psi)) * dt
        dy   = (u*math.sin(psi) + v_sway*math.cos(psi)) * dt
        dpsi = r * dt
        du   = (self.A2*u + self.A3*u*abs(u) + self.A4*u**3
                + self.A18*TX - v_sway*r) * dt
        A12_eff_k = self.A12 + self.A16 * abs(u)
        dr   = (A12_eff_k*r + self.A13*r*abs(r) + self.A14*r**3 + self.A19*TN) * dt
        dv   = (self.A5*v_sway + self.A6*v_sway*abs(v_sway) + self.A7*v_sway**3
                - (1.0/self.A1)*u*r) * dt
        v_new = float(np.clip(v_sway + dv, -2.5, 2.5))
        psi_new = (psi+dpsi+math.pi) % (2*math.pi) - math.pi
        u_new   = float(np.clip(u+du, self.u_min, self.u_max))
        r_new   = float(np.clip(r+dr, -self.r_max, self.r_max))
        return np.array([x+dx, y+dy, psi_new, u_new, r_new]), v_new

    def _predict_trajectory(self, xi0, U_flat, v_sway=0.0):
        U    = U_flat.reshape(self.Nc, 2)
        traj    = np.zeros((self.N+1, 6))
        traj[0,:5] = xi0
        traj[0, 5]  = v_sway
        for k in range(self.N):
            uk = min(k, self.Nc-1)
            xi_new, v_k = self._predict_one_step(traj[k,:5], U[uk], traj[k,5])
            traj[k+1,:5] = xi_new
            traj[k+1, 5] = v_k
        return traj

    def _predict_obstacle_pos(self, obs, k):
        ox, oy, obs_r = float(obs[0]), float(obs[1]), float(obs[2])
        vx = float(obs[3]) if len(obs)>3 else 0.0
        vy = float(obs[4]) if len(obs)>4 else 0.0
        return ox+vx*k*self.dt_mpc, oy+vy*k*self.dt_mpc, obs_r

    def _wrap_angle(self, a):
        return (a+math.pi) % (2*math.pi) - math.pi

    def _compute_TX_ff(self, u_ref):
        drag = self.A2*u_ref + self.A3*u_ref*abs(u_ref) + self.A4*u_ref**3
        return float(np.clip(-drag/self.A18, self.TX_min, self._TX_SOLVER_MAX))

    def reset(self):
        self._opt_u          = None
        self.u_prev          = np.array([self._TX_FF_NOMINAL, 0.0])
        self.active          = False
        self.mode            = 'GLOBAL'
        self._return_step    = 0
        self._total_steps    = 0
        self._active_obs_idx = None
        self._chosen_side    = None   # v10: reset cache sisi

    # ── Obs sudah lewat ───────────────────────────────────────────────────
    def _active_obs_has_passed(self, state, obstacles):
        if self._active_obs_idx is None: return False
        if self._active_obs_idx >= len(obstacles): return True

        x,y,psi,phi,u,v,r,p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)

        obs    = obstacles[self._active_obs_idx]
        ox,oy  = float(obs[0]), float(obs[1])
        obs_r  = float(obs[2])
        ox_vel = float(obs[3]) if len(obs)>3 else 0.0
        oy_vel = float(obs[4]) if len(obs)>4 else 0.0

        dx  = ox-x;  dy  = oy-y
        dvx = ox_vel-vx_usv;  dvy = oy_vel-vy_usv
        dist_edge = math.hypot(dx,dy) - obs_r
        v_rel_sq  = dvx**2+dvy**2
        tcpa      = -(dx*dvx+dy*dvy)/v_rel_sq if v_rel_sq>1e-6 else -999.0
        return tcpa < -2.0 and dist_edge > 5.0

    # ── Check obstacles ───────────────────────────────────────────────────
    def check_obstacles(self, state, obstacles):
        x,y,psi,phi,u,v,r,p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)

        obs_status       = 'CLEAR'
        min_dist_overall = float('inf')
        closest_threat   = None
        closest_dist     = float('inf')

        for idx, obs in enumerate(obstacles):
            ox,oy,obs_r = float(obs[0]),float(obs[1]),float(obs[2])
            ox_vel = float(obs[3]) if len(obs)>3 else 0.0
            oy_vel = float(obs[4]) if len(obs)>4 else 0.0

            dx=ox-x; dy=oy-y
            dvx=ox_vel-vx_usv; dvy=oy_vel-vy_usv
            dist_now  = math.hypot(dx,dy)
            dist_edge = dist_now - obs_r
            min_dist_overall = min(min_dist_overall, dist_edge)

            if dist_edge < self.d_safe:
                return 'CRITICAL', min_dist_overall

            in_gate  = dist_edge < self._d_gate
            v_rel_sq = dvx**2+dvy**2
            tcpa = -(dx*dvx+dy*dvy)/v_rel_sq if v_rel_sq>1e-6 else 0.0
            dcpa = (math.hypot(dx+dvx*tcpa,dy+dvy*tcpa)-obs_r
                    if tcpa>0 else dist_edge)

            is_direct = dist_edge < self.d_warn
            # v11: tcpa window dikembalikan ke 6s (12s terlalu sensitif → false alarm)
            # Tambah filter dcpa < d_safe+1.5 (bukan +0.5) agar obs dengan
            # DCPA besar (>2m) tidak trigger LMPC — obs2 dcpa=4.8m tidak akan trigger
            is_tcpa   = in_gate and (0<tcpa<6.0 and dcpa<self.d_safe+1.5)
            # LOCK_DIST dikembalikan ke 5.0m (7.0m terlalu jauh → lock obs yang
            # sudah tidak berbahaya dan menyebabkan false avoidance lanjutan)
            is_locked = (self.active
                         and self._active_obs_idx==idx
                         and dist_now < 5.0
                         and tcpa > -15.0)

            if is_direct or is_tcpa or is_locked:
                obs_status = 'WARN'
                if dist_edge < closest_dist:
                    closest_dist   = dist_edge
                    closest_threat = idx

        if obs_status=='WARN' and self._active_obs_idx is None:
            self._active_obs_idx = closest_threat

        return obs_status, min_dist_overall

    def _is_real_danger(self, state, obstacles):
        x,y,psi,phi,u,v,r,p = state
        vx_usv = u*math.cos(psi) - v*math.sin(psi)
        vy_usv = u*math.sin(psi) + v*math.cos(psi)
        for obs in obstacles:
            ox,oy,obs_r = float(obs[0]),float(obs[1]),float(obs[2])
            ox_vel = float(obs[3]) if len(obs)>3 else 0.0
            oy_vel = float(obs[4]) if len(obs)>4 else 0.0
            dx=ox-x; dy=oy-y
            dvx=ox_vel-vx_usv; dvy=oy_vel-vy_usv
            dist_edge = math.hypot(dx,dy) - obs_r
            in_gate   = dist_edge < self._d_gate
            if dist_edge < self.d_safe: return True
            v_rel_sq = dvx**2+dvy**2
            tcpa = -(dx*dvx+dy*dvy)/v_rel_sq if v_rel_sq>1e-6 else 0.0
            dcpa = math.hypot(dx+dvx*tcpa,dy+dvy*tcpa)-obs_r if tcpa>0 else dist_edge
            if dist_edge < self.d_warn: return True
            if in_gate and (0<tcpa<6.0 and dcpa<self.d_safe+1.5): return True
        return False

    # ── Ref trajektori (SELALU jalur global) ──────────────────────────────
    def _build_ref_traj(self, state, full_path, wp_idx, target_u):
        x0, y0, psi0, u0, r0 = state[0], state[1], state[2], state[4], state[6]
        s_start = max(0, wp_idx-20)
        s_end   = min(len(full_path), wp_idx+100)
        dists   = [math.hypot(x0-float(full_path[i][0]), y0-float(full_path[i][1]))
                   for i in range(s_start, s_end)]
        closest = s_start + int(np.argmin(dists))
        ref       = np.zeros((self.N+1, 5))
        ref[0]    = [x0, y0, psi0, u0, r0]
        path_step = max(1, int(round(target_u * self.dt_mpc)))
        for k in range(1, self.N+1):
            idx = min(closest+k*path_step, len(full_path)-1)
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

    # ══════════════════════════════════════════════════════════════════════
    # v10: MULTI-OBS SIDE SELECTION
    # ══════════════════════════════════════════════════════════════════════

    def _predict_clearance_side(self, state, obstacles, side_sign, n_steps=18):
        """
        Simulasikan lintasan kasar ke sisi side_sign selama n_steps langkah
        dt_mpc. Hitung clearance minimum ke SEMUA obs (posisi obs di masa depan).

        side_sign: +1 = kiri (port, TN>0), -1 = kanan (stbd, TN<0)
        Return: minimum clearance (m) dari semua obs di semua langkah.
        """
        x,y,psi,phi,u,v,r,p = state
        TN_sim = side_sign * 80.0   # hint TN sedang
        TX_sim = self._TX_FF_NOMINAL

        xi = np.array([x, y, psi, u, r])
        v_sw = float(v)
        min_cl = float('inf')

        for k in range(n_steps):
            xi, v_sw = self._predict_one_step(xi, np.array([TX_sim, TN_sim]), v_sw)
            px, py = xi[0], xi[1]
            for obs in obstacles:
                ox_k, oy_k, obs_r = self._predict_obstacle_pos(obs, k+1)
                dist_edge = math.hypot(px-ox_k, py-oy_k) - obs_r
                if dist_edge < min_cl:
                    min_cl = dist_edge

        return min_cl

    def _get_TN_hint_multi(self, state, obstacles):
        """
        v10: Pilih sisi dengan clearance TERBAIK ke SEMUA obs.

        Algoritma:
          1. Hitung clearance sisi KIRI (TN>0) ke semua obs
          2. Hitung clearance sisi KANAN (TN<0) ke semua obs
          3. Pilih sisi yang clearancenya lebih besar

        Jika satu sisi mengakibatkan tabrakan (clearance < d_safe),
        PAKSA sisi lain meski clearancenya kecil.

        Gunakan cache _chosen_side agar tidak flip-flop per step.
        """
        # Jika sudah ada pilihan sisi sebelumnya, evaluasi dulu apakah masih valid
        if self._chosen_side is not None:
            cl_chosen = self._predict_clearance_side(state, obstacles, self._chosen_side)
            if cl_chosen > self.d_safe:
                # Sisi yang sama masih aman → pertahankan
                TN_hint    = self._chosen_side * 80.0
                force_sign = cl_chosen < self.d_safe + 1.0
                return TN_hint, force_sign

        # Pilih ulang berdasarkan multi-obs clearance
        cl_left  = self._predict_clearance_side(state, obstacles, +1)
        cl_right = self._predict_clearance_side(state, obstacles, -1)

        # Kedua sisi unsafe → pilih yang clearance lebih besar (lesser evil)
        if cl_left < self.d_safe and cl_right < self.d_safe:
            chosen = +1 if cl_left >= cl_right else -1
            force_sign = False
        elif cl_left < self.d_safe:
            # Kiri berbahaya → paksa kanan
            chosen = -1
            force_sign = True
        elif cl_right < self.d_safe:
            # Kanan berbahaya → paksa kiri
            chosen = +1
            force_sign = True
        else:
            # Keduanya aman → pilih yang lebih lega
            chosen = +1 if cl_left >= cl_right else -1
            # force_sign hanya jika perbedaannya signifikan (>1m)
            force_sign = abs(cl_left - cl_right) > 1.0

        self._chosen_side = chosen
        TN_hint = chosen * 80.0
        return TN_hint, force_sign

    # ── Fallback hint geometris (untuk satu obs, backward compat) ─────────
    def _get_TN_hint(self, state, obstacles):
        """Wrapper: gunakan versi multi-obs."""
        return self._get_TN_hint_multi(state, obstacles)

    # ── Main solver ───────────────────────────────────────────────────────
    def solve(self, state, wp_prev, wp_next, obstacles, target_u,
              full_path, wp_idx, dist_goal, static_obs=None):

        if dist_goal < 5.0:
            self.reset()
            return None, 'GLOBAL', {'obs_status':'CLEAR','min_dist':0.0}

        if self._total_steps > self._MAX_TOTAL_STEPS:
            self.reset()
            return None, 'GLOBAL', {'obs_status':'HARD_TIMEOUT','min_dist':0.0}

        x,y,psi,phi,u,v,r,p = state
        xi0    = np.array([x,y,psi,u,r])
        v_sway = float(v)

        if self.active and self._active_obs_has_passed(state, obstacles):
            self.reset()
            return None, 'GLOBAL', {'obs_status':'PASSED','min_dist':999.0}

        obs_status, min_dist = self.check_obstacles(state, obstacles)

        if obs_status == 'CLEAR':
            self.reset()
            return None, 'GLOBAL', {'obs_status':obs_status,'min_dist':min_dist}

        self.active        = True
        self._total_steps += 1

        v_abs   = abs(v_sway)
        avoid_u = min(target_u, 1.4)
        TX_ff   = self._compute_TX_ff(avoid_u)

        real_danger = self._is_real_danger(state, obstacles)

        ref_traj = self._build_ref_traj(state, full_path, wp_idx, avoid_u)

        TN_lo_solve = -self._TN_SOLVER_MAX
        TN_hi_solve =  self._TN_SOLVER_MAX

        if real_danger:
            self.mode = 'LMPC_AVOID'
            pos_weight       = 5.0
            heading_weight   = 3.0
            u_weight         = 15.0
            # v10: obs_penalty dikurangi 500→350 agar arc lebih kecil
            obs_penalty      = 350.0
            sway_damp        = 15.0 + 30.0 * max(0.0, v_abs-0.3)
            sway_v_weight    = 10.0

            all_obs = []
            for obs in obstacles:
                all_obs.append((float(obs[0]),float(obs[1]),float(obs[2]),
                                float(obs[3]) if len(obs)>3 else 0.0,
                                float(obs[4]) if len(obs)>4 else 0.0))
            if static_obs:
                for obs in static_obs:
                    all_obs.append((float(obs[0]),float(obs[1]),float(obs[2]),0.0,0.0))

            # v10: gunakan multi-obs hint
            TN_hint, force_sign = self._get_TN_hint_multi(state, obstacles)
            u0_guess = np.tile([TX_ff, TN_hint], self.Nc)

            if force_sign:
                TN_lo_solve = 0.0 if TN_hint > 0 else -self._TN_SOLVER_MAX
                TN_hi_solve = self._TN_SOLVER_MAX if TN_hint > 0 else 0.0
                self._opt_u = None
                self.u_prev[1] = float(TN_hint)

        else:
            self._return_step += 1
            self.mode = 'LMPC_RETURN'

            if self._return_step > self._RETURN_MAX_STEPS:
                self.reset()
                return None, 'GLOBAL', {'obs_status':'RETURN_TIMEOUT','min_dist':min_dist}

            # v11: pos_weight RETURN 12→18 — tarik lebih tegas ke jalur
            # heading_weight 8→10 — koreksi heading lebih cepat
            # obs_penalty 0→150 untuk static obs saja — cegah USV masuk O6
            # sway_damp tetap tinggi agar tidak spike
            pos_weight     = 18.0
            heading_weight = 10.0
            u_weight       = 10.0
            obs_penalty    = 150.0   # aktif hanya untuk static obs (dyn obs = kosong di RETURN)
            sway_damp      = 40.0 + 20.0 * max(0.0, v_abs-0.2)
            sway_v_weight  = 10.0
            # RETURN: hanya masukkan static obs ke all_obs, bukan dyn obs
            # (dyn obs sudah terlewati saat masuk RETURN mode)
            all_obs = []
            if static_obs:
                for obs in static_obs:
                    all_obs.append((float(obs[0]),float(obs[1]),float(obs[2]),0.0,0.0))
            u0_guess = np.tile([TX_ff, 0.0], self.Nc)

        if self._opt_u is not None and len(self._opt_u)==2*self.Nc:
            cs = self._compute_cost(self._opt_u, xi0, ref_traj, all_obs,
                                    pos_weight, heading_weight, u_weight,
                                    sway_damp, obs_penalty, v_sway, sway_v_weight)
            cg = self._compute_cost(u0_guess,   xi0, ref_traj, all_obs,
                                    pos_weight, heading_weight, u_weight,
                                    sway_damp, obs_penalty, v_sway, sway_v_weight)
            if cs < cg * 1.05:
                u0_guess = self._opt_u.copy()

        bounds    = []
        min_thr   = max(self.TX_min, TX_ff * 0.85)
        for _ in range(self.Nc):
            bounds.append((min_thr, self._TX_SOLVER_MAX))
            bounds.append((TN_lo_solve, TN_hi_solve))

        result = minimize(
            fun=self._compute_cost,
            x0=u0_guess,
            args=(xi0, ref_traj, all_obs, pos_weight, heading_weight,
                  u_weight, sway_damp, obs_penalty, v_sway, sway_v_weight),
            method='SLSQP', bounds=bounds,
            options={'maxiter':100, 'ftol':5e-4, 'disp':False}
        )

        cg_val = self._compute_cost(u0_guess, xi0, ref_traj, all_obs,
                                    pos_weight, heading_weight, u_weight,
                                    sway_damp, obs_penalty, v_sway, sway_v_weight)
        if result.success or result.fun < cg_val:
            U_opt       = result.x.reshape(self.Nc, 2)
            self._opt_u = result.x.copy()
        else:
            U_opt       = np.tile([TX_ff, TN_hint if real_danger else 0.0],
                                  self.Nc).reshape(self.Nc, 2)
            self._opt_u = None

        TX_opt, TN_opt = float(U_opt[0][0]), float(U_opt[0][1])

        TX_opt = float(np.clip(
            self.u_prev[0]+np.clip(TX_opt-self.u_prev[0],-self.dTX_max,self.dTX_max),
            min_thr, self._TX_SOLVER_MAX))
        TN_opt = float(np.clip(
            self.u_prev[1]+np.clip(TN_opt-self.u_prev[1],-self.dTN_max,self.dTN_max),
            self.TN_min, self.TN_max))

        self.u_prev = np.array([TX_opt, TN_opt])
        Tcmd        = np.array([TX_opt, 0.0, TN_opt, 0.0])

        U_vis    = self._opt_u if self._opt_u is not None else np.tile([TX_ff,0.0],self.Nc)
        opt_traj = self._predict_trajectory(xi0, U_vis, v_sway)

        psi_opt = float(opt_traj[min(2, self.N), 2])

        info = {
            'obs_status'  : obs_status,
            'min_dist'    : min_dist,
            'avoid_side'  : 'LEFT' if (TN_opt > 0) else 'RIGHT',
            'psi_opt'     : psi_opt,
            'ref_traj'    : ref_traj,
            'opt_traj'    : opt_traj,
        }
        return Tcmd, self.mode, info

    # ── Cost function ─────────────────────────────────────────────────────
    def _compute_cost(self, U_flat, xi0, ref_traj, all_obstacles,
                      pos_weight=8.0, heading_weight=3.0, u_weight=5.0,
                      sway_damp_weight=0.0, obs_penalty=500.0, v_sway=0.0,
                      sway_v_weight=0.0):
        U    = U_flat.reshape(self.Nc, 2)
        traj = self._predict_trajectory(xi0, U_flat, v_sway)
        cost = 0.0

        for k in range(self.N):
            xi  = traj[k,:5]
            ref = ref_traj[k] if k < len(ref_traj) else ref_traj[-1]

            e_x   = xi[0]-ref[0];  e_y  = xi[1]-ref[1]
            e_psi = self._wrap_angle(xi[2]-ref[2])
            e_u   = xi[3]-ref[3];  r_now = xi[4]

            cost += (pos_weight     * e_x**2
                   + pos_weight     * e_y**2
                   + heading_weight * e_psi**2
                   + u_weight       * e_u**2
                   + 0.5            * (xi[4]-ref[4])**2)

            cost += sway_damp_weight * r_now**2

            v_sw = traj[k, 5]
            cost += sway_v_weight * v_sw**2
            sway_excess = max(0.0, abs(v_sw) - 0.5)
            cost += 500.0 * sway_excess**2

            uk  = min(k, self.Nc-1)
            cost += 0.005*U[uk][0]**2 + 0.01*U[uk][1]**2

            du0 = U[uk][0]-(U[uk-1][0] if uk>0 else self.u_prev[0])
            du1 = U[uk][1]-(U[uk-1][1] if uk>0 else self.u_prev[1])
            cost += 0.1*du0**2 + 0.5*du1**2

            # Obstacle penalty
            if obs_penalty > 0:
                for obs in all_obstacles:
                    ox_k, oy_k, obs_r = self._predict_obstacle_pos(obs, k)
                    dist = math.hypot(xi[0]-ox_k, xi[1]-oy_k)
                    obs_vx    = float(obs[3]) if len(obs)>3 else 0.0
                    obs_vy    = float(obs[4]) if len(obs)>4 else 0.0
                    is_static = math.hypot(obs_vx,obs_vy) < 0.05

                    # v11: margin dyn 1.5→0.8m → arc lebih kecil → CTE max ~1.5m
                    # (CTE minimum geometri obs1 sudah ~1.75m, margin 0.8 cukup)
                    # Keamanan tetap dijaga multi-obs hint yang memilih sisi benar
                    margin = 0.3 if is_static else 0.8
                    pen    = obs_penalty * (1.2 if is_static else 1.0)
                    safe_m = obs_r + self.d_safe + margin

                    if dist < safe_m:
                        penetration = safe_m - dist

                        dcpa_boost = 1.0
                        if not is_static:
                            vx_xi = xi[3]*math.cos(xi[2])
                            vy_xi = xi[3]*math.sin(xi[2])
                            ddx = ox_k-xi[0]; ddy = oy_k-xi[1]
                            ddvx = obs_vx-vx_xi; ddvy = obs_vy-vy_xi
                            vr2 = ddvx**2+ddvy**2
                            if vr2 > 1e-6:
                                t_cpa_k = -(ddx*ddvx+ddy*ddvy)/vr2
                                if t_cpa_k > 0:
                                    dcpa = math.hypot(
                                        ddx+ddvx*t_cpa_k,
                                        ddy+ddvy*t_cpa_k) - obs_r
                                    dcpa_boost = 1.0 + 3.0/(max(dcpa,0.1)+0.5)

                        cost += pen * dcpa_boost * penetration**2 \
                              + pen * 0.1 * penetration

        # Terminal cost
        e_xt = traj[-1,0]-ref_traj[-1][0]
        e_yt = traj[-1,1]-ref_traj[-1][1]
        e_pt = self._wrap_angle(traj[-1,2]-ref_traj[-1][2])
        cost += pos_weight*2.0*(e_xt**2+e_yt**2)
        cost += heading_weight*2.0*e_pt**2
        return cost