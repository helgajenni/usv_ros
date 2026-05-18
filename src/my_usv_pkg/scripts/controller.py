#!/usr/bin/env python3
"""
USV Controller — ILOS + PID 4-DOF  (v11)
Perubahan dari versi sebelumnya:
  - compute_control() tambah argumen lmpc_mode='GLOBAL'
  - alpha psi_d_filtered: 0.1 (GLOBAL/AVOID) → 0.25 (LMPC_RETURN)
    Efek: saat return, heading referensi mengejar jalur global 2.5×
    lebih cepat → CTE kembali <1m dalam ~4s bukan ~8s
  - Tidak ada perubahan lain: PID gains, ILOS, rate limiter tetap sama
"""
import numpy as np
import math


class USVController:
    def __init__(self):
        import rospy
        c = rospy.get_param('/ctrl', {})
        l = rospy.get_param('/usv/lims', {})

        self.delta    = float(c.get('ILOS_Delta', 2.5))
        self.gamma    = float(c.get('ILOS_gamma', 0.5))
        self.kappa    = float(c.get('ILOS_kappa', 0.45))
        self.beta_hat = 0.0
        self.psi_d_filtered = 0.0

        self.Ku   = {'p': float(c.get('Ku_p',   55.0)),
                     'i': float(c.get('Ku_i',     0.5)),
                     'd': float(c.get('Ku_d',    7.0))}
        self.Kpsi = {'p': float(c.get('Kpsi_p',   2250.0)),
                     'i': float(c.get('Kpsi_i',    30.0)),
                     'd': float(c.get('Kpsi_d',   2000.0))}
        self.Kphi = {'p': float(c.get('Kphi_p',    120.0)),
                     'i': float(c.get('Kphi_i',    1.0)),
                     'd': float(c.get('Kphi_d',    80.0))}

        self.eInt_u   = 0.0
        self.eInt_psi = 0.0
        self.eInt_phi = 0.0
        self.intMax   = {'u':   float(c.get('intMax_u',   25.0)),
                         'psi': float(c.get('intMax_psi',  4.0)),
                         'phi': float(c.get('intMax_phi',  1.0))}
        self.psi_e_prev = 0.0

        self.U0_saved     = float(c.get('U0_target', 1.5))
        self.filt_r       = 0.0
        self.filt_p       = 0.0
        self.phi_des_prev = 0.0

        self.k_bank  = 0.4
        self.phi_max = math.radians(10.0)
        self.g       = 9.81

        self.lims = {
            'TX':  float(l.get('TX',   200.0)),
            'TY':  float(l.get('TY',     60.0)),
            'TN':  float(l.get('TN',     1750.0)),
            'TK':  float(l.get('TK',     0.7)),
            'dTX': float(l.get('dTX',   200.0)),
            'dTN': float(l.get('dTN',   7500.0)),
            'dTK': float(l.get('dTK',   5.0)),
        }

        self.TX_prev = 0.0
        self.TN_prev = 0.0
        self.TK_prev = 0.0

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # lmpc_mode ditambahkan — default 'GLOBAL' agar backward compatible
    def compute_control(self, state, wp_prev, wp_next, dt, t,
                        dist_goal, target_U0=1.5, lmpc_mode='GLOBAL'):
        x, y, psi, phi, u, v, r, p = state

        # ── 1. ILOS GUIDANCE ─────────────────────────────────────
        dx_path = float(wp_next[0]) - float(wp_prev[0])
        dy_path = float(wp_next[1]) - float(wp_prev[1])

        if math.hypot(dx_path, dy_path) < 1e-6:
            alpha = math.atan2(float(wp_next[1]) - y, float(wp_next[0]) - x)
        else:
            alpha = math.atan2(dy_path, dx_path)

        ye = (-math.sin(alpha) * (x - float(wp_prev[0]))
              + math.cos(alpha) * (y - float(wp_prev[1])))

        delta_adaptive = max(self.delta, abs(ye) * 1.5)

        if abs(ye) > 1.5:
            self.beta_hat = 0.0
        else:
            max_ye = self.delta * 1.2
            ye_clamped = float(np.clip(ye, -max_ye, max_ye))
            self.beta_hat += dt * self.kappa * (ye_clamped / self.delta)
            self.beta_hat = float(np.clip(self.beta_hat, -1.0, 1.0))

        psi_d_raw = self.wrap_to_pi(
            alpha - math.atan2(ye + self.gamma * self.beta_hat, delta_adaptive))

        # ── v11: alpha adaptif berdasarkan mode ───────────────────
        # GLOBAL/AVOID : 0.1 — smooth, tidak overshoot
        # LMPC_RETURN  : 0.25 — 2.5× lebih cepat mengejar jalur
        #   Kenapa 0.25 dan bukan lebih besar (misal 0.5)?
        #   0.5 → psi_d bisa lompat 10-15 deg per step → TN spike
        #   0.25 → perubahan ~5 deg per step → smooth tapi terarah
        alpha_filter = 0.25 if lmpc_mode == 'LMPC_RETURN' else 0.1
        self.psi_d_filtered += alpha_filter * self.wrap_to_pi(
            psi_d_raw - self.psi_d_filtered)
        psi_d = self.wrap_to_pi(self.psi_d_filtered)

        # ── 2. ERRORS ────────────────────────────────────────────
        psi_e   = self.wrap_to_pi(psi_d - psi)
        fade_in = min(1.0, t / 4.0)

        # ── 3. SPEED SETPOINT ────────────────────────────────────
        if dist_goal < 6.0:
            self.U0_saved = target_U0
        else:
            self.U0_saved += 0.15 * (target_U0 - self.U0_saved)

        e_u = self.U0_saved - u

        # ── 4. ROLL REFERENCE ────────────────────────────────────
        phi_des = 0.0
        self.phi_des_prev = phi_des
        e_phi = phi_des - phi

        self.eInt_phi = float(np.clip(
            self.eInt_phi + e_phi * dt,
            -self.intMax['phi'], self.intMax['phi']))

        # ── 5. ANTI-WINDUP ───────────────────────────────────────
        if t > 0 and (psi_e * self.psi_e_prev < 0):
            self.eInt_psi *= 0.5
        self.psi_e_prev = psi_e

        self.eInt_u   = float(np.clip(self.eInt_u   + e_u   * dt,
                                      -self.intMax['u'],   self.intMax['u']))
        self.eInt_psi = float(np.clip(self.eInt_psi + psi_e * dt,
                                      -self.intMax['psi'], self.intMax['psi']))
        self.eInt_phi = float(np.clip(self.eInt_phi + e_phi  * dt,
                                      -self.intMax['phi'], self.intMax['phi']))

        # ── 6. DERIVATIVE FILTER ─────────────────────────────────
        self.filt_r += 0.8 * (r - self.filt_r)
        self.filt_p += 0.8 * (p - self.filt_p)

        # ── 7. PID CONTROL OUTPUTS ───────────────────────────────
        u_ref  = self.U0_saved
        TX_ff  = (0.7405*u_ref
                  - 0.4219*abs(u_ref)*u_ref
                  + 0.1397*(u_ref**2)*u_ref) / 0.0178 + self.Ku['d'] * u_ref
        TX = TX_ff + self.Ku['p']*e_u + self.Ku['i']*self.eInt_u - self.Ku['d']*u

        max_psi_e = math.radians(20.0)
        psi_e_clamped = float(np.clip(psi_e, -max_psi_e, max_psi_e))

        TN = (self.Kpsi['p'] * psi_e_clamped
              + self.Kpsi['i'] * self.eInt_psi
              - self.Kpsi['d'] * self.filt_r)

        TK = (self.Kphi['p'] * e_phi
            + self.Kphi['i'] * self.eInt_phi
            - self.Kphi['d'] * self.filt_p)

        TY = 0.0

        # ── 8. ACTIVE BRAKING ────────────────────────────────────
        if target_U0 <= 0.01 and dist_goal < 2.0:
            TX = -180.0 if u > 0.15 else 0.0
            TN = 0.0
            TK = 0.0

        # ── 9. CLIPPING + FADE-IN ────────────────────────────────
        TX = fade_in * float(np.clip(TX, -self.lims['TX'], self.lims['TX']))
        TY = fade_in * float(np.clip(TY, -self.lims['TY'], self.lims['TY']))
        TN = fade_in * float(np.clip(TN, -self.lims['TN'], self.lims['TN']))
        TK = fade_in * float(np.clip(TK, -self.lims['TK'], self.lims['TK']))

        # ── 10. RATE LIMITER ─────────────────────────────────────
        dTX = self.lims['dTX'] * dt
        dTN = self.lims['dTN'] * dt
        dTK = self.lims['dTK'] * dt
        TX = self.TX_prev + float(np.clip(TX - self.TX_prev, -dTX, dTX))
        TN = self.TN_prev + float(np.clip(TN - self.TN_prev, -dTN, dTN))
        TK = self.TK_prev + float(np.clip(TK - self.TK_prev, -dTK, dTK))

        self.TX_prev = TX
        self.TN_prev = TN
        self.TK_prev = TK

        return np.array([TX, TY, TN, TK]), ye, psi_e, psi_d