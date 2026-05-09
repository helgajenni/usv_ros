#!/usr/bin/env python3
"""
USV Controller — ILOS + PID 4-DOF
Setara dengan MATLAB: ILOS Guidance + PID Controller (A1-A27 Taylor Model)
"""
import numpy as np
import math


class USVController:
    def __init__(self):
        import rospy
        # ── Baca dari ROS Parameter Server (params.yaml) ─────────
        c = rospy.get_param('/ctrl', {})
        l = rospy.get_param('/usv/lims', {})

        # ILOS Guidance
        self.delta    = float(c.get('ILOS_Delta', 1.8))
        self.gamma    = float(c.get('ILOS_gamma', 0.5))
        self.kappa    = float(c.get('ILOS_kappa', 0.45))
        self.beta_hat = 0.0
        self.psi_d_filtered = 0.0  # Diinisialisasi dari main loop

        # PID Gains
        self.Ku   = {'p': float(c.get('Ku_p',   55.0)),
                     'i': float(c.get('Ku_i',     0.5)),
                     'd': float(c.get('Ku_d',    7.0))}
        self.Kpsi = {'p': float(c.get('Kpsi_p',   2250.0)),
                     'i': float(c.get('Kpsi_i',    75.0)),
                     'd': float(c.get('Kpsi_d',   2000.0))}
        self.Kphi = {'p': float(c.get('Kphi_p',    120.0)),
                     'i': float(c.get('Kphi_i',    0.5)),
                     'd': float(c.get('Kphi_d',    80.0))}

        self.eInt_u   = 0.0
        self.eInt_psi = 0.0
        self.eInt_phi = 0.0
        self.intMax   = {'u':   float(c.get('intMax_u',   25.0)),
                         'psi': float(c.get('intMax_psi',  8.0)),
                         'phi': float(c.get('intMax_phi',  1.0))}
        self.psi_e_prev = 0.0

        self.U0_saved     = float(c.get('U0_target', 1.5))
        self.filt_r       = 0.0
        self.filt_p       = 0.0
        self.phi_des_prev = 0.0

        self.k_bank  = 0.4
        self.phi_max = math.radians(10.0) #ini tuh 0.7 rad
        self.g       = 9.81

        # Actuator Limits (dari params.yaml /usv/lims)
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

    # ─────────────────────────────────────────────────────────────
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ─────────────────────────────────────────────────────────────
    def compute_control(self, state, wp_prev, wp_next, dt, t,
                        dist_goal, target_U0=1.5):
        """
        ILOS Guidance + PID 4-DOF Controller.

        Parameters
        ----------
        state      : [x, y, psi, phi, u, v, r, p]
        wp_prev    : titik awal segmen jalur (nearest path point)
        wp_next    : titik akhir segmen jalur (lookahead point)
        dt         : time step [s]
        t          : waktu simulasi [s]
        dist_goal  : jarak ke goal [m]
        target_U0  : kecepatan target berbasis curvature (dari main loop)

        Returns
        -------
        Tcmd  : [TX, TY, TN, TK]
        ye    : cross-track error [m]
        psi_e : heading error [rad]
        psi_d : desired heading [rad]
        """
        x, y, psi, phi, u, v, r, p = state

        # ── 1. ILOS GUIDANCE ─────────────────────────────────────
        dx_path = float(wp_next[0]) - float(wp_prev[0])
        dy_path = float(wp_next[1]) - float(wp_prev[1])

        if math.hypot(dx_path, dy_path) < 1e-6:
            alpha = math.atan2(float(wp_next[1]) - y, float(wp_next[0]) - x)
        else:
            alpha = math.atan2(dy_path, dx_path)

        # Cross-track error
        ye = (-math.sin(alpha) * (x - float(wp_prev[0]))
              + math.cos(alpha) * (y - float(wp_prev[1])))

        if abs(ye) > 5.0:
            self.beta_hat = 0.0

        delta_adaptive = self.delta  # fixed lookahead, mencegah ILOS osilasi saat ye besar

        max_ye       = delta_adaptive * 1.2
        ye_clamped   = float(np.clip(ye, -max_ye, max_ye))

        self.beta_hat += dt * self.kappa * (ye_clamped / delta_adaptive)
        self.beta_hat  = float(np.clip(self.beta_hat, -1.0, 1.0))

        psi_d_raw = self.wrap_to_pi(
            alpha - math.atan2(ye_clamped + self.gamma * self.beta_hat,
                               delta_adaptive))

        # Low-pass filter psi_d (alpha=0.7): lebih responsif untuk antisipasi tikungan
        self.psi_d_filtered += 0.7 * self.wrap_to_pi(psi_d_raw - self.psi_d_filtered)
        psi_d = self.wrap_to_pi(self.psi_d_filtered)

        # ── 2. ERRORS ────────────────────────────────────────────
        psi_e   = self.wrap_to_pi(psi_d - psi)
        fade_in = min(1.0, t / 4.0)   # 4 detik ramp (sesuai MATLAB)

        # ── 3. SPEED SETPOINT ────────────────────────────────────
        # target_U0 sudah dihitung berbasis curvature di main loop
        if dist_goal < 6.0:
            self.U0_saved = target_U0
        else:
            self.U0_saved += 0.15 * (target_U0 - self.U0_saved)  # alpha_u=0.15

        e_u = self.U0_saved - u

        ## ── 4. ROLL REFERENCE (banking) ──────────────────────────────
        phi_des = 0.0                        # ← Selalu ingin tegak
        self.phi_des_prev = phi_des
        e_phi = phi_des - phi                # ← Error = seberapa miring kapal

        # Integral dengan anti-windup
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
        # α=0.1 (bukan 0.3): lebih agresif mengeliminasi noise roll (ωn≈3.7 rad/s)
        # yang masuk ke D-term yaw via filt_r → mengurangi feedback roll-yaw-CTE
        self.filt_r += 0.1 * (r - self.filt_r)
        self.filt_p += 0.05 * (p - self.filt_p)

        # ── 7. PID CONTROL OUTPUTS ───────────────────────────────
        # TX: feedforward (steady-state thrust) + PID correction
        # TX_ff = -(A2*u + A3*|u|u + A4*|u|²u) / A18  saat steady state (du=0)
        # Parameter baru: A2=-0.7405, A3=+0.4219, A4=-0.1397, A18=0.0178
        u_ref  = self.U0_saved
        TX_ff  = (0.7405*u_ref
                  - 0.4219*abs(u_ref)*u_ref
                  + 0.1397*(u_ref**2)*u_ref) / 0.0178 + self.Ku['d'] * u_ref
        TX = TX_ff + self.Ku['p']*e_u + self.Ku['i']*self.eInt_u - self.Ku['d']*u

        # TN: yaw control force [N] (bow thruster equivalent, limit 1750 N)
        TN = (self.Kpsi['p']*psi_e
              + self.Kpsi['i']*self.eInt_psi
              - self.Kpsi['d']*self.filt_r)

        # TK: 0 (rudder tidak dipakai untuk roll)
        TK = (self.Kphi['p'] * e_phi
            + self.Kphi['i'] * self.eInt_phi
            - self.Kphi['d'] * self.filt_p)

        TY = 0.0

        # ── 8. CLIPPING + FADE-IN ────────────────────────────────
        TX = fade_in * float(np.clip(TX, -self.lims['TX'], self.lims['TX']))
        TY = fade_in * float(np.clip(TY, -self.lims['TY'], self.lims['TY']))
        TN = fade_in * float(np.clip(TN, -self.lims['TN'], self.lims['TN']))
        TK = fade_in * float(np.clip(TK, -self.lims['TK'], self.lims['TK']))

        # ── 9. RATE LIMITER ──────────────────────────────────────
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