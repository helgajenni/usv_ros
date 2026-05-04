#!/usr/bin/env python3
import rospy
import numpy as np
import math

class USVController:
    def __init__(self):
        self.delta = 2.5
        self.gamma = 0.5
        self.kappa = 0.02
        self.beta_hat = 0.0
        self.psi_d_filtered = 0.0
        
        self.Ku   = {'p': 800.0, 'i': 10.0, 'd': 150.0}
        self.Kpsi = {'p': 45.0,  'i': 1.5,  'd': 20.0}
        self.Kphi = {'p': 5.0,   'i': 0.1,  'd': 2.0}
        
        self.eInt_u, self.eInt_psi, self.eInt_phi = 0.0, 0.0, 0.0
        self.intMax = {'u': 25.0, 'psi': 8.0, 'phi': 3.0}
        self.psi_e_prev = 0.0 # Memori Anti-windup
        
        self.U0_saved = 1.5
        self.filt_r, self.filt_p = 0.0, 0.0
        self.phi_des_prev = 0.0
        
        self.k_bank = 0.3
        self.phi_max = math.radians(4.0)
        self.g = 9.81

        self.lims = {'TX': 40.0, 'TY': 60.0, 'TN': 35.0, 'TK': 60.0, 
                     'dTX': 200.0, 'dTN': 150.0, 'dTK': 200.0}
        
        self.TX_prev, self.TN_prev, self.TK_prev = 0.0, 0.0, 0.0

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_control(self, state, wp_prev, wp_next, dt, t, dist_goal):
        x, y, psi, phi, u, v, r, p = state
        
        dx_path = wp_next[0] - wp_prev[0]
        dy_path = wp_next[1] - wp_prev[1]
        
        if math.hypot(dx_path, dy_path) < 1e-6:
            alpha = math.atan2(wp_next[1] - y, wp_next[0] - x)
        else:
            alpha = math.atan2(dy_path, dx_path)
        
        ye = -math.sin(alpha)*(x - wp_prev[0]) + math.cos(alpha)*(y - wp_prev[1])
        
        if abs(ye) > 5.0:
            self.beta_hat = 0.0
            
        ye_clamped = np.clip(ye, -self.delta * 1.2, self.delta * 1.2)
        self.beta_hat += dt * self.kappa * (ye_clamped / self.delta)
        self.beta_hat = np.clip(self.beta_hat, -0.8, 0.8)
        
        psi_d_raw = self.wrap_to_pi(alpha - math.atan2(ye_clamped + self.gamma * self.beta_hat, self.delta))
        self.psi_d_filtered += 1.0 * self.wrap_to_pi(psi_d_raw - self.psi_d_filtered)
        psi_d = self.wrap_to_pi(self.psi_d_filtered)
        
        psi_e = self.wrap_to_pi(psi_d - psi)
        fade_in = min(1.0, t / 4.0)
        
        if dist_goal < 1.5:
            target_U0 = max(0.6, 1.5 * (dist_goal / 1.5))
        else:
            target_U0 = 1.5
            
        self.U0_saved += 0.15 * (target_U0 - self.U0_saved)
        e_u = self.U0_saved - u
        
        U_eff = max(0.3, math.hypot(u, v))
        kappa_approx = 2 * math.sin(psi_e) / max(0.5, self.delta)
        a_y_cmd = (U_eff**2) * kappa_approx
        phi_cmd = self.k_bank * math.atan(a_y_cmd / self.g)
        phi_cmd = np.clip(phi_cmd, -self.phi_max, self.phi_max)
        
        alpha_phi = dt / (1.0 + dt)
        phi_des = self.phi_des_prev + alpha_phi * (phi_cmd - self.phi_des_prev)
        self.phi_des_prev = phi_des
        e_phi = phi_des - phi
        
        # --- PERBAIKAN: Anti-windup Integrator Reset ---
        if t > 0 and (psi_e * self.psi_e_prev < 0):
            self.eInt_psi *= 0.5
        self.psi_e_prev = psi_e

        self.eInt_u = np.clip(self.eInt_u + e_u*dt, -self.intMax['u'], self.intMax['u'])
        self.eInt_psi = np.clip(self.eInt_psi + psi_e*dt, -self.intMax['psi'], self.intMax['psi'])
        self.eInt_phi = np.clip(self.eInt_phi + e_phi*dt, -self.intMax['phi'], self.intMax['phi'])
        
        self.filt_r += 0.3 * (r - self.filt_r)
        self.filt_p += 0.3 * (p - self.filt_p)
        
        TX = self.Ku['p']*e_u     + self.Ku['i']*self.eInt_u   - self.Ku['d']*u
        TN = self.Kpsi['p']*psi_e + self.Kpsi['i']*self.eInt_psi - self.Kpsi['d']*self.filt_r
        TK = self.Kphi['p']*e_phi + self.Kphi['i']*self.eInt_phi - self.Kphi['d']*self.filt_p
        TY = 0.0
        
        TX = fade_in * np.clip(TX, -self.lims['TX'], self.lims['TX'])
        TY = fade_in * np.clip(TY, -self.lims['TY'], self.lims['TY'])
        TN = fade_in * np.clip(TN, -self.lims['TN'], self.lims['TN'])
        TK = fade_in * np.clip(TK, -self.lims['TK'], self.lims['TK'])
        
        max_dTX, max_dTN, max_dTK = self.lims['dTX']*dt, self.lims['dTN']*dt, self.lims['dTK']*dt
        TX = self.TX_prev + np.clip(TX - self.TX_prev, -max_dTX, max_dTX)
        TN = self.TN_prev + np.clip(TN - self.TN_prev, -max_dTN, max_dTN)
        TK = self.TK_prev + np.clip(TK - self.TK_prev, -max_dTK, max_dTK)
        
        self.TX_prev, self.TN_prev, self.TK_prev = TX, TN, TK
        
        return np.array([TX, TY, TN, TK]), ye, psi_e, psi_d