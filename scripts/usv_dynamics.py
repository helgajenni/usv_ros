#!/usr/bin/env python3
import numpy as np
import rospy
import math

class USVDynamics:
    def __init__(self):
        # Mengambil dictionary '/usv' dari ROS Parameter Server
        usv_p = rospy.get_param('/usv', {})
        
        # Dictionary nilai default A1-A22 (Model Identifikasi Baru)
        default_A = {
            'A1':   1.5066, 'A2':  -0.7405, 'A3':   0.4219, 'A4':  -0.1397,
            'A5':  -0.1464, 'A6':  -3.1952, 'A7':   4.1189, 'A8':   0.0000,
            'A9':   0.0000, 'A10':  0.0845, 'A11':  0.0561, 'A12': -1.0495,
            'A13':  1.4038, 'A14': -2.0764, 'A15':  0.0010, 'A16':  0.9671,
            'A17':  0.0021, 'A18':  0.0178, 'A19':  0.0010, 'A20':  0.0000,
            'A21':  0.0000, 'A22':  0.0000,
        }

        # Dictionary nilai default parameter roll (menggantikan A23-A27)
        default_K = {
            'KpLin':  0.0000,   # Redaman roll linier
            'KpAbs':  0.0000,   # Redaman roll kuadratik
            'KpCub':  0.0000,   # Redaman roll kubik
            'Kphi':  13.5523,   # Momen righting: -Kphi * sin(phi)
            'Kfy':   -0.0175,   # Coupling sway force ke roll
            'Kv':    -3.3096,   # Coupling sway velocity ke roll
            'Kr':    -2.7576,   # Coupling yaw rate ke roll
            'Kdelta': 0.1738,   # Efek sudut kemudi (delta) ke roll
            'Kbias': -0.3631,   # Bias roll konstan
        }

        # Loop otomatis untuk mengisi self.A dan self.K dari YAML
        self.A = {}
        for key, default_val in default_A.items():
            self.A[key] = float(usv_p.get(key, default_val))

        self.K = {}
        for key, default_val in default_K.items():
            self.K[key] = float(usv_p.get(key, default_val))

        # State awal kapal (diambil dari parameter start: [1.0, 8.0])
        start_pt = rospy.get_param('/mission/start', [1.0, 8.0])

        # ✅ PERBAIKAN: START DARI DIAM (u=0), BUKAN u=1.5
        # state = [x, y, psi, phi, u, v, r, p]
        self.state = np.array([
            start_pt[0],  # x = 1.0
            start_pt[1],  # y = 8.0
            0.0,          # psi (heading) = 0
            0.0,          # phi (roll) = 0
            0.0,          # u (surge velocity) = 0 ← DIUBAH DARI 1.5!
            0.0,          # v (sway velocity) = 0
            0.0,          # r (yaw rate) = 0
            0.0           # p (roll rate) = 0
        ])

    def taylor_4dof(self, V, T, psi, phi):
        u, v, r, p = V[0], V[1], V[2], V[3]
        
        # Clamp states agar tidak blow-up
        u = np.clip(u, -5, 5)
        v = np.clip(v, -3, 3)
        r = np.clip(r, -3, 3)
        p = np.clip(p, -5, 5)
        phi = np.clip(phi, -math.radians(30), math.radians(30))
        
        # Pemetaan input aktuator:
        Fx    = T[0]   # surge force
        Fy    = T[1]   # sway force (TY, = 0 dari controller)
        Fy_yn = T[2]   # yaw control: masuk ke dr sebagai A19 * Fy_yn
        delta = T[3]   # sudut kemudi [rad]
        A = self.A
        K = self.K

        # ── Surge ────────────────────────────────────────────────────
        du = (A['A1']*v*r
              + A['A2']*u + A['A3']*abs(u)*u + A['A4']*(abs(u)**2)*u
              + A['A18']*Fx)

        # ── Sway ─────────────────────────────────────────────────────
        dv = (-(1.0/A['A1'])*u*r
              + A['A5']*v + A['A6']*abs(v)*v + A['A7']*(abs(v)**2)*v
              + A['A8']*abs(r)*v + A['A9']*abs(v)*r)

        # ── Roll (Model Baru: sin(phi), Kv, Kr, Kdelta, Kbias) ───────
        dp = (-K['KpLin']*p
              - K['KpAbs']*abs(p)*p
              - K['KpCub']*(abs(p)**2)*p
              - K['Kphi']*math.sin(phi)
              + K['Kfy']*Fy
              + K['Kv']*v
              + K['Kr']*r
              + K['Kdelta']*delta
              + K['Kbias'])

        # ── Yaw (tambah A20, A21, A22; kontrol melalui A19*Fy_yn) ────
        dr = (-A['A10']*v*u + A['A11']*u*v
              + A['A12']*r + A['A13']*abs(r)*r + A['A14']*(abs(r)**2)*r
              + A['A15']*abs(r)*u + A['A16']*abs(u)*r + A['A17']*abs(u)*u
              + A['A20']*abs(r)*u + A['A21']*abs(u)*r + A['A22']*abs(u)*u
              + A['A19']*Fy          # coupling sway force ke yaw (fisik)
              + A['A19']*Fy_yn)      # kontrol yaw (TN → melalui A19)
        
        # Anti-windup / Acceleration bounding
        du = np.clip(du, -10, 10)
        dv = np.clip(dv, -10, 10)
        dr = np.clip(dr, -10, 10)
        dp = np.clip(dp, -15, 15)
        
        Vdot = np.array([du, dv, dr, dp])
        
        # Kinematika (Rotation Matrix)
        R = np.array([
            [math.cos(psi), -math.sin(psi)*math.cos(phi), 0, 0],
            [math.sin(psi),  math.cos(psi)*math.cos(phi), 0, 0],
            [0,              0,                           math.cos(phi), 0],
            [0,              0,                           0, 1]
        ])
        eta_dot = np.dot(R, np.array([u, v, r, p]))
        
        return Vdot, eta_dot

    def step(self, Tcmd, dt):
        V = self.state[4:8]
        psi = self.state[2]
        phi = self.state[3]
        
        Vdot, eta_dot = self.taylor_4dof(V, Tcmd, psi, phi)
        
        # Euler Integration
        new_x = self.state[0] + eta_dot[0] * dt
        new_y = self.state[1] + eta_dot[1] * dt
        new_psi = (self.state[2] + eta_dot[2] * dt + np.pi) % (2 * np.pi) - np.pi
        new_phi = self.state[3] + eta_dot[3] * dt
        
        new_u = np.clip(self.state[4] + Vdot[0] * dt, -5, 5)
        new_v = np.clip(self.state[5] + Vdot[1] * dt, -3, 3)
        new_r = np.clip(self.state[6] + Vdot[2] * dt, -3, 3)
        new_p = np.clip(self.state[7] + Vdot[3] * dt, -5, 5)
        new_phi = np.clip(new_phi, -math.radians(30), math.radians(30))
        
        self.state = np.array([new_x, new_y, new_psi, new_phi, new_u, new_v, new_r, new_p])
        return self.state