#!/usr/bin/env python3
import numpy as np
import rospy
import math

class USVDynamics:
    def __init__(self):
        # Mengambil dictionary '/usv' dari ROS Parameter Server
        usv_p = rospy.get_param('/usv', {})
        
        # Dictionary nilai default untuk A1-A27 sebagai fallback pengaman
        default_A = {
            'A1': 1.0059,  'A2': -0.0087, 'A3': -0.2041, 'A4': -0.1302,
            'A5': -0.8849, 'A6': 2.9728,  'A7': -0.3969, 'A8': 3.4876,
            'A9': 6.8460,  'A10': 0.0000, 'A11': 0.0000, 'A12': -0.3062,
            'A13': 0.5764, 'A14': -0.9204, 'A15': 0.0000, 'A16': 0.2530,
            'A17': 0.0003, 'A18': 0.0007, 'A19': 0.0500, 'A20': 0.0000,
            'A21': 0.0000, 'A22': 0.0000, 'A23': 1.4384, 'A24': 0.0066,
            'A25': 0.0291, 'A26': -0.0001, 'A27': 0.0000
        }
        
        # Loop otomatis untuk mengisi self.A dari YAML
        self.A = {}
        for key, default_val in default_A.items():
            self.A[key] = float(usv_p.get(key, default_val))
            
        # State awal kapal (diambil dari parameter start: [1.0, 8.0])
        start_pt = rospy.get_param('/mission/start', [1.0, 8.0])
        self.state = np.array([start_pt[0], start_pt[1], 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])

    def taylor_4dof(self, V, T, psi, phi):
        u, v, r, p = V[0], V[1], V[2], V[3]
        
        # Clamp states agar tidak blow-up
        u = np.clip(u, -5, 5)
        v = np.clip(v, -3, 3)
        r = np.clip(r, -3, 3)
        p = np.clip(p, -5, 5)
        phi = np.clip(phi, -math.radians(30), math.radians(30))
        
        Fx, Fy = T[0], T[2] # TX (Surge), TN (Yaw)
        A = self.A
        
        # Dinamika Taylor 4-DOF
        du = A['A1']*v*r + A['A2']*u + A['A3']*abs(u)*u + A['A4']*(abs(u)**2)*u + A['A18']*Fx
        dv = -(1/A['A1'])*u*r + A['A5']*v + A['A6']*abs(v)*v + A['A7']*(abs(v)**2)*v + A['A8']*abs(r)*v + A['A9']*abs(v)*r
        dp = A['A20']*p + A['A21']*abs(p)*p + A['A22']*(abs(p)**2)*p - A['A23']*phi + A['A24']*u*r + A['A25']*v + A['A26']*Fy
        dr = -A['A10']*v*u + A['A11']*u*v + A['A12']*r + A['A13']*abs(r)*r + A['A14']*(abs(r)**2)*r + A['A15']*abs(r)*u + A['A16']*abs(u)*r + A['A17']*abs(u)*u + A['A19']*Fy
        
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
        new_psi = (self.state[2] + eta_dot[2] * dt + np.pi) % (2 * np.pi) - np.pi # wrapToPi
        new_phi = self.state[3] + eta_dot[3] * dt
        
        new_u = np.clip(self.state[4] + Vdot[0] * dt, -5, 5)
        new_v = np.clip(self.state[5] + Vdot[1] * dt, -3, 3)
        new_r = np.clip(self.state[6] + Vdot[2] * dt, -3, 3)
        new_p = np.clip(self.state[7] + Vdot[3] * dt, -5, 5)
        new_phi = np.clip(new_phi, -math.radians(30), math.radians(30))
        
        self.state = np.array([new_x, new_y, new_psi, new_phi, new_u, new_v, new_r, new_p])
        return self.state