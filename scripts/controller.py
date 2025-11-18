import numpy as np

def controller_guard_3dof_ilos(eta, Vb, psi_ref, u_ref, e_align_deg,
                               Kpsi_align, obstacles, safeDist, avoid,
                               Ku, Kr, Tmax, Nmax, r_max_cmd, Kv, Kdv, Ymax, v_ref_sway):
    x, y, psi = eta
    u, v, r = Vb
    # nearest obstacle
    d_min = np.inf
    avoid_dir = np.zeros(2)
    r0 = 0
    for obs in obstacles:
        cx, cy, rr = obs
        vv = np.array([x - cx, y - cy])
        d = np.linalg.norm(vv)
        if d < d_min:
            d_min = d
            avoid_dir = vv / max(d, 1e-9)
            r0 = rr
    clr = d_min - r0

    # guard
    yaw_avoid = 0
    u_ref_eff = u_ref
    if clr < 2.0*safeDist:
        psi_away = np.arctan2(avoid_dir[1], avoid_dir[0])
        dpsiAway = np.arctan2(np.sin(psi_away - psi), np.cos(psi_away - psi))
        yaw_avoid = avoid['yaw_gain'] * dpsiAway
        if clr < safeDist:
            u_ref_eff = u_ref * max(0.3, clr/safeDist)
    e_psi = np.arctan2(np.sin(psi_ref - psi), np.cos(psi_ref - psi))
    r_cmd = np.clip(Kpsi_align*e_psi + yaw_avoid, -r_max_cmd, r_max_cmd)
    Fx = Ku*(u_ref_eff - u)
    Fx = np.clip(Fx, -Tmax, Tmax)
    Mz = Kr*(r_cmd - r)
    Mz = np.clip(Mz, -Nmax, Nmax)
    Fy = Kv*(v_ref_sway - v) - Kdv*v
    Fy = np.clip(Fy, -Ymax, Ymax)
    return np.array([Fx, Fy, Mz])

def ilos_heading(psi_path, ye, ilos, dt):
    Delta = max(1e-3, ilos['Delta_h'])
    ilos['z'] = ilos['z'] + dt * (ye / np.sqrt(ye*ye + Delta*Delta))
    ilos['z'] = np.clip(ilos['z'], -3.0, 3.0)
    psi_ref = psi_path - np.arctan2(ye, Delta) - ilos['kappa'] * ilos['z']
    return psi_ref, ilos