#!/usr/bin/env python3
"""
USV Path Planning Node — ROS Implementation
RRT* + G2CBS C^2 + ILOS + PID Taylor 4-DOF
"""
import rospy
import numpy as np
import math
import random
import csv
import os
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
from usv_dynamics import USVDynamics
from controller import USVController
from lmpc_controller import LMPCController
from visualization_msgs.msg import Marker, MarkerArray

# ================================================================
#  G2CBS
# ================================================================

def _natural_spline_M(t_arr, y_arr):
    n = len(y_arr)
    if n <= 2:
        return np.zeros(n)
    h = np.diff(t_arr)
    A = np.zeros((n, n))
    d = np.zeros(n)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2.0 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        d[i] = 6.0 * ((y_arr[i+1] - y_arr[i]) / h[i]
                      - (y_arr[i] - y_arr[i-1]) / h[i-1])
    return np.linalg.solve(A, d)


def smooth_path_g2cbs_c2(raw_path, n_per_seg=40):
    keep = np.r_[True, np.linalg.norm(np.diff(raw_path, axis=0), axis=1) > 1e-8]
    P = raw_path[keep]
    if len(P) <= 2:
        return P
    n = len(P)
    t_arr = np.zeros(n)
    for i in range(1, n):
        t_arr[i] = t_arr[i-1] + np.linalg.norm(P[i] - P[i-1])
    if t_arr[-1] < 1e-9:
        return P
    h  = np.diff(t_arr)
    Mx = _natural_spline_M(t_arr, P[:, 0])
    My = _natural_spline_M(t_arr, P[:, 1])
    sx = np.diff(P[:, 0]) / h
    sy = np.diff(P[:, 1]) / h
    mx = np.zeros(n); my = np.zeros(n)
    mx[0] = sx[0]  - h[0] * (2*Mx[0]  + Mx[1])  / 6.0
    my[0] = sy[0]  - h[0] * (2*My[0]  + My[1])  / 6.0
    for i in range(1, n - 1):
        mx[i] = 0.5 * (sx[i-1] + h[i-1]*(Mx[i-1] + 2*Mx[i]) / 6.0
                       + sx[i]  - h[i]  *(2*Mx[i] + Mx[i+1]) / 6.0)
        my[i] = 0.5 * (sy[i-1] + h[i-1]*(My[i-1] + 2*My[i]) / 6.0
                       + sy[i]  - h[i]  *(2*My[i] + My[i+1]) / 6.0)
    mx[-1] = sx[-1] + h[-1] * (Mx[-2] + 2*Mx[-1]) / 6.0
    my[-1] = sy[-1] + h[-1] * (My[-2] + 2*My[-1]) / 6.0
    tau = np.linspace(0, 1, n_per_seg).reshape(-1, 1)
    segments = []
    for i in range(n - 1):
        hi = h[i]
        b0 = P[i]; b3 = P[i+1]
        b1 = b0 + (hi / 3.0) * np.array([mx[i],   my[i]])
        b2 = b3 - (hi / 3.0) * np.array([mx[i+1], my[i+1]])
        pts = ((1-tau)**3*b0 + 3*(1-tau)**2*tau*b1
               + 3*(1-tau)*tau**2*b2 + tau**3*b3)
        segments.append(pts[1:] if i > 0 else pts)
    result = np.vstack(segments)
    keep2  = np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-8]
    return result[keep2]


def downsample_path(path, min_dist=3.0):
    if len(path) < 3:
        return path
    res = [path[0]]
    for pt in path[1:-1]:
        if np.linalg.norm(pt - res[-1]) > min_dist:
            res.append(pt)
    res.append(path[-1])
    return np.array(res)


def smooth_path_chaikin(path, iterations=4):
    if len(path) < 3:
        return path
    curr_path = path
    for _ in range(iterations):
        smoothed = [curr_path[0]]
        for i in range(len(curr_path) - 1):
            p0 = curr_path[i]; p1 = curr_path[i+1]
            smoothed.extend([0.75*p0 + 0.25*p1, 0.25*p0 + 0.75*p1])
        smoothed.append(curr_path[-1])
        curr_path = np.array(smoothed)
    return curr_path


def compute_path_curvature(path):
    n = len(path)
    if n < 3:
        return np.zeros(n)
    x, y = path[:, 0], path[:, 1]
    d = np.maximum(np.sqrt(np.diff(x)**2 + np.diff(y)**2), 1e-12)
    s = np.r_[0, np.cumsum(d)]
    xp  = np.zeros(n); yp  = np.zeros(n)
    xpp = np.zeros(n); ypp = np.zeros(n)
    for i in range(1, n - 1):
        h1 = s[i]-s[i-1]; h2 = s[i+1]-s[i]
        xp[i]  = (x[i+1]*h1**2 + x[i]*(h2**2-h1**2) - x[i-1]*h2**2) / (h1*h2*(h1+h2))
        yp[i]  = (y[i+1]*h1**2 + y[i]*(h2**2-h1**2) - y[i-1]*h2**2) / (h1*h2*(h1+h2))
        xpp[i] = 2*(x[i+1]*h1 - x[i]*(h1+h2) + x[i-1]*h2) / (h1*h2*(h1+h2))
        ypp[i] = 2*(y[i+1]*h1 - y[i]*(h1+h2) + y[i-1]*h2) / (h1*h2*(h1+h2))
    xp[0]  = (x[1]-x[0])/(s[1]-s[0]);     yp[0]  = (y[1]-y[0])/(s[1]-s[0])
    xp[-1] = (x[-1]-x[-2])/(s[-1]-s[-2]); yp[-1] = (y[-1]-y[-2])/(s[-1]-s[-2])
    xpp[0] = xpp[1];  ypp[0]  = ypp[1]
    xpp[-1]= xpp[-2]; ypp[-1] = ypp[-2]
    den = np.maximum((xp**2 + yp**2)**1.5, 1e-9)
    return (xp*ypp - yp*xpp) / den


def repair_path_obstacles(path, obs, margin):
    path = path.copy()
    for _ in range(30):
        any_fixed = False
        for i in range(1, len(path) - 1):
            for o in obs:
                dx   = path[i, 0] - o[0]; dy = path[i, 1] - o[1]
                dist = math.hypot(dx, dy); excl = o[2] + margin
                if dist < excl:
                    if dist < 1e-6: dx, dy, dist = 1.0, 0.0, 1.0
                    path[i, 0] = o[0] + excl * dx / dist
                    path[i, 1] = o[1] + excl * dy / dist
                    any_fixed  = True
        if not any_fixed:
            break
    return path


# ================================================================
#  RRT*
# ================================================================

def is_collision_free(p1, p2, obs, margin):
    dist  = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    steps = max(20, int(dist / 0.5))
    for s in np.linspace(0, 1, steps):
        px = p1[0] + s*(p2[0]-p1[0]); py = p1[1] + s*(p2[1]-p1[1])
        for o in obs:
            if math.hypot(px-o[0], py-o[1]) < (o[2]+margin):
                return False
    return True


def shortcut_path(path, obs, margin, n_iter=600, max_shortcut_m=15.0):
    path = list(path)
    for _ in range(n_iter):
        if len(path) < 3: break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, min(i + 20, len(path) - 1))
        dist_ij = np.linalg.norm(np.array(path[j]) - np.array(path[i]))
        if dist_ij <= max_shortcut_m and is_collision_free(
                np.array(path[i]), np.array(path[j]), obs, margin):
            path = path[:i+1] + path[j:]
    return np.array(path)


def rrt_star(start_pt, goal_pt, obs, map_sz, rrt_p, margin):
    nodes   = [np.array(start_pt)]
    parents = [0]
    costs   = [0.0]
    for _ in range(rrt_p['maxIter']):
        if random.random() < rrt_p['goalBias']:
            sample = np.array(goal_pt)
        else:
            sample = np.array([random.uniform(0, map_sz[1]),
                               random.uniform(0, map_sz[0])])
        dists   = np.linalg.norm(np.array(nodes) - sample, axis=1)
        n_idx   = np.argmin(dists); nearest = nodes[n_idx]
        dir_vec = sample - nearest; dist = np.linalg.norm(dir_vec)
        new_pt  = nearest + rrt_p['stepSize']*(dir_vec/dist) if dist > rrt_p['stepSize'] else sample
        if not (0 <= new_pt[0] <= map_sz[1] and 0 <= new_pt[1] <= map_sz[0]): continue
        if not is_collision_free(nearest, new_pt, obs, margin): continue
        dists_new   = np.linalg.norm(np.array(nodes) - new_pt, axis=1)
        near_idx    = np.where(dists_new <= rrt_p['rewireRad'])[0]
        best_parent = n_idx
        best_cost   = costs[n_idx] + np.linalg.norm(new_pt - nodes[n_idx])
        for ni in near_idx:
            c = costs[ni] + np.linalg.norm(new_pt - nodes[ni])
            if c < best_cost and is_collision_free(nodes[ni], new_pt, obs, margin):
                best_parent = ni; best_cost = c
        nodes.append(new_pt); parents.append(best_parent); costs.append(best_cost)
        new_i = len(nodes) - 1
        for ni in near_idx:
            nc = costs[new_i] + np.linalg.norm(nodes[ni] - new_pt)
            if nc < costs[ni] and is_collision_free(new_pt, nodes[ni], obs, margin):
                parents[ni] = new_i; costs[ni] = nc
        if np.linalg.norm(new_pt - goal_pt) < rrt_p['goalTol']: break
    dists_goal = np.linalg.norm(np.array(nodes) - goal_pt, axis=1)
    goal_idx   = np.argmin(dists_goal)
    path = [np.array(goal_pt)]; idx = goal_idx
    while idx != 0:
        path.append(nodes[idx]); idx = parents[idx]
    path.append(np.array(start_pt)); path.reverse()
    return np.array(path)


# ================================================================
#  LOGGING & PLOT
# ================================================================

def save_and_plot_results(log_data):
    filename = os.path.expanduser("~/catkin_ws/usv_simulation_results.csv")
    with open(filename, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Time','Psi_deg','Psi_err_deg','CTE_m',
                    'Surge_u','Sway_v','Yaw_Rate_r_deg',
                    'Yaw_Torque_TN','Roll_phi_deg','Roll_Rate_p_deg',
                    'Mode','Obs_MinDist_m'])
        for row in log_data:
            w.writerow(row)

    t        = [r[0]  for r in log_data]
    psi      = [r[1]  for r in log_data]
    psi_err  = [r[2]  for r in log_data]
    cte      = [r[3]  for r in log_data]
    u_log    = [r[4]  for r in log_data]
    v_log    = [r[5]  for r in log_data]
    r_log    = [r[6]  for r in log_data]
    tn_log   = [r[7]  for r in log_data]
    phi_log  = [r[8]  for r in log_data]
    p_log    = [r[9]  for r in log_data]
    mode_arr = [r[10] for r in log_data]
    obs_dist = [r[11] for r in log_data]

    cte_arr = np.array(cte)
    rospy.loginfo("=== ANALISIS ERROR ===")
    rospy.loginfo(f"Max |CTE|  : {np.max(np.abs(cte_arr)):.3f} m")
    rospy.loginfo(f"MAE CTE    : {np.mean(np.abs(cte_arr)):.3f} m")
    rospy.loginfo(f"RMSE CTE   : {np.sqrt(np.mean(cte_arr**2)):.3f} m")
    n_lmpc = sum(1 for m in mode_arr if m != 'GLOBAL')
    rospy.loginfo(f"LMPC aktif : {n_lmpc}/{len(mode_arr)} steps "
                  f"({100*n_lmpc/max(len(mode_arr),1):.1f}%)")

    mode_color = {
        'GLOBAL':      'lightblue',
        'LMPC_AVOID':  'lightyellow',
        'LMPC_RETURN': 'lightsalmon',
    }

    def shade_modes(ax, t_arr, m_arr):
        if len(t_arr) < 2: return
        i0 = 0
        for i in range(1, len(m_arr)):
            if m_arr[i] != m_arr[i-1] or i == len(m_arr)-1:
                c = mode_color.get(m_arr[i0], 'white')
                ax.axvspan(t_arr[i0], t_arr[min(i, len(t_arr)-1)],
                           alpha=0.25, color=c, zorder=0)
                i0 = i

    def make_plot(ax, t_arr, y_arr, m_arr, title, color='steelblue',
                  hline=None, hline_label=None, hline_color='r'):
        shade_modes(ax, t_arr, m_arr)
        ax.plot(t_arr, y_arr, color=color, linewidth=1.2, zorder=2)
        if hline is not None:
            ax.axhline(y=hline, color=hline_color, linestyle='--',
                       linewidth=1, label=hline_label, zorder=1)
            if hline_label: ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('t [s]', fontsize=8)
        ax.grid(True, alpha=0.4)

    import matplotlib.patches as mpatches
    fig, axs = plt.subplots(3, 4, figsize=(22, 12))
    fig.canvas.manager.set_window_title('USV States Performance — LMPC + ILOS')

    make_plot(axs[0,0], t, psi,     mode_arr, 'Heading ψ [deg]')
    make_plot(axs[0,1], t, psi_err, mode_arr, 'Heading Error [deg]', color='green')
    make_plot(axs[0,2], t, cte,     mode_arr, 'Cross-Track Error [m]', color='purple')

    shade_modes(axs[0,3], t, mode_arr)
    axs[0,3].plot(t, obs_dist, color='darkred', linewidth=1.2, zorder=2)
    axs[0,3].axhline(y=1.0, color='red',    linestyle='--', label='d_safe=1.0m')
    axs[0,3].axhline(y=2.5, color='orange', linestyle='--', label='d_warn=2.5m')
    axs[0,3].legend(fontsize=7)
    axs[0,3].set_title('Min Obstacle Distance [m]', fontsize=9)
    axs[0,3].set_xlabel('t [s]', fontsize=8)
    axs[0,3].grid(True, alpha=0.4)

    make_plot(axs[1,0], t, u_log, mode_arr, 'Surge u [m/s]',
              hline=1.5, hline_label='u_target=1.5')
    make_plot(axs[1,1], t, v_log, mode_arr, 'Sway v [m/s]',       color='cyan')
    make_plot(axs[1,2], t, r_log, mode_arr, 'Yaw Rate r [deg/s]', color='green')

    mode_num = [0 if m=='GLOBAL' else (1 if m=='LMPC_AVOID' else 2) for m in mode_arr]
    axs[1,3].fill_between(t, mode_num, step='post', alpha=0.6, color='orange')
    axs[1,3].set_yticks([0, 1, 2])
    axs[1,3].set_yticklabels(['GLOBAL', 'AVOID', 'RETURN'], fontsize=8)
    axs[1,3].set_title('Mode Timeline', fontsize=9)
    axs[1,3].set_xlabel('t [s]', fontsize=8)
    axs[1,3].grid(True, alpha=0.4)

    make_plot(axs[2,0], t, tn_log,  mode_arr, 'Yaw Torque TN [Nm]', color='purple')
    make_plot(axs[2,1], t, phi_log, mode_arr, 'Roll φ [deg]')
    make_plot(axs[2,2], t, p_log,   mode_arr, 'Roll Rate p [deg/s]', color='cyan')

    legend_els = [
        mpatches.Patch(facecolor='lightblue',   alpha=0.5, label='GLOBAL (RRT* + ILOS)'),
        mpatches.Patch(facecolor='lightyellow', alpha=0.5, label='LMPC_AVOID'),
        mpatches.Patch(facecolor='lightsalmon', alpha=0.5, label='LMPC_RETURN'),
    ]
    axs[2,3].axis('off')
    axs[2,3].legend(handles=legend_els, loc='center', fontsize=11,
                    title='Mode Legend', title_fontsize=12)

    plt.tight_layout()
    save_path = os.path.expanduser("~/catkin_ws/usv_performance_lmpc.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    rospy.loginfo(f"Plot disimpan: {save_path}")
    plt.show()


def get_obstacle_states(dyn_obs, t):
    obs_list = []
    for o in dyn_obs:
        ox = o['x0'] + o['vx'] * t
        oy = o['y0'] + o['vy'] * t
        obs_list.append((ox, oy, o['r'], o['vx'], o['vy']))
    return obs_list


def publish_dynamic_obstacles(pub, dyn_obs, t, frame_id="map"):
    ma = MarkerArray()
    for idx, o in enumerate(dyn_obs):
        ox = o['x0'] + o['vx'] * t
        oy = o['y0'] + o['vy'] * t
        m = Marker()
        m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
        m.ns = "dynamic_obstacles"; m.id = idx
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position.x = ox; m.pose.position.y = oy
        m.pose.position.z = 0.1; m.pose.orientation.w = 1.0
        m.scale.x = o['r']*2; m.scale.y = o['r']*2; m.scale.z = 0.2
        m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.2; m.color.a = 0.85
        ma.markers.append(m)
        label = Marker()
        label.header.frame_id = frame_id; label.header.stamp = rospy.Time.now()
        label.ns = "dynamic_obstacles_label"; label.id = idx + 100
        label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
        label.pose.position.x = ox; label.pose.position.y = oy
        label.pose.position.z = 0.5; label.pose.orientation.w = 1.0
        label.scale.z = 0.4
        label.color.r = 1.0; label.color.g = 1.0; label.color.b = 1.0; label.color.a = 1.0
        label.text = f"DYN{idx+1}"
        ma.markers.append(label)
    pub.publish(ma)


# ================================================================
#  MAIN ROS LOOP
# ================================================================

def main_ros_loop():
    rospy.init_node('usv_core_node', anonymous=False)
    pub_pose      = rospy.Publisher('/usv/pose',              Pose2D,      queue_size=10)
    pub_path      = rospy.Publisher('/usv/planned_path',      Path,        queue_size=1, latch=True)
    pub_raw       = rospy.Publisher('/usv/raw_path',          Path,        queue_size=1, latch=True)
    pub_dyn_obs   = rospy.Publisher('/usv/dynamic_obstacles', MarkerArray, queue_size=1)
    pub_local_ref = rospy.Publisher('/usv/local_ref_path',    Path,        queue_size=1)
    pub_local_opt = rospy.Publisher('/usv/local_opt_path',    Path,        queue_size=1)
    random.seed(10); np.random.seed(10)

    start    = rospy.get_param('/mission/start',    [1.0,  8.0])
    waypoint = rospy.get_param('/mission/waypoint', [25.0, 20.0])
    goal     = rospy.get_param('/mission/goal',     [48.0, 13.0])
    map_size = rospy.get_param('/map/size',         [33.0, 50.0])
    obs_raw  = rospy.get_param('/map/obstacles',    [])
    margin   = rospy.get_param('/safety_margin',    1.5)
    obs      = [list(o) for o in obs_raw]

    rrt_cfg = rospy.get_param('/rrt', {})
    rrt_p = {
        'maxIter':   int(rrt_cfg.get('maxIter',  800)),
        'stepSize':  float(rrt_cfg.get('stepSize', 1.0)),
        'goalBias':  float(rrt_cfg.get('goalBias', 0.15)),
        'goalTol':   float(rrt_cfg.get('goalTol',  1.0)),
        'rewireRad': float(rrt_cfg.get('rewireRad',3.0)),
    }

    rospy.loginfo("=== RRT* Segment 1: Start -> Waypoint ===")
    path1 = rrt_star(start, waypoint, obs, map_size, rrt_p, margin)
    rospy.loginfo("=== RRT* Segment 2: Waypoint -> Goal ===")
    path2 = rrt_star(waypoint, goal, obs, map_size, rrt_p, margin)

    raw_path_all    = np.vstack((path1, path2[1:]))
    rospy.loginfo("=== Repair → Shortcut → Downsample → Chaikin → Warp → G2CBS ===")
    raw_repaired    = repair_path_obstacles(raw_path_all, obs, margin)
    raw_shortcut    = shortcut_path(raw_repaired, obs, margin, n_iter=800, max_shortcut_m=8.0)
    raw_downsampled = downsample_path(raw_shortcut, min_dist=2.5)
    raw_chaikin     = smooth_path_chaikin(raw_downsampled, iterations=4)

    idx_closest = np.argmin(np.linalg.norm(raw_chaikin - np.array(waypoint), axis=1))
    shift_vec   = np.array(waypoint) - raw_chaikin[idx_closest]
    sigma       = 8.0
    for i in range(len(raw_chaikin)):
        dist_idx = abs(i - idx_closest)
        weight   = math.exp(-(dist_idx**2) / (2 * sigma**2))
        raw_chaikin[i] += shift_vec * weight

    full_path = smooth_path_g2cbs_c2(raw_chaikin, n_per_seg=50)

    path_curv = compute_path_curvature(full_path)
    _win      = min(500, len(path_curv) // 4)
    path_curv = np.convolve(np.abs(path_curv), np.ones(_win) / _win, mode='same')
    n_wp      = len(full_path)

    raw_len    = float(np.sum(np.linalg.norm(np.diff(raw_path_all, axis=0), axis=1)))
    smooth_len = float(np.sum(np.linalg.norm(np.diff(full_path,    axis=0), axis=1)))
    rospy.loginfo("=" * 45)
    rospy.loginfo(f"  Total RRT* raw     : {raw_len:.2f} m")
    rospy.loginfo(f"  Total G2CBS smooth : {smooth_len:.2f} m")
    rospy.loginfo(f"  Jumlah titik jalur : {n_wp}")
    rospy.loginfo("=" * 45)

    def _make_path_msg(pts):
        msg = Path(); msg.header.frame_id = "map"; msg.header.stamp = rospy.Time.now()
        for pt in pts:
            ps = PoseStamped()
            ps.pose.position.x = float(pt[0]); ps.pose.position.y = float(pt[1])
            msg.poses.append(ps)
        return msg

    pub_path.publish(_make_path_msg(full_path))
    pub_raw.publish(_make_path_msg(raw_path_all))

    # ── Inisialisasi ─────────────────────────────────────────────────────
    dynamics = USVDynamics()
    look_idx = min(1 + 12, n_wp - 1)
    dx_init  = full_path[look_idx][0] - full_path[1][0]
    dy_init  = full_path[look_idx][1] - full_path[1][1]
    dynamics.state[2] = math.atan2(dy_init, dx_init)

    dyn_obs_cfg = rospy.get_param('/dynamic_obstacles', [])
    DYN_OBS = [
        {'x0': float(o['x0']), 'y0': float(o['y0']),
         'r':  float(o['r']),  'vx': float(o['vx']), 'vy': float(o['vy'])}
        for o in dyn_obs_cfg
    ]
    rospy.loginfo(f"=== Dynamic Obstacles: {len(DYN_OBS)} ===")
    for i, o in enumerate(DYN_OBS):
        rospy.loginfo(f"  Obs{i+1}: pos=({o['x0']},{o['y0']}) v=({o['vx']:.3f},{o['vy']:.3f})")

    lmpc_cfg    = rospy.get_param('/lmpc', {})
    usv_p       = rospy.get_param('/usv',  {})
    lmpc_params = dict(lmpc_cfg)
    for k in ['A2', 'A12', 'A16', 'A18', 'A19']:
        if usv_p.get(k) is not None:
            lmpc_params[k] = float(usv_p[k])

    lmpc = LMPCController(lmpc_params)
    rospy.loginfo(f"=== LMPC Initialized: N={lmpc.N}, Nc={lmpc.Nc}, "
                  f"d_safe={lmpc.d_safe}m, d_warn={lmpc.d_warn}m ===")

    current_mode  = 'GLOBAL'
    mode_log_list = []
    min_dist_obs  = 999.0
    cte           = 0.0
    psi_e         = 0.0

    controller = USVController()
    controller.psi_d_filtered = dynamics.state[2]

    dt          = 0.05
    rate        = rospy.Rate(1.0 / dt)
    t           = 0.0
    wp_idx      = 1
    wp_reached  = False
    wp_logged   = False
    in_terminal = False

    ctrl_cfg  = rospy.get_param('/ctrl', {})
    WP_RADIUS = float(ctrl_cfg.get('WP_RADIUS', 2.0))
    GOAL_TOL  = float(ctrl_cfg.get('GOAL_TOL',  0.5))

    log_data = []
    rospy.loginfo("=== Memulai Simulasi 4-DOF ===")

    # ════════════════════════════════════════════════════════════════════
    while not rospy.is_shutdown():
        state = dynamics.state
        x, y, psi, phi, u, v, r, p = state

        # ── TIMEOUT ──────────────────────────────────────────────────────
        MAX_SIM_TIME = float(ctrl_cfg.get('max_sim_time', 70.0))
        if t >= MAX_SIM_TIME:
            rospy.logwarn(f">>> TIMEOUT {MAX_SIM_TIME:.0f}s pada t={t:.1f}s | "
                          f"pos=({x:.1f},{y:.1f}) | dist_goal={dist_goal:.2f}m")
            dynamics.step(np.array([0.0, 0.0, 0.0, 0.0]), dt)
            pub_pose.publish(Pose2D(x=dynamics.state[0], y=dynamics.state[1],
                                   theta=dynamics.state[2]))
            save_and_plot_results(log_data)
            rospy.signal_shutdown("Timeout")
            break

        # ── GOAL ─────────────────────────────────────────────────────────
        dist_goal = math.hypot(x - goal[0], y - goal[1])
        if dist_goal < GOAL_TOL:
            rospy.loginfo(f">>> GOAL TERCAPAI pada t = {t:.1f}s")
            dynamics.step(np.array([0.0, 0.0, 0.0, 0.0]), dt)
            pub_pose.publish(Pose2D(x=dynamics.state[0], y=dynamics.state[1],
                                   theta=dynamics.state[2]))
            save_and_plot_results(log_data)
            rospy.signal_shutdown("Selesai")
            break

        # ── WAYPOINT ─────────────────────────────────────────────────────
        dist_wp = math.hypot(x - waypoint[0], y - waypoint[1])
        if not wp_reached and dist_wp < WP_RADIUS:
            wp_reached = True
            controller.beta_hat = 0.0
            controller.eInt_psi = 0.0
            controller.eInt_u   = 0.0
        if not wp_logged and dist_wp < 0.5:
            rospy.loginfo(f">>> Waypoint TEPAT dilewati pada t = {t:.1f}s")
            wp_logged = True

        # ── ADVANCE wp_idx ────────────────────────────────────────────────
        search_end = min(wp_idx + 100, n_wp)
        dists_near = [math.hypot(full_path[i][0]-x, full_path[i][1]-y)
                      for i in range(wp_idx, search_end)]
        new_idx    = wp_idx + int(np.argmin(dists_near))
        if new_idx >= wp_idx:
            wp_idx = new_idx

        # ── LOOKAHEAD ─────────────────────────────────────────────────────
        curv_now  = abs(path_curv[min(wp_idx, len(path_curv)-1)])
        lookahead = max(2.0, 2.5 - 3.0 * curv_now)
        idx_next  = wp_idx
        for i in range(wp_idx, n_wp):
            if math.hypot(full_path[i][0]-x, full_path[i][1]-y) >= lookahead:
                idx_next = i; break
        if idx_next == wp_idx:
            idx_next = min(wp_idx + 30, n_wp - 1)

        wp_prev = full_path[wp_idx]
        wp_next = full_path[idx_next]

        # ── SPEED TARGET ──────────────────────────────────────────────────
        U0_target   = float(ctrl_cfg.get('U0_target',      1.5))
        curv_thresh = float(ctrl_cfg.get('U0_curv_thresh', 0.12))
        curv_speed  = float(ctrl_cfg.get('U0_curv_speed',  1.1))

        curv_end    = min(wp_idx + 120, n_wp)
        future_curv = float(np.max(np.abs(path_curv[wp_idx:curv_end]))) \
                      if wp_idx < curv_end else 0.0

        # ── TERMINAL ──────────────────────────────────────────────────────
        if dist_goal < WP_RADIUS and not in_terminal:
            in_terminal = True
            controller.beta_hat = 0.0
            controller.eInt_psi = 0.0
            controller.eInt_u   = 0.0

        if dist_goal < GOAL_TOL:
            target_U0 = 0.0
            controller.eInt_u = 0.0
        elif in_terminal:
            target_U0 = max(0.0, U0_target * math.sqrt(
                max(0.0, dist_goal - GOAL_TOL) / max(1e-3, WP_RADIUS)))
        elif dist_goal < 10.0:
            target_U0 = max(0.0, U0_target * math.sqrt(
                max(0.0, dist_goal - GOAL_TOL) / 10.0))
        elif future_curv > curv_thresh:
            target_U0 = curv_speed
        else:
            target_U0 = U0_target

        # ════════════════════════════════════════════════════════════════
        #  CONTROL — LMPC + ILOS
        # ════════════════════════════════════════════════════════════════

        # ── LANGKAH 1: Cek obstacle DULU sebelum compute_control ─────────
        obs_now = get_obstacle_states(DYN_OBS, t)
        obs_status, min_dist_obs = lmpc.check_obstacles(state, obs_now)

        lmpc_should_activate = (
            t > 4.0
            and dist_goal > float(ctrl_cfg.get('GOAL_TOL', 1.5)) * 2
            and (
                obs_status in ['WARN', 'CRITICAL']
                or (current_mode != 'GLOBAL' and min_dist_obs < lmpc.d_exit)
            )
        )

        # ════════════════════════════════════════════════════════════════
        # ARSITEKTUR BARU: LMPC-guided ILOS
        # ─────────────────────────────────────────────────────────────
        # LMPC  → menghitung heading optimal (hindari obs / kembali ke jalur)
        # ILOS  → menghasilkan TN smooth dari heading error (rate-limited, PID)
        # TX    → dari LMPC (speed control 1.1–1.5 m/s)
        # TN    → dari ILOS (tidak pernah spike, tidak ada mode-switch TN)
        #
        # Cara kerja:
        #   1. LMPC solve() → psi_opt (heading 0.6s ke depan di opt_traj)
        #   2. Set psi_d_filtered = psi_opt SEBELUM compute_control()
        #   3. ILOS hitung TN smooth menuju psi_opt (damp 90% LMPC + 10% jalur)
        #   4. Gabung: Tcmd[0]=TX_lmpc, Tcmd[2]=TN_ilos
        # ════════════════════════════════════════════════════════════════

        # ── LANGKAH 2: LMPC solve (jika diperlukan) ──────────────────────
        psi_lmpc = None   # heading optimal dari LMPC
        TX_lmpc  = None   # TX optimal dari LMPC

        if lmpc_should_activate:
            static_obs_for_lmpc = [
                (float(o[0]), float(o[1]), float(o[2]), 0.0, 0.0) for o in obs
            ]
            Tcmd_lmpc, lmpc_mode, lmpc_info = lmpc.solve(
                state      = state,
                wp_prev    = wp_prev,
                wp_next    = wp_next,
                obstacles  = obs_now,
                target_u   = target_U0,
                full_path  = full_path,
                wp_idx     = wp_idx,
                dist_goal  = dist_goal,
                static_obs = static_obs_for_lmpc,
            )
            if Tcmd_lmpc is not None:
                psi_lmpc = lmpc_info.get('psi_opt')   # heading optimal
                TX_lmpc  = float(Tcmd_lmpc[0])

                if 'ref_traj' in lmpc_info and 'opt_traj' in lmpc_info:
                    pub_local_ref.publish(_make_path_msg(lmpc_info['ref_traj']))
                    pub_local_opt.publish(_make_path_msg(lmpc_info['opt_traj']))

                if len(mode_log_list) == 0 or mode_log_list[-1][1] != lmpc_mode:
                    mode_log_list.append((t, lmpc_mode, min_dist_obs))
                    side_str = lmpc_info.get('avoid_side', 'N/A')
                    if lmpc_mode == 'LMPC_AVOID':
                        rospy.logwarn(
                            f"[t={t:.1f}s] ⚡ HINDARI | sisi={side_str} | "
                            f"obs_dist={min_dist_obs:.2f}m")
                    elif lmpc_mode == 'LMPC_RETURN':
                        rospy.loginfo(
                            f"[t={t:.1f}s] ↩ RETURN ke jalur | "
                            f"obs_dist={min_dist_obs:.2f}m")

        # ── LANGKAH 3: Set heading ILOS dari LMPC optimal ────────────────
        # Saat LMPC aktif: psi_d_filtered ← psi_lmpc (LMPC steering ILOS)
        # filter internal ILOS (+= 0.1*(psi_d_raw - psi_d_filtered)) akan menjaga
        # transisi halus: 90% psi_lmpc + 10% natural ILOS heading
        if psi_lmpc is not None:
            controller.psi_d_filtered = float(psi_lmpc)
            if lmpc_mode == 'LMPC_AVOID':
                # Neutralize D-term selama manuver HINDARI:
                # Saat USV belok (r tumbuh), filt_r ikut tumbuh → D-term = -Kpsi_d*filt_r
                # menjadi besar negatif → TN turun atau balik negatif → USV berhenti belok!
                # Dengan reset filt_r=0 setiap step: TN = Kpsi_p*psi_e + Kpsi_i*eInt
                # → selalu positif (P+I only) → USV belok utara konsisten sampai obs lewat.
                controller.filt_r = 0.0

        # ── LANGKAH 4: ILOS compute_control (SELALU dipanggil sekali) ────
        # TN yang dihasilkan selalu smooth (rate-limited dTN, anti-windup PID)
        # Heading referensi sudah di-steer oleh LMPC atau ILOS natural
        Tcmd, cte, psi_e, psi_d = controller.compute_control(
            state, wp_prev, wp_next, dt, t, dist_goal, target_U0,
            lmpc_mode=current_mode)

        # ── LANGKAH 5: Override TX dari LMPC (speed), TN tetap dari ILOS ─
        if TX_lmpc is not None:
            Tcmd[0]      = float(np.clip(TX_lmpc, lmpc.TX_min, lmpc._TX_SOLVER_MAX))
            current_mode = lmpc_mode
        else:
            prev_mode    = current_mode
            current_mode = 'GLOBAL'
            if prev_mode != 'GLOBAL':
                lmpc.reset()
                # Kurangi integral windup yaw yang terakumulasi selama avoidance.
                # psi_d_filtered TIDAK direset ke psi — biarkan nilai terakhir LMPC
                # (heading kembali-ke-jalur dari RETURN mode) tetap aktif agar ILOS
                # langsung lanjut smooth menuju jalur, bukan mulai dari heading avoidance.
                controller.eInt_psi *= 0.3
                # Reset filt_r: selama RETURN mode heading berubah (misal balik dari +65°
                # ke 0°) sehingga filt_r ≈ -0.2 rad/s. Saat GLOBAL aktif,
                # D-term = -Kpsi_d × filt_r = -2000×(-0.2) = +400 Nm spike →
                # USV belok kiri berlebih → CTE osilasi. Dengan reset ke 0, spike hilang.
                controller.filt_r = 0.0
                # Opsi C: reset ke arah jalur global tapi dibatasi max 15 deg dari psi saat ini
                # → tidak ada P-term spike, tapi langsung mengarah ke jalur bukan ke heading menyimpang
                _psi_path = math.atan2(
                    float(wp_next[1]) - float(wp_prev[1]),
                    float(wp_next[0]) - float(wp_prev[0]))
                _psi_now  = float(state[2])
                _delta    = (_psi_path - _psi_now + math.pi) % (2*math.pi) - math.pi
                _psi_reset = _psi_now + float(np.clip(_delta, -math.radians(15), math.radians(15)))
                controller.psi_d_filtered = _psi_reset
                mode_log_list.append((t, 'GLOBAL', min_dist_obs))
                rospy.loginfo(
                    f"[t={t:.1f}s] ✅ KEMBALI KE GLOBAL | obs_dist={min_dist_obs:.2f}m")

        # ── DYNAMICS STEP ────────────────────────────────────────────────
        dynamics.step(Tcmd, dt)

        # ── SURGE HARD CLAMP ─────────────────────────────────────────────
        U_MAX_HARD = 1.5
        if dynamics.state[4] > U_MAX_HARD:
            dynamics.state[4] = U_MAX_HARD

        # ── LOGGING ──────────────────────────────────────────────────────
        if 30.0 <= t <= 40.0: rospy.loginfo(f"[POS] t={t:.1f}s x={x:.2f} y={y:.2f}")
        log_data.append([
            t,
            math.degrees(psi),
            math.degrees(psi_e),
            cte,
            u, v,
            math.degrees(r),
            Tcmd[2],
            math.degrees(phi),
            math.degrees(p),
            current_mode,
            min_dist_obs,
        ])

        pub_pose.publish(Pose2D(x=dynamics.state[0],
                                y=dynamics.state[1],
                                theta=dynamics.state[2]))
        publish_dynamic_obstacles(pub_dyn_obs, DYN_OBS, t)
        t += dt
        rate.sleep()


if __name__ == '__main__':
    try:
        main_ros_loop()
    except rospy.ROSInterruptException:
        pass