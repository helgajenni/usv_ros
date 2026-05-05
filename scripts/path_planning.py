#!/usr/bin/env python3
import rospy
import numpy as np
import math
import random
import csv
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
from usv_dynamics import USVDynamics
from controller import USVController

def is_collision_free(p1, p2, obs, margin):
    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    steps = max(20, int(dist / 0.5))
    for s in np.linspace(0, 1, steps):
        px = p1[0] + s * (p2[0] - p1[0])
        py = p1[1] + s * (p2[1] - p1[1])
        for o in obs:
            if math.hypot(px - o[0], py - o[1]) < (o[2] + margin):
                return False
    return True

def rrt_star(start_pt, goal_pt, obs, map_sz, rrt_p, margin):
    nodes = [np.array(start_pt)]
    parents = [0]
    costs = [0.0] 

    for i in range(rrt_p['maxIter']):
        if random.random() < rrt_p['goalBias']: 
            sample = np.array(goal_pt)
        else: 
            sample = np.array([random.uniform(0, map_sz[1]), random.uniform(0, map_sz[0])])
            
        dists = np.linalg.norm(np.array(nodes) - sample, axis=1)
        n_idx = np.argmin(dists)
        nearest_node = nodes[n_idx]
        
        dir_vec = sample - nearest_node
        dist = np.linalg.norm(dir_vec)
        if dist > rrt_p['stepSize']: 
            new_pt = nearest_node + rrt_p['stepSize'] * (dir_vec / dist)
        else: 
            new_pt = sample
            
        if not (0 <= new_pt[0] <= map_sz[1] and 0 <= new_pt[1] <= map_sz[0]): 
            continue
        if not is_collision_free(nearest_node, new_pt, obs, margin): 
            continue

        dists_to_new = np.linalg.norm(np.array(nodes) - new_pt, axis=1)
        near_indices = np.where(dists_to_new <= rrt_p['rewireRad'])[0]

        best_parent = n_idx
        best_cost = costs[n_idx] + np.linalg.norm(new_pt - nodes[n_idx])

        for ni in near_indices:
            c = costs[ni] + np.linalg.norm(new_pt - nodes[ni])
            if c < best_cost and is_collision_free(nodes[ni], new_pt, obs, margin):
                best_parent = ni
                best_cost = c

        nodes.append(new_pt)
        parents.append(best_parent)
        costs.append(best_cost)
        new_node_idx = len(nodes) - 1

        for ni in near_indices:
            nc = costs[new_node_idx] + np.linalg.norm(nodes[ni] - new_pt)
            if nc < costs[ni] and is_collision_free(new_pt, nodes[ni], obs, margin):
                parents[ni] = new_node_idx
                costs[ni] = nc

        if np.linalg.norm(new_pt - goal_pt) < rrt_p['goalTol']: 
            break

    dists_to_goal = np.linalg.norm(np.array(nodes) - goal_pt, axis=1)
    goal_idx = np.argmin(dists_to_goal)
    
    path = [np.array(goal_pt)]
    curr_idx = goal_idx
    while curr_idx != 0:
        path.append(nodes[curr_idx])
        curr_idx = parents[curr_idx]
    path.append(np.array(start_pt))
    path.reverse()
    return np.array(path)

# PERBAIKAN: Konfigurasi G2CBS C^2 Continuous Ekuivalen MATLAB
def smooth_path_g2cbs_scipy(raw_path):
    # 1. FILTERING (Rahasia MATLAB): Saring titik RRT* yang terlalu rapat
    # Kita hanya mengambil titik yang berjarak minimal 2.5 meter agar 
    # tersedia "ruang" untuk membuat lengkungan (corner) yang sangat landai.
    filtered_path = [raw_path[0]]
    for pt in raw_path:
        if np.linalg.norm(pt - filtered_path[-1]) >= 2.5:
            filtered_path.append(pt)
            
    # Pastikan Goal (titik akhir) tidak terbuang
    if np.linalg.norm(raw_path[-1] - filtered_path[-1]) > 0.1:
        filtered_path.append(raw_path[-1])
        
    clean = np.array(filtered_path)
    
    # Keamanan jika jalurnya ternyata terlalu pendek
    if len(clean) < 4:
        return raw_path 

    # 2. KONFIGURASI G2CBS Python
    # k=3 : Menjamin sifat C^2 Continuous (turunan kedua mulus)
    # s=2.0 : Smoothing factor, memaksa kurva membulat mengabaikan zig-zag kecil
    tck, u = splprep([clean[:, 0], clean[:, 1]], s=2.0, k=3)
    
    # 3. Render 800 titik baru agar lintasan terlihat SANGAT licin seperti jalan tol
    u_new = np.linspace(0, 1, 800)
    x_new, y_new = splev(u_new, tck)
    
    return np.vstack((x_new, y_new)).T

def save_and_plot_results(log_data):
    filename = os.path.expanduser("~/catkin_ws/usv_simulation_results.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Psi_deg', 'Psi_err_deg', 'CTE_m', 'Surge_u', 'Sway_v', 'Yaw_Rate_r_deg', 'Yaw_Torque_TN', 'Roll_phi_deg', 'Roll_Rate_p_deg'])
        for row in log_data: writer.writerow(row)

    t = [row[0] for row in log_data]; psi = [row[1] for row in log_data]
    psi_err = [row[2] for row in log_data]; cte = [row[3] for row in log_data]
    u = [row[4] for row in log_data]; v = [row[5] for row in log_data]
    r = [row[6] for row in log_data]; tn = [row[7] for row in log_data]
    phi = [row[8] for row in log_data]; p = [row[9] for row in log_data]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.canvas.manager.set_window_title('USV States Performance')
    axs[0,0].plot(t, psi, 'b-'); axs[0,0].set_title('Heading \u03c8 [deg]')
    axs[0,1].plot(t, psi_err, 'g-'); axs[0,1].set_title('Heading Error [deg]')
    axs[0,2].plot(t, cte, 'm-'); axs[0,2].set_title('Cross-Track Error [m]')
    axs[1,0].plot(t, u, 'b-'); axs[1,0].set_title('Surge u [m/s]'); axs[1,0].axhline(y=1.5, color='r', linestyle='--')
    axs[1,1].plot(t, v, 'c-'); axs[1,1].set_title('Sway v [m/s]')
    axs[1,2].plot(t, r, 'g-'); axs[1,2].set_title('Yaw Rate r [deg/s]')
    axs[2,0].plot(t, tn, 'purple'); axs[2,0].set_title('Yaw Torque TN [Nm]')
    axs[2,1].plot(t, phi, 'b-'); axs[2,1].set_title('Roll \u03d5 [deg]')
    axs[2,2].plot(t, p, 'c-'); axs[2,2].set_title('Roll Rate p [deg/s]')
    for ax in axs.flat: ax.grid(True); ax.set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()

def main_ros_loop():
    rospy.init_node('usv_core_node', anonymous=True)
    pub_pose = rospy.Publisher('/usv/pose', Pose2D, queue_size=10)
    pub_path = rospy.Publisher('/usv/planned_path', Path, queue_size=1, latch=True)
    pub_raw = rospy.Publisher('/usv/raw_path', Path, queue_size=1, latch=True) 
    
    random.seed(10)
    np.random.seed(10)
    
    start = rospy.get_param('/mission/start', [1.0, 8.0])
    waypoint = rospy.get_param('/mission/waypoint', [25.0, 20.0])
    goal = rospy.get_param('/mission/goal', [48.0, 13.0])
    map_size = rospy.get_param('/map/size', [33.0, 50.0])
    obs = rospy.get_param('/map/obstacles', [])
    margin = rospy.get_param('/safety_margin', 2.0)
    
    rrt_p = {'maxIter': 1000, 'stepSize': 1.0, 'goalBias': 0.15, 'goalTol': 1.0, 'rewireRad': 4.0}
    path1 = rrt_star(start, waypoint, obs, map_size, rrt_p, margin)
    path2 = rrt_star(waypoint, goal, obs, map_size, rrt_p, margin)
    
    raw_path_all = np.vstack((path1, path2[1:]))
    full_path = smooth_path_g2cbs_scipy(raw_path_all)
    
    path_msg = Path()
    path_msg.header.frame_id = "map"
    for pt in full_path:
        ps = PoseStamped(); ps.pose.position.x = pt[0]; ps.pose.position.y = pt[1]
        path_msg.poses.append(ps)
    pub_path.publish(path_msg)

    raw_msg = Path()
    raw_msg.header.frame_id = "map"
    for pt in raw_path_all:
        ps = PoseStamped(); ps.pose.position.x = pt[0]; ps.pose.position.y = pt[1]
        raw_msg.poses.append(ps)
    pub_raw.publish(raw_msg)
    
    dynamics = USVDynamics()
    
    dx_init = full_path[min(20, len(full_path)-1)][0] - full_path[0][0]
    dy_init = full_path[min(20, len(full_path)-1)][1] - full_path[0][1]
    dynamics.state[2] = math.atan2(dy_init, dx_init)
    
    controller = USVController()
    
    dt = 0.05
    rate = rospy.Rate(1.0 / dt)
    t = 0.0
    wp_idx = 1
    n_wp = len(full_path)
    
    log_data = []
    rospy.loginfo("=== Memulai Simulasi 4-DOF ===")
    
    while not rospy.is_shutdown():
        state = dynamics.state
        x, y, psi, phi, u, v, r, p = state
        
        dist_goal = np.linalg.norm([x - goal[0], y - goal[1]])
        if dist_goal < 0.5:
            rospy.loginfo(f">>> TARGET PRESISI [48,13] TERCAPAI pada t = {t:.1f}s")
            save_and_plot_results(log_data)
            break
            
        # PERBAIKAN: Index hanya maju perlahan (maksimal 10) agar TIDAK MELOMPAT tikungan
        search_range = range(wp_idx, min(wp_idx + 10, n_wp))
        dists = [math.hypot(full_path[i][0] - x, full_path[i][1] - y) for i in search_range]
        if dists:
            local_idx = np.argmin(dists)
            new_idx = search_range[local_idx]
            if new_idx >= wp_idx:
                wp_idx = new_idx
                
        # Lookahead yang ketat untuk menempel pada jalur
        lookahead_dist = 1.5
        idx_next = wp_idx
        for i in range(wp_idx, n_wp):
            if math.hypot(full_path[i][0] - full_path[wp_idx][0], full_path[i][1] - full_path[wp_idx][1]) >= lookahead_dist:
                idx_next = i
                break
        
        if idx_next == wp_idx:
            idx_next = min(wp_idx + 5, n_wp - 1)
        
        wp_prev = full_path[wp_idx]
        wp_next = full_path[idx_next]
        
        if wp_idx >= n_wp - 10 or dist_goal < 6.0:
            wp_next = goal
            
        Tcmd, cte, psi_e, psi_d = controller.compute_control(state, wp_prev, wp_next, dt, t, dist_goal)
        dynamics.step(Tcmd, dt)
        
        log_data.append([
            t, math.degrees(psi), math.degrees(psi_e), cte, u, v, 
            math.degrees(r), Tcmd[2], math.degrees(phi), math.degrees(p)
        ])

        pose_msg = Pose2D(x=dynamics.state[0], y=dynamics.state[1], theta=dynamics.state[2])
        pub_pose.publish(pose_msg)
        
        t += dt
        rate.sleep()

if __name__ == '__main__':
    try:
        main_ros_loop()
    except rospy.ROSInterruptException:
        pass