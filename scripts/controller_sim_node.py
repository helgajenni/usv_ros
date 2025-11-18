import numpy as np
import matplotlib.pyplot as plt

from planner_node import rrt_star_grid, smooth_path
from controller import controller_guard_3dof_ilos, ilos_heading
from utils_usv import cumulativeArc, wrapToPi, validateGoalGrid
# ...tambahkan import lain jika ada

def usv(nu, T, Te, psi):
    u, v, r = nu
    u4, u5 = T[0], T[2]
    m = 11.8
    Xu = -44.3
    Yv = -187.5
    Nr = -83.6
    du2 = 16.7
    dv2 = 242.3
    dr2 = 103.8

    earth_d = np.dot(
        np.array([[np.cos(psi), -np.sin(psi), 0],
                  [np.sin(psi),  np.cos(psi), 0],
                  [        0,            0, 1]]),
        nu)
    u_dot = (Xu*u + du2*np.abs(u)*u + u4)/m
    v_dot = (Yv*v + dv2*np.abs(v)*v + Te[1])/m
    r_dot = (Nr*r + dr2*np.abs(r)*r + u5)/m
    acc = np.array([u_dot, v_dot, r_dot])
    return acc, earth_d

def main():
    # ==== Map and obstacles
    mapSize = [100, 100]
    obstacles = np.array([[30, 70, 10],
                         [55, 70, 10],
                         [30, 42, 10],
                         [55, 42, 10]])
    map_ = np.zeros(mapSize)
    xg, yg = np.meshgrid(np.arange(1, mapSize[1]+1), np.arange(1, mapSize[0]+1))
    for k in range(obstacles.shape[0]):
        mask = (xg - obstacles[k,0])**2 + (yg - obstacles[k,1])**2 <= (obstacles[k,2]+1.5)**2
        map_[mask] = 1

    # ===== Start, Waypoint, Goal
    start    = np.array([15, 41])
    waypoint = np.array([43, 61])
    goal     = np.array([70, 43])
    safeDistPlan = 1.0
    waypoint = validateGoalGrid(waypoint, obstacles, safeDistPlan)
    goal     = validateGoalGrid(goal,     obstacles, safeDistPlan)

    plt.figure(1); plt.clf(); plt.grid(True); plt.axis('equal')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.xlim([0, mapSize[1]]); plt.ylim([0, mapSize[0]])
    plt.title('RRT* Growing Tree + Raw & Smooth Path')
    theta = np.linspace(0,2*np.pi,80)
    for obs in obstacles:
        plt.fill(obs[0]+obs[2]*np.cos(theta), obs[1]+obs[2]*np.sin(theta),
                 'r', alpha=0.3, edgecolor='none')
    plt.plot(start[0],start[1],'g^', markersize=8, label='Start')
    plt.plot(waypoint[0], waypoint[1], 'mo', markersize=8, label='Waypoint')
    plt.plot(goal[0], goal[1], 'bo', markersize=8, label='Goal')

    # === RRT* Planning
    np.random.seed(50)
    pathSW, nodesSW = rrt_star_grid(map_, start, waypoint, 5000, 4.0, 3.0, safeDistPlan)
    np.random.seed(65)
    pathWG, nodesWG = rrt_star_grid(map_, waypoint, goal, 5000, 4.0, 3.0, safeDistPlan)
    plt.plot(pathSW[:,0], pathSW[:,1], 'm--', linewidth=1.5)
    plt.plot(pathWG[:,0], pathWG[:,1], 'k--', linewidth=1.5)
    plt.legend()
    plt.show()

    # Smoothing
    pSW = smooth_path(pathSW, 40)
    pWG = smooth_path(pathWG, 40)
    pSW[-1,:] = waypoint
    pWG[0,:] = waypoint

    plt.figure(2) ; plt.clf(); plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.xlim([0, mapSize[1]]); plt.ylim([0, mapSize[0]])
    plt.title('USV 3-DOF: ILOS + PID-Fuzzy Guard (Stable)')
    plt.plot(pSW[:,0],pSW[:,1],'c-',linewidth=2)
    plt.plot(pWG[:,0],pWG[:,1],'b-',linewidth=2)

    # ... lanjutkan dengan implementasi loop simulasi dynamics & controller
    # (lihat portingan MATLAB ke Python step-by-step pada jawaban sebelumnya)
    # Lanjutkan segment 1 dan segment 2 sesuai algoritma MATLAB

if __name__ == "__main__":
    main()