import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline

def rrt_star_grid(map_, start_xy, goal_xy, maxIter, step, goalTol, safeDist):
    maxR, maxC = map_.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    distmap = ndimage.distance_transform_edt(map_==0)
    traversable = distmap >= safeDist

    sx_round = int(round(sx))
    sy_round = int(round(sy))
    gx_round = int(round(gx))
    gy_round = int(round(gy))
    if not inbounds(sx_round, sy_round, maxC, maxR) or not traversable[sy_round, sx_round]:
        raise ValueError('Start tidak valid / terlalu dekat obstacle')
    if not inbounds(gx_round, gy_round, maxC, maxR) or not traversable[gy_round, gx_round]:
        gx, gy = auto_fix_point(gx, gy, traversable, distmap, safeDist)
        print(f'Goal diperbaiki ke ({gx:.1f}, {gy:.1f})')

    nodes = [{
        'x': sx,
        'y': sy,
        'parent': None,
        'cost': 0
    }]
    gamma = 20
    goalFound = False
    goalIdx = 0
    for it in range(maxIter):
        if np.random.rand() < 0.2:
            xs, ys = gx, gy
        else:
            xs = np.random.rand() * maxC
            ys = np.random.rand() * maxR
        # nearest node
        dists = [np.hypot(xs - n['x'], ys - n['y']) for n in nodes]
        idxNear = np.argmin(dists)

        th = np.arctan2(ys-nodes[idxNear]['y'], xs-nodes[idxNear]['x'])
        xn = nodes[idxNear]['x'] + step*np.cos(th)
        yn = nodes[idxNear]['y'] + step*np.sin(th)

        if not inbounds(xn, yn, maxC, maxR): continue
        if not edge_free_grid([nodes[idxNear]['x'], nodes[idxNear]['y']],
                              [xn,yn], traversable): continue

        new_cost = nodes[idxNear]['cost'] + np.hypot(xn-nodes[idxNear]['x'], yn-nodes[idxNear]['y'])
        new = {'x': xn, 'y': yn, 'parent': idxNear, 'cost': new_cost}

        # rewiring
        rad = min(gamma * np.sqrt(np.log(len(nodes)+1)/(len(nodes)+1)), 15)
        Nei = []
        for i, n in enumerate(nodes):
            if np.hypot(n['x']-xn, n['y']-yn) <= rad:
                Nei.append(i)
        for j in Nei:
            if edge_free_grid([nodes[j]['x'], nodes[j]['y']], [xn,yn], traversable):
                c = nodes[j]['cost'] + np.hypot(nodes[j]['x']-xn, nodes[j]['y']-yn)
                if c < new['cost']:
                    new['cost'] = c
                    new['parent'] = j
        nodes.append(new)
        newIdx = len(nodes) - 1

        # rewire others
        for j in Nei:
            if edge_free_grid([nodes[j]['x'], nodes[j]['y']], [xn,yn], traversable):
                c = nodes[newIdx]['cost'] + np.hypot(nodes[j]['x']-xn, nodes[j]['y']-yn)
                if c < nodes[j]['cost']:
                    nodes[j]['parent'] = newIdx
                    nodes[j]['cost'] = c

        # goal check
        if np.hypot(xn - gx, yn - gy) < goalTol:
            if not goalFound or new['cost'] < nodes[goalIdx]['cost']:
                goalFound = True
                goalIdx = newIdx

    if not goalFound:
        dists = [np.hypot(n['x']-gx, n['y']-gy) for n in nodes]
        goalIdx = np.argmin(dists)

    # build path from goalIdx to start
    path = []
    idx = goalIdx
    while idx is not None:
        n = nodes[idx]
        path.insert(0, [n['x'], n['y']])
        idx = n['parent']

    # extend to goal if reachable
    if edge_free_grid(path[-1], [gx, gy], traversable):
        path.append([gx, gy])
    path = np.array(path)
    # nodes as list of dict
    return path, nodes

def inbounds(x, y, mc, mr):
    return x>=0 and y>=0 and x<mc and y<mr

def edge_free_grid(a, b, trv):
    N = max(2, int(np.hypot(b[0]-a[0], b[1]-a[1])))
    xs = np.linspace(a[0], b[0], N)
    ys = np.linspace(a[1], b[1], N)
    idx = np.vstack((np.round(ys), np.round(xs))).astype(int).T
    idx[:,0] = np.clip(idx[:,0], 0, trv.shape[0]-1)
    idx[:,1] = np.clip(idx[:,1], 0, trv.shape[1]-1)
    return np.all(trv[idx[:,0], idx[:,1]] > 0)

def auto_fix_point(x, y, trav, distmap, safeDist):
    gxg, gyg = np.gradient(distmap.astype(float))
    r = np.clip(int(round(y)), 0, trav.shape[0]-1)
    c = np.clip(int(round(x)), 0, trav.shape[1]-1)
    gxn = gxg[r,c]
    gyn = gyg[r,c]
    gnorm = np.hypot(gxn, gyn)
    if gnorm < 1e-6:
        dir = np.random.rand(2)
        dir = dir / np.linalg.norm(dir)
    else:
        dir = np.array([gyn, gxn]) / gnorm
    shift = max(5, 3*safeDist)
    xOut = np.clip(x + shift*dir[0], 0, trav.shape[1]-1)
    yOut = np.clip(y + shift*dir[1], 0, trav.shape[0]-1)
    return xOut, yOut

def smooth_path(P, nPerSeg):
    if len(P) <= 2: return P
    t = np.zeros(P.shape[0])
    for i in range(1, len(P)):
        t[i] = t[i-1] + np.linalg.norm(P[i] - P[i-1])
    if t[-1] < 1e-9: return P[:1,:]
    cs_x = CubicSpline(t, P[:,0], bc_type='natural')
    cs_y = CubicSpline(t, P[:,1], bc_type='natural')
    ts = np.linspace(0, t[-1], nPerSeg*(len(P)-1))
    out = np.stack([cs_x(ts), cs_y(ts)], axis=-1)
    return out