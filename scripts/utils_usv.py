import numpy as np

def cumulativeArc(P):
    return np.hstack(([0], np.cumsum(np.sqrt(np.sum(np.diff(P, axis=0)**2, axis=1)))))

def validateGoalGrid(g, obs, safeDist):
    gNew = np.array(g, dtype=float)
    for i in range(obs.shape[0]):
        cx, cy, r = obs[i]
        d = np.hypot(gNew[0] - cx, gNew[1] - cy)
        if d < (r + safeDist):
            dir = (gNew - np.array([cx, cy])) / max(d, 1e-6)
            gNew = gNew + dir * ((r + safeDist + 8) - d)
    return gNew

def wrapToPi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def crossTrackError(p, p1, p2):
    A = p2-p1
    B = p-p1
    return (A[0]*B[1] - A[1]*B[0]) / max(1e-6, np.linalg.norm(A))