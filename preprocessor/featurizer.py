import numpy as np

class HandPoseFeaturizer:
    def __init__(self, normalizer):
        self.norm = normalizer
        self.FINGERS = {"thumb":[1,2,3,4],"index":[5,6,7,8],"middle":[9,10,11,12],"ring":[13,14,15,16],"pinky":[17,18,19,20]}
        self.TIPS = [4,8,12,16,20]
    def __call__(self, k):
        nk = self.norm(k)
        a = np.asarray(nk, np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 3) if (a.size % 3) == 0 else a.reshape(-1, 2)
        if a.shape[1] == 2:
            v = np.ones((a.shape[0], 1), np.float32)
            a = np.concatenate([a, v], axis=1)
        return a.astype(np.float32)
    def get_middlepoint(self, k, tips_only=True):
        nk = self.norm(k)
        xy = nk[:, :2]
        v  = nk[:, 2] if nk.shape[1] >= 3 else np.ones(len(nk), np.float32)
        if tips_only:
            idx = [i for i in self.TIPS if i < len(xy)]
        else:
            idx = list(range(len(xy)))
        vis = [np.isfinite(xy[i]).all() and (v[i] > 0) for i in idx]
        if any(vis):
            pts = np.stack([xy[i] for i, ok in zip(idx, vis) if ok], 0)
            c = pts.mean(0)
        else:
            c = np.array([0.0, 0.0], np.float32)
        return float(c[0]), float(c[1])
    def _bone_lengths(self, xy):
        out = []
        for f in self.FINGERS.values():
            for a,b in zip(f[:-1], f[1:]):
                if max(a,b) < len(xy) and np.isfinite(xy[[a,b]]).all():
                    out.append(float(np.linalg.norm(xy[b]-xy[a])))
                else:
                    out.append(0.0)
        return np.array(out, np.float32)
    def _curl_angles(self, xy):
        def ang(u,v):
            nu,nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu<1e-6 or nv<1e-6:
                return 0.0
            c = float(np.clip(np.dot(u,v)/(nu*nv), -1.0, 1.0))
            return float(np.arccos(c))
        out = []
        for f in self.FINGERS.values():
            if max(f) < len(xy) and np.isfinite(xy[f]).all():
                a,b,c,d = xy[f[0]], xy[f[1]], xy[f[2]], xy[f[3]]
                out.append(ang(b-a, c-b))
                out.append(ang(c-b, d-c))
            else:
                out += [0.0, 0.0]
        return np.array(out, np.float32)
    def _abduction(self, xy):
        axis = xy[9] if 9 < len(xy) and np.isfinite(xy[9]).all() else (xy[5] if 5 < len(xy) and np.isfinite(xy[5]).all() else np.array([1.0,0.0], np.float32))
        def a2(v):
            nu,na = np.linalg.norm(v), np.linalg.norm(axis)
            if nu<1e-6 or na<1e-6:
                return 0.0
            c = float(np.clip(np.dot(v,axis)/(nu*na), -1.0, 1.0))
            s = np.cross(np.array([axis[0],axis[1],0.0], np.float32), np.array([v[0],v[1],0.0], np.float32))[2]
            a = float(np.arccos(c))
            return float(np.copysign(a, s))
        bases = [5,9,13,17]
        out = []
        for b in bases:
            if b < len(xy) and np.isfinite(xy[b]).all():
                out.append(a2(xy[b]))
            else:
                out.append(0.0)
        return np.array(out, np.float32)
    def _tips_pairwise(self, xy):
        idx = [i for i in self.TIPS if i < len(xy)]
        P = xy[idx]
        m = np.isfinite(P).all(axis=1)
        P = P[m]
        if len(P)==0:
            return np.zeros((10,), np.float32)
        d = []
        for i in range(len(P)):
            for j in range(i+1, len(P)):
                d.append(float(np.linalg.norm(P[i]-P[j])))
        return np.array(d, np.float32)

def get_middlepoint_xy(img, kpts, tips=(4,8,12,16,20)):
    xy = kpts[:, :2].copy()
    v  = kpts[:, 2] if kpts.shape[1] >= 3 else np.ones(len(kpts), np.float32)
    H, W = img.shape[1], img.shape[2]
    idx = [i for i in tips if i < len(xy)]
    vis = [np.isfinite(xy[i]).all() and (v[i] > 0) for i in idx]
    if any(vis):
        pts = np.stack([xy[i] for i, ok in zip(idx, vis) if ok], 0)
        c = pts.mean(0)
    else:
        c = np.array([0.0, 0.0], np.float32)
    return float(c[0]*W), float(c[1]*H)