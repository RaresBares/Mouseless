import numpy as np

class HandPoseFeaturizer:
    def __init__(self, normalizer):
        self.norm = normalizer
        self.FINGERS = {"thumb":[1,2,3,4],"index":[5,6,7,8],"middle":[9,10,11,12],"ring":[13,14,15,16],"pinky":[17,18,19,20]}
        self.TIPS = [4,8,12,16,20]
    def __call__(self, k):
        nk = self.norm(k)
        xy = nk[:, :2]
        f = [xy.reshape(-1), self._bone_lengths(xy), self._curl_angles(xy), self._abduction(xy), self._tips_pairwise(xy)]
        return np.concatenate(f).astype(np.float32)
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