import numpy as np

class HandPoseNormalizer:
    def __init__(self, denorm_size=None, mirror_canonical=True, eps=1e-6):
        self.denorm_size = denorm_size
        self.mirror_canonical = mirror_canonical
        self.eps = eps
        self.MCP = [5, 9, 13, 17]
        self.TIPS = [4, 8, 12, 16, 20]
    def __call__(self, k):
        a = np.asarray(k, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 3)
        xy = a[:, :2].copy()
        v = a[:, 2:3] if a.shape[1] >= 3 else np.ones((a.shape[0], 1), np.float32)
        if self.denorm_size is not None:
            W, H = self.denorm_size
            xy[:, 0] *= W
            xy[:, 1] *= H
        pts = [xy[i] for i in self.MCP if i < len(xy) and np.isfinite(xy[i]).all()]
        if not pts:
            for i in [0] + self.TIPS:
                if i < len(xy) and np.isfinite(xy[i]).all():
                    pts.append(xy[i])
        c = np.mean(np.stack(pts, 0), 0) if pts else np.array([0.0, 0.0], np.float32)
        xy -= c
        if all(j < len(xy) and np.isfinite(xy[j]).all() for j in [5, 17]):
            s = float(np.linalg.norm(xy[5] - xy[17])) + self.eps
        else:
            r = np.linalg.norm(xy, axis=1)
            r = r[np.isfinite(r)]
            s = float(np.mean(r)) + self.eps if r.size else 1.0
        xy /= s
        base = xy[9] if 9 < len(xy) and np.isfinite(xy[9]).all() else (xy[5] if 5 < len(xy) and np.isfinite(xy[5]).all() else np.array([1.0, 0.0], np.float32))
        phi = float(np.arctan2(base[1], base[0]))
        cp, sp = np.cos(-phi), np.sin(-phi)
        R = np.array([[cp, -sp], [sp, cp]], np.float32)
        xy = xy @ R.T
        if self.mirror_canonical and all(j < len(xy) and np.isfinite(xy[j]).all() for j in [5, 17]):
            if xy[5, 0] < xy[17, 0]:
                xy[:, 0] *= -1.0
        out = np.concatenate([xy, v], axis=1)
        return out