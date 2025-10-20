import numpy as np
from torchvision.transforms.functional import hflip as tv_hflip, affine as tv_affine, InterpolationMode


#Random Augmenter for randomizing Trainingsdata

class RandomHandAugment:
    def __init__(self, degrees=30.0, scale=(0.9,1.1), translate=0.1, hflip=0.5):
        self.degrees = float(degrees)
        self.scale = tuple(scale)
        self.translate = float(translate)
        self.hflip = float(hflip)
    def __call__(self, img_t, k):
        C,H,W = img_t.shape
        x = np.asarray(k, dtype=np.float32).copy()
        pts = x[:, :2]
        pts[:, 0] *= W
        pts[:, 1] *= H
        do_flip = np.random.rand() < self.hflip
        if do_flip:
            img_t = tv_hflip(img_t)
            pts[:, 0] = (W - 1) - pts[:, 0]
        ang = float(np.random.uniform(-self.degrees, self.degrees))
        sc = float(np.random.uniform(self.scale[0], self.scale[1]))
        tx = float(np.random.uniform(-self.translate, self.translate) * W)
        ty = float(np.random.uniform(-self.translate, self.translate) * H)
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        img_t = tv_affine(img_t, angle=ang, translate=[int(tx), int(ty)], scale=sc, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, center=[cx, cy])
        rad = np.deg2rad(ang)
        ca, sa = np.cos(rad), np.sin(rad)
        R = np.array([[ca, -sa], [sa, ca]], np.float32)
        pts -= np.array([cx, cy], np.float32)
        pts = (pts @ R.T) * sc
        pts += np.array([cx + tx, cy + ty], np.float32)
        pts[:, 0] /= W
        pts[:, 1] /= H
        x[:, :2] = pts
        return img_t, x