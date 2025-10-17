from enum import Enum

import torch

from data.loader import YoloHandKptsDataset
from visualizer.picture import show_Skeleton, show_middle_point
import numpy as np


class Finger(Enum):
    WRIST = 0
    THUMB = 4
    INDEX = 8
    MIDDLE = 12
    RING = 16
    PINKY = 20

class Frame:
    def __init__(self, img, keypoints):
        self.img = img #Tensor mit RGB
        self.keypoints = keypoints

    def draw(self):
        show_Skeleton(self.img, self.keypoints)

    def avarage_distance(self):
        if hasattr(self.keypoints, 'detach'):
            k = self.keypoints.detach().cpu().numpy()
        else:
            k = np.asarray(self.keypoints)
        xy = k[:, :2]
        n = len(xy)
        if n < 2:
            return 0.0, 0.0
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(xy[i] - xy[j]))
        avg_all = float(np.mean(dists))
        tips = [4, 8, 12, 16, 20]
        tip_sum = 0.0
        for i in range(len(tips)):
            a = xy[tips[i]]
            b = xy[tips[(i + 1) % len(tips)]]
            tip_sum += float(np.linalg.norm(a - b))
        return avg_all, tip_sum

    def tips_distance(self, first, second):
        if hasattr(self.keypoints, 'detach'):
            k = self.keypoints.detach().cpu().numpy()
        else:
            k = np.asarray(self.keypoints)
        return float(np.linalg.norm(k[first, :2] - k[second, :2]))

    def get_middle_point(self):
        if hasattr(self.keypoints, 'detach'):
            k = self.keypoints.detach().cpu().numpy()
        else:
            k = np.asarray(self.keypoints)
        xy = k[:, :2]
        n = len(xy)
        tips = [4, 8, 12, 16, 20]
        valid = [i for i in tips if i < n and not np.isnan(xy[i]).any() and not np.isinf(xy[i]).any()]
        if len(valid) > 0:
            c = xy[valid].mean(axis=0)
        else:
            mask = ~np.isnan(xy).any(axis=1) & ~np.isinf(xy).any(axis=1)
            if mask.any():
                c = xy[mask].mean(axis=0)
            else:
                c = np.array([0.0, 0.0], dtype=float)
        return float(c[0]), float(c[1])

yolo = YoloHandKptsDataset(root="../data/raw/hand-keypoints/train", size=192)
img, kp = yolo[78]

frame = Frame(img, kp)
show_middle_point(frame.img, frame.keypoints,frame.get_middle_point()[0],frame.get_middle_point()[1])