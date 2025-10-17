from torch import nn


class GestureClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GestureClassifier, self).__init__()
        #Input tensor mit 21 keypoints, 15 distances, Avarage Distance without wrist
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(37, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )