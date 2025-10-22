from data.loader import YoloHandKptsDataset
from preprocessor.featurizer import HandPoseFeaturizer
from preprocessor.normalizer import HandPoseNormalizer
from visualizer.hand_visualizer import show_Skeleton, show_middle_point_wrapper, show_middle_point


class Frame:
    def __init__(self, img, keypoints):
        self.img = img
        self.keypoints = keypoints
    def draw(self, fn):
        return fn(self.img, self.keypoints)


ds = YoloHandKptsDataset("../data/raw/hand-keypoints/train")
fr = Frame(*ds[88])

normalizer = HandPoseFeaturizer(HandPoseNormalizer())
print(normalizer(fr.keypoints))
show_middle_point(fr.img,normalizer(fr.keypoints),
                  normalizer.get_middlepoint(fr.keypoints),
                  normalizer.get_middlepoint(fr.keypoints))