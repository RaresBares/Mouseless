class Frame:
    def __init__(self, img, keypoints):
        self.img = img
        self.keypoints = keypoints
    def draw(self, fn):
        return fn(self.img, self.keypoints)