import numpy as np


class FaceStability:

    def __init__(self, threshold=5):

        self.previous = None
        self.threshold = threshold

    def is_stable(self, point):

        if self.previous is None:

            self.previous = point
            return False

        d = np.linalg.norm(
            np.array(point) -
            np.array(self.previous)
        )

        self.previous = point

        return d < self.threshold

    def reset(self):

        self.previous = None