__author__ = "Polyarnyi Nickolay"


import numpy as np


_circle_dx = np.array([0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1] * 2)
_circle_dy = np.array([3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3] * 2)

class DummyFastDetector:
    """
    @see http://www.edwardrosten.com/work/fast.html
    """

    def __init__(self, n=9, threshold=30):
        self.n = n
        self.threshold = threshold

    def _has_n_trues_in_a_row(self, a):
        assert len(a) > 0
        a = np.array(np.concatenate(([False], a, [False])), np.uint8)
        from_true_to_false = np.nonzero((a[1:] - a[:-1]) == 255)[0]
        from_false_to_true = np.nonzero((a[1:] - a[:-1]) == 1)[0]
        count_in_a_row = from_true_to_false - from_false_to_true
        return count_in_a_row.max() >= self.n

    def detect(self, img):
        h, w = img.shape[:2]
        points = []
        for x in range(3, w-3):
            for y in range(3, h-3):
                intencity = img[y, x]

                intencities = img[_circle_dy + y, _circle_dx + x]

                lighter = intencities > intencity + self.threshold
                darker = intencities < intencity - self.threshold

                if lighter.sum() >= 2 * self.n and self._has_n_trues_in_a_row(lighter):
                    points.append([x, y])
                elif darker.sum() >= 2 * self.n and self._has_n_trues_in_a_row(darker):
                    points.append([x, y])
        return np.array(points) / w
