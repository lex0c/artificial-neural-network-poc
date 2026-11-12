import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from spirals import make_spirals


class SpiralsTest(unittest.TestCase):
    def test_one_spiral_per_class(self):
        np.random.seed(0)
        points, labels = make_spirals(num_samples=200, noise=0.0)

        self.assertEqual(points.shape, (400, 2))
        self.assertEqual(list(labels), [0] * 200 + [1] * 200)
        self.assertLessEqual(np.abs(points).max(), 1.0)  # within the plot of train-spirals.py

        # Without noise, the second spiral is the first one turned by 180 degrees
        np.testing.assert_allclose(points[200:], -points[:200])

    def test_no_straight_line_separates_them(self):
        # The best straight line between the classes (a logistic regression) gets only about 2 points in 3 right
        np.random.seed(1)
        points, labels = make_spirals(num_samples=2000)

        accuracy = LogisticRegression().fit(points, labels).score(points, labels)
        self.assertLess(accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
