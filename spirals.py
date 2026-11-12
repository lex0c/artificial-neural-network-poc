import numpy as np


# Synthetic data for the decision boundary example: two spirals that wind around each other, one per class. No
# straight line separates them, so a model without hidden layers can't classify them, and a network with hidden
# layers has to bend its boundary to follow the arms.


# Generates num_samples points per spiral: the coordinates [x, y] (between about -1 and 1) and the class (0 or 1).
# Each spiral makes one and a half turns, and noise moves each point a little away from its arm.
def make_spirals(num_samples=500, noise=0.03):
    angle = np.sqrt(np.random.uniform(0, 1, num_samples)) * 3 * np.pi  # the square root spreads the points along the arm
    arm = np.column_stack([angle * np.cos(angle), angle * np.sin(angle)]) / (3 * np.pi)

    points = np.concatenate([arm, -arm]) + np.random.normal(0, noise, (2 * num_samples, 2))  # the second arm is the first one turned by 180 degrees
    labels = np.concatenate([np.zeros(num_samples), np.ones(num_samples)])
    return points, labels
