import numpy as np


# Checks gradients written by hand against the definition of the derivative: each value of each array is nudged a
# little up and down, and the change of the loss gives its gradient (central differences):
#
#   dL/dw ~ (L(w + eps) - L(w - eps)) / (2 * eps)
#
# loss() computes the loss from the current values of the arrays in params, which this changes in place and puts back.
# It costs two evaluations of the loss per value, so it is for small models in tests. Use float64: in float32 the
# differences are too coarse to check a gradient. Used by the tests of feedforward.py and of ssnn/.
def numerical_gradients(loss, params, eps=1e-6):
    gradients = []

    for param in params:
        grad = np.zeros_like(param)

        for idx in np.ndindex(param.shape):
            original = param[idx]

            param[idx] = original + eps
            loss_plus = loss()

            param[idx] = original - eps
            loss_minus = loss()

            param[idx] = original
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        gradients.append(grad)

    return gradients
