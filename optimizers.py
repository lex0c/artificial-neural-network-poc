import numpy as np


# An optimizer decides how the gradients change the parameters in each update (the last step of train_step).
# Each one has update(layers, gradients, learning_rate), where gradients holds (dL/dweights, dL/dbiases) per layer.


# Plain gradient descent: every parameter moves by -learning_rate * gradient. The step is proportional to the gradient,
# so parameters with small gradients learn slowly, and one learning rate has to suit every parameter.
class SGD:
    def update(self, layers, gradients, learning_rate):
        for layer, (grad_weights, grad_biases) in zip(layers, gradients):
            layer["weights"] -= learning_rate * grad_weights
            layer["biases"] -= learning_rate * grad_biases


# Adam (adaptive moment estimation) keeps two running averages for every parameter:
#   m: the mean of its recent gradients (momentum: steps keep their direction when the gradients agree, and noisy
#      gradients that change sign cancel out)
#   v: the mean of its recent squared gradients (how large its gradients usually are)
# and moves each parameter by learning_rate * m / sqrt(v). Dividing by sqrt(v) gives every parameter a step of about
# learning_rate, whatever the scale of its gradients, so parameters with small gradients still learn.
#
# m and v start at 0, so in the first updates they are too small; dividing by (1 - beta^t) corrects that (bias
# correction). In the first update this makes the step learning_rate * gradient / (|gradient| + epsilon): about
# learning_rate against the direction of the gradient, unless the gradient is close to epsilon. The averages carry over
# between train() calls.
class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1      # how much of the previous mean of the gradients each update keeps
        self.beta2 = beta2      # the same, for the mean of the squared gradients
        self.epsilon = epsilon  # avoids a division by 0
        self.m = None
        self.v = None
        self.t = 0              # number of updates so far

    def update(self, layers, gradients, learning_rate):
        if self.m is None:
            self.m = [{"weights": np.zeros_like(l["weights"]), "biases": np.zeros_like(l["biases"])} for l in layers]
            self.v = [{"weights": np.zeros_like(l["weights"]), "biases": np.zeros_like(l["biases"])} for l in layers]

        self.t += 1

        # The textbook form divides m by (1 - beta1^t) and v by (1 - beta2^t) for every parameter:
        #   parameter -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
        # Both corrections are the same number for all the parameters of an update, so they can be folded into the
        # step size (and epsilon scaled to match), which gives exactly the same update with fewer passes over the arrays
        # (as noted in section 2 of the Adam paper).
        step_size = learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        epsilon = self.epsilon * np.sqrt(1 - self.beta2 ** self.t)

        for layer, m, v, layer_gradients in zip(layers, self.m, self.v, gradients):
            for key, grad in zip(("weights", "biases"), layer_gradients):
                # The running averages, updated in place: m = beta1 * m + (1 - beta1) * grad, and the same for v
                m[key] *= self.beta1
                m[key] += (1 - self.beta1) * grad
                v[key] *= self.beta2
                v[key] += (1 - self.beta2) * grad * grad

                layer[key] -= step_size * m[key] / (np.sqrt(v[key]) + epsilon)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


def make_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"Invalid optimizer: {name}. Use one of {list(OPTIMIZERS)}")

    return OPTIMIZERS[name]()
