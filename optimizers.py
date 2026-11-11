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
# correction). In the first update this makes the step exactly learning_rate * gradient / |gradient|, about
# learning_rate in the direction of the gradient. The averages carry over between train() calls.
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

        for layer, m, v, layer_gradients in zip(layers, self.m, self.v, gradients):
            for key, grad in zip(("weights", "biases"), layer_gradients):
                m[key] = self.beta1 * m[key] + (1 - self.beta1) * grad
                v[key] = self.beta2 * v[key] + (1 - self.beta2) * grad ** 2

                m_corrected = m[key] / (1 - self.beta1 ** self.t)
                v_corrected = v[key] / (1 - self.beta2 ** self.t)

                layer[key] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


def make_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"Invalid optimizer: {name}. Use one of {list(OPTIMIZERS)}")

    return OPTIMIZERS[name]()
