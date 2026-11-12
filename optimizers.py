import numpy as np


# An optimizer decides how the gradients change the parameters in each update (the last step of train_step).
#
# Each one has update(params, grads, learning_rate): params is a list of arrays, which it changes in place, and grads
# the gradient of the loss with respect to each of them, in the same order. Any model can use them: FeedForward passes
# the weights and the biases of its layers (see FeedForward.update), and ssnn/ the arrays of its own modules.


# The state that momentum, RMSprop and Adam keep for every parameter: an array of zeros shaped like each array of
# params. It starts over when the number of arrays changes (e.g. a layer added after some training), so the new ones
# get a state of their own instead of being left out.
def zeros_like_params(params):
    return [np.zeros_like(param) for param in params]


def needs_new_state(state, params):
    return state is None or len(state) != len(params)


# Plain gradient descent: every parameter moves by -learning_rate * gradient. The step is proportional to the gradient,
# so parameters with small gradients learn slowly, and one learning rate has to suit every parameter.
class SGD:
    def update(self, params, grads, learning_rate):
        for param, grad in zip(params, grads):
            param -= learning_rate * grad


# Momentum: each parameter keeps a velocity, the running sum of its gradients with the older ones fading by beta in
# every update, and moves by -learning_rate * velocity. Like a ball rolling downhill, it keeps going in the directions
# where the gradients agree from step to step, and gradients that change sign (zigzagging across a narrow valley)
# cancel out. With beta 0.9, a gradient that stays the same ends up moving the parameter 10 times as far as with SGD.
class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta  # how much of the previous velocity each update keeps
        self.velocity = None

    def update(self, params, grads, learning_rate):
        if needs_new_state(self.velocity, params):
            self.velocity = zeros_like_params(params)

        for param, velocity, grad in zip(params, self.velocity, grads):
            velocity *= self.beta  # velocity = beta * velocity + grad
            velocity += grad
            param -= learning_rate * velocity


# RMSprop: each parameter keeps the running mean of its squared gradients (with rho, how much of it each update keeps),
# and moves by -learning_rate * gradient / sqrt(that mean). Dividing by the usual size of its gradients gives every
# parameter a step of about learning_rate: parameters with small gradients still learn, and those with large ones
# don't jump too far. It is the second half of Adam, without the momentum.
class RMSprop:
    def __init__(self, rho=0.9, epsilon=1e-8):
        self.rho = rho
        self.epsilon = epsilon  # avoids a division by 0
        self.mean_square = None

    def update(self, params, grads, learning_rate):
        if needs_new_state(self.mean_square, params):
            self.mean_square = zeros_like_params(params)

        for param, mean_square, grad in zip(params, self.mean_square, grads):
            mean_square *= self.rho  # mean_square = rho * mean_square + (1 - rho) * grad^2
            mean_square += (1 - self.rho) * grad * grad
            param -= learning_rate * grad / (np.sqrt(mean_square) + self.epsilon)


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

    def update(self, params, grads, learning_rate):
        if needs_new_state(self.m, params):
            self.m, self.v = zeros_like_params(params), zeros_like_params(params)
            self.t = 0  # the bias correction starts over with the averages

        self.t += 1

        # The textbook form divides m by (1 - beta1^t) and v by (1 - beta2^t) for every parameter:
        #   parameter -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
        # Both corrections are the same number for all the parameters of an update, so they can be folded into the
        # step size (and epsilon scaled to match), which gives exactly the same update with fewer passes over the arrays
        # (as noted in section 2 of the Adam paper).
        # Python floats (not NumPy scalars), so float32 parameters are updated in float32
        step_size = float(learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))
        epsilon = float(self.epsilon * np.sqrt(1 - self.beta2 ** self.t))

        for param, m, v, grad in zip(params, self.m, self.v, grads):
            # The running averages, updated in place: m = beta1 * m + (1 - beta1) * grad, and the same for v
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad * grad

            param -= step_size * m / (np.sqrt(v) + epsilon)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSprop, "adam": Adam}


def make_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"Invalid optimizer: {name}. Use one of {list(OPTIMIZERS)}")

    return OPTIMIZERS[name]()
