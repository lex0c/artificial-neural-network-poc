import time

import numpy as np

from optimizers import make_optimizer
from progress import format_duration, format_number
from ssnn.gradients import add_task_gradients, balance_gradients, clip_global_norm, l2_gradients
from ssnn.model import PARAMETER_NAMES


# Training on a task whose examples are independent (experiment A): each example is one event of a session of its
# own, which starts from the zero state. One minibatch:
#   1. for each example: forward (model.step) and the gradients of its cross-entropy (task_gradients)
#   2. the mean of those gradients, plus the balance loss of the router and the L2 penalty
#   3. gradient clipping by global norm, and one step of the optimizer (Adam, shared with the rest of the repository)
# After each epoch, the loss and accuracy on the validation set; the model ends with the parameters of the epoch with
# the lowest validation loss, so the test set is only used for the final evaluation.


# Runs the model on every example, each from the zero state: the probabilities, and the experts selected for each one
def evaluate(model, inputs):
    results = [model.step(x, model.new_state()) for x in inputs]
    probabilities = np.array([probabilities for probabilities, _, _ in results])
    selected = np.array([trace["selected"] for _, _, trace in results])
    return probabilities, selected


def cross_entropy(probabilities, labels):
    return float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12))))


def train(model, train_set, val_set, verbose=False):
    config = model.config
    inputs, labels = train_set[0], train_set[1]
    optimizer = make_optimizer("adam")
    order_rng = np.random.default_rng([config["seed"], 1])  # the order of the examples, apart from the initialization
    batch_size = config["batch_size"]
    history = {"loss": [], "val_loss": [], "val_accuracy": [], "grad_norm": []}
    best_val_loss, best_params, best_epoch = np.inf, None, 0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        order = order_rng.permutation(len(inputs))
        epoch_loss, norms = 0.0, []

        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            grads = {name: np.zeros_like(model.params[name]) for name in PARAMETER_NAMES}
            router_logits, router_inputs = [], []

            for index in batch:
                _, _, trace = model.step(inputs[index], model.new_state())
                epoch_loss += add_task_gradients(model, trace, labels[index], grads)
                if trace["router_logits"] is not None:
                    router_logits.append(trace["router_logits"])
                    router_inputs.append(trace["q"])

            for name in PARAMETER_NAMES:
                grads[name] /= len(batch)

            if router_logits and config["balance_weight"] > 0:
                _, balance = balance_gradients(np.array(router_logits), np.array(router_inputs))
                for name, grad in balance.items():
                    grads[name] += config["balance_weight"] * grad

            if config["l2_weight"] > 0:
                _, l2 = l2_gradients(model.params)
                for name, grad in l2.items():
                    grads[name] += config["l2_weight"] * grad

            norms.append(clip_global_norm(grads, config["clip_grad_norm"]))
            optimizer.update(model.parameters(), [grads[name] for name in PARAMETER_NAMES], config["learning_rate"])

        probabilities, _ = evaluate(model, val_set[0])
        val_loss = cross_entropy(probabilities, val_set[1])
        val_accuracy = float(np.mean(probabilities.argmax(axis=1) == val_set[1]))
        history["loss"].append(epoch_loss / len(inputs))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["grad_norm"].append(float(np.median(norms)))

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_params = {name: array.copy() for name, array in model.params.items()}

        if verbose:
            print(f"  epoch {epoch}/{config['epochs']} - loss {format_number(history['loss'][-1])} - val_loss "
                  f"{format_number(val_loss)} - val_accuracy {val_accuracy:.4f} - {format_duration(time.time() - start_time)}")

    if best_params is not None:
        model.params.update(best_params)
    history["best_epoch"] = best_epoch
    return history
