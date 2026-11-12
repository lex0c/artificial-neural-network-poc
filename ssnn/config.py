import hashlib
import json


# The starting configuration. These are starting points, not validated hyperparameters: the data, the training and the
# metrics have to be checked before tuning them.
DEFAULT_CONFIG = {
    "seed": 7,
    "dtype": "float32",
    "input_dim": 8,               # d
    "state_dim": 32,              # H: of each expert, and of the global state
    "num_experts": 16,            # E
    "top_k": 2,                   # k experts run per event
    "num_classes": 2,             # C
    "rho": 0.35,                  # how much of its new value an expert takes when it runs: h = (1 - rho) h + rho n
    "beta": 0.60,                 # the same for the global state: g = (1 - beta) g + beta v
    "routing_temperature": 1.0,   # tau: the weights of the selected experts are softmax(z / tau)
    "router": "learned",          # or "random": k experts drawn at random per event, with equal weights (a baseline)
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-3,
    "balance_weight": 1e-2,       # lambda_bal of the balance loss of the router (gradients.balance_gradients)
    "l2_weight": 0.0,             # lambda_l2, on the weight matrices
    "clip_grad_norm": 1.0,        # the gradients of a minibatch are scaled down to this global norm at most
}


# A copy of DEFAULT_CONFIG with some values changed, e.g. make_config(top_k=16), checked
def make_config(**changes):
    unknown = set(changes) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

    config = {**DEFAULT_CONFIG, **changes}
    validate(config)
    return config


def validate(config):
    checks = [
        (config["dtype"] in ("float32", "float64"), "dtype must be float32 or float64"),
        (1 <= config["top_k"] <= config["num_experts"], "top_k must be between 1 and num_experts"),
        (0 < config["rho"] <= 1 and 0 < config["beta"] <= 1, "rho and beta must be in (0, 1]"),
        (config["routing_temperature"] > 0, "routing_temperature must be positive"),
        (config["router"] in ("learned", "random"), "router must be learned or random"),
        (min(config["input_dim"], config["state_dim"], config["num_classes"]) >= 1, "the dimensions must be at least 1"),
    ]

    for ok, message in checks:
        if not ok:
            raise ValueError(message)


# A short fingerprint of a configuration, to record with every result
def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
