import numpy as np


# Quality and routing metrics


def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


# Accuracy within each group (e.g. each regime of experiment A), as {group: accuracy}
def accuracy_by_group(predictions, labels, groups):
    predictions, labels, groups = map(np.asarray, (predictions, labels, groups))
    return {int(group): accuracy(predictions[groups == group], labels[groups == group]) for group in np.unique(groups)}


def entropy_bits(probabilities):
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


# How the router used the experts, from the experts selected for each event (one row of k indices per event):
#   usage: the share of the selections that went to each expert
#   unused: how many experts were never selected
#   usage_entropy: the entropy of usage in bits, log2(E) when every expert is used equally
#   group_information: with groups (e.g. the regime of each event), the mutual information in bits between the group
#     of an event and the experts it selects: 0 when every group uses the experts in the same proportions, up to
#     log2(number of groups) when each group has experts of its own (specialization)
def routing_summary(selected, num_experts, groups=None):
    selected = np.asarray(selected)
    usage = np.bincount(selected.ravel(), minlength=num_experts) / selected.size
    summary = {"usage": usage, "unused": int(np.sum(usage == 0)), "usage_entropy": entropy_bits(usage)}

    if groups is not None:
        groups = np.repeat(np.asarray(groups), selected.shape[1])  # one group per selection
        joint = np.zeros((groups.max() + 1, num_experts))
        np.add.at(joint, (groups, selected.ravel()), 1)
        joint /= joint.sum()
        information = entropy_bits(joint.sum(axis=1)) + entropy_bits(joint.sum(axis=0)) - entropy_bits(joint.ravel())
        summary["group_information"] = max(0.0, information)  # never below 0, which rounding could give

    return summary
