# SSNN (Sparse Stateful Neural Network): a proof of concept of a modular recurrent architecture. A router picks a few
# recurrent modules (experts) per event, only those update their state, and a global state integrates their outputs.
# See ssnn/README.md.
#
# It shares activations.py, optimizers.py, progress.py and gradcheck.py with the rest of the repository. Only
# baselines.py imports feedforward.py, for the dense baselines of quality: FeedForward limits the BLAS threads, so the
# benchmarks don't use it (they choose the threads themselves).
