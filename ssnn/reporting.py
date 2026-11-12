import csv
import os
import platform
import subprocess

import numpy as np


# What the scripts record with every result, so it can be traced to the code and the machine that gave it

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# The commit of the code, with -dirty when any file has changes that aren't committed, new files included
def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return commit + ("-dirty" if changes.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def cpu_name():
    try:
        with open("/proc/cpuinfo") as file:
            return next(line.split(":", 1)[1].strip() for line in file if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def environment():
    return {"commit": git_commit(), "cpu": cpu_name(), "numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "")}


# Adds a row to a CSV file, writing the header if the file is new. A file with other columns raises an error: the row
# would otherwise land under the wrong headers without any warning.
def append_row(filepath, row):
    new_file = not os.path.exists(filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if not new_file:
        with open(filepath, newline="") as file:
            header = next(csv.reader(file), [])
        if header != list(row):
            raise ValueError(f"{filepath} has other columns than this row: move it away or use another file")

    with open(filepath, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
