import os
import time


# Helpers to report the progress of a training.


def format_duration(seconds):
    seconds = int(round(seconds))

    if seconds < 60:
        return f"{seconds}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# "1m 05s elapsed, 2m 10s left", where the time left assumes the rest goes at the same pace as what is done.
def timing(start_time, done, total):
    elapsed = time.time() - start_time
    left = elapsed / done * (total - done) if done else 0.0
    return f"{format_duration(elapsed)} elapsed, {format_duration(left)} left"


# A path for messages: relative to the current directory, so it can be copied into the next command.
def display_path(path):
    return os.path.relpath(path)


# 4 significant digits (0.04842, 1.855e-30), or "-" when there is no value yet.
def format_number(value):
    return "-" if value is None else f"{value:.4g}"
