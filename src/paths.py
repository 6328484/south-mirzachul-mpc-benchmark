"""Locations of the input data, stored results and figures, relative to the repository root.

All scripts in src/ read and write files through these helpers, so they can be started from any
working directory, e.g. ``python src/evaluate.py``.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
POLICIES = os.path.join(RESULTS, "policies")
LOGS = os.path.join(RESULTS, "training_logs")
FIGURES = os.path.join(ROOT, "figures")


def data_file(name):
    return os.path.join(DATA, name)


def result_file(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def policy_file(name):
    os.makedirs(POLICIES, exist_ok=True)
    return os.path.join(POLICIES, name)


def log_file(name):
    os.makedirs(LOGS, exist_ok=True)
    return os.path.join(LOGS, name)


def figure_file(name):
    os.makedirs(FIGURES, exist_ok=True)
    return os.path.join(FIGURES, name)
