"""Causal detector-cost accounting and conservative LOCAL admission.

No labels, future frames, fitted model, or detector inference is used here.
The LOCAL prior is an unvalidated configuration choice. Online updates use
whether an attempted crop pass avoided FULL, never ground-truth accuracy.
"""
from collections import deque

import numpy as np


class DetectorBudget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.costs = deque(maxlen=self.cfg.budget_window)
        self.dense_costs = deque(maxlen=self.cfg.budget_window)

    def projected_fraction(self, cost, dense_cost):
        # Account for eviction before a new frame enters the rolling window.
        spent, dense = sum(self.costs), sum(self.dense_costs)
        if len(self.costs) == self.costs.maxlen:
            spent -= self.costs[0]
            dense -= self.dense_costs[0]
        return (spent + cost) / max(dense + dense_cost, 1e-9)

    def allows(self, cost, dense_cost):
        return self.projected_fraction(cost, dense_cost) <= self.cfg.detector_budget_fraction

    def record(self, cost, dense_cost):
        self.costs.append(float(cost))
        self.dense_costs.append(float(dense_cost))
        return sum(self.costs) / max(sum(self.dense_costs), 1e-9)


class LocalCostGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.attempts = self.successes = 0
        self.full_times = deque(maxlen=32)
        self.roi_times = {}

    @property
    def success_estimate(self):
        c = self.cfg
        return (c.local_success_prior * c.local_prior_strength + self.successes) / (c.local_prior_strength + self.attempts)

    def observe_full(self, milliseconds):
        if milliseconds > 0:
            self.full_times.append(float(milliseconds))

    def observe_local(self, milliseconds, avoided_full, n_crops=1):
        self.attempts += 1
        self.successes += bool(avoided_full)
        if milliseconds > 0:
            self.roi_times.setdefault(n_crops, deque(maxlen=32)).append(float(milliseconds))

    def admit(self, n_crops, roi_cost, full_cost):
        ratio = roi_cost / max(full_cost, 1e-9)
        if n_crops == 0:
            return False, "local_empty", ratio
        if n_crops > self.cfg.local_max_crops or ratio > self.cfg.local_max_cost_fraction:
            return False, "local_cost_limit", ratio
        times = self.roi_times.get(n_crops, ())
        time_ratio = float(np.median(times) / np.median(self.full_times)) \
            if times and self.full_times else 0.0
        required = max(ratio, time_ratio) + self.cfg.local_cost_margin
        if self.success_estimate <= required:
            return False, "local_low_success", ratio
        return True, "local_admitted", ratio
