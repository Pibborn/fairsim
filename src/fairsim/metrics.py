"""
fairsim metrics: turn a firm's audit log into group-wise fairness numbers.

This is the measurement layer. It depends on fairlearn (a first-class dependency:
it is the right tool and central to the problem) and on the fairsim DataStore,
whose frame() is the point-in-time-correct audit table these metrics read.

Why this is a module and not a DataStore method: the DataStore owns memory and its
job ends at frame(); turning that table into fairness numbers is measurement, a
different responsibility. Keeping them apart is a "who owns what" line, not a
dependency one. We hand back a fairlearn MetricFrame rather than a bespoke result,
because MetricFrame is the vocabulary the rest of the library already speaks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
from fairlearn.metrics import MetricFrame

from fairsim.core import DataStore


def group_metrics_over_time(
    store: DataStore,
    metrics: Callable[..., Any] | Mapping[str, Callable[..., Any]],
    *,
    y_pred: str = "decision",
    y_true: str | None = None,
    group: str = "group",
) -> MetricFrame:
    """Per-(step, group) metric(s) over a firm's audit log.

    step is a fairlearn CONTROL feature (condition on it) and group is the
    SENSITIVE feature (the fairness axis), so fairlearn does the per-step slicing:
      * mf.by_group    -> the metric indexed by (step, group)
      * mf.difference() -> the group gap, one value per step

    `metrics` is a single metric or a name->metric mapping (fairlearn handles both).
    Pass `y_true` to use a label-based metric (e.g. true_positive_rate over the
    "repaid" column); it defaults to `y_pred`, which suits label-free metrics like
    selection_rate. N-group general; no per-group bookkeeping on the caller's side.

    When `y_true` names a real label, rows whose label has not been revealed yet are
    DROPPED before computing. This is not a convenience: a label-based rate (TPR,
    FPR, ...) is only defined on answered questions, and the audit log carries NaN
    for both delayed rows (outcome still maturing) and selectively-unlabelled rows
    (denied applicants who never generate an outcome). Feeding those to fairlearn is
    an error, so an error rate reported here is always conditioned on "of the loans
    that matured", which is the honest population to report it over.
    """
    frame = store.frame()
    if y_true is not None:
        frame = frame[frame[y_true].notna()]
    steps = pd.Series(
        frame.index.get_level_values("step"), index=frame.index, name="step"
    )
    y_true_col = frame[y_true] if y_true is not None else frame[y_pred]
    return MetricFrame(
        metrics=metrics,
        y_true=y_true_col,
        y_pred=frame[y_pred],
        sensitive_features=frame[group],
        control_features=steps,
    )
