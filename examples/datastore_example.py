"""
DataStore, exercised: delayed labels, selective labels, point-in-time training.

Run:
    PYTHONPATH=/home/mattia/projects/fairsim .venv/bin/python examples/datastore_example.py

Scenario: a lender observes applicants each step and approves some of them. It
only ever learns whether a loan was repaid FOR THE APPLICANTS IT APPROVED
(selective labels), and it learns that a few steps LATER (delayed labels). The
DataStore is the firm's memory that makes this bookkeeping disappear.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fairsim.core import Batch, DataStore


def observe(step, rng):
    """Fake an Observer: 4 applicants per step, features indexed by agent_id."""
    ids = [f"s{step}_a{i}" for i in range(4)]
    X = pd.DataFrame(
        {
            "income": rng.normal(0, 1, size=4),
            "history": rng.normal(0, 1, size=4),
        },
        index=pd.Index(ids, name="agent_id"),
    )
    groups = pd.Series(rng.integers(0, 2, size=4), index=X.index, name="group")
    return Batch(X=X, groups=groups)


rng = np.random.default_rng(0)
store = DataStore()

# True (hidden) repayment rule: high income + history repay. The firm doesn't
# know this; it will try to learn it from the outcomes it gets to see.
def repaid(row):
    logit = 1.2 * row["income"] + 0.8 * row["history"]
    return int(1 / (1 + np.exp(-logit)) > 0.5)


PENDING = {}  # outcomes we've "approved" but not yet observed: due_step -> [(id, step_seen, y)]

for step in range(1, 7):
    # 1. OBSERVE + RECORD (the "question")
    batch = observe(step, rng)
    store.record(batch, step=step)

    # 2. DECIDE: approve the 2 highest-income applicants (selective!)
    approved = batch.X["income"].nlargest(2).index
    # Their true outcome will only be observable 2 steps from now.
    for aid in approved:
        y = repaid(batch.X.loc[aid])
        PENDING.setdefault(step + 2, []).append((aid, step, y))

    # 3. REVEAL (the "answer") for anything that came due THIS step.
    due = PENDING.pop(step, [])
    if due:
        # point-in-time: attach each label to the exact observation it came from
        by_step = {}
        for aid, seen, y in due:
            by_step.setdefault(seen, {})[aid] = y
        for seen, mapping in by_step.items():
            store.reveal("repaid", mapping, step=seen)

print("Store after 6 steps:")
print(f"  {len(store)} feature rows recorded")
print(f"  labels revealed: {store.frame()['repaid'].notna().sum()} "
      f"(only approved & matured loans)")
print()
print("A peek at the firm's memory (frame()):")
print(store.frame())
print()

# 4. TRAIN on the point-in-time join. Only answered questions come back.
X, y, groups = store.training_set("repaid", with_groups=True)
print(f"training_set: X={X.shape}, y={len(y)} labels, groups aligned={len(groups)}")
print("  X keeps its (agent_id, step) index, so every training row is traceable:")
print(X.head(3))

model = LogisticRegression().fit(X.to_numpy(), y)
print(f"\nfitted LogisticRegression on the harvested labels; coef={np.round(model.coef_[0], 2)}")
print("(true rule was income=1.2, history=0.8 — recovered directionally from selective data)")
