"""
Lending, Step 4 (mesa side): the SAME learned bank, by hand.

This is the A side of the Step-4 A/B. It does everything lending_fairsim_learned.py
does (delayed outcomes, selective labels, a LogisticRegression bank with a random
bootstrap), but with no fairsim primitives. The point of the comparison is what a
learned bank needs that a fixed-threshold bank did not: a POINT-IN-TIME training
table. A loan decided at step 3 on that step's score is only labelled at step 6
when it matures. The row we train on must pair the score AS SEEN AT STEP 3 with the
outcome from step 6. If we instead read the borrower's score at maturity we leak
the outcome (the score already moved because of it).

So this file hand-rolls a tiny DataStore: `self.log`, keyed by (agent_id, step),
holding the score and group we saw at decision time, with the label written back
later. Watch how much of `step()` is loan-timing bookkeeping. In the fairsim
version the DataStore owns all of it and `step()` stays about the domain.

Run:
    python src/fairsim/domains/lending_mesa_learned.py
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from mesa import Agent, DataCollector, Model
from sklearn.linear_model import LogisticRegression


class Borrower(Agent):
    """Owns its true repay_prob and the timing of its own loan. When a loan
    matures it settles it and reports the outcome back to the model, tagged with
    the ORIGINATION step so the model can pin the label onto the right log row.
    """

    def __init__(self, model, repay_prob, group):
        super().__init__(model)
        self.repay_prob = repay_prob
        self.group = group
        self.approved = False
        self.issued_at = None
        self.last_outcome = None

    def step(self):
        if self.issued_at is None:
            return
        if self.model.steps - self.issued_at < self.model.repay_lag:
            return
        repaid = int(self.model.rng.random() < self.repay_prob)
        self.last_outcome = "repaid" if repaid else "defaulted"
        self.repay_prob = _clip(
            self.repay_prob + (self.model.shift if repaid else -self.model.shift)
        )
        # Hand the outcome back tagged with WHEN the loan was granted, so the model
        # can find the matching feature row. This tag is the manual equivalent of
        # the DataStore's (agent_id, step) key.
        self.model.record_outcome(self.unique_id, self.issued_at, repaid)
        self.issued_at = None


class Bank(Agent):
    """The lender. Measures a score (identity reading of repay_prob) and, once it
    has labels, predicts P(repaid) with a LogisticRegression. Before it can fit, it
    lends at random (bootstrap).
    """

    def __init__(self, model, min_train, bootstrap_rate):
        super().__init__(model)
        self.model_ml = LogisticRegression()
        self.fitted = False
        self.min_train = min_train
        self.bootstrap_rate = bootstrap_rate

    def score(self, borrower):
        return borrower.repay_prob     # unbiased, monotone, identity

    def decide(self, pool):
        # Cold start: no model, lend at random. This branch is the same idea as the
        # fairsim bootstrap, but here it also has to stay in sync with how the log is
        # written below, because there is no primitive tying decision to record.
        if not self.fitted:
            approve = self.model.rng.random(len(pool)) < self.bootstrap_rate
            return {b.unique_id: int(a) for b, a in zip(pool, approve)}
        # Warm: predict P(repaid) from the measured score and threshold at 0.5.
        X = np.array([[self.score(b)] for b in pool])
        proba = self.model_ml.predict_proba(X)[:, 1]
        return {b.unique_id: int(p >= 0.5) for b, p in zip(pool, proba)}


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class LendingModel(Model):
    """Same dynamics as the fairsim Step-4 model. The extra weight here is `self.log`
    and the three places that touch it: record a row at decision time, write a label
    back at maturity, and rebuild (X, y) to fit. That is a hand-rolled DataStore.
    """

    def __init__(
        self,
        n_per_group=200,
        group_means=(0.6, 0.4),
        group_std=0.15,
        shift=0.05,
        repay_lag=3,
        apply_rate=0.25,
        min_train=50,
        bootstrap_rate=0.8,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.shift = shift
        self.repay_lag = repay_lag
        self.apply_rate = apply_rate
        self.bank = Bank(self, min_train, bootstrap_rate)

        for group, mean in enumerate(group_means):
            for _ in range(n_per_group):
                p = _clip(self.rng.normal(mean, group_std))
                Borrower(self, p, group)

        # The hand-rolled DataStore. One row per applicant per step, keyed by
        # (agent_id, step): the score and group we SAW then, and a `repaid` slot that
        # stays None until that loan matures. This is exactly what the fairsim
        # DataStore does for us, spelled out because mesa has no such thing.
        self.log: dict[tuple[int, int], dict] = {}

        self._selection_rate: dict = {}
        self._dp_diff = float("nan")

        self.datacollector = DataCollector(
            model_reporters={
                "mean_repay_prob_g0": lambda m: m.mean_repay(0),
                "mean_repay_prob_g1": lambda m: m.mean_repay(1),
                "repay_gap": lambda m: m.mean_repay(0) - m.mean_repay(1),
                "approval_rate_g0": lambda m: m._selection_rate.get(0, float("nan")),
                "approval_rate_g1": lambda m: m._selection_rate.get(1, float("nan")),
                "dp_diff": lambda m: m._dp_diff,
                "trained": lambda m: int(m.bank.fitted),
            }
        )

    def borrowers(self):
        return self.agents_by_type[Borrower]

    def eligible(self):
        free = self.borrowers().select(lambda b: b.issued_at is None)
        if self.apply_rate >= 1.0:
            return list(free)
        return [b for b in free if self.rng.random() < self.apply_rate]

    def _members(self, group):
        return [b for b in self.borrowers() if b.group == group]

    def mean_repay(self, group):
        members = self._members(group)
        return sum(b.repay_prob for b in members) / len(members)

    # --- the hand-rolled DataStore: record, write-back, and train ------------

    def record_decision(self, pool, decisions) -> None:
        # Snapshot the score and group AS SEEN NOW, before any outcome moves them.
        # Miss this ordering and the training row later leaks the outcome.
        for b in pool:
            self.log[(b.unique_id, self.steps)] = {
                "score": self.bank.score(b),
                "group": b.group,
                "decision": decisions[b.unique_id],
                "repaid": None,
            }

    def record_outcome(self, agent_id, issued_at, repaid) -> None:
        # Point-in-time write-back: land the label on the row from the origination
        # step, not on a fresh row. Selective by construction, since only approved
        # loans ever call this.
        row = self.log.get((agent_id, issued_at))
        if row is not None:
            row["repaid"] = repaid

    def _training_xy(self):
        # Rebuild (X, y) from every row whose label has come back. This is the
        # dropna-on-label join the DataStore's training_set() does in one call.
        rows = [r for r in self.log.values() if r["repaid"] is not None]
        X = np.array([[r["score"]] for r in rows])
        y = np.array([r["repaid"] for r in rows])
        return X, y

    def learn(self) -> None:
        X, y = self._training_xy()
        if len(y) < self.bank.min_train:
            return
        self.bank.model_ml.fit(X, y)
        self.bank.fitted = True

    # --- domain truth --------------------------------------------------------

    def apply(self, decisions) -> None:
        by_id = {b.unique_id: b for b in self.borrowers()}
        for aid, approve in decisions.items():
            b = by_id[aid]
            b.approved = bool(approve)
            if approve:
                b.issued_at = self.steps

    def _record_fairness(self, pool, decisions) -> None:
        approved = pd.Series({b.unique_id: decisions[b.unique_id] for b in pool})
        groups = pd.Series({b.unique_id: b.group for b in pool})
        if approved.empty:
            self._selection_rate, self._dp_diff = {}, float("nan")
            return
        mf = MetricFrame(
            metrics=selection_rate,
            y_true=approved,
            y_pred=approved,
            sensitive_features=groups,
        )
        self._selection_rate = mf.by_group.to_dict()
        self._dp_diff = mf.difference()

    def step(self):
        self.borrowers().shuffle_do("step")   # settle matured loans, write outcomes back into the log
        self.learn()                           # rebuild (X, y) from the log and refit
        pool = self.eligible()
        decisions = self.bank.decide(pool)     # bootstrap or predict
        self.record_decision(pool, decisions)  # snapshot score+group NOW, before outcomes move them
        self.apply(decisions)
        self._record_fairness(pool, decisions)
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = LendingModel(rng=0)
    for _ in range(40):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().round(3)
    print("repay-prob + approval trajectory; `trained`=1 once the bank has fit:")
    print(df[["mean_repay_prob_g0", "mean_repay_prob_g1", "repay_gap",
              "approval_rate_g0", "approval_rate_g1", "dp_diff", "trained"]].iloc[::5])

    labelled = [r for r in model.log.values() if r["repaid"] is not None]
    repaid = sum(r["repaid"] for r in labelled)
    print(
        f"\nlearned bank fit on {len(labelled)} hand-joined rows "
        f"({repaid} repaid / {len(labelled) - repaid} defaulted); "
        f"fitted={model.bank.fitted}. The log carries {len(model.log)} rows total."
    )
