"""
Lending, Step 1: plain mesa, every party an agent, no fairsim yet.

This is the persistent-borrower port of Liu et al.'s delayed impact (see
env-plans/lending.md for the fork). Each borrower owns a true repay_prob
(construct space). The bank MEASURES a score from it (observation space) and
decides. Repaying nudges a borrower's true repay_prob up, defaulting nudges it
down, and denied applicants are left untouched. We watch whether two groups with
different starting creditworthiness drift apart under a single hand-set threshold.

No fairsim imports on purpose. This is the A side of an A/B: the same domain is
rebuilt on the fairsim primitives in `lending_fairsim.py`, so we can compare how
much bookkeeping each costs the user. Once the dynamics look right here, the
observe -> record -> decide -> reveal -> learn machinery gets swapped in there.

Run:
    python src/fairsim/domains/lending_mesa.py
"""

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from mesa import Agent, DataCollector, Model


class Borrower(Agent):
    """A loan applicant. Owns its TRUE creditworthiness; the bank never sees it
    directly, only a measured score. There is deliberately no `score` attribute:
    the score is not something the borrower has, it is what the bank reads.
    """

    def __init__(self, model, repay_prob, group):
        super().__init__(model)
        self.repay_prob = repay_prob   # construct space: true chance of repaying
        self.group = group
        self.approved = False
        self.last_outcome = None       # "repaid" / "defaulted" / None (not lent to)


class Bank(Agent):
    """The lender. Holds a per-group cutoff and decides on a measured score.
    score() is the measurement crossing (construct -> observation): for now an
    identity reading of repay_prob, i.e. the unbiased, monotone case. This is the
    exact seam the Observer takes over in Step 2.
    """

    def __init__(self, model, thresholds):
        super().__init__(model)
        self.thresholds = thresholds   # {group: cutoff}

    def score(self, borrower):
        return borrower.repay_prob     # unbiased, monotone, identity for now

    def approves(self, borrower):
        return self.score(borrower) >= self.thresholds[borrower.group]


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class LendingModel(Model):
    """One tick: every borrower applies, the bank decides, approved loans resolve,
    and a resolved outcome moves that borrower's true repay_prob. Denied applicants
    are untouched, which is where the feedback loop bites: you only move if you
    were lent to, so a group that gets fewer loans gets fewer chances to improve.
    """

    def __init__(
        self,
        n_per_group=200,
        thresholds=None,
        group_means=(0.6, 0.4),
        group_std=0.15,
        shift=0.05,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.thresholds = thresholds if thresholds is not None else {0: 0.5, 1: 0.5}
        self.bank = Bank(self, self.thresholds)
        self.shift = shift

        # Two groups differ only in their STARTING creditworthiness. That is the
        # Liu setup: same scoring, same threshold, different distributions.
        for group, mean in enumerate(group_means):
            for _ in range(n_per_group):
                p = _clip(self.rng.normal(mean, group_std))
                Borrower(self, p, group)

        # Group-wise DECISION metrics come from fairlearn (selection rate by group
        # plus the demographic-parity gap). They are computed once per step() over
        # that step's decisions and stashed here; the reporters just read them back,
        # so the MetricFrame is built once, not once per reporter.
        self._selection_rate: dict = {}   # {group: approval rate this step}
        self._dp_diff = float("nan")      # selection-rate gap across groups

        # mesa-native metrics into a DataCollector (batch_run requires one). The
        # per-group repay-prob mean is a construct-space population statistic, not a
        # metric(y_true, y_pred), so it stays a plain group mean, not a MetricFrame.
        self.datacollector = DataCollector(
            model_reporters={
                "mean_repay_prob_g0": lambda m: m.mean_repay(0),
                "mean_repay_prob_g1": lambda m: m.mean_repay(1),
                "repay_gap": lambda m: m.mean_repay(0) - m.mean_repay(1),
                "approval_rate_g0": lambda m: m._selection_rate.get(0, float("nan")),
                "approval_rate_g1": lambda m: m._selection_rate.get(1, float("nan")),
                "dp_diff": lambda m: m._dp_diff,   # fairlearn demographic-parity gap
            }
        )

    def borrowers(self):
        return self.agents_by_type[Borrower]

    def _members(self, group):
        return [b for b in self.borrowers() if b.group == group]

    def mean_repay(self, group):
        members = self._members(group)
        return sum(b.repay_prob for b in members) / len(members)

    def _record_fairness(self, applicants) -> None:
        # fairlearn wants aligned arrays: the decision (approved) as y_pred and the
        # protected attribute as sensitive_features. selection_rate ignores y_true,
        # so approved doubles as a dummy there. by_group covers every group at once;
        # difference() is the demographic-parity gap, free and N-group general.
        approved = pd.Series({b.unique_id: int(b.approved) for b in applicants})
        groups = pd.Series({b.unique_id: b.group for b in applicants})
        mf = MetricFrame(
            metrics=selection_rate,
            y_true=approved,
            y_pred=approved,
            sensitive_features=groups,
        )
        self._selection_rate = mf.by_group.to_dict()
        self._dp_diff = mf.difference()

    def step(self):
        applicants = self.borrowers()

        # decide: read a score, book the approval (no mutation of construct yet)
        for b in applicants:
            b.approved = self.bank.approves(b)

        # resolve approved loans and move the borrower's TRUE repay_prob
        for b in applicants:
            if not b.approved:
                b.last_outcome = None
                continue
            repaid = self.rng.random() < b.repay_prob
            b.last_outcome = "repaid" if repaid else "defaulted"
            b.repay_prob = _clip(b.repay_prob + (self.shift if repaid else -self.shift))

        self._record_fairness(applicants)
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = LendingModel(rng=0)
    for _ in range(40):
        model.step()

    # DataCollector hands back a DataFrame indexed by step; print every 5th row.
    df = model.datacollector.get_model_vars_dataframe().round(3)
    print(df.iloc[::5])
