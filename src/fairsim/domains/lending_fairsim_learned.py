""" Lending, Step 4 (fairsim side): a LEARNED bank.

Same domain as lending_fairsim.py, but the bank no longer reads the score
straight through (ColumnScore). It fits a LogisticRegression on repaid vs the
observed score and approves on predicted P(repaid). The only lines that change
from Step 3 are the Policy, the rule, and one bootstrap branch for the cold
start; the delayed/selective label machinery is untouched, because the training
table comes out of the DataStore for free.

This is the B side of the Step-4 A/B. The A side, lending_mesa_learned.py, has to
hand-build the same point-in-time training table. Compare the two `step()` bodies
and the amount of loan-timing bookkeeping each one carries.

Run from the repo root:
    python -m fairsim.domains.lending_fairsim_learned
"""

import pandas as pd
from mesa import Agent, DataCollector, Model
from sklearn.linear_model import LogisticRegression

from fairsim.core import ColumnScore, DeclarativeObserver, MLAgent, SklearnPolicy, threshold

# ml-fairness-gym's DELAYED_IMPACT credit-cluster distribution (per group cluster
# probabilities and per cluster repayment probabilities). Used only when the
# LendingModel is built with population="credit_clusters".
MLFG_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),   # group 0
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),   # group 1
)
MLFG_SUCCESS_PROBS = (0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7)


class Borrower(Agent):
    """Identical to Step 3: owns its repay_prob (construct) and the timing of its
    own loan. It settles a matured loan in step() and hands the outcome to the
    bank; whether the bank was analytical or learned makes no difference here.
    """

    def __init__(self, model, repay_prob, group, cluster=None):
        super().__init__(model)
        self.repay_prob = repay_prob
        self.group = group
        # cluster is set only in the "credit_clusters" population: a discrete
        # credit bucket (0..6) whose success_probs[cluster] IS the repay_prob.
        # Drift then moves the bucket up/down instead of nudging a float, which
        # mirrors ml-fairness-gym's cluster-shift dynamics.
        self.cluster = cluster
        self.approved = False
        self.issued_at = None     # origination step of an outstanding loan; None = eligible
        self.settlement = None    # (issued_at, repaid) produced at maturity, consumed by the bank
        self.last_outcome = None

    def step(self):
        if self.issued_at is None:
            return
        if self.model.steps - self.issued_at < self.model.repay_lag:
            return
        repaid = self.model.rng.random() < self.repay_prob
        self.last_outcome = "repaid" if repaid else "defaulted"
        if self.cluster is not None:
            # Discrete cluster drift: repay bumps the bucket up, default down,
            # and the repay_prob is read off the fixed success_probs table.
            probs = self.model.success_probs
            self.cluster = min(self.cluster + 1, len(probs) - 1) if repaid \
                else max(self.cluster - 1, 0)
            self.repay_prob = probs[self.cluster]
        else:
            self.repay_prob = _clip(
                self.repay_prob + (self.model.shift if repaid else -self.model.shift)
            )
        self.settlement = int(repaid)   # just the outcome; the store knows the grant step
        self.issued_at = None


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class LendingModel(Model):
    """The bank is an MLAgent carrying a SklearnPolicy. The domain adds exactly one
    thing over Step 3: a bootstrap rule to make decisions before the model has seen
    any labels. Everything about WHEN outcomes arrive stays in reveal_due().
    """

    def __init__(
        self,
        n_per_group=200,
        group_means=(0.6, 0.4),
        group_std=0.15,
        shift=0.05,
        repay_lag=3,
        apply_rate=0.25,
        min_train=50,          # cold-start floor: fit only once this many labels exist
        bootstrap_rate=0.8,    # while the model can't predict, lend to this fraction at random
        population="gaussian",  # or "credit_clusters" for the ml-fairness-gym distribution
        lender="learned",       # or "oracle" for an analytic max-utility threshold
        breakeven=0.5,          # oracle threshold on the true repay probability
        cluster_probabilities=MLFG_CLUSTER_PROBS,
        success_probabilities=MLFG_SUCCESS_PROBS,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.shift = shift
        self.repay_lag = repay_lag
        self.apply_rate = apply_rate
        self.bootstrap_rate = bootstrap_rate
        self.population = population
        self.success_probs = tuple(success_probabilities)

        if population == "credit_clusters":
            # Each borrower is placed in a discrete credit cluster drawn from its
            # group's cluster distribution; its repay_prob is the cluster's
            # success probability. Drift (in Borrower.step) moves the bucket.
            for group in (0, 1):
                probs = cluster_probabilities[group]
                for _ in range(n_per_group):
                    cluster = int(self.rng.choice(len(probs), p=probs))
                    Borrower(self, self.success_probs[cluster], group, cluster=cluster)
        else:
            for group, mean in enumerate(group_means):
                for _ in range(n_per_group):
                    p = _clip(self.rng.normal(mean, group_std))
                    Borrower(self, p, group)

        observer = DeclarativeObserver(
            features=["repay_prob"],
            feature_names=["score"],
            groups="group",
            seed=0,
        )
        # The lender is the experimental variable:
        #   "learned" -> a logistic model fit on delayed/selective labels, then a
        #                group-blind threshold on predicted P(repaid).
        #   "oracle"  -> ml-fairness-gym's analytic max-utility lender: score the
        #                true repayment probability directly (ColumnScore) and
        #                approve at the break-even threshold. No learning.
        if lender == "oracle":
            policy, rule, label = ColumnScore("score"), threshold(breakeven), None
        else:
            policy, rule, label = SklearnPolicy(LogisticRegression()), threshold(0.5), "repaid"
        self.bank = MLAgent(
            self,
            observer,
            policy,
            rule,
            label=label,
            min_train=min_train,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "mean_repay_prob_g0": lambda m: m.mean_repay(0),
                "mean_repay_prob_g1": lambda m: m.mean_repay(1),
                "repay_gap": lambda m: m.mean_repay(0) - m.mean_repay(1),
                "trained": lambda m: int(m.bank.policy.is_ready()),
            }
        )

    def borrowers(self):
        return self.agents_by_type[Borrower]

    def eligible(self):
        free = self.borrowers().select(lambda b: b.issued_at is None)
        if self.apply_rate >= 1.0:
            return free
        return free.select(lambda b: self.rng.random() < self.apply_rate)

    def _members(self, group):
        return [b for b in self.borrowers() if b.group == group]

    def mean_repay(self, group):
        members = self._members(group)
        return sum(b.repay_prob for b in members) / len(members)

    # --- domain truth, plain Python ------------------------------------------

    def bootstrap(self, pool) -> pd.Series:
        # No model yet: lend generously and at random to gather the first labels.
        # These approvals are the domain's call (the Policy made no decision), so the
        # domain also logs them, keeping the audit log complete during the cold start.
        ids = [b.unique_id for b in pool]
        approve = (self.rng.random(len(ids)) < self.bootstrap_rate).astype(int)
        return pd.Series(approve, index=pd.Index(ids, name="agent_id"))

    def apply(self, decisions: pd.Series) -> None:
        by_id = {b.unique_id: b for b in self.borrowers()}
        for aid, approve in decisions.items():
            b = by_id[aid]
            b.approved = bool(approve)
            if approve:
                b.issued_at = self.steps

    def reveal_due(self) -> None:
        # step="last_action" lands each outcome on the step the bank last lent to that
        # borrower, read from the store's decision log, so the domain does not group by
        # grant step. Runs before the bank acts again this step, so "last action" is
        # unambiguous. Delayed and selective as before: only settled loans appear.
        outcomes = {}
        for b in self.borrowers():
            if b.settlement is None:
                continue
            outcomes[b.unique_id] = b.settlement
            b.settlement = None
        self.bank.datastore.reveal("repaid", outcomes, step="last_action")

    def step(self):
        self.borrowers().shuffle_do("step")   # subjects settle matured loans, move repay_prob
        self.reveal_due()                      # write matured outcomes into memory
        self.bank.learn()                      # refit on everything revealed so far (a no-op until min_train)
        applicants = self.eligible()
        _, decisions = self.bank.step(applicants)   # observe -> record -> predict/decide if ready
        if decisions is None:                  # cold start: model not ready, domain decides and logs
            decisions = self.bootstrap(applicants)
            self.bank.datastore.record_decision(decisions, step=self.steps)
        self.apply(decisions)
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = LendingModel(rng=0)
    for _ in range(40):
        model.step()

    print("repay-prob trajectory (construct); `trained`=1 once the bank has fit:")
    print(model.datacollector.get_model_vars_dataframe().round(3).iloc[::5])

    from fairlearn.metrics import selection_rate

    from fairsim.metrics import group_metrics_over_time

    mf = group_metrics_over_time(model.bank.datastore, selection_rate)
    print("\nselection rate by (step, group):")
    print(mf.by_group.round(3).unstack().iloc[::10])
    print("\ndemographic-parity gap per step:")
    print(mf.difference().round(3).iloc[::10])

    # The training table is not something the domain assembles: it is the DataStore's
    # point-in-time join of features to later labels. Show its size and that the bank fit.
    X, y = model.bank.datastore.training_set("repaid")
    print(
        f"\nlearned bank fit on {len(y)} point-in-time rows "
        f"({int(y.sum())} repaid / {int((1 - y).sum())} defaulted); "
        f"is_ready={model.bank.policy.is_ready()}"
    )
