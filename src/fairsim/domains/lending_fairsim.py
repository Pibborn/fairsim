"""
Lending, Step 3: delayed and selective labels on the fairsim primitives.

Step 2 resolved repayment the same step a loan was granted. Real lending does not:
a loan granted at step t is repaid or defaulted at t + repay_lag, and you only ever
learn the outcome for loans you actually granted. This is where the DataStore stops
being decorative:

  * a loan's features are recorded at grant time (step t);
  * the outcome is revealed later, at maturity, back onto the step-t row
    (reveal(step=issued_at)), so the label describes the features it was decided on;
  * only approved borrowers are ever revealed (selective labels).

The bank still does not learn yet (ColumnScore), so the store is not feeding a
model. But it is now a correct point-in-time log of delayed, selective outcomes,
which is exactly the structure a learned bank needs in Step 4 and which the mesa
version would have to hand-build.

Run from the repo root:
    python -m fairsim.domains.lending_fairsim
"""

import pandas as pd
from mesa import Agent, DataCollector, Model

from fairsim.core import ColumnScore, DeclarativeObserver, MLAgent, by_group


class Borrower(Agent):
    """Owns its TRUE repay_prob (construct space), no `score` attribute. Also owns
    the timing of its own loan: when a loan it took matures, it settles it in
    step() (subject autonomy), moving its own repay_prob and handing the realized
    outcome to the bank to record.
    """

    def __init__(self, model, repay_prob, group):
        super().__init__(model)
        self.repay_prob = repay_prob
        self.group = group
        self.approved = False
        self.issued_at = None     # origination step of an outstanding loan; None = eligible
        self.settlement = None    # repaid (0/1) produced at maturity, consumed by the bank
        self.last_outcome = None  # "repaid" / "defaulted" / None, for readability

    def step(self):
        # Subject autonomy: settle my loan the step it matures, and move my own
        # repay_prob. The realized outcome is handed to the bank via `settlement`;
        # writing it into the bank's memory is the institution's job, not mine.
        if self.issued_at is None:
            return
        if self.model.steps - self.issued_at < self.model.repay_lag:
            return  # loan not matured yet
        repaid = self.model.rng.random() < self.repay_prob
        self.last_outcome = "repaid" if repaid else "defaulted"
        self.repay_prob = _clip(
            self.repay_prob + (self.model.shift if repaid else -self.model.shift)
        )
        # Just the outcome. I no longer stamp the grant step onto the handoff: the
        # bank's DataStore already knows when it last lent to me, so reveal_due()
        # attaches this to that step via step="last_action".
        self.settlement = int(repaid)
        self.issued_at = None      # loan closed; eligible to apply again


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class LendingModel(Model):
    """Same tick as lending_mesa, but the observe -> decide half is fairsim's. The
    bank is an MLAgent (Observer + Policy + DecisionRule + DataStore); apply(),
    resolve() and reveal() are domain truth and stay plain Python here.
    """

    def __init__(
        self,
        n_per_group=200,
        thresholds=None,
        group_means=(0.6, 0.4),
        group_std=0.15,
        shift=0.05,
        repay_lag=3,
        apply_rate=0.25,
        rng=None,
    ):
        super().__init__(rng=rng)
        thresholds = thresholds if thresholds is not None else {0: 0.5, 1: 0.5}
        self.shift = shift
        self.repay_lag = repay_lag   # steps between granting a loan and its outcome
        # Each eligible borrower applies with this probability per step. Below 1.0
        # it staggers applications so the population does not move in lockstep; in
        # lockstep the approved cohort would all lock into loans at once and the
        # applicant pool would swing between "everyone" and "only past rejects".
        self.apply_rate = apply_rate

        for group, mean in enumerate(group_means):
            for _ in range(n_per_group):
                p = _clip(self.rng.normal(mean, group_std))
                Borrower(self, p, group)

        # The measurement crossing, now declared instead of hand-written. Reading
        # "repay_prob" straight through is the unbiased, monotone (identity) score;
        # swap the string for a callable to model a noisy or group-biased proxy,
        # and nothing else in the file changes. groups is parked out of X.
        observer = DeclarativeObserver(
            features=["repay_prob"],
            feature_names=["score"],
            groups="group",
            seed=0,
        )
        # The bank: no subclass. ColumnScore is the no-ML baseline (score = the
        # "score" column, verbatim), by_group is the per-group threshold rule. This
        # triple is the entire replacement for the mesa version's Bank class.
        self.bank = MLAgent(
            self, observer, ColumnScore("score"), by_group(thresholds), label="repaid"
        )

        # Only the CONSTRUCT-space population statistic is collected per step here,
        # because true repay_prob is not in the audit log (the log holds the OBSERVED
        # score). The decision/fairness metrics no longer need per-step reporters:
        # they come post-hoc from the bank's audit log. See fairness_over_time().
        self.datacollector = DataCollector(
            model_reporters={
                "mean_repay_prob_g0": lambda m: m.mean_repay(0),
                "mean_repay_prob_g1": lambda m: m.mean_repay(1),
                "repay_gap": lambda m: m.mean_repay(0) - m.mean_repay(1),
            }
        )

    def borrowers(self):
        return self.agents_by_type[Borrower]

    def eligible(self):
        # WHO applies this step: borrowers without an outstanding loan (a borrower
        # cannot take a second loan before the first resolves), thinned by apply_rate
        # so applications stagger. The Observer records a row only for applicants,
        # which is the selective-observation half of the story.
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

    def apply(self, decisions: pd.Series) -> None:
        # What "approved" means (the old Effect). Book the decision and, for
        # approvals, originate a loan now; it matures repay_lag steps later.
        by_id = {b.unique_id: b for b in self.borrowers()}
        for aid, approve in decisions.items():
            b = by_id[aid]
            b.approved = bool(approve)
            if approve:
                b.issued_at = self.steps

    def reveal_due(self) -> None:
        # WHEN and for WHOM outcomes are known. Subjects settled their matured loans
        # in their own step(); the institution now writes those outcomes into memory.
        # step="last_action" lands each on the step the bank last lent to that
        # borrower, so the label falls on the features the loan was decided on
        # (point-in-time correct) without the domain grouping by grant step itself.
        # Only approved loans ever settle, so the labels stay delayed and selective.
        # This runs before the bank acts again this step, which is what makes
        # "last action" unambiguous (no newer loan yet to confuse it with).
        outcomes = {}
        for b in self.borrowers():
            if b.settlement is None:
                continue
            outcomes[b.unique_id] = b.settlement
            b.settlement = None
        self.bank.datastore.reveal("repaid", outcomes, step="last_action")

    def step(self):
        self.borrowers().shuffle_do("step")   # subjects: settle matured loans, move repay_prob
        self.reveal_due()                      # institution: record matured outcomes into memory
        applicants = self.eligible()
        _, decisions = self.bank.step(applicants)   # observe -> record -> decide (+ logs the decision)
        if decisions is not None:
            self.apply(decisions)              # book approvals, originate loans
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = LendingModel(rng=0)
    for _ in range(40):
        model.step()

    print("repay-prob trajectory (construct, per-step DataCollector):")
    print(model.datacollector.get_model_vars_dataframe().round(3).iloc[::10])

    # Fairness comes post-hoc from the bank's audit log via the shared helper: no
    # per-step, per-group bookkeeping in the domain at all.
    from fairlearn.metrics import selection_rate

    from fairsim.metrics import group_metrics_over_time

    mf = group_metrics_over_time(model.bank.datastore, selection_rate)
    print("\nselection rate by (step, group):")
    print(mf.by_group.round(3).unstack().iloc[::10])
    print("\ndemographic-parity gap per step:")
    print(mf.difference().round(3).iloc[::10])

    # Show the labels are genuinely delayed and selective in the audit log.
    frame = model.bank.datastore.frame()
    approved = frame[frame["decision"] == 1]
    denied = frame[frame["decision"] == 0]
    print(
        f"\naudit log: {len(approved)} approved rows, "
        f"{approved['repaid'].notna().sum()} with a revealed outcome "
        f"(rest still maturing = delayed);"
    )
    print(
        f"           {len(denied)} denied rows, "
        f"{denied['repaid'].notna().sum()} revealed (0 = selective)."
    )
