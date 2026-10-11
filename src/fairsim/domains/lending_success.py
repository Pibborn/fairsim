"""
Lending, Step 6: the success test.

Confirms the port reproduces Liu's qualitative result and the no-feedback baseline,
using the ANALYTICAL max-utility bank (the faithful reproduction, per the "Fidelity
to Liu et al." section in env-plans/lending.md), not the learned bank from Step 4.

The bank here is `ColumnScore` plus a global `threshold(break_even)`. Because the
Observer reads `repay_prob` straight through, the score IS P(repay), so a global cut
at the profit break-even is exactly Liu's `max_profit_loans = trunc(perf + 1 -
break_even)`. No learning, no selective labels: ColumnScore passes the score through
and `label=None`, matching Liu's full-knowledge analytical rule.

Two checks:
  1. Feedback ON: the two groups' mean repay-prob drift APART over the run. This is
     the Liu qualitative result.
  2. Feedback OFF (`score_dynamics=False`): approvals still happen, but no outcome
     moves anyone, so the distributions and the gap stay put. This reproduces
     `SimpleLoans` (approvals with no feedback).

Note on scope. This is the persistent-borrower port with an identity score, so it
shows the gap WIDENING because the advantaged group improves faster. It does not
claim Liu's stronger "active harm" result (the disadvantaged group's mean declining),
which needs lending below the mean-movement-neutral threshold and, for the numbers,
the distribution-level variant. See the fidelity section.

Run:
    python -m fairsim.domains.lending_success
"""

from mesa import Agent, DataCollector, Model

from fairsim.core import ColumnScore, DeclarativeObserver, MLAgent, threshold


class Borrower(Agent):
    def __init__(self, model, repay_prob, group):
        super().__init__(model)
        self.repay_prob = repay_prob   # construct: true chance of repaying
        self.group = group
        self.approved = False


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class LendingModel(Model):
    """Persistent borrowers, an analytical max-utility bank, immediate resolution.

    Resolution is same-step on purpose here: the delayed/selective label machinery
    was already validated in Step 3, and Liu's own dynamic resolves and shifts each
    step, so keeping it immediate makes the drift result easy to read.
    """

    def __init__(
        self,
        n_per_group=500,
        break_even=0.8,               # Liu's canonical max-profit break-even (u_def=-4)
        group_means=(0.75, 0.6),      # chosen so break_even=0.8 lends meaningfully to both
        group_std=0.15,
        shift=0.05,
        score_dynamics=True,          # False -> SimpleLoans: approvals but no feedback
        rng=None,
    ):
        super().__init__(rng=rng)
        self.shift = shift
        self.score_dynamics = score_dynamics

        for group, mean in enumerate(group_means):
            for _ in range(n_per_group):
                p = _clip(self.rng.normal(mean, group_std))
                Borrower(self, p, group)

        observer = DeclarativeObserver(
            features=["repay_prob"], feature_names=["score"], groups="group", seed=0
        )
        # Analytical max-utility bank: score = P(repay) (identity, unbiased), so a
        # global threshold at break_even reproduces max_profit_loans. label=None
        # because it does not learn; ColumnScore already knows the true curve.
        self.bank = MLAgent(self, observer, ColumnScore("score"), threshold(break_even), label=None)

        self.datacollector = DataCollector(
            model_reporters={
                "mean_repay_g0": lambda m: m.mean_repay(0),
                "mean_repay_g1": lambda m: m.mean_repay(1),
                "repay_gap": lambda m: m.mean_repay(0) - m.mean_repay(1),
                "approval_g0": lambda m: m.approval_rate(0),
                "approval_g1": lambda m: m.approval_rate(1),
            }
        )

    def borrowers(self):
        return self.agents_by_type[Borrower]

    def _members(self, group):
        return [b for b in self.borrowers() if b.group == group]

    def mean_repay(self, group):
        members = self._members(group)
        return sum(b.repay_prob for b in members) / len(members)

    def approval_rate(self, group):
        members = self._members(group)
        return sum(int(b.approved) for b in members) / len(members)

    def apply(self, decisions):
        by_id = {b.unique_id: b for b in self.borrowers()}
        for aid, approve in decisions.items():
            by_id[aid].approved = bool(approve)

    def resolve(self):
        # Institutional consequence plus feedback: approved loans resolve now, and a
        # resolved outcome moves the borrower's true repay_prob (the score's cause).
        # With score_dynamics off, we still lend but never move anyone: SimpleLoans.
        for b in self.borrowers():
            if not b.approved:
                continue
            repaid = self.rng.random() < b.repay_prob
            if self.score_dynamics:
                b.repay_prob = _clip(b.repay_prob + (self.shift if repaid else -self.shift))

    def step(self):
        applicants = self.borrowers()
        _, decisions = self.bank.step(applicants)   # observe -> record -> decide (always ready)
        self.apply(decisions)
        self.resolve()
        self.datacollector.collect(self)


def run(score_dynamics, steps=40, seed=0):
    model = LendingModel(score_dynamics=score_dynamics, rng=seed)
    for _ in range(steps):
        model.step()
    return model.datacollector.get_model_vars_dataframe()


def success_test():
    on = run(score_dynamics=True)
    off = run(score_dynamics=False)

    gap_on_start, gap_on_end = on["repay_gap"].iloc[0], on["repay_gap"].iloc[-1]
    gap_off_start, gap_off_end = off["repay_gap"].iloc[0], off["repay_gap"].iloc[-1]

    print("Feedback ON (max-utility bank):")
    print(on.round(3).iloc[::10])
    print(f"\n  repay-prob gap: {gap_on_start:.3f} -> {gap_on_end:.3f} "
          f"(widened by {gap_on_end - gap_on_start:+.3f})")

    print("\nFeedback OFF (SimpleLoans, score_dynamics=False):")
    print(off.round(3).iloc[::10])
    print(f"\n  repay-prob gap: {gap_off_start:.3f} -> {gap_off_end:.3f} "
          f"(changed by {gap_off_end - gap_off_start:+.3f})")

    # Check 1: with feedback the two groups drift apart (the gap widens).
    assert gap_on_end > gap_on_start + 0.01, (
        f"expected the gap to widen under feedback, got {gap_on_start:.3f} -> {gap_on_end:.3f}"
    )
    # Check 2: without feedback the gap is essentially static (SimpleLoans).
    assert abs(gap_off_end - gap_off_start) < 0.01, (
        f"expected a static gap without feedback, got {gap_off_start:.3f} -> {gap_off_end:.3f}"
    )
    print("\nPASS: feedback drives the groups apart; SimpleLoans (no feedback) holds them static.")


if __name__ == "__main__":
    success_test()
