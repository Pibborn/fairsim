"""
College admission with a COLLECTIVE: the worked example of the coordination
pattern. Builds on college_fairsim.py by subclassing only; the original file
and its A/B are untouched.

The pattern (this is the whole of it, there is no Collective primitive):

  * The collective is a plain mesa Agent (Coalition below). It reads the
    institution's Disclosure like any subject-side actor and holds a strategy.
  * Members consult it in their own step(). SubjectAgent behavior stays the
    user's, per the library's rule; the coalition is just a shared brain the
    members choose to ask.
  * What the coalition can KNOW comes through the Disclosure, the named
    subject-side crossing from core.py: parameter disclosure here (the jury
    publishes its threshold), query access in the probe demo at the bottom
    (the jury is a black box the coalition is allowed to poke k times a step).

The research question the example answers: can collective RESTRAINT stop the
gaming ratchet? college_fairsim shows individually-rational gaming ratchets
the learned threshold to the ceiling (the gamed mass sits at the cutoff and
is mostly ineligible, so the accuracy-max threshold steps over it, and the
gamers follow). A coalition that tells its INELIGIBLE members not to game is
deliberately declining to poison the jury's training signal, trading those
members' short-run admissions for the group's long-run access. Strategies:

  * None            : everyone games individually (the baseline ratchet).
  * "no_gaming"     : members never game. Maximal restraint.
  * "eligible_only" : members game only if truly eligible (a member knows its
                      own quality). The gamed mass at the cutoff is then
                      genuinely eligible, so stepping over it no longer buys
                      the jury accuracy.
  * "boycott_above" : members game normally until the published threshold
                      exceeds `boycott_cut`, then stop entirely. This one
                      actually USES the disclosure: the coalition reacts to
                      what the institution announces.

Run from the repo root:
    python -m fairsim.domains.college_collective
"""

import numpy as np
import pandas as pd
from mesa import Agent, DataCollector

from fairsim.core import Disclosure
from fairsim.domains.college_fairsim import (
    AdmissionModel,
    Applicant,
    ContinuousJury,
    LearnedThreshold,
)


class Coalition(Agent):
    """The collective: one mesa Agent that reads the Disclosure and answers
    members' "should I game?" question. It has no special powers: everything
    it knows arrived through the same Disclosure any subject could read, and
    everything it does happens inside its members' own step()."""

    def __init__(self, model, *, groups=(1,), strategy=None, boycott_cut=0.9):
        super().__init__(model)
        self.groups = set(groups)
        self.strategy = strategy
        self.boycott_cut = boycott_cut
        self.threshold_estimate = None  # what the coalition believes the cutoff is

    def step(self):
        """Read the institution's disclosure. Parameter disclosure is a plain
        read; under query access the coalition PROBES instead (see
        estimate_boundary_by_probing below, used in the demo)."""
        d = self.model.disclosure
        if d.param is not None:
            self.threshold_estimate = d.published

    def is_member(self, applicant) -> bool:
        return applicant.group in self.groups

    def advises_gaming(self, applicant) -> bool:
        """Members ask this from their own step(). Non-members always get
        True (they behave like the original individually-gaming Applicant)."""
        if self.strategy is None or not self.is_member(applicant):
            return True
        if self.strategy == "no_gaming":
            return False
        if self.strategy == "eligible_only":
            return bool(applicant.eligible)
        if self.strategy == "boycott_above":
            t = self.threshold_estimate
            return t is None or t <= self.boycott_cut
        raise ValueError(f"unknown strategy {self.strategy!r}")


class CoalitionApplicant(Applicant):
    """Identical to Applicant except it asks the coalition before gaming.
    Consulting the shared brain is one line; everything else is inherited."""

    def step(self):
        if self.model.coalition.advises_gaming(self):
            super().step()


class CoalitionAdmissionModel(AdmissionModel):
    """college_fairsim's AdmissionModel with (a) the smeared publication
    convention replaced by the named Disclosure primitive, and (b) a Coalition
    in the loop. The Stackelberg ordering gains one sub-step: publish ->
    coalition reads the disclosure -> subjects react (consulting it) -> ...
    """

    def __init__(self, *args, coalition_groups=(1,), coalition_strategy=None,
                 boycott_cut=0.9, queries_per_step=0, **kw):
        super().__init__(*args, **kw)
        # The named crossing: what subjects may see of the jury's policy.
        self.disclosure = Disclosure(
            self.jury.policy, param="threshold", queries_per_step=queries_per_step
        )
        self.coalition = Coalition(
            self, groups=coalition_groups, strategy=coalition_strategy,
            boycott_cut=boycott_cut,
        )
        # Re-key the agent-level collector to the subclassed applicant type
        # (mesa's agents_by_type is keyed by exact class).
        self.datacollector = DataCollector(
            model_reporters={
                "threshold": lambda m: m.published_threshold,
                "manipulation_rate": lambda m: float(
                    np.mean([a.manipulated for a in m.cohort()])
                ),
                "admit_rate": lambda m: float(
                    np.mean([a.selected for a in m.cohort()])
                ),
            },
            agenttype_reporters={
                CoalitionApplicant: {
                    "group": "group",
                    "quality": "quality",
                    "score": "score",
                    "burden": "burden",
                    "eligible": "eligible",
                    "selected": "selected",
                }
            },
        )

    def spawn_cohort(self):
        for a in self.cohort():
            a.remove()
        for _ in range(self.num_applicants):
            group = int(self.rng.random() > self.mix_weight[0])
            quality = float(
                np.clip(self.rng.normal(self.mu[group], self.sigma[group]), 0, 1)
            )
            CoalitionApplicant(self, quality, group)

    def cohort(self):
        return list(self.agents_by_type.get(CoalitionApplicant, []))

    def step(self):
        self.spawn_cohort()
        cohort_set = self.agents_by_type[CoalitionApplicant]

        # 1. publish. The rule keeps reading model.published_threshold (the
        #    institution knows its own cutoff); the DISCLOSURE is what the
        #    subject side gets to read, and here they coincide (param
        #    disclosure). An opaque institution would drop param= and the
        #    coalition would have to probe or infer.
        self.published_threshold = self.jury.policy.threshold
        self.disclosure.publish()

        # 2. the coalition reads the disclosure, then subjects react.
        self.coalition.step()
        cohort_set.shuffle_do("step")

        # 3-4. observe/decide, apply, reveal (admitted only), learn: unchanged.
        _, decisions = self.jury.step(cohort_set)
        if decisions is None:
            decisions = pd.Series(
                {a.unique_id: int(a.score >= self.published_threshold) for a in self.cohort()}
            )
            self.jury.datastore.record_decision(decisions, step=self.steps)
        self.apply(decisions)
        self.reveal_due()
        self.jury.learn()

        self.datacollector.collect(self)


def run(strategy=None, steps=50, seed=0, coalition_groups=(1,), **kw):
    model = CoalitionAdmissionModel(
        policy=LearnedThreshold(0.5), coalition_strategy=strategy,
        coalition_groups=coalition_groups, rng=seed, **kw,
    )
    for _ in range(steps):
        model.step()
    return model


def summarize(model, label):
    agents = model.datacollector.get_agenttype_vars_dataframe(CoalitionApplicant)
    late = agents[agents.index.get_level_values("Step") > 25]
    t_final = model.published_threshold
    out = {"final_threshold": t_final}
    for g in (0, 1):
        gg = late[late["group"] == g]
        elig = gg[gg["eligible"] == 1]
        out[f"admit_g{g}"] = float(gg["selected"].mean())
        out[f"eligible_admit_g{g}"] = float(elig["selected"].mean()) if len(elig) else float("nan")
        out[f"burden_g{g}"] = float(gg["burden"].mean())
    print(f"{label:28s} threshold {out['final_threshold']:.3f} | "
          f"admit g0 {out['admit_g0']:.2f} g1 {out['admit_g1']:.2f} | "
          f"eligible admit g0 {out['eligible_admit_g0']:.2f} g1 {out['eligible_admit_g1']:.2f} | "
          f"burden g1 {out['burden_g1']:.3f}")
    return out


def estimate_boundary_by_probing(disclosure, grid_size=101):
    """The QUERY channel, demonstrated: the jury is a black box (no published
    parameter), but the coalition may probe it. One query = one batch of
    hypothetical applicants swept over the score range; the boundary estimate
    is where the returned score crosses 0.5. This is 'shopping around' as a
    measurement instrument."""
    grid = pd.DataFrame(
        {"score": np.linspace(0.0, 1.0, grid_size)},
        index=pd.Index(range(grid_size), name="agent_id"),
    )
    scores = np.asarray(disclosure.probe(grid), dtype=float)
    above = np.nonzero(scores >= 0.5)[0]
    return float(grid["score"].iloc[above[0]]) if len(above) else None


if __name__ == "__main__":
    # --- sanity: a strategy-less coalition changes nothing --------------------
    from fairsim.domains.college_fairsim import run as run_original

    base = run_original(policy=LearnedThreshold(0.5), steps=50, seed=0)
    base_traj = base.datacollector.get_model_vars_dataframe()["threshold"].tolist()
    none_model = run(strategy=None, steps=50, seed=0)
    none_traj = none_model.datacollector.get_model_vars_dataframe()["threshold"].tolist()
    assert none_traj == base_traj, "a no-op coalition should be invisible"
    print("sanity: coalition with no strategy is bit-identical to college_fairsim (50 rounds).\n")

    # --- can collective restraint stop the ratchet? ---------------------------
    print("learned jury, 50 rounds, coalition = group 1 unless noted:")
    results = {}
    results["baseline (no coalition)"] = summarize(none_model, "baseline (no coalition)")
    for strat in ("no_gaming", "eligible_only", "boycott_above"):
        results[strat] = summarize(run(strategy=strat, steps=50, seed=0), f"g1 coalition: {strat}")
    both = run(strategy="eligible_only", steps=50, seed=0, coalition_groups=(0, 1))
    results["eligible_only (all)"] = summarize(both, "all-group: eligible_only")

    assert (
        results["eligible_only (all)"]["final_threshold"]
        < results["baseline (no coalition)"]["final_threshold"]
    ), "expected all-group eligible-only restraint to hold the threshold down"

    # --- the query channel: inferring a hidden boundary -----------------------
    # An OPAQUE jury (no published parameter), but the institution allows one
    # probe per step. The coalition locates the decision boundary to grid
    # precision without ever being told it. The jury is fit on a cohort with a
    # known eligibility cutoff at 0.75; the coalition never sees that number.
    rng = np.random.default_rng(0)
    n = 500
    scores = rng.uniform(0, 1, n)
    Xfit = pd.DataFrame({"score": scores}, index=pd.Index(range(n), name="agent_id"))
    yfit = ((scores + rng.normal(0, 0.02, n)) >= 0.75).astype(int)
    opaque_jury = ContinuousJury(0.5)
    opaque_jury.update(Xfit, yfit)

    hidden = Disclosure(opaque_jury, queries_per_step=1)  # no param= : opaque
    est = estimate_boundary_by_probing(hidden)
    true_t = opaque_jury.threshold
    print(f"\nquery channel: coalition probes an opaque ContinuousJury once "
          f"(101-applicant sweep):\n  estimated boundary {est:.2f} vs implied threshold {true_t:.2f} "
          f"(fit cutoff 0.75, never disclosed)")
    assert est is not None and abs(est - true_t) < 0.05
    try:
        estimate_boundary_by_probing(hidden)
        raise SystemExit("second probe should have exhausted the query budget")
    except PermissionError:
        print("  second probe correctly denied: query budget is 1 per step")
