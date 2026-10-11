"""
Attention allocation, Steps 2-6: the fairsim rebuild of attention_mesa.py.

The three spaces, named:
  * Construct:   incident_rate (the true latent rate) and incidents_occurred
                 (true events this tick). mesa owns both; the allocator never
                 reads them.
  * Observation: the allocator's map, incidents discovered where attention went.
                 CENSORED: an unattended location reads 0 on the map no matter
                 what happened there. Note the censoring cannot live in the
                 Observer callable: it is caused by THIS tick's decision, which
                 happens after the observation. So the Observer reads the map
                 (feature "seen_last") and the censoring is applied in apply(),
                 where discovery actually happens. See the Step 7 reflections in
                 env-plans/attention.md.
  * Decision:    an allocation, a real-valued pd.Series over ALL locations
                 summing to the budget. Not a per-agent 0/1: this is the port
                 that tests whether the decision type survives allocation.

The decision rule (allocate_budget) lives HERE, not in core: DecisionRule is a
callable alias, so a domain-shaped rule is a plain function, no registration.

Two policies:
  * ColumnScore("seen_last"): the myopic proportional allocator from Step 1,
    for free. This is the one that produces the runaway loop.
  * MeanSeenScore: the first policy that learns from a revealed OUTCOME rather
    than fitting on labels; it scores each location by its historical mean
    discovered incidents, learned through the ordinary DataStore reveal path.

The oracle pattern reappears: a second DataStore records the TRUE occurred
incidents for every location every tick, so the discovery ratio (how much of a
group's true incidents the institution ever sees) is a join of the two frames.

Run from the repo root:
    python -m fairsim.domains.attention_fairsim
"""

import numpy as np
import pandas as pd
from mesa import Agent, DataCollector, Model

from fairsim.core import ColumnScore, DataStore, DeclarativeObserver, MLAgent


class Location(Agent):
    """A place incidents happen. Owns its true incident_rate and drifts it
    itself (subject autonomy): attention suppresses the rate, neglect lets it
    climb. Drift runs at the END of the tick, on this tick's attention.
    """

    def __init__(self, model, incident_rate, group):
        super().__init__(model)
        self.incident_rate = incident_rate   # construct: true latent rate
        self.group = group
        self.attention = 0                   # units allocated this tick
        self.incidents_occurred = 0          # construct: true events this tick
        self.incidents_seen = 0              # the allocator's map entry (censored)

    def step(self):
        if self.attention == 0:
            self.incident_rate += self.model.dynamic_rate
        else:
            self.incident_rate = max(
                0.0, self.incident_rate - self.model.dynamic_rate * self.attention
            )


def allocate_budget(budget: int, *, stochastic: bool = False, rng=None):
    """Domain-shaped DecisionRule: scores -> an integer allocation summing to
    `budget`, proportional to score. All-zero scores fall back to a uniform
    spread, which doubles as the bootstrap: a policy with no information
    allocates evenly, no separate cold-start branch.

    Two allocation modes:
      * deterministic (default): largest-remainder rounding of the proportional
        share. Every location with positive score gets its rounded share.
      * stochastic: draw the allocation from a multinomial with the proportional
        probabilities, matching ml-fairness-gym's NaiveProbabilityMatchingAgent.
        This concentrates units on the top locations and can leave low-score
        locations with exactly zero. Needs an rng (pass the model's for
        reproducibility): allocate_budget(b, stochastic=True, rng=model.rng).
    """
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        s = scores.clip(lower=0).astype(float)
        if s.sum() == 0:
            s = pd.Series(1.0, index=scores.index)
        p = (s / s.sum()).to_numpy()
        if stochastic:
            if rng is None:
                raise ValueError("stochastic allocation needs an rng")
            counts = rng.multinomial(budget, p)
            return pd.Series(counts, index=scores.index)
        share = pd.Series(p * budget, index=scores.index)
        units = np.floor(share).astype(int)
        remainder = budget - int(units.sum())
        order = (share - units).sort_values(ascending=False).index
        units.loc[order[:remainder]] += 1
        return units
    return rule


class MeanSeenScore:
    """Score each location by its historical mean DISCOVERED incidents, learned
    from the revealed "seen" outcome. This is the first Policy whose update()
    consumes a reward-like signal instead of classification labels; it flows
    through training_set() like any label. Locations never measured fall back to
    `prior` (an optimism knob: a high prior sends units back to dark spots).
    """

    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self._means: dict = {}

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return np.array([self._means.get(aid, self.prior) for aid in X.index])

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        s = pd.Series(y, index=X.index.get_level_values("agent_id"), dtype=float)
        self._means = s.groupby(level=0).mean().to_dict()

    def is_ready(self) -> bool:
        return True   # the prior makes it always ready; no bootstrap needed


class AttentionModel(Model):
    """One tick: the allocator splits the budget on its censored map, discovery
    happens where the units went, the oracle logs the truth, rates drift.
    """

    def __init__(
        self,
        incident_rates=(8.0, 6.0, 4.0, 2.0),
        groups=(0, 0, 1, 1),
        budget=6,
        dynamic_rate=0.02,
        policy=None,
        stochastic_allocation=False,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.dynamic_rate = dynamic_rate
        for rate, group in zip(incident_rates, groups):
            Location(self, rate, group)

        # The Observer reads the MAP, not the world: "seen_last" is what the
        # allocator discovered last tick, already censored. Reading construct
        # (incident_rate) here would silently delete the entire research question.
        observer = DeclarativeObserver(
            features=["incidents_seen"],
            feature_names=["seen_last"],
            groups="group",
            seed=0,
        )
        # stochastic_allocation switches the largest-remainder allocator for a
        # multinomial one (ml-fairness-gym's NaiveProbabilityMatchingAgent),
        # seeded off the model rng so a seeded run stays reproducible.
        alloc = allocate_budget(budget, stochastic=stochastic_allocation, rng=self.rng)
        self.allocator = MLAgent(
            self,
            observer,
            policy if policy is not None else ColumnScore("seen_last"),
            alloc,
            label="seen",
        )

        # The oracle pattern: the researcher's ledger of the truth the allocator
        # never sees. Same primitives, different vantage point.
        self.oracle = DataStore()
        self.oracle_obs = DeclarativeObserver(
            features=["incident_rate"], feature_names=["true_rate"], groups="group", seed=0
        )

        self.datacollector = DataCollector(
            model_reporters={
                **{f"rate_{i}": (lambda m, i=i: m.locations()[i].incident_rate)
                   for i in range(len(incident_rates))},
                **{f"attn_{i}": (lambda m, i=i: m.locations()[i].attention)
                   for i in range(len(incident_rates))},
            }
        )

    def locations(self):
        return list(self.agents_by_type[Location])

    # --- domain truth, plain Python ------------------------------------------

    def apply(self, decisions: pd.Series) -> None:
        # What the allocation MEANS: attention goes to a location, incidents
        # occur from the TRUE rate, and each unit discovers at most one. This is
        # where the censoring physically happens, so it is Effect, not Observer.
        by_id = {loc.unique_id: loc for loc in self.locations()}
        discovered = {}
        for aid, units in decisions.items():
            loc = by_id[aid]
            loc.attention = int(units)
            loc.incidents_occurred = int(self.rng.poisson(loc.incident_rate))
            loc.incidents_seen = min(loc.incidents_occurred, loc.attention)
            if loc.attention > 0:
                discovered[aid] = loc.incidents_seen
        # Censored reveal: only attended locations get an outcome; unattended
        # rows stay NaN in the audit log ("not measured", distinct from the
        # map's misleading 0). Same tick, so not delayed: censoring is the WHO,
        # not the WHEN. step="last_action" resolves to this tick's allocation.
        self.allocator.datastore.reveal("seen", discovered, step="last_action")

    def audit(self) -> None:
        # Oracle: record the true rate and reveal the true occurred count for
        # EVERY location, attended or not. The institution cannot write this
        # table; the simulation can, because it authored the truth.
        locations = self.locations()
        truth = self.oracle_obs(locations, self, step=self.steps)
        self.oracle.record(truth, step=self.steps)
        self.oracle.reveal(
            "occurred",
            {loc.unique_id: loc.incidents_occurred for loc in locations},
            step=self.steps,
        )

    def step(self):
        locations = self.agents_by_type[Location]
        _, decisions = self.allocator.step(locations)   # observe map -> record -> allocate
        self.apply(decisions)                           # attention, occurrence, censored discovery
        self.allocator.learn()                          # refit on revealed outcomes (no-op for ColumnScore)
        self.audit()                                    # oracle logs the truth
        locations.shuffle_do("step")                    # subjects drift their own true rates
        self.datacollector.collect(self)


def run(policy=None, steps=100, seed=0, **kw):
    model = AttentionModel(policy=policy, rng=seed, **kw)
    for _ in range(steps):
        model.step()
    return model


if __name__ == "__main__":
    # --- the runaway loop under the myopic proportional allocator (Step 6) ----
    model = run()
    df = model.datacollector.get_model_vars_dataframe().round(2)
    print("true rates and attention under the myopic allocator (rates 8/6/4/2, budget 6):")
    print(df.iloc[::20])

    total_attn = {i: df[f"attn_{i}"].sum() for i in range(4)}
    coldest = min(total_attn, key=total_attn.get)
    r = df[f"rate_{coldest}"]
    print(
        f"\nleast-attended location: {coldest} "
        f"({total_attn[coldest]} of {sum(total_attn.values())} unit-ticks); "
        f"true rate {r.iloc[0]:.2f} -> {r.iloc[-1]:.2f} while unseen"
    )
    assert r.iloc[-1] > r.iloc[0] + 0.2, "expected the neglected location's true rate to drift up"

    # --- metrics off the two stores (Step 5) ----------------------------------
    from fairsim.metrics import group_metrics_over_time

    mean_attention = lambda y_true, y_pred: float(np.mean(y_pred))
    attn = group_metrics_over_time(model.allocator.datastore, mean_attention)
    print("\nmean attention units by (step, group):")
    print(attn.by_group.round(2).unstack().iloc[::20])

    # Discovery ratio: what fraction of each group's TRUE incidents the
    # institution ever saw. A join of the institution's frame (seen) with the
    # oracle's (occurred) on their shared (agent_id, step) index.
    inst = model.allocator.datastore.frame()
    orc = model.oracle.frame()
    joined = inst.join(orc[["occurred"]])
    by_group = joined.groupby("group").agg(
        seen=("seen", lambda s: s.fillna(0).sum()), occurred=("occurred", "sum")
    )
    by_group["discovery_ratio"] = (by_group["seen"] / by_group["occurred"]).round(3)
    print("\ndiscovery ratio by group (seen / truly occurred, whole run):")
    print(by_group)

    unmeasured = inst["seen"].isna().sum()
    print(f"\naudit log: {len(inst)} location-ticks, {unmeasured} unmeasured (NaN = censored,")
    print("           distinct from the map's 0), the rest with a discovered count.")

    # --- the learning allocator through the reveal path -----------------------
    bandit = MeanSeenScore(prior=1.0)
    model2 = run(policy=bandit)
    final_rates = {loc.unique_id: round(loc.incident_rate, 2) for loc in model2.locations()}
    print("\nMeanSeenScore learned means vs final true rates (means are censored DOWN,")
    print("a location can never look busier than the attention it was given):")
    print(f"  learned means:    { {k: round(v, 2) for k, v in bandit._means.items()} }")
    print(f"  final true rates: {final_rates}")
