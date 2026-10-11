"""
Attention allocation, Step 1: plain mesa, every party an agent, no fairsim yet.

Ports the dynamics of ml-fairness-gym's LocationAllocationEnv. Think pest
control or mobile health clinics, the source's own framing: its docstring
explicitly disclaims predictive policing, citing Richardson et al. on dirty
data. Each tick the allocator splits a fixed budget of attention units over
locations, proportional to LAST tick's observed incidents. Incidents occur from a
true latent rate (construct space); an incident is only SEEN where an attention
unit was present (one unit discovers at most one incident). Attention suppresses a
location's true rate; neglect lets it climb.

The runaway loop this is built to show: a location that gets no attention shows
zero seen incidents, so it never wins attention back, while its TRUE rate drifts
up unmeasured. The allocator's map of the city diverges from the city.

No fairsim imports on purpose. This is the A side of an A/B with
attention_fairsim.py.

Run:
    python src/fairsim/domains/attention_mesa.py
"""

import numpy as np
from mesa import Agent, DataCollector, Model


class Location(Agent):
    """A place incidents happen. Owns its TRUE incident_rate; the allocator never
    sees it, only the incidents its attention units happened to discover.
    """

    def __init__(self, model, incident_rate, group):
        super().__init__(model)
        self.incident_rate = incident_rate   # construct space: true latent rate
        self.group = group
        self.attention = 0                   # units allocated this tick
        self.incidents_occurred = 0          # construct truth this tick
        self.incidents_seen = 0              # what the allocator discovered (censored)


class Allocator(Agent):
    """The institution. Splits a fixed budget proportional to last tick's SEEN
    incidents (largest-remainder rounding). Seen, not occurred: it allocates on
    its own censored map, which is the whole problem.
    """

    def __init__(self, model, budget):
        super().__init__(model)
        self.budget = budget

    def allocate(self, locations):
        claims = np.array([l.incidents_seen for l in locations], dtype=float)
        if claims.sum() == 0:
            claims = np.ones(len(locations))   # first tick / total blackout: spread evenly
        share = claims / claims.sum() * self.budget
        units = np.floor(share).astype(int)
        remainder = self.budget - units.sum()
        order = np.argsort(-(share - units))   # biggest fractional parts win the leftovers
        units[order[:remainder]] += 1
        return {l.unique_id: int(u) for l, u in zip(locations, units)}


class AttentionModel(Model):
    """One tick: allocate on last tick's seen counts, incidents occur, attention
    discovers some of them, true rates drift (attended down, neglected up).
    """

    def __init__(
        self,
        incident_rates=(8.0, 6.0, 4.0, 2.0),
        groups=(0, 0, 1, 1),
        budget=6,
        dynamic_rate=0.02,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.dynamic_rate = dynamic_rate
        self.allocator = Allocator(self, budget)
        for rate, group in zip(incident_rates, groups):
            Location(self, rate, group)

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

    def step(self):
        locations = self.locations()

        # decide: split the budget on the allocator's censored map
        allocation = self.allocator.allocate(locations)
        for loc in locations:
            loc.attention = allocation[loc.unique_id]

        # world: incidents occur from the TRUE rate; attention discovers at most
        # one incident per unit. Where no attention went, nothing is seen.
        for loc in locations:
            loc.incidents_occurred = int(self.rng.poisson(loc.incident_rate))
            loc.incidents_seen = min(loc.incidents_occurred, loc.attention)

        # drift: attention suppresses the true rate, neglect lets it climb
        for loc in locations:
            if loc.attention == 0:
                loc.incident_rate += self.dynamic_rate
            else:
                loc.incident_rate = max(
                    0.0, loc.incident_rate - self.dynamic_rate * loc.attention
                )

        self.datacollector.collect(self)


if __name__ == "__main__":
    model = AttentionModel(rng=0)
    for _ in range(100):
        model.step()

    df = model.datacollector.get_model_vars_dataframe().round(2)
    print("true rates and attention (start rates 8/6/4/2, budget 6):")
    print(df.iloc[::20])

    # The loop, stated: the location the allocator attends LEAST should show a
    # rising true rate over its unattended stretch, invisible to the allocator.
    total_attn = {i: df[f"attn_{i}"].sum() for i in range(4)}
    coldest = min(total_attn, key=total_attn.get)
    r = df[f"rate_{coldest}"]
    print(
        f"\nleast-attended location: {coldest} "
        f"({total_attn[coldest]} of {sum(total_attn.values())} unit-ticks); "
        f"true rate {r.iloc[0]:.2f} -> {r.iloc[-1]:.2f} while unseen"
    )
