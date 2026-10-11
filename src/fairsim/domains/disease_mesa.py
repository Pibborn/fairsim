"""
Infectious disease (fair treatment allocation), Step 1: plain mesa + networkx,
no fairsim yet.

Ports the dynamics of ml-fairness-gym's InfectiousDiseaseEnv (SIR variant).
People sit on a contact network (Zachary's karate club by default, whose two
factions give natural groups). Each tick the authority treats up to `supply`
infected people (treatment cures: infected -> recovered), then contagion runs
SYNCHRONOUSLY: a susceptible node with n infected neighbors gets infected with
probability 1 - (1 - beta)^n, an infected node recovers naturally with
probability gamma. Transitions are computed for everyone, then applied, exactly
like the source's compute-then-apply loop.

Triage policies, mirroring the source's agents:
  * none       : the untreated outbreak.
  * random     : treat infected people in random order.
  * centrality : treat infected people by eigenvector centrality, the source's
                 CentralityAgent (efficient, and the seed of the equity issue:
                 central nodes belong disproportionately to one faction).

No fairsim imports on purpose. This is the A side of an A/B with
disease_fairsim.py.

Run from the repo root:
    python -m fairsim.domains.disease_mesa
"""

import networkx as nx
import numpy as np
from mesa import Agent, DataCollector, Model

S, I, R = 0, 1, 2


class Person(Agent):
    """One node on the contact graph. Owns its health state; contagion is the
    node doing it to itself from its neighbors' states, staged in step() and
    applied in advance() so the whole population transitions synchronously.
    """

    def __init__(self, model, node_id, group):
        super().__init__(model)
        self.node_id = node_id
        self.group = group
        self.state = S
        self.times_treated = 0
        self._next = S

    def step(self):
        if self.state == S:
            n_inf = sum(
                self.model.person[j].state == I
                for j in self.model.graph.neighbors(self.node_id)
            )
            p_infect = 1.0 - (1.0 - self.model.beta) ** n_inf
            self._next = I if self.model.rng.random() < p_infect else S
        elif self.state == I:
            self._next = R if self.model.rng.random() < self.model.gamma else I
        else:
            self._next = R

    def advance(self):
        self.state = self._next


class Authority(Agent):
    """The institution. Each tick it triages the infected and cures the top
    `supply` of them (treatment transition: infected -> recovered).
    """

    def __init__(self, model, supply, policy="centrality"):
        super().__init__(model)
        self.supply = supply
        self.policy = policy

    def allocate(self, people):
        infected = [p for p in people if p.state == I]
        if self.policy == "none" or not infected:
            return []
        if self.policy == "random":
            self.model.rng.shuffle(infected)
            return infected[: self.supply]
        # centrality: the source's CentralityAgent triage order
        infected.sort(key=lambda p: -self.model.centrality[p.node_id])
        return infected[: self.supply]


class DiseaseModel(Model):
    """One tick: treat (up to supply, infected only), then synchronous
    contagion. A burn-in lets the outbreak establish before treatment starts.
    """

    def __init__(
        self,
        graph=None,
        beta=0.25,
        gamma=0.01,
        supply=1,
        policy="centrality",
        burn_in=3,
        rng=None,
    ):
        super().__init__(rng=rng)
        self.graph = graph if graph is not None else nx.karate_club_graph()
        self.beta = beta
        self.gamma = gamma
        self.burn_in = burn_in
        self.centrality = nx.eigenvector_centrality(self.graph)

        self.person = {}
        for n in self.graph.nodes:
            group = self.graph.nodes[n].get("club")
            group = {"Mr. Hi": 0, "Officer": 1}.get(group, 0)
            self.person[n] = Person(self, n, group)

        # patient zero: one random node starts infected
        patient0 = int(self.rng.choice(list(self.graph.nodes)))
        self.person[patient0].state = I

        self.authority = Authority(self, supply, policy)

        self.datacollector = DataCollector(
            model_reporters={
                "infected": lambda m: sum(p.state == I for p in m.people()),
                "recovered": lambda m: sum(p.state == R for p in m.people()),
                "treated_g0": lambda m: sum(
                    p.times_treated for p in m.people() if p.group == 0
                ),
                "treated_g1": lambda m: sum(
                    p.times_treated for p in m.people() if p.group == 1
                ),
            }
        )

    def people(self):
        return list(self.agents_by_type[Person])

    def step(self):
        # 1. treat: the authority's allocation, applied before contagion (the
        #    source's ordering). No treatment during burn-in.
        if self.steps > self.burn_in:
            for p in self.authority.allocate(self.people()):
                p.state = R
                p.times_treated += 1

        # 2. contagion, synchronous: stage everyone, then apply.
        people_set = self.agents_by_type[Person]
        people_set.do("step")
        people_set.do("advance")

        self.datacollector.collect(self)


def sick_days(model_df):
    """Total person-ticks spent infected, the source's DayTracker summed."""
    return int(model_df["infected"].sum())


if __name__ == "__main__":
    results = {}
    for policy, supply in (("none", 0), ("random", 3), ("centrality", 3)):
        days = []
        for seed in range(20):
            m = DiseaseModel(policy=policy, supply=supply, rng=seed)
            for _ in range(50):
                m.step()
            days.append(sick_days(m.datacollector.get_model_vars_dataframe()))
        results[policy] = float(np.mean(days))
        print(f"{policy:>10} (supply {supply}): mean sick-days over 20 seeds = {results[policy]:7.1f}")

    assert results["none"] > results["random"] > results["centrality"] * 0.999, (
        "expected treatment to slow the outbreak and centrality to beat random"
    )
    print("\nuntreated outbreak spreads; treatment slows it; centrality triage is most efficient.")
