"""
Infectious disease (fair treatment allocation), Steps 2-6: the fairsim rebuild
of disease_mesa.py.

The three spaces, named:
  * Construct:   the true health state and the transmission process. mesa and
                 the graph own them.
  * Observation: health per node plus GRAPH FEATURES (degree, eigenvector
                 centrality, infected-neighbor count). These are the new thing:
                 they are not attributes of one agent, they are computed from
                 the agent's place in the network. They fit the existing
                 FeatureSpec because a mesa agent carries a reference to its
                 model, so a feature callable reaches the graph through
                 `a.model`. Note that in this environment observation is
                 COMPLETE (the source exposes true health states and the whole
                 graph to its agents), so construct and observation coincide;
                 the fairness question here is allocation, not measurement.
  * Decision:    treat / don't treat per node, at most `supply` treatments per
                 tick. Unit-capacity allocation, so it is top-k shaped, not
                 attention's real-valued budget split.

Policies:
  * CentralityTriage : score = infected x eigenvector centrality (the source's
                       CentralityAgent), with the core `treat-top-k`-style rule.
  * The parity RULE  : same scores, but the supply is split evenly across
                       groups; equity enters through the DecisionRule, which is
                       what the rule's `groups` argument exists for.
  * RiskTriage       : LogisticRegression predicting next-tick infection from
                       the graph features, learned through the ordinary
                       reveal -> training_set path (label: infected_next).

Run from the repo root:
    python -m fairsim.domains.disease_fairsim
"""

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Agent, DataCollector, Model
from sklearn.linear_model import LogisticRegression

from fairsim.core import DeclarativeObserver, MLAgent, SklearnPolicy

S, I, R = 0, 1, 2


class Person(Agent):
    """One node on the contact graph. Contagion is subject autonomy: staged in
    step(), applied in advance(), so the population transitions synchronously.
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


# --- policies and rules -------------------------------------------------------


class CentralityTriage:
    """Treat the infected, most central first: score = infected x centrality.
    Both are columns of the observed X, so this is a two-column ColumnScore.
    """

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return (X["infected"] * X["centrality"]).to_numpy(dtype=float)

    def update(self, X, y, groups=None) -> None:
        pass

    def is_ready(self) -> bool:
        return True


class RiskTriage(SklearnPolicy):
    """ML triage: predict next-tick infection risk from the graph features and
    treat the highest-risk nodes. Learns through the ordinary reveal path.
    """

    def __init__(self):
        super().__init__(LogisticRegression(max_iter=1000))

    def update(self, X, y, groups=None) -> None:
        if len(np.unique(y)) < 2:
            return  # LogisticRegression cannot fit a single class
        super().update(X, y, groups)


def treat_top_k(k: int):
    """Domain rule: treat the k highest scores, but never a zero score (a zero
    score means 'not a candidate', and treating it would waste a dose on a node
    the policy declined to nominate). core.top_k lacks that guard: it always
    picks exactly k rows.
    """
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        pos = scores[scores > 0]
        chosen = pos.nlargest(k).index
        return pd.Series(scores.index.isin(chosen).astype(int), index=scores.index)
    return rule


def parity_top_k(k: int):
    """Equity as a DecisionRule: split the supply evenly across groups, then
    treat top scores within each group. Leftover capacity (a group with too few
    candidates) spills over to the others. This is what the rule's `groups`
    argument exists for.
    """
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        if groups is None:
            raise ValueError("parity_top_k needs groups")
        out = pd.Series(0, index=scores.index)
        gs = sorted(groups.unique())
        quota, spare = divmod(k, len(gs))
        remaining = 0
        for i, g in enumerate(gs):
            budget = quota + (1 if i < spare else 0) + remaining
            pos = scores[(groups == g) & (scores > 0)]
            chosen = pos.nlargest(budget).index
            out.loc[chosen] = 1
            remaining = budget - len(chosen)
        return out
    return rule


# --- the world ------------------------------------------------------------


class DiseaseModel(Model):
    """One tick: the authority observes (health + graph features), triages and
    treats up to supply, contagion runs synchronously, and this tick's health
    outcome is revealed onto the rows just observed.
    """

    def __init__(
        self,
        graph=None,
        beta=0.25,
        gamma=0.01,
        supply=3,
        policy=None,
        rule=None,
        burn_in=3,
        treatment="cure",
        rng=None,
    ):
        super().__init__(rng=rng)
        self.graph = graph if graph is not None else nx.karate_club_graph()
        self.beta = beta
        self.gamma = gamma
        self.burn_in = burn_in
        # What a treatment DOES, as a world parameter (not an agent choice):
        #   "cure"      -> a treated INFECTED node moves to recovered (default).
        #   "vaccinate" -> a treated SUSCEPTIBLE node moves to recovered; an
        #                  infected node is unaffected. This matches the
        #                  ml-fairness-gym transition matrix [[0,0,1],[0,1,0],
        #                  [0,0,1]] hardcoded in its scenario_builder.
        if treatment not in ("cure", "vaccinate"):
            raise ValueError(f"treatment must be 'cure' or 'vaccinate', got {treatment!r}")
        self.treatment = treatment
        self.centrality = nx.eigenvector_centrality(self.graph)

        self.person = {}
        for n in self.graph.nodes:
            group = self.graph.nodes[n].get("club")
            group = {"Mr. Hi": 0, "Officer": 1}.get(group, self.graph.nodes[n].get("block", 0))
            self.person[n] = Person(self, n, group)

        self.person_by_uid = {p.unique_id: p for p in self.person.values()}

        patient0 = int(self.rng.choice(list(self.graph.nodes)))
        self.person[patient0].state = I

        # The graph features: callables reach the network through the agent's
        # model backref. The Observer never needed to know a graph exists.
        observer = DeclarativeObserver(
            features=[
                lambda a, rng: int(a.state == I),
                lambda a, rng: a.model.graph.degree(a.node_id),
                lambda a, rng: a.model.centrality[a.node_id],
                lambda a, rng: sum(
                    a.model.person[j].state == I
                    for j in a.model.graph.neighbors(a.node_id)
                ),
            ],
            feature_names=["infected", "degree", "centrality", "infected_neighbors"],
            groups="group",
            seed=0,
        )
        self.authority = MLAgent(
            self,
            observer,
            policy if policy is not None else CentralityTriage(),
            rule if rule is not None else treat_top_k(supply),
            label="infected_next",
            min_train=100,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "infected": lambda m: sum(p.state == I for p in m.people()),
                "recovered": lambda m: sum(p.state == R for p in m.people()),
            }
        )

    def people(self):
        return list(self.agents_by_type[Person])

    # --- domain truth, plain Python ------------------------------------------

    def apply(self, decisions: pd.Series) -> None:
        # What treatment MEANS is a world parameter (self.treatment). A cure
        # moves a treated INFECTED node to recovered; a vaccination moves a
        # treated SUSCEPTIBLE node to recovered and does nothing to an infected
        # one. Treating any other state is a wasted dose.
        for aid, treat in decisions.items():
            if not treat:
                continue
            p = self.person_by_uid[aid]
            if self.treatment == "vaccinate":
                if p.state == S:
                    p.state = R
            else:  # "cure"
                if p.state == I:
                    p.state = R
            p.times_treated += 1

    def reveal_due(self) -> None:
        # The outcome of the situation observed this tick is the health state
        # AFTER this tick's contagion: delayed by half a tick, complete (this
        # environment is fully observed, so there are no selective labels and
        # no oracle store is needed; the authority's audit log IS the truth).
        outcomes = {p.unique_id: int(p.state == I) for p in self.people()}
        self.authority.datastore.reveal("infected_next", outcomes, step=self.steps)

    def step(self):
        people_set = self.agents_by_type[Person]

        if self.steps > self.burn_in:
            _, decisions = self.authority.step(people_set)
            if decisions is not None:   # cold-start: RiskTriage before first fit
                self.apply(decisions)

        people_set.do("step")
        people_set.do("advance")

        if self.steps > self.burn_in:
            self.reveal_due()
            self.authority.learn()

        self.datacollector.collect(self)


def run(steps=50, seed=0, **kw):
    model = DiseaseModel(rng=seed, **kw)
    for _ in range(steps):
        model.step()
    return model


def summarize(model):
    """Efficiency and equity off the ONE store: total sick-days, plus per-group
    sick-days and treatment shares from the audit log's own columns.
    """
    frame = model.authority.datastore.frame()
    sick = int(model.datacollector.get_model_vars_dataframe()["infected"].sum())
    by_group = frame.groupby("group").agg(
        sick_days=("infected", "sum"), treatments=("decision", "sum")
    )
    return sick, by_group


if __name__ == "__main__":
    # --- A/B: fairsim centrality triage against the mesa Step 1 version -------
    from fairsim.domains.disease_mesa import DiseaseModel as MesaModel

    mesa_m = MesaModel(policy="centrality", supply=3, rng=0)
    for _ in range(50):
        mesa_m.step()
    mesa_traj = mesa_m.datacollector.get_model_vars_dataframe()["infected"].tolist()

    fs = run(steps=50, seed=0, supply=3)
    fs_traj = fs.datacollector.get_model_vars_dataframe()["infected"].tolist()
    assert fs_traj == mesa_traj, "fairsim and mesa centrality runs diverged"
    print("A/B: fairsim centrality triage is bit-identical to the mesa version (50 ticks).")

    # --- success test (Step 6): the equity / efficiency tradeoff --------------
    # Karate's two factions both contain hubs, so the disparity is mild there;
    # the tradeoff is sharper on a planted two-block graph where group 0's
    # community is denser (more central AND more combustible) than group 1's.
    def blocks_graph(seed):
        g = nx.stochastic_block_model([17, 17], [[0.30, 0.02], [0.02, 0.10]], seed=seed)
        for n in g.nodes:
            g.nodes[n]["block"] = int(g.nodes[n]["block"])
        return g

    # Equity metric: ACCESS, P(treated | infected) per group, off the audit
    # log. Raw treatment share turned out to be confounded (outbreaks localize
    # in one block per run, so even the parity rule's realized doses follow the
    # outbreak; the spillover clause guarantees it). Access conditional on need
    # is the quantity the parity rule actually moves.
    rows = {}
    for name, rule_fn in (("centrality", treat_top_k), ("parity", parity_top_k)):
        sick_all, access = [], []
        for seed in range(20):
            m = run(steps=50, seed=seed, graph=blocks_graph(seed), rule=rule_fn(2), supply=2)
            sick, _ = summarize(m)
            sick_all.append(sick)
            f = m.authority.datastore.frame()
            inf = f[f["infected"] == 1]
            a = inf.groupby("group")["decision"].mean()
            access.append((a.get(0, np.nan), a.get(1, np.nan)))
        access = np.array(access, dtype=float)
        rows[name] = (
            float(np.mean(sick_all)),
            float(np.nanmean(access[:, 0])),
            float(np.nanmean(access[:, 1])),
        )
        sick_m, a0, a1 = rows[name]
        print(f"{name:>10}: mean sick-days {sick_m:6.1f}, "
              f"P(treated|infected) group 0 {a0:.2f}, group 1 {a1:.2f}  (20 seeds, supply 2)")

    eff_c, _, acc1_c = rows["centrality"]
    eff_p, _, acc1_p = rows["parity"]
    assert eff_p > eff_c, "expected the parity rule to cost efficiency"
    assert acc1_p > acc1_c, "expected the parity rule to raise the sparse group's access"
    print("the tradeoff: centrality is the efficient allocation; the parity rule buys the")
    print(f"sparse group access ({acc1_c:.2f} -> {acc1_p:.2f}) at {eff_p - eff_c:+.1f} sick-days.")

    # --- ML triage through the reveal path -------------------------------------
    ml = run(steps=50, seed=0, policy=RiskTriage(), rule=None, supply=3)
    sick_ml, by_group_ml = summarize(ml)
    frame = ml.authority.datastore.frame()
    treated = frame[frame["decision"] == 1]
    wasted = int((treated["infected"] == 0).sum())
    print(f"\nRiskTriage (logistic on graph features, label=infected_next): "
          f"{sick_ml} sick-days,")
    print(f"  {len(treated)} doses given, {wasted} to nodes that were not infected "
          f"(the model buys risk, not diagnosis).")

    # --- scale (Step 7): many nodes, many ticks --------------------------------
    import time

    big = nx.barabasi_albert_graph(300, 2, seed=1)
    for n in big.nodes:
        big.nodes[n]["block"] = n % 2
    t0 = time.perf_counter()
    m = run(steps=100, seed=0, graph=big, supply=5)
    dt = time.perf_counter() - t0
    n_rows = len(m.authority.datastore)
    print(f"\nscale: 300 nodes x 100 ticks = {n_rows} store rows in {dt:.1f}s "
          f"({1000 * dt / 100:.0f} ms/tick).")
