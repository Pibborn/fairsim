"""
Hiring, end to end: Observer -> MLAgent (+ rule) -> DataStore, driven by a World.

Run (with fairsim installed, e.g. `uv pip install -e .` from the repo root):
    python examples/hiring_example.py
"""

import mesa

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, selection_rate

from fairsim.core import (
    Batch,
    DeclarativeObserver,
    MLAgent,
    DataStore,
    ColumnScore,
    SklearnPolicy,
    top_k,
    threshold,
    by_group,
    passthrough,
)


# mesa-style simulation + some fairsim primitives
# the worker looks the same whether you do fairsim or pure mesa
class Worker(mesa.Agent):

    def __init__(self, model, skills, experience, group, attrition=0.1):
        super().__init__(model)
        self.skills = skills
        self.experience = experience
        self.group = group
        self.status = 'unemployed'
        self.attrition = attrition

    def leave(self):
        # attrition via mesa's own seeded RNG (self.model.random), so a seeded
        # model is reproducible -- gymnasium's check_env determinism probe needs this.
        if self.status == 'employed' and self.model.random.random() < self.attrition:
            self.status = 'unemployed'


class BigMan(mesa.Agent):

    def __init__(self, max_hires, model):
        super().__init__(model)
        self.max_hires = max_hires
        self.obs = DeclarativeObserver(features=["experience", "skills"],
                                       feature_names=["experience_hat",
                                                      "skill_hat"],
                                       groups="group",
                                       seed=123)
        self.rule = top_k(max_hires)

    def hire(self):
        batch = self.obs(self.model.workers, self.model, step=self.model.steps)
        scores = pd.Series(batch.X["skill_hat"])
        return self.rule(scores, batch.groups)


class HiringSim(mesa.Model):

    def __init__(self, n):
        super().__init__()
        self.num_agents = n
        self.workers = Worker.create_agents(model=self, n=n,
                                            skills=np.random.normal(size=n),
                                            experience=np.random.normal(size=n),
                                            group=np.random.binomial(1, 0.1, n))
        self.bigman = BigMan(10, self)

    def step(self):
        self.bigman.hire()
        self.workers.shuffle_do("leave")


# fairsim-style simulation.
#
# The whole loop below is ORDINARY mesa written by hand. fairsim ships no World
# and no step() for you: you bring your own mesa.Model and call the primitives
# where your domain wants them. What fairsim contributes is only the four nouns
# you inject / call: an Observer, a Policy, a DecisionRule, and the MLAgent that
# owns the DataStore. Everything else here (who is eligible, what "hired" DOES,
# when an outcome is known) is domain truth and stays in your hands.
#
# There is also no BigMLMan subclass: we build a plain MLAgent(observer, policy,
# rule, label). Swap ColumnScore for SklearnPolicy / FairlearnPolicy and not one
# other line moves.
class FairSimSim(mesa.Model):

    def __init__(self, n, observer, policy, rule):
        super().__init__()
        self.num_agents = n
        Worker.create_agents(model=self, n=n,
                             skills=np.random.normal(size=n),
                             experience=np.random.normal(size=n),
                             group=np.random.binomial(1, 0.1, n))
        self.bigman = MLAgent(self, observer, policy, rule,
                              label="good_hire", min_train=30)

        # METRICS are not a fairsim primitive. Fairness measurement is just
        # fairlearn's MetricFrame over each step's decisions + groups, collected
        # into mesa's own DataCollector. We stash the per-step MetricFrame result
        # on the model in step(); these reporters read it back. Nothing to import
        # from fairsim, no Metric class to subclass.
        self.datacollector = mesa.DataCollector(model_reporters={
            "ready":     lambda m: m.bigman.policy.is_ready(),
            "dp_diff":   lambda m: getattr(m, "sel_gap", float("nan")),
            "sel_g0":    lambda m: getattr(m, "sel_by_group", {}).get(0, float("nan")),
            "sel_g1":    lambda m: getattr(m, "sel_by_group", {}).get(1, float("nan")),
        })

    # --- domain truth, all plain Python -------------------------------------
    def eligible(self):
        # WHO can be hired this step. Pure mesa selection.
        return self.agents_by_type[Worker].select(lambda w: w.status == "unemployed")

    def apply(self, decisions):
        # WHAT "hired" does to a worker (the old Effect). Mutates construct space.
        by_id = {w.unique_id: w for w in self.agents_by_type[Worker]}
        for aid, hired in decisions.items():
            if hired:
                by_id[aid].status = "employed"
                by_id[aid].hired_at = self.steps

    def reveal_due(self):
        # WHEN the outcome is known, and what it is. Here: a hire's quality is
        # observed one step after hiring (delayed), and ONLY for people we hired
        # (selective labels -- we never see how rejected applicants would have
        # performed). Ground truth: skilled workers are good hires.
        matured = self.agents_by_type[Worker].select(
            lambda w: getattr(w, "hired_at", None) is not None
            and self.steps > w.hired_at
        )
        if len(matured) == 0:
            return
        outcome = {w.unique_id: int(w.skills > 0) for w in matured}
        self.bigman.datastore.reveal("good_hire", outcome, step=None)

    def bootstrap(self, pool):
        # COLD START is the domain's problem, not the library's. A learning policy
        # can't predict until it has labels, and labels only come from decisions --
        # a deadlock unless someone seeds it. We hire a few at RANDOM (not by
        # observed skill) so the seed labels don't bake in the very heuristic we're
        # here to study. This is where you'd put explore/epsilon-greedy logic too.
        # Return a full-pool 0/1 Series (not just the hires) so the selection-rate
        # metric has the right denominator.
        ids = [w.unique_id for w in pool]
        chosen = set(self.random.sample(ids, k=min(5, len(ids))))
        return pd.Series({i: int(i in chosen) for i in ids}, dtype=int)

    def record_fairness(self, decisions):
        # Fairness measurement = fairlearn MetricFrame over THIS step's decisions.
        # selection_rate by group needs no labels (labels are delayed + selective
        # here anyway), so it's the honest per-step signal: are the two groups
        # hired at equal rates? difference() is the demographic-parity gap.
        by_id = {w.unique_id: w for w in self.agents_by_type[Worker]}
        groups = pd.Series({i: by_id[i].group for i in decisions.index})
        mf = MetricFrame(metrics=selection_rate,
                         y_true=decisions, y_pred=decisions,   # y_true unused by selection_rate
                         sensitive_features=groups)
        self.sel_by_group = mf.by_group.to_dict()
        self.sel_gap = mf.difference()

    # --- the loop: your mesa.Model.step(), calling fairsim primitives ---------
    def step(self):
        pool = self.eligible()
        agents, decisions = self.bigman.step(pool)   # observe -> record -> maybe decide
        if decisions is None and len(pool) > 0:      # policy not ready -> domain seeds
            decisions = self.bootstrap(pool)
        if decisions is not None:
            self.apply(decisions)                    # decision space -> construct space
            self.record_fairness(decisions)          # measure THIS step (fairlearn)
        self.reveal_due()                            # answers that matured this step
        self.bigman.learn()                          # refit from the store
        self.datacollector.collect(self)             # mesa collects the metrics
        self.agents_by_type[Worker].shuffle_do("leave")


# ===========================================================================
# RL variant: same domain, but an external RL agent sets the hiring policy.
#
# Nothing about the loop changes shape. The firm still scores with a Policy and
# decides with a rule. The ONLY difference from the supervised runs above: the
# rule's per-group thresholds are read from an ExternalPolicy mailbox that a
# gymnasium wrapper (MesaEnv) fills each tick from the RL library's action.
#
# The learning problem: the firm's TEST is biased against group 1 (measured skill
# is 0.6 lower than true skill). A single group-blind threshold therefore
# under-hires good group-1 workers. A smart agent lowers group 1's threshold,
# recovering both quality (true skill of hires) and parity (equal selection).
# ===========================================================================
class HiringRLModel(mesa.Model):

    def __init__(self, mailbox, n=120, spawn=15, lam=1.0, seed=None):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)   # all sampling derives from the seed
        self.spawn = spawn
        self.lam = lam
        self._make_workers(n)

        # Biased test: group 1's measured skill is 0.6 below the truth. The firm
        # scores on this biased measurement (skill_hat); the reward judges on the
        # TRUE skill. That gap is the whole reason the agent has anything to learn.
        observer = DeclarativeObserver(
            features=["experience",
                      lambda a, rng: a.skills - (0.6 if a.group == 1 else 0.0)
                      + rng.normal(0, 0.05)],
            feature_names=["experience_hat", "skill_hat"],
            groups="group", seed=1,
        )

        def rule(scores, groups):
            # per-group thresholds arrive from OUTSIDE via the mailbox. The domain
            # step() never changes; only what the mailbox holds does.
            th = mailbox.pending
            if th is None:                       # before the first action
                return pd.Series(0, index=scores.index, dtype=int)
            cuts = groups.map(lambda g: float(th[int(g)]))
            return (scores >= cuts).astype(int)

        self.bigman = MLAgent(self, observer, ColumnScore("skill_hat"), rule)
        self.last_quality = 0.0     # mean TRUE skill of this step's hires
        self.sel_gap = 0.0          # demographic-parity gap this step

    def _make_workers(self, n):
        Worker.create_agents(self, n=n,
                             skills=self.rng.normal(size=n),
                             experience=self.rng.normal(size=n),
                             group=self.rng.binomial(1, 0.35, n))

    def eligible(self):
        return self.agents_by_type[Worker].select(lambda w: w.status == "unemployed")

    def apply(self, decisions):
        by_id = {w.unique_id: w for w in self.agents_by_type[Worker]}
        for aid, hired in decisions.items():
            if hired:
                by_id[aid].status = "employed"

    def measure(self, decisions):
        by_id = {w.unique_id: w for w in self.agents_by_type[Worker]}
        hired = [by_id[i] for i in decisions.index if decisions[i] == 1]
        self.last_quality = float(np.mean([w.skills for w in hired])) if hired else 0.0
        groups = pd.Series({i: by_id[i].group for i in decisions.index})
        mf = MetricFrame(metrics=selection_rate, y_true=decisions, y_pred=decisions,
                         sensitive_features=groups)
        self.sel_gap = float(mf.difference()) if len(mf.by_group) > 1 else 0.0

    def step(self):
        pool = self.eligible()
        if len(pool) > 0:
            _, decisions = self.bigman.step(pool)
            if decisions is not None:
                self.apply(decisions)
                self.measure(decisions)
        self.agents_by_type[Worker].shuffle_do("leave")
        self._make_workers(self.spawn)          # top up the pool so episodes don't drain


def run_rl():
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from gymnasium.utils.env_checker import check_env
        from fairsim.mesa_env import MesaEnv
    except ImportError as e:
        print(f"\n[RL demo skipped: {e}]")
        return

    def observe(m):
        # a small aggregate view: pool composition + per-group mean true skill +
        # last parity gap. (A toy observation; real domains would use the Observer.)
        pool = list(m.eligible())
        g = np.array([w.group for w in pool]); s = np.array([w.skills for w in pool])
        frac1 = float((g == 1).mean()) if len(pool) else 0.0
        m0 = float(s[g == 0].mean()) if (g == 0).any() else 0.0
        m1 = float(s[g == 1].mean()) if (g == 1).any() else 0.0
        return [frac1, m0, m1, m.sel_gap]

    def reward(m):
        # the objective: hire quality (true skill) minus a fairness penalty.
        return m.last_quality - m.lam * m.sel_gap

    factory = lambda mailbox: HiringRLModel(mailbox, seed=0)
    make_env = lambda: MesaEnv(
        factory,
        observe=observe, reward=reward,
        action_space=gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32),
        observation_space=gym.spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32),
        max_steps=40,
    )

    print("\n--- RL: MesaEnv + stable-baselines3 PPO ---")
    check_env(make_env(), skip_render_check=True)
    print("gymnasium check_env: PASS (MesaEnv is a valid gym.Env)")

    def episode_return(model, env, n=40):
        obs, _ = env.reset(); tot = 0.0; gaps = []
        for _ in range(n):
            action = env.action_space.sample() if model is None else model.predict(obs, deterministic=True)[0]
            obs, r, term, trunc, _ = env.step(action); tot += r; gaps.append(env.model.sel_gap)
            if term or trunc: break
        return tot, float(np.mean(gaps))

    base_ret, base_gap = episode_return(None, make_env())     # random-action baseline
    ppo = PPO("MlpPolicy", make_env(), n_steps=256, batch_size=64, n_epochs=4,
              seed=0, verbose=0)
    ppo.learn(total_timesteps=4096)
    ppo_ret, ppo_gap = episode_return(ppo, make_env())

    print(f"random-action policy: return={base_ret:6.2f}  mean parity gap={base_gap:.3f}")
    print(f"PPO-trained policy:   return={ppo_ret:6.2f}  mean parity gap={ppo_gap:.3f}")


if __name__ == '__main__':
    h = HiringSim(10)
    h.run_for(100)
    print('OK')

    # fairsim: no subclass, no World, just inject observer + policy + rule + label.
    n = 200
    observer = DeclarativeObserver(features=["experience", "skills"],
                                   feature_names=["experience_hat",
                                                  "skill_hat"],
                                   groups="group",
                                   seed=123)
    h2 = FairSimSim(n, observer, SklearnPolicy(LogisticRegression()), top_k(5))
    h2.run_for(60)
    store = h2.bigman.datastore
    print("OK again:",
          f"ready={h2.bigman.policy.is_ready()}",
          f"store_rows={len(store)}",
          f"labels_revealed={int(store.frame()['good_hire'].notna().sum())}")

    # The metrics came straight out of mesa's DataCollector -- a plain DataFrame,
    # one row per step, ready for pandas / plotting. No fairsim metric objects.
    metrics = h2.datacollector.get_model_vars_dataframe()
    print("\nfairness over time (mesa DataCollector + fairlearn MetricFrame):")
    print(metrics[["ready", "sel_g0", "sel_g1", "dp_diff"]]
          .iloc[::12].round(3).to_string())

    run_rl()
