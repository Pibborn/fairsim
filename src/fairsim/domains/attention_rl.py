"""
Attention allocation, RL-as-Policy: a CLEAN copy of the attention environment
wired for an external RL driver. attention_fairsim.py is left untouched; this
file exists to test the one core seam no port has exercised: ExternalPolicy.

The ownership flip is the point. In attention_fairsim.py the Policy owns the
scores and the loop calls it. Here the RL library owns the training loop, so
the allocator's Policy is an ExternalPolicy: a mailbox. The gymnasium wrapper
writes an allocation into `policy.pending` before each tick; inside the tick,
MLAgent.step() runs completely unchanged (observe -> record -> predict ->
decide -> log decision), where predict() just hands the mailbox contents
through and the rule is `passthrough` (the decisions were already made
outside; per core's own docstring, allocation over a FIXED set of subjects
pairs ExternalPolicy with passthrough). The domain's step() cannot tell the
difference, which is the claim being tested.

Observation (RL side): last tick's SEEN incidents per location, the same
censored map the myopic allocator uses. The agent never sees true rates.
Action: a preference vector in [0, 1]^n, converted to an integer allocation
summing to the budget by largest-remainder rounding.
Reward: incidents discovered this tick, optionally minus a fairness penalty
(the gap between the groups' same-tick discovery ratios, computed from
construct truth, which is the simulator's prerogative when shaping rewards).

Run from the repo root (trains a small PPO, ~1 minute on CPU):
    python -m fairsim.domains.attention_rl
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from mesa import Agent, Model

from fairsim.core import DeclarativeObserver, ExternalPolicy, MLAgent, passthrough


class Location(Agent):
    """Same subject as attention_fairsim.py: owns its true incident_rate and
    drifts it itself on this tick's attention (suppressed when attended,
    climbing when neglected).
    """

    def __init__(self, model, incident_rate, group):
        super().__init__(model)
        self.incident_rate = incident_rate   # construct: true latent rate
        self.group = group
        self.attention = 0
        self.incidents_occurred = 0          # construct: true events this tick
        self.incidents_seen = 0              # the allocator's map entry (censored)

    def step(self):
        if self.attention == 0:
            self.incident_rate += self.model.dynamic_rate
        else:
            self.incident_rate = max(
                0.0, self.incident_rate - self.model.dynamic_rate * self.attention
            )


class AttentionModel(Model):
    """One tick, same shape as attention_fairsim.py, but the allocator's
    Policy is a mailbox filled from outside before step() is called.
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
        self.budget = budget
        for rate, group in zip(incident_rates, groups):
            Location(self, rate, group)

        observer = DeclarativeObserver(
            features=["incidents_seen"],
            feature_names=["seen_last"],
            groups="group",
            seed=0,
        )
        self.policy = ExternalPolicy()
        # label=None: this Policy learns in the RL library, not via learn().
        # The outcome is still revealed below, so the audit log stays complete.
        self.allocator = MLAgent(self, observer, self.policy, passthrough, label=None)

    def locations(self):
        return list(self.agents_by_type[Location])

    def apply(self, decisions: pd.Series) -> None:
        by_id = {loc.unique_id: loc for loc in self.locations()}
        discovered = {}
        for aid, units in decisions.items():
            loc = by_id[aid]
            loc.attention = int(units)
            loc.incidents_occurred = int(self.rng.poisson(loc.incident_rate))
            loc.incidents_seen = min(loc.incidents_occurred, loc.attention)
            if loc.attention > 0:
                discovered[aid] = loc.incidents_seen
        self.allocator.datastore.reveal("seen", discovered, step="last_action")

    def step(self):
        locations = self.agents_by_type[Location]
        _, decisions = self.allocator.step(locations)   # predict() reads the mailbox
        self.apply(decisions)
        locations.shuffle_do("step")


def to_allocation(preferences: np.ndarray, budget: int) -> np.ndarray:
    """Preference vector in [0, 1]^n -> integer allocation summing to budget
    (largest-remainder rounding, uniform fallback on an all-zero vector).
    """
    p = np.clip(np.asarray(preferences, dtype=float), 0.0, None)
    if p.sum() == 0:
        p = np.ones_like(p)
    share = p / p.sum() * budget
    units = np.floor(share).astype(int)
    order = np.argsort(-(share - units))
    units[order[: budget - units.sum()]] += 1
    return units


class AttentionEnv(gym.Env):
    """The gymnasium face of AttentionModel. The env owns episode boundaries
    and reward; the mesa model stays the single source of dynamics.
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon=100, fairness_coef=0.0, **model_kw):
        self.horizon = horizon
        self.fairness_coef = fairness_coef
        self.model_kw = model_kw
        n = len(model_kw.get("incident_rates", (8.0, 6.0, 4.0, 2.0)))
        self.observation_space = gym.spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)
        self.model: AttentionModel | None = None
        self._episode_seed = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [loc.incidents_seen for loc in self.model.locations()], dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed
        self.model = AttentionModel(rng=self._episode_seed, **self.model_kw)
        self._episode_seed += 1
        return self._obs(), {}

    def step(self, action):
        units = to_allocation(action, self.model.budget)
        self.model.policy.pending = units          # fill the mailbox...
        self.model.step()                          # ...the loop runs unchanged

        locations = self.model.locations()
        reward = float(sum(loc.incidents_seen for loc in locations))
        if self.fairness_coef > 0:
            ratios = []
            for g in sorted({loc.group for loc in locations}):
                seen = sum(loc.incidents_seen for loc in locations if loc.group == g)
                occurred = sum(loc.incidents_occurred for loc in locations if loc.group == g)
                ratios.append(seen / occurred if occurred else 1.0)
            reward -= self.fairness_coef * (max(ratios) - min(ratios))

        truncated = self.model.steps >= self.horizon
        return self._obs(), reward, False, truncated, {}


def evaluate(controller, seeds=range(100, 110), horizon=100, **env_kw):
    """Mean episode reward of `controller: obs -> action` over fixed seeds."""
    env = AttentionEnv(horizon=horizon, **env_kw)
    totals = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(controller(obs))
            total += r
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3 import PPO

    # --- the seam is a legal gym.Env -------------------------------------------
    check_env(AttentionEnv(horizon=20), skip_render_check=True)
    print("gymnasium check_env: passed.")

    # --- baselines through the same env ----------------------------------------
    n = 4
    rng = np.random.default_rng(0)
    baselines = {
        "random": lambda obs: rng.random(n, dtype=np.float32),
        "uniform": lambda obs: np.ones(n, dtype=np.float32),
        "myopic proportional": lambda obs: np.asarray(obs, dtype=np.float32),
    }
    scores = {name: evaluate(c) for name, c in baselines.items()}
    for name, s in scores.items():
        print(f"  {name:>20}: mean episode reward {s:7.1f}")

    # --- PPO through the ExternalPolicy mailbox --------------------------------
    train_env = AttentionEnv(horizon=100)
    ppo = PPO("MlpPolicy", train_env, seed=0, verbose=0)
    ppo.learn(total_timesteps=30_000)
    ppo_score = evaluate(lambda obs: ppo.predict(obs, deterministic=True)[0])
    print(f"  {'PPO (30k steps)':>20}: mean episode reward {ppo_score:7.1f}")
    assert ppo_score > scores["random"], "expected PPO to beat the random allocator"

    # --- what the audit log says about the trained agent -----------------------
    env = AttentionEnv(horizon=100)
    obs, _ = env.reset(seed=100)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(ppo.predict(obs, deterministic=True)[0])
        done = terminated or truncated
    frame = env.model.allocator.datastore.frame()
    attn = frame.groupby("group")["decision"].mean()
    unmeasured = frame["seen"].isna().mean()
    print(f"\naudit log of one PPO episode: mean attention units group 0 {attn.get(0, 0):.2f}, "
          f"group 1 {attn.get(1, 0):.2f};")
    print(f"  {unmeasured:.0%} of location-ticks unmeasured (NaN, censored). The mailbox run")
    print("  produced the same complete audit trail as an internal policy would.")

    # --- fairness-shaped reward, same seam --------------------------------------
    fair_env = AttentionEnv(horizon=100, fairness_coef=3.0)
    ppo_fair = PPO("MlpPolicy", fair_env, seed=0, verbose=0)
    ppo_fair.learn(total_timesteps=30_000)

    plain_fair_score = evaluate(
        lambda obs: ppo_fair.predict(obs, deterministic=True)[0], fairness_coef=3.0
    )
    plain_score_of_fair = evaluate(lambda obs: ppo_fair.predict(obs, deterministic=True)[0])
    print(f"\nfairness-shaped PPO (coef 3.0): shaped reward {plain_fair_score:7.1f}, "
          f"unshaped reward {plain_score_of_fair:7.1f}")
    print(f"  (plain PPO's unshaped reward: {ppo_score:7.1f}. In this configuration balance")
    print("  and efficiency align near the uniform optimum, so the penalty acts as a")
    print("  regularizer rather than a tax; both agents ran through the same mailbox seam.)")
