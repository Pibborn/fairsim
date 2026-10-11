"""
MesaEnv: make a mesa.Model into a gymnasium.Env

Note that compared to supervised ML, we do not let a firm/agent own a RL env. Instead we
invert the ownership:

  * The env owns the progress, by calling one model tick per env.step().
  * The mesa.Model's step() content stays the domain's, unchanged from the non-RL runs.
  * The RL library computes the action BEFORE calling env.step(action). So by the
    time the model advances, the decision already exists. MesaEnv just STAGES it
    into the ExternalPolicy mailbox the domain wired into its firm, advances one
    tick, and reads back observation + reward.

Lives in its own module on purpose so that gymnasium can be an optional dependency. Importing
core_v3 (Observer, Policy, DataStore, MLAgent) never drags gymnasium in; only this
bridge does.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym

from fairsim.core import ExternalPolicy


class MesaEnv(gym.Env):
    """Wrap a mesa.Model behind the gymnasium contract.

    You provide the domain-specific pieces; the env provides the plumbing:

      factory(mailbox) -> mesa.Model
          Build a mesa.Model each episode, prepared to consume `mailbox` (an
          ExternalPolicy). Either hand the mailbox in as the firm's policy
          (allocation -> passthrough), or capture it in a rule that reads
          mailbox.pending (per-group thresholds). The env writes mailbox.pending
          before every tick.
      observe(model) -> array-like
          The RL observation. What the learner sees.
      reward(model) -> float
          The RL objective (e.g. efficiency minus a fairness penalty). Be careful
          to model is so that it is reasonable: it will be used for learning,
          and thus should represent something that the firm/agent can reasonably
          observe.
      terminated(model) -> bool   (optional)
          Early episode end. Truncation at max_steps is handled here.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        factory: Callable[[ExternalPolicy], Any],
        *,
        observe: Callable[[Any], Any],
        reward: Callable[[Any], float],
        action_space: gym.Space,
        observation_space: gym.Space,
        max_steps: int,
        terminated: Optional[Callable[[Any], bool]] = None,
    ):
        super().__init__()
        self._factory = factory
        self._observe = observe
        self._reward = reward
        self._terminated = terminated
        self._max_steps = max_steps
        self.action_space = action_space
        self.observation_space = observation_space
        self.mailbox: ExternalPolicy | None = None
        self.model: Any = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.mailbox = ExternalPolicy()
        self.model = self._factory(self.mailbox)   # fresh model, wired to the mailbox
        return self._obs(), {}

    def step(self, action):
        self.mailbox.pending = action    
        self.model.step() # note: this will "consume" the action
        obs = self._obs()
        reward = float(self._reward(self.model))
        terminated = bool(self._terminated(self.model)) if self._terminated else False
        truncated = self.model.steps >= self._max_steps
        return obs, reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        # Clip into the declared box so a slightly out-of-range aggregate never
        # trips the RL library's bounds checks.
        obs = np.asarray(self._observe(self.model), dtype=np.float32)
        space = self.observation_space
        if isinstance(space, gym.spaces.Box):
            obs = np.clip(obs, space.low, space.high)
        return obs
