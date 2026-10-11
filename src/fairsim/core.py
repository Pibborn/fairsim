# FairSim core primitives
# Welcome! Here are the central tenets of fairsim:
# 1. I want to make it as clear as possible what is "construct space",
#    "observation space" and "decision space". In a sense, I think
#    writing code should be more or less natural once you keep this
#    in mind. "Who owns what?" questions (in terms of OOP) should be
#    answered naturally by thinking about that. 
# 2. Avoid putting all relevant logic in agents or custom primitives.
#    I want to make it easier to do the ML-related plumbing / data tracking,
#    not replace the base simulation library (mesa)
# 3. I want the right primitives. If writing the actual code shows that
#    a ton of code is needed, then we do not have the right primitives.
# 4. The "right primitives" are those that i) make it easy to think about code
#    ii) encourage to think in terms of "who owns what" in the ML loop
#    iii) encourage code re-usage and do not need to be subclassed every
#    single time by library users.
# 5. Should play nice with other libraries: fairlearn and mesa most of all.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from mesa import Agent, Model
from mesa.agent import AgentSet


# ============================================================================
# OBSERVATION SPACE
#
# The Observer is the crossing from construct space 
# (the agents, with their true, hidden attributes) into
# observation space (a table of what the institution actually measured).
# The whole point of tenet #1 is that this crossing is a NAMED, single step,
# not something smeared across the domain code.
#
# A note on the river inspiration: river thinks of a stream as a sequence of
# "questions" (features now) and "answers" (labels later). The Observer is the
# question side. It emits what the Big Man sees at decision time. The answer
# (the label / outcome) shows up later and is the DataStore's problem, not the
# Observer's. So the Observer never touches labels.
# ============================================================================


@dataclass(frozen=True)
class Batch:
    """
    Observation space, made concrete. This is what the Observer hands to the
    Big Man so he can make his decisions with the magical power of Performative
    Statistics!

    Three things live here, and the whole reason Batch exists (instead of just
    passing a bare DataFrame around) is to keep them aligned and to keep the
    protected attribute OUT of the feature matrix:

      * X       : the measured features. Indexed by agent id, so row <-> agent
                  alignment is free and I never have to zip() ids and rows by
                  hand. sklearn and fairlearn both eat a DataFrame directly.
      * groups  : the protected attribute, kept as a SEPARATE Series (also
                  indexed by agent id). Separate on purpose: if it lived as a
                  column in X, it might get trained on by accident.
                  fairlearn wants it as sensitive_features anyway, so.
      * meta    : an escape hatch. Empty until some domain actually needs it
                  (sample weights, an "was this imputed?" mask, timestamps...).
                  I don't want to decide up front whether metadata is essential,
                  so I make it optional. Note that this is a `dataclass.field`, 
                  which just means that it's a class parameter with a fixed type
                  annotation (here, dict).
    
    FAQs: (nobody has asked these actually, but)

    Q: Is it good that the same agent, observed twice, can give a different Batch?
    A: Yes. Observation is a measurement, and measurement can be noisy / stochastic.
    If you take the same hiring test twice, your results might differ. 
    If a particular observation needs to be remembered, that's the DataStore's
    job, not Batch's.
    """

    X: pd.DataFrame
    groups: pd.Series | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_ids(self) -> pd.Index:
        # Convenience: the ids are just the index. Exposed so decide() / effects
        # can write results back against the same keys.
        return self.X.index

    def __len__(self) -> int:
        return len(self.X)


# A feature spec is either:
#   * a str  -> read that attribute, as-is, deterministically. Pure sugar.
#   * a callable f(agent, rng) -> value -> compute the OBSERVED value. This is
#     the one and only place measurement happens: bias, proxies, transforms, and
#     noise all live here. The callable gets the rng so stochastic measurement
#     (including group-dependent noise) needs nothing but numpy.
# There is deliberately no separate "noise" mechanism. One agent, observed twice,
# can differ because the callable used the rng — not because a second system
# perturbed it afterwards. Observation happens in exactly one place per feature.
FeatureSpec = str | Callable[[Any, np.random.Generator], Any]


@runtime_checkable
class Observer(Protocol):
    """
    An Observer models the measurement process, that is,
    how the "state of the world is reduced to a set of rows,
    columns and values in a dataset". See Chapter 1 of Barocas, Hardt,
    Narayanan, and Figure 1 in particular.

    In more concrete terms, fairsim allows you to model how e.g. an institution
    may be able to measure certain characteristics of individuals only noisily,
    or by any other abstractions, valid or not. Some important caveats and points
    of reflection need to apply here:

    * Noise, even per-group noise, may not be a realistic way to model measurement
    bias. If you want a concrete model of measurement bias, you most likely need to
    commit to a worldview in the sense of Friedler et al. 2016/2021 and decide whether
    your assumption is WAE or WYSIWYG or somewhere in between.
    * Any concepts which the agents themselves are designed to have (by you, the library
    user, or me, the developer and example maker) are numerical or categorized immediately,
    which may not be an appropriate way to measure unobservable properties of individuals
    such as gender identity.
    * Please consider if the concepts you are measuring have been demonstrated to be
    operationalized with significant validity. See for instance Abigail Z. Jacobs
    and Hanna Wallach. Measurement and fairness. FAccT'21.

    The bottom line I am trying to convey here is that modelling measurement with noise
    is neither necessary nor sufficient to obtain a realistic model of the
    real world. Exercise caution, both while programming a simulation and when
    discussing the results you obtained.

    An Observer owns: which features to extract, how to extract them,
    what noise to apply, which attribute is the group label.

    This is a Protocol, not a base class: anything callable with this signature
    is an Observer. DeclarativeObserver below is the one I ship, but if a domain
    wants something weirder it just writes a callable and nobody has to subclass...
    unless they want to for clarity.
    """

    def __call__(
        self,
        agents: AgentSet,
        model: Any,
        *,
        step: int,
    ) -> Batch:
        """Reduce a set of agents to a Batch (construct space -> observation space).

        An observer takes AgentSet in (mesa / construct space), and gives you a
        Batch out (ML / observation space). If you're passing something that
        isn't an AgentSet, you probably skipped the World's selection step.
        """
        ...



class DeclarativeObserver:
    """
    A basic Observer with minimal logic. You declare what to measure; you don't write
    a loop. That is the simplest possible observer: the measurement process is a short spec,
    so this helps you avoiding having to do plumbing throughout your simulation.
    Example usage:

        obs = DeclarativeObserver(
            features=[
                "experience",                                        # read the attribute
                lambda a, rng: a.skill + rng.normal(0, 0.3),         # noisy measurement
                lambda a, rng: biased_test(a.skill, a.group),        # biased proxy (no noise)
            ],
            feature_names=["experience", "skill_hat", "test_hat"],
            groups="group",                                          # kept out of X
            seed=0,
        )
        batch = obs(candidates, model, step=model.steps)

    FAQs: (nobody asked for)
    * features are str (read attribute) or callable f(agent, rng) -> value. The
      callable is how I say "the firm sees a biased/noisy test, not the true
      skill" without ever writing that onto the agent. Construct space stays clean.
    * ALL measurement lives in the callable: bias, proxies, transforms, noise.
      There's no second noise mechanism. Group-dependent noise is just an `if`
      inside the callable, because the callable sees the whole agent.
    * groups is extracted the same way but parked in its own Series. It never
      enters X. See Batch for why I'm precious about this. (Group is a true label,
      so its spec is agent-only: str or f(agent) -> value, no rng.)
    * the rng lives on the Observer, seeded once, and is handed to every callable.
      Same Observer, stepped many times, gives a reproducible stream of measurements.
    """

    def __init__(
        self,
        features: Sequence[FeatureSpec],
        *,
        feature_names: Sequence[str] | None = None,
        groups: str | Callable[[Any], Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if not features:
            raise ValueError("features must be a non-empty sequence")

        self._features = list(features)
        self._groups_spec = groups
        self._rng = np.random.default_rng(seed)

        # Resolve column names
        if feature_names is not None:
            if len(feature_names) != len(self._features):
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) must match "
                    f"features length ({len(self._features)})"
                )
            self._names = list(feature_names)
        else:
            self._names = self._auto_names(self._features)

    def __call__(
        self,
        agents: AgentSet,
        model: Any,
        *,
        step: int,
    ) -> Batch:
        agents = list(agents)  # AgentSet is iterable; pin it so we can index twice

        if not agents:
            # Empty but well-formed: same columns, empty index. Downstream code
            # (predict, decide) should survive an empty step without special-casing.
            empty_groups = (
                pd.Series([], dtype=object, name="group")
                if self._groups_spec is not None
                else None
            )
            return Batch(
                X=pd.DataFrame(columns=self._names),
                groups=empty_groups,
            )

        # The callables run here, each getting the rng: this IS the measurement
        # step. After this, X is what the institution "saw", not the truth.
        agent_ids = [self._agent_id(a) for a in agents]
        rows = [self._extract_row(a) for a in agents]

        X = pd.DataFrame(rows, columns=self._names, index=pd.Index(agent_ids, name="agent_id"))

        groups = None
        if self._groups_spec is not None:
            group_vals = [self._extract_group(a) for a in agents]
            groups = pd.Series(group_vals, index=X.index, name="group")

        return Batch(X=X, groups=groups)

    
    # the functions that actually get into the agents and read their attributes

    def _extract_row(self, agent: Any) -> list[Any]:
        values: list[Any] = []
        for spec in self._features:
            if isinstance(spec, str):
                try:
                    values.append(getattr(agent, spec))
                except AttributeError:
                    raise AttributeError(
                        f"Agent {self._agent_id(agent)!r} has no attribute {spec!r}"
                    ) from None
            else:
                values.append(spec(agent, self._rng))  # callable owns its own noise
        return values

    def _extract_group(self, agent: Any) -> Any:
        spec = self._groups_spec
        if isinstance(spec, str):
            try:
                return getattr(agent, spec)
            except AttributeError:
                raise AttributeError(
                    f"Agent {self._agent_id(agent)!r} has no attribute {spec!r}"
                ) from None
        return spec(agent)

    @staticmethod
    def _auto_names(features: list[FeatureSpec]) -> list[str]:
        # If you didn't name your lambdas' outputs, I'll name them f_0, f_1, ...
        # You'll regret it when you read the DataFrame, but I won't stop you.
        names: list[str] = []
        lambda_idx = 0
        for spec in features:
            if isinstance(spec, str):
                names.append(spec)
            else:
                names.append(f"f_{lambda_idx}")
                lambda_idx += 1
        return names

    @staticmethod
    def _agent_id(agent: Any) -> Any:
        # Tolerate both mesa agents (unique_id) and plain objects (id()).
        # Explicit None check: mesa's first agent has unique_id == 0, which is
        # falsy -- `x or id(agent)` would silently swap it for id(). Don't.
        uid = getattr(agent, "unique_id", None)
        return uid if uid is not None else id(agent)



# ============================================================================
# DECISION SPACE
#
# A decision is just a pd.Series of actions indexed by agent_id (the same
# alignment trick as Batch.X, so decision rows line up with observation rows for
# free). No dataclass: one action per agent is one column, and there is no
# protected attribute to keep out of it, so the Series IS the decision space.
# (If a domain ever needs multi-field actions it becomes a DataFrame indexed by
# agent_id, still no new type. Held off until something actually asks for it.)
#
# A DecisionRule is likewise not a class, just a function (scores, groups) ->
# actions. It is a Callable alias: documentation only, zero runtime. `groups` is
# in the signature because the rule that MATTERS for this library is the
# group-aware one (per-group thresholds); top_k / threshold ignore it. Ship a
# few; a domain writes its own the same way.
# ============================================================================
DecisionRule = Callable[[pd.Series, "pd.Series | None"], pd.Series]


def top_k(k: int) -> DecisionRule:
    """Act on the k highest-scoring agents (a hard budget). Group-blind."""
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        chosen = scores.nlargest(k).index
        return pd.Series(scores.index.isin(chosen).astype(int), index=scores.index)
    return rule


def threshold(t: float) -> DecisionRule:
    """Act on everyone scoring at or above t. Group-blind."""
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        return (scores >= t).astype(int)
    return rule


def by_group(thresholds: Mapping[Any, float], *, default: float = 0.5) -> DecisionRule:
    """Per-group thresholds: the fairness-aware rule the `groups` arg exists for.
    `thresholds` maps a group value to its cutoff; unlisted groups use `default`.
    """
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        if groups is None:
            raise ValueError("by_group needs groups, but the Batch carried none")
        cut = groups.map(lambda g: thresholds.get(g, default))
        return (scores >= cut).astype(int)
    return rule


def passthrough(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """The do-nothing rule: the 'scores' are ALREADY 0/1 decisions, hand them
    through untouched. This is what you pair with a fairness POST-PROCESSOR
    (fairlearn ThresholdOptimizer), because the post-processor already did the
    group-aware thresholding itself. There is nothing left to decide, and you
    CAN'T also impose a top_k budget on top: re-thresholding a group-calibrated
    decision is nonsense, and fairlearn doesn't support 'DP-fair AND exactly-k'
    either. passthrough makes that boundary explicit instead of hiding it.
    """
    return scores


# ============================================================================
# THE POLICY  (how scores get made, and learned)
#
# 
# In FairSim, supervised / RL / no-ML-baseline / fairness-mitigated differ ONLY in which
# Policy you inject.
# 
# A policy is any object with these three methods:
#   predict(X, groups) -> a score per row. Side-effect free. Continuous scores
#                         by default (the DecisionRule owns the cutoff), OR final
#                         0/1 decisions for a post-processor (pair with passthrough).
#   update(X, y, groups) -> learn from revealed labels. Gets the FULL training set
#                         each call (pulled from the DataStore), so sklearn refits
#                         and river loops behind one signature.
#   is_ready() -> can I predict yet? The model knows, not the World. Before this
#                 is True the loop falls back to something simpler.
#
# `groups` is threaded through both because every fairlearn / aif360 mitigator
# needs the protected attribute (they call it sensitive_features). It's free and
# leak-safe: the Batch already carries groups as a separate Series, so the MLAgent
# just hands it over; it never sat inside X to begin with.
# ============================================================================


@runtime_checkable
class Policy(Protocol):
    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        ...

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        ...

    def is_ready(self) -> bool:
        ...


class ColumnScore:
    """The no-ML baseline: score = one feature column, verbatim. This
    is not a super useful method in experimentation, but it's simple 
    and useful for debugging.
    """

    def __init__(self, col: str):
        self.col = col

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return X[self.col].to_numpy(dtype=float)

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        pass

    def is_ready(self) -> bool:
        return True


class SklearnPolicy:
    """Wraps a sklearn estimator (including a Pipeline, so you can use e.g. fairlearn
    pre-processing like CorrelationRemover). update() refits;
    predict() returns a continuous score. Not ready until fit once.

    scorer picks how a fitted estimator becomes a score:
      "predict_proba"     -> P(class 1)          (classifiers)
      "decision_function" -> signed distance     (SVMs, etc.)
      "predict"           -> the raw output      (regressors)
    """

    def __init__(self, estimator: Any, *, scorer: str = "predict_proba"):
        self.estimator = estimator
        self.scorer = scorer
        self._fitted = False

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        if self.scorer == "predict_proba":
            return self.estimator.predict_proba(X)[:, 1]
        if self.scorer == "decision_function":
            return self.estimator.decision_function(X)
        return self.estimator.predict(X)

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        self.estimator.fit(X, y)
        self._fitted = True

    def is_ready(self) -> bool:
        return self._fitted


class FairlearnPolicy:
    """Wraps a fairlearn mitigator: an in-processor (reductions, e.g.
    ExponentiatedGradient) or a post-processor (ThresholdOptimizer). The group is
    passed as sensitive_features. Two flags cover the ways fairlearn's API varies:

      * groups_at_predict: post-processors need the group to predict (they
        threshold per group); reductions need it only to fit. Set True for
        ThresholdOptimizer, False for reductions.
      * scorer: "pmf" -> probability via _pmf_predict (reductions -> continuous
        scores -> use a normal rule like top_k/threshold). "predict" -> the hard
        0/1 output (post-processors -> pair with the `passthrough` rule).

    The base model lives inside the mitigator (its estimator= argument), so
    'post-processing needs a model first' is satisfied by construction: update()
    fits base + adjustment together each round (or the base is prefit).
    """

    def __init__(self, mitigator: Any, *, groups_at_predict: bool = False, scorer: str = "predict"):
        self.mitigator = mitigator
        self.groups_at_predict = groups_at_predict
        self.scorer = scorer
        self._fitted = False

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        if self.scorer == "pmf":
            return np.asarray(self.mitigator._pmf_predict(X))[:, 1]
        kw = {"sensitive_features": groups} if self.groups_at_predict else {}
        return np.asarray(self.mitigator.predict(X, **kw))

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        self.mitigator.fit(X, y, sensitive_features=groups)
        self._fitted = True

    def is_ready(self) -> bool:
        return self._fitted


class ExternalPolicy:
    """A Policy whose 'prediction' is fed from outside the model. It has no
    algorithm and does not learn. Think of it as a mailbox. Something outside the loop
    (a gymnasium wrapper driving an RL library, a human in the loop, a replay of
    logged decisions) writes to `pending`; predict() hands it to the mesa.Model.

    This is the seam that lets an external driver control the firm WITHOUT the
    loop knowing: the domain's step() still just calls firm.step(). Ownership of
    the policy follows ownership of the training loop, and here the RL library
    owns the loop, so this end holds only a relay. See src/fairsim/mesa_env.py.

    Two shapes for `pending`, both used in practice:
      * per-row decisions already made (allocation over a FIXED set of subjects)
        -> pair this policy with the `passthrough` rule.
      * a parameter for a rule (e.g. per-group thresholds over a DYNAMIC pool)
        -> keep a real scoring Policy and read `pending` from your rule instead.
        (This is why the name is External, not RL: it does not do RL, and the
        same relay serves non-RL external drivers too.)
    """

    def __init__(self):
        self.pending: Any = None

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return self.pending

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        pass  # learning happens in the external driver, not here

    def is_ready(self) -> bool:
        return self.pending is not None


# ============================================================================
# THE REVERSE CROSSING: what subjects perceive of the institution.
#
# The Observer names the institution-measures-subjects crossing (construct ->
# observation). Co-adaptive domains need the crossing in the OTHER direction:
# subjects measuring the institution. Before this existed, that crossing was
# smeared across domain code as a duck-typed convention (every jury Policy
# exposes `.threshold`, and Model.step() copies it to
# `model.published_threshold` for applicants to read). Disclosure gives it a
# name and one owner, same as tenet #1 demands for the forward crossing.
# ============================================================================


class Disclosure:
    """What subjects can perceive of an institution's model. The mirror of
    Observer. Three transparency levels, from most to least open:

      * PARAMETER disclosure: pass `param="threshold"` and publish() snapshots
        that attribute off the policy into `.published`. This formalizes the
        Stackelberg convention from the college domain: the institution
        announces a parameter, subjects best-respond to the announcement.
      * QUERY access: pass queries_per_step > 0 and subjects may call
        probe(X) to see what the CURRENT policy would score, up to the budget.
        predict() is side-effect free by contract, so probing trains nothing
        and logs nothing; the Disclosure only limits and meters it. This is
        the "shop around / A-B test the gatekeeper" channel.
      * OUTCOME-ONLY: construct with neither. Subjects can then only learn
        from decisions actually applied to them (agent attributes after the
        World's apply), which needs no primitive at all. Constructing an
        explicit outcome-only Disclosure just documents the worldview.

    The WORLD owns publish timing: call publish() once per tick, at the point
    in Model.step() where the announcement happens (typically the top: publish,
    subjects react, institution observes the reacted state). `.published` is a
    snapshot, so every subject reading it mid-tick sees the same value even if
    the policy is refit later in the same tick. publish() also resets the
    query meter.

    Deliberately NOT here: any model of how subjects reason about what they
    see. Inference from probes or outcomes is subject behavior and lives in
    the agents' own step() (or a coordinating collective agent), per the
    "behavior is the user's" rule.
    """

    def __init__(
        self,
        policy: Policy,
        *,
        param: str | None = None,
        queries_per_step: int = 0,
    ):
        self.policy = policy
        self.param = param
        self.queries_per_step = queries_per_step
        self.published: Any = None
        self._queries_used = 0

    def publish(self) -> Any:
        """Snapshot the published parameter (if any) and reset the query
        meter. Returns the published value for convenience."""
        if self.param is not None:
            self.published = getattr(self.policy, self.param)
        self._queries_used = 0
        return self.published

    @property
    def queries_left(self) -> int:
        return max(self.queries_per_step - self._queries_used, 0)

    def probe(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        """Ask the current policy what it would score for X. Costs one query
        from this step's budget. Raises PermissionError when the institution
        offers no query access or the budget is spent; strategic agents that
        want to degrade gracefully should check `queries_left` first."""
        if self.queries_per_step <= 0:
            raise PermissionError("this institution offers no query access")
        if self._queries_used >= self.queries_per_step:
            raise PermissionError("query budget exhausted for this step")
        self._queries_used += 1
        return self.policy.predict(X, groups)


class MLAgent(Agent):
    """
    Base class for agents involved in learning a ML model of some kind.
    This should represent, in some sense, the Big Man in control of the
    situation, that is making decisions about other people (Agents) with
    the magical power of Performative Statistics!

    A standard domain builds an MLAgent without subclassing anything.
    You can just pass it an observer and a rule.
    """

    def __init__(
        self,
        world: Model,
        observer: Observer,
        policy: Policy,
        rule: DecisionRule,
        *,
        label: str | None = None,
        min_train: int = 1,
    ):
        super().__init__(world)
        self.world = world
        self.policy = policy      # ColumnScore / SklearnPolicy / FairlearnPolicy / ...
        self.observer = observer
        self.rule = rule
        # `label` is the column this agent trains on: the SAME key the World later
        # hands to datastore.reveal(...). A no-ML baseline (ColumnScore) never learns, so 
        # you can leave label=None in that case. But then, why use fairsim and not pure mesa?
        self.label = label
        # cold-start floor: don't fit until this many answers have been revealed.
        self.min_train = min_train
        self.datastore = DataStore()
        # TODO: consider accepting the DataStore (and its max_rows) from the caller
        #   rather than always newing an unbounded one here.

    def learn(self) -> None:
        """Pull the training set from the store and refit the policy.

        Takes no data: the store IS the agent's memory. Refit-from-scratch each
        call, which is what sklearn / fairlearn want; an incremental (river) policy
        hides that inside its own update(). Until `min_train` answers exist we skip,
        so is_ready() stays False and the World's fallback covers the cold start.
        """
        if self.label is None:
            return  # a non-learning policy (baseline) has nothing to train
        X, y, groups = self.datastore.training_set(self.label, with_groups=True)
        if len(y) < self.min_train:
            return
        self.policy.update(X, y, groups)

    def decide(self, batch: Batch, scores: pd.Series) -> pd.Series:
        """
        Scores -> per-agent actions, via the injected rule. Not
        meant to be overridden for standard domains: swap the rule instead
        (top_k, threshold, by_group). The genuinely domain-specific part is what
        the action MEANS to a subject, and that lives in World.apply(), not here.
        """
        return self.rule(scores, batch.groups)

    def step(self, agents: AgentSet):
        """
        The default loop, in case you need one. The WORLD drives this (it passes
        the agents in); this Big Man does not self-activate. Observe -> record ->
        predict -> decide, and hand the decisions back for the World to apply. No
        mutation here by design.
        """
        batch = self.observer(agents, self.world, step=self.world.steps)
        self.datastore.record(batch, step=self.world.steps)  # the "question" side (T2)

        # Cold-start gate: the Policy says whether it can predict yet (a fresh
        # sklearn model can't until its first fit; ColumnScore always can). When
        # it can't, we make no decision this step and the World falls back (T5).
        if not self.policy.is_ready():
            return agents, None

        # groups are handed to predict for the fairlearn mitigators that need them
        # (post-processors threshold per group). Score-based policies ignore it.
        scores = pd.Series(self.policy.predict(batch.X, batch.groups), index=batch.agent_ids)
        decisions = self.decide(batch, scores)
        # Log the action against the same features we just recorded, so the store
        # is a full audit log. This is the "under-the-hood bookkeeping" the domain
        # would otherwise write by hand.
        self.datastore.record_decision(decisions, step=self.world.steps)
        return agents, decisions


# ============================================================================
# THE LOOP: deliberately NOT shipped.
#
# There is no World base class and no fairsim-provided step(). That's on purpose.
# The simulation loop IS mesa.Model.step(), and it's the user's to write. If we
# shipped a World.step() it would contain nothing but a fixed calling order:
#
#     subjects = self.some_selection()               # <- your mesa selection
#     for firm in self.agents_by_type[MLAgent]:
#         agents, decisions = firm.step(subjects)     # <- fairsim primitive
#         if decisions is not None:
#             self.apply(agents, decisions)           # <- your effect (domain-specific)
#     self.reveal_due_outcomes()                      # <- your delayed/selective labels
#     for firm in self.agents_by_type[MLAgent]:
#         firm.learn()                                # <- fairsim primitive
#
# Two of those lines are already MLAgent methods; the other three are your domain.
# So a World.step() would only reimplement mesa's Model.step() and charge you a
# base class for it. Instead: bring your own mesa.Model, call firm.step() /
# firm.learn() where you want them, and write apply / reveal in plain Python.
# `apply` (the old Effect) and `reveal_due` (when an outcome matures: a loan taken
# at step 3 defaults at step 15) are domain truth and stay yours. See
# hiring_example.py for the whole loop written out by hand.
# ============================================================================


# DataStore
#
# The DataStore is the one piece that justifies fairsim over "mesa + sklearn".
# It's the firm's memory: it remembers what it saw (features, at the step it saw
# them) and, later, what it learned (the outcome). Those two arrive at different
# times.
#
# This is inspired a bit by `river`: the Observer emitted the "question" (features now).
# The DataStore is where the "answer" (label) lands whenever it comes.
# This is also inspired by `feast`: features are indexed by
# (entity, event_time) = (agent_id, step), and training data is a point-in-time-correct
# join of features to later labels.
#
# Two properties that this implies:
#   * DELAYED labels  -> reveal() can be called any step, long after record().
#   * SELECTIVE labels -> you only reveal() outcomes for the agents you acted on
#     (you only learn if a loan was repaid for applicants you approved). That's
#     the selective-labels problem, and it's just "reveal a subset".


class DataStore:
    """
    The firm's database. Accumulate observed features per step, attach labels
    later, hand back a point-in-time (X, y) for training.

    Internally, this has two aligned dataframes indexed by (agent_id, step):
      * _features : what was measured, snapshotted at record() time.
      * _labels   : outcomes, written later by reveal(). NaN until an answer lands.
    Plus _groups: the protected attribute at record() time, kept OUT of the
    feature matrix (same no-leakage rule as Batch) but available for fairness-aware
    training / auditing.

    A note on the index: I keep the (agent_id, step) MultiIndex all the way onto
    the training X. sklearn and fairlearn don't care, and it means every training
    row is traceable back to who was seen and when, which may be useful when you want
    to study e.g. performativity or drift across retraining rounds.
    """

    def __init__(self, *, max_rows: int | None = None):
        # max_rows: optional sliding window.
        # None = remember everything.
        # if you want a more advanced memory method (keep all agents of a certain group,
        # discard based on relevance...) you will have to code that up yourself
        self.max_rows = max_rows
        self._features = pd.DataFrame()
        self._labels = pd.DataFrame()
        self._groups = pd.Series(dtype=object)
        # The action the firm took, logged at decision time. Kept SEPARATE from
        # _labels on purpose: a decision is known immediately and completely, an
        # outcome (label) arrives later and only for the agents you acted on. Mixing
        # them would let training_set() mistake the decision for a target.
        self._decisions = pd.Series(dtype=object)

    # "write side" methods

    def record(self, batch: Batch, *, step: int) -> None:
        """Snapshot a Batch's features (and groups) as observed at `step`."""
        if len(batch) == 0:
            return

        idx = pd.MultiIndex.from_arrays(
            [batch.X.index.to_numpy(), np.full(len(batch), step)],
            names=["agent_id", "step"],
        )
        chunk = batch.X.copy()
        chunk.index = idx
        self._features = chunk if self._features.empty else pd.concat([self._features, chunk])

        # Always append a groups chunk, one entry per feature row, padding with
        # NaN when this batch carried none. That keeps _groups 1:1 with _features
        # even if some batches have groups and others don't -- so the reindex in
        # training_set() never silently misaligns.
        vals = batch.groups.to_numpy() if batch.groups is not None else np.full(len(batch), np.nan)
        g = pd.Series(vals, index=idx, name="group")
        self._groups = g if self._groups.empty else pd.concat([self._groups, g])

        self._enforce_window()

    def record_decision(self, decisions: pd.Series, *, step: int) -> None:
        """Log the action taken at `step`, aligned to the features recorded that
        same step. Unlike reveal(), this is not an outcome and needs no waiting: the
        firm knows what it did the instant it decides. The MLAgent calls this for
        you, so the DataStore frame becomes a complete audit log (features, group,
        decision, and later the label) with no bookkeeping on the domain's side.
        """
        if decisions is None or len(decisions) == 0:
            return
        idx = pd.MultiIndex.from_arrays(
            [np.asarray(decisions.index), np.full(len(decisions), step)],
            names=["agent_id", "step"],
        )
        chunk = pd.Series(np.asarray(decisions), index=idx, name="decision")
        self._decisions = (
            chunk if self._decisions.empty else pd.concat([self._decisions, chunk])
        )
        self._enforce_window()

    def reveal(
        self,
        label: str,
        values: Mapping[Any, Any] | pd.Series,
        *,
        step: int | Literal["last_action"] | None = None,
    ) -> None:
        """Attach an outcome to a row. This is the 'answer', and it can arrive any time.

        values maps agent_id -> outcome. `step` picks WHICH feature-rows it lands on:
          * step=k             -> the specific observation at step k (point-in-time
                                  correct: the label describes the decision made on
                                  THOSE features).
          * step="last_action" -> the most recent step this agent was ACTED ON, read
                                  from the decision log. The store already knows when
                                  it last acted on an agent, so the domain does not
                                  have to carry that step around just to reveal
                                  against it. This is the selective-labels shortcut:
                                  an outcome is about the loan you last granted, and
                                  the store finds which step that was. It assumes you
                                  reveal before acting on the same agent again (true
                                  when reveal runs before the decision each step); if
                                  you act on an agent more than once before its
                                  outcome lands, pass the explicit step instead.
          * step=None          -> every row for that agent (an outcome about the agent
                                  as such, regardless of when it was observed).

        Idempotent / last-write-wins. Revealing for a subset of agents is exactly
        how you express selective labels.
        """
        if self._features.empty:
            return
        items = list(values.items())  # dict and pd.Series both support .items()

        # Lazily create / realign the label frame to the feature index.
        self._labels = self._labels.reindex(self._features.index)
        if label not in self._labels.columns:
            self._labels[label] = np.nan

        if step == "last_action":
            for aid, val in items:
                k = self._last_action_step(aid)
                if k is not None and (aid, k) in self._labels.index:
                    self._labels.loc[(aid, k), label] = val
        elif step is not None:
            for aid, val in items:
                key = (aid, step)
                if key in self._labels.index:
                    self._labels.loc[key, label] = val
        else:
            agent_ids = self._labels.index.get_level_values("agent_id")
            for aid, val in items:
                self._labels.loc[agent_ids == aid, label] = val

    def _last_action_step(self, aid: Any) -> int | None:
        """The most recent step where a POSITIVE decision was logged for `aid`, or
        None if the store never acted on it. 'Acted on' means a truthy decision (a
        loan granted, a candidate hired): an outcome only exists for those, so a
        denied-only agent correctly has nowhere to attach a label.
        """
        if self._decisions.empty:
            return None
        agent_ids = self._decisions.index.get_level_values("agent_id").to_numpy()
        steps = self._decisions.index.get_level_values("step").to_numpy()
        acted = (agent_ids == aid) & (self._decisions.to_numpy() != 0)
        hit = steps[acted]
        return int(hit.max()) if len(hit) else None


    def training_set(
        self,
        label: str,
        *,
        dropna: bool = True,
        with_groups: bool = False,
        with_decisions: bool = False,
    ) -> tuple[Any, ...]:
        """Point-in-time join of features to one revealed label.

        Returns (X, y), extended by `groups` and/or `decisions` in that order
        when the flags ask for them: (X, y[, groups][, decisions]). By default
        drops rows whose label hasn't been revealed yet, so you train only on
        answered questions. In a sense, this does not support semi-supervised
        learning. Also note that X keeps its (agent_id, step) index for
        traceability.

        with_decisions=True returns the logged action aligned to the same
        (agent_id, step) rows (NaN where no decision was logged). This is the
        affordance an action-aware learner needs: when the outcome is
        action-contaminated (attention's seen = min(occurred, attention)) or
        the training rows are decision-censored (lending's approved-only
        labels), de-censoring / propensity weighting needs to know what the
        firm DID, not just what it saw and learned.
        """
        empty_X = pd.DataFrame(columns=self._features.columns)

        def _package(X: pd.DataFrame, y: np.ndarray) -> tuple[Any, ...]:
            out: list[Any] = [X, y]
            if with_groups:
                out.append(self._groups.reindex(X.index))
            if with_decisions:
                dec = (
                    self._decisions.reindex(X.index)
                    if not self._decisions.empty
                    else pd.Series(np.nan, index=X.index, name="decision")
                )
                out.append(dec.rename("decision"))
            return tuple(out)

        if self._features.empty or label not in self._labels.columns:
            return _package(empty_X, np.array([]))

        combined = self._features.join(self._labels[[label]])
        if dropna:
            combined = combined.dropna(subset=[label])
        if combined.empty:
            return _package(empty_X, np.array([]))

        y = combined[label].to_numpy()
        X = combined.drop(columns=[label])
        return _package(X, y)

    def frame(self) -> pd.DataFrame:
        """The complete audit log: features, group, decision, and whatever labels
        exist, all joined on (agent_id, step). This is what makes group-level
        metrics a post-hoc one-liner (a MetricFrame over frame(), sliced by step)
        instead of per-step, per-group DataCollector reporters. Useful for auditing
        or quick debugging of the simulation.
        """
        out = self._features.copy()
        if not self._groups.empty:
            out = out.join(self._groups.rename("group"), how="left")
        if not self._decisions.empty:
            out = out.join(self._decisions, how="left")
        if not self._labels.empty:
            out = out.join(self._labels, how="left")
        return out

    def __len__(self) -> int:
        return len(self._features)


    def _enforce_window(self) -> None:
        """Drops lines according to `self.max_rows`. Useful to not let
        the DataStore size get too big across simulations.
        """
        if self.max_rows is None or len(self._features) <= self.max_rows:
            return
        self._features = self._features.iloc[-self.max_rows:]
        keep = self._features.index
        # DROP rows that fell out of the window; never reindex-to-features here.
        # Reindexing would INSERT placeholder rows for feature rows that have no
        # label/decision yet, and a later record_decision()/reveal() for the same
        # (agent_id, step) would then append a duplicate key. (Found by the
        # lifecycle experiment, the first user of max_rows with an MLAgent.)
        if not self._labels.empty:
            self._labels = self._labels[self._labels.index.isin(keep)]
        if not self._groups.empty:
            self._groups = self._groups[self._groups.index.isin(keep)]
        if not self._decisions.empty:
            self._decisions = self._decisions[self._decisions.index.isin(keep)]
