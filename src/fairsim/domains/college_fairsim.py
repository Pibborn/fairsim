"""
College admission with strategic gaming.

  * Construct:   `quality`, the true test score, and `eligible` derived from it
                 against a per-group threshold which we assume to be "true".
  * Observation: `score`, the gameable test score. Note that the agents act
                 on this and then the Observer reads it without any noise.
  * Decision:    admit or not, 0/1

Simulation loop:
publish -> subjects react -> observe/decide -> apply -> reveal -> learn.
The applicants read the published threshold as a parameter of the
jury's rule (not the admit decisions the rule outputs).
Every jury Policy exposes a `.threshold` attribute (the publishable parameter)
and the model copies it to `model.published_threshold` at the top of the tick.
Applicants read the model field.

Juries (all Policies, per the plan's Step 4):
  * FixedThreshold    : NaiveJury, a constant published threshold.
  * LearnedThreshold  : refits the accuracy-max threshold each round on
                        (gamed score, true eligibility) of ADMITTED applicants
                        only (selective labels), like the source's NaiveJury.
  * RobustThreshold   : burnin at threshold 0 (admit everyone, so it sees
                        UNMANIPULATED scores), then one fit on worst-case
                        gamed scores, frozen. The source's RobustJury
                        (Hardt et al. 2015, Algorithm 1).
  * ContinuousJury    : SklearnPolicy(LogisticRegression) predicting
                        eligibility from the observed score. Its publishable
                        threshold is the IMPLIED decision boundary, recovered
                        by inverting the fitted logistic.

Run from the repo root:
    python -m fairsim.domains.college_fairsim
"""

import numpy as np
import pandas as pd
from mesa import Agent, DataCollector, Model
from sklearn.linear_model import LogisticRegression

from fairsim.core import DataStore, DeclarativeObserver, MLAgent, SklearnPolicy


def best_threshold(scores, labels):
    """Accuracy-maximizing single threshold, scanned over the observed scores.
    Deterministic counterpart of ml-fairness-gym's single_threshold with its
    default cost matrix (tp=1, tn=1, fp=-1, fn=-1).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.concatenate([[0.0], np.unique(scores), [1.0 + 1e-9]])
    correct = [(np.int64((scores >= t)) == labels).sum() for t in candidates]
    return float(candidates[int(np.argmax(correct))])


# --- the juries: Policies whose learned quantity IS the published parameter ---


class FixedThreshold:
    """NaiveJury: a constant threshold. predict() hands the observed score
    through; the cutoff lives in the rule (which reads the model's published
    field). `.threshold` is the publishable parameter every jury must expose.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return X["score"].to_numpy(dtype=float)

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        pass

    def is_ready(self) -> bool:
        return True


class LearnedThreshold(FixedThreshold):
    """Refits the accuracy-max threshold each round on the full revealed
    training set (gamed scores vs true eligibility, admitted applicants only).
    Like the source's _can_train, it waits until both labels have been seen.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        self._fitted = False

    def is_ready(self) -> bool:
        """A threshold becomes ready only after a two-class fitted update.

        This keeps the college self-annotation wrapper aligned with the
        lending learner: its first update uses human labels only, rather than
        pseudo-labeling from an initial hand-set threshold.
        """
        return self._fitted

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        if len(np.unique(y)) == 2:
            self.threshold = best_threshold(X["score"], y)
            self._fitted = True


class RobustThreshold(FixedThreshold):
    """Anticipates gaming: publishes threshold 0 for `burnin` rounds (everyone
    admitted, nobody games at 0, so the revealed scores are unmanipulated),
    then fits ONE threshold on the scores shifted up by each group's maximum
    affordable gaming, and freezes.
    """

    def __init__(self, max_change: dict, burnin: int = 10):
        super().__init__(threshold=0.0)
        self.max_change = max_change   # group -> max affordable score change
        self.burnin = burnin
        self._updates = 0
        self._frozen = False

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        if self._frozen:
            return
        self._updates += 1
        if self._updates < self.burnin:
            return
        shifted = np.clip(
            X["score"].to_numpy(dtype=float) + groups.map(self.max_change).to_numpy(dtype=float),
            0.0, 1.0,
        )
        self.threshold = best_threshold(shifted, y)
        self._frozen = True


class ContinuousJury(SklearnPolicy):
    """A jury that models P(eligible | score) instead of scanning thresholds.
    Its published parameter is the IMPLIED decision boundary, recovered by
    inverting the fitted 1D logistic (the score where P = 0.5). This inversion
    is the price of the Stackelberg game: the applicants need a scalar to react
    to, and a model-based policy only has one implicitly.
    """

    def __init__(self, initial_threshold: float = 0.5):
        super().__init__(LogisticRegression())
        self._initial = initial_threshold

    @property
    def threshold(self) -> float:
        if not self.is_ready():
            return self._initial
        b0 = float(self.estimator.intercept_[0])
        b1 = float(self.estimator.coef_[0][0])
        if b1 <= 0:  # non-monotone fit: no meaningful boundary to publish
            return self._initial
        return float(np.clip(-b0 / b1, 0.0, 1.0))

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        if len(np.unique(y)) < 2:
            return  # LogisticRegression cannot fit a single class
        super().update(X, y, groups)


def admit_above_published(model):
    """Domain-shaped DecisionRule: admit where the score clears the model's
    CURRENTLY PUBLISHED threshold, the same field the applicants read. The
    cutoff deliberately does not live in the rule's closure, so subjects and
    rule cannot disagree about what was published.
    """
    def rule(scores: pd.Series, groups: pd.Series | None = None) -> pd.Series:
        return (scores >= model.published_threshold).astype(int)
    return rule


# --- the world ---------------------------------------------------------------


class Applicant(Agent):
    """One applicant, alive for one round. Gaming is subject autonomy, so it
    lives in step(): read the published threshold, move the OBSERVED score up
    to it if the group-dependent cost is affordable. `quality` never moves.
    """

    def __init__(self, model, quality, group):
        super().__init__(model)
        self.quality = quality                       # construct: true test score
        self.group = group
        self.eligible = int(quality >= model.true_thresholds[group])
        self.score = quality                         # observed; gaming moves this
        # Keep the disclosed-threshold requirement separate from an action the
        # applicant actually takes.  The former is a counterfactual requirement;
        # the latter is the simulation's incurred manipulation cost.
        self.required_disclosed_threshold_cost = 0.0
        self.incurred_gaming_cost = 0.0
        self.affordability_failure = False
        self.burden = 0.0  # backward-compatible alias for incurred_gaming_cost
        self.manipulated = False
        self.selected = False

    def step(self):
        t = self.model.subject_threshold(self)
        if t is None:
            return
        if self.score >= t:
            return
        cost = self.model.gaming_cost(self.group) * (t - self.score)
        self.required_disclosed_threshold_cost = cost
        if (
            self.model.gaming
            and cost <= 1.0
            and (t - self.score) <= self.model.gaming_control
        ):
            self.score = t
            self.manipulated = True
            self.incurred_gaming_cost = cost
            self.burden = cost
        elif self.model.gaming:
            self.affordability_failure = True


class AdmissionModel(Model):
    """One round: fresh cohort, publish, applicants game, jury observes the
    gamed scores and decides, admit, reveal eligibility of the admitted only,
    learn.
    """

    def __init__(
        self,
        num_applicants=20,
        mix_weight=(0.6, 0.4),
        mu=(0.8, 0.6),
        sigma=(0.1, 0.3),
        group_cost=(0.8, 1.2),
        true_threshold_factor=0.1,
        policy=None,
        gaming=True,
        gaming_control=np.inf,
        subsidize=False,
        subsidy_beta=0.8,
        label_regime="admitted_only",
        label_delay=0,
        retrain_every=1,
        max_rows=None,
        disclosure="public",
        rng=None,
    ):
        super().__init__(rng=rng)
        self.num_applicants = num_applicants
        self.mix_weight = mix_weight
        self.mu = mu
        self.sigma = sigma
        self.group_cost = group_cost
        self.true_thresholds = tuple(m + true_threshold_factor * m for m in mu)
        self.gaming = gaming
        self.gaming_control = gaming_control
        self.subsidize = subsidize
        self.subsidy_beta = subsidy_beta
        if label_regime not in ("all", "admitted_only"):
            raise ValueError("label_regime must be 'all' or 'admitted_only'")
        if label_delay < 0:
            raise ValueError("label_delay must be non-negative")
        if retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        self.label_regime = label_regime
        self.label_delay = label_delay
        self.retrain_every = retrain_every
        if disclosure not in ("public", "opaque"):
            raise ValueError("disclosure must be 'public' or 'opaque'")
        self.disclosure = disclosure
        self._pending_labels = []
        self._labels_matured = 0
        self._used_fallback = False

        policy = policy if policy is not None else FixedThreshold(0.5)
        self.published_threshold = policy.threshold

        # The Observer reads the GAMED score: measurement itself is honest here,
        # the distortion was made by the subjects before observation.
        observer = DeclarativeObserver(features=["score"], groups="group", seed=0)
        self.jury = MLAgent(self, observer, policy, admit_above_published(self), label="eligible")
        if max_rows is not None:
            self.jury.datastore.max_rows = max_rows
        # Store-bound policy wrappers (for example SelfAnnotating) need the
        # jury's record of unrevealed rows. Bind every wrapper in a nested stack
        # while leaving ordinary policies untouched.
        layer = policy
        while layer is not None:
            bind = getattr(layer, "bind", None)
            if bind is not None:
                bind(self.jury.datastore, "eligible")
            layer = getattr(layer, "base", None)

        self.datacollector = DataCollector(
            model_reporters={
                "threshold": lambda m: m.published_threshold,
                "manipulation_rate": lambda m: float(
                    np.mean([a.manipulated for a in m.cohort()])
                ),
                "manipulation_rate_g0": lambda m: m.group_rate("manipulated", 0),
                "manipulation_rate_g1": lambda m: m.group_rate("manipulated", 1),
                "affordability_failure_rate_g0": lambda m: m.group_rate("affordability_failure", 0),
                "affordability_failure_rate_g1": lambda m: m.group_rate("affordability_failure", 1),
                "admit_rate": lambda m: float(
                    np.mean([a.selected for a in m.cohort()])
                ),
                "observed_count": lambda m: len(m.cohort()),
                "decision_count": lambda m: int(
                    sum(a.selected for a in m.cohort())
                ),
                "labels_matured": lambda m: m._labels_matured,
                "training_rows": lambda m: m.training_rows(),
                "retained_feature_rows": lambda m: len(m.jury.datastore),
                "trained": lambda m: int(m.jury.policy.is_ready()),
                "used_fallback": lambda m: int(m._used_fallback),
                "selection_g0": lambda m: m.selection_rate(0),
                "selection_g1": lambda m: m.selection_rate(1),
                "eligible_admit_g0": lambda m: m.eligible_admit_rate(0),
                "eligible_admit_g1": lambda m: m.eligible_admit_rate(1),
                "eligible_incurred_cost_g0": lambda m: m.eligible_incurred_cost(0),
                "eligible_incurred_cost_g1": lambda m: m.eligible_incurred_cost(1),
                "eligible_required_cost_g0": lambda m: m.eligible_required_cost(0),
                "eligible_required_cost_g1": lambda m: m.eligible_required_cost(1),
                # Legacy reporter names retain the new, incurred-cost meaning.
                "eligible_burden_g0": lambda m: m.eligible_incurred_cost(0),
                "eligible_burden_g1": lambda m: m.eligible_incurred_cost(1),
            },
            agenttype_reporters={
                Applicant: {
                    "group": "group",
                    "quality": "quality",
                    "score": "score",
                    "burden": "burden",
                    "required_disclosed_threshold_cost": "required_disclosed_threshold_cost",
                    "incurred_gaming_cost": "incurred_gaming_cost",
                    "manipulated": "manipulated",
                    "affordability_failure": "affordability_failure",
                    "eligible": "eligible",
                    "selected": "selected",
                }
            },
        )

    def gaming_cost(self, group):
        cost = self.group_cost[group]
        if self.subsidize and group == 1:
            cost *= self.subsidy_beta
        return cost

    def subject_threshold(self, applicant):
        """Return the threshold this applicant can use for strategic response.

        Standard admissions experiments are public: every applicant sees the
        published threshold. Subclasses can implement a coalition-specific
        channel while an opaque institution returns no target to ordinary
        applicants.
        """
        return self.published_threshold if self.disclosure == "public" else None

    def before_subjects_act(self):
        """Extension hook for disclosure or coalition state updates."""

    def after_round(self):
        """Extension hook after decisions, labels, learning, and logging."""

    def cohort(self):
        return list(self.agents_by_type.get(Applicant, []))

    def cohort_agentset(self):
        """The schedulable cohort; subclasses may change applicant type."""
        return self.agents_by_type[Applicant]

    # --- domain truth, plain Python ------------------------------------------

    def spawn_cohort(self):
        for a in self.cohort():
            a.remove()
        for _ in range(self.num_applicants):
            group = int(self.rng.random() > self.mix_weight[0])
            quality = float(
                np.clip(self.rng.normal(self.mu[group], self.sigma[group]), 0, 1)
            )
            Applicant(self, quality, group)

    def apply(self, decisions: pd.Series) -> None:
        by_id = {a.unique_id: a for a in self.cohort()}
        for aid, admit in decisions.items():
            by_id[aid].selected = bool(admit)

    def selection_rate(self, group):
        applicants = [a for a in self.cohort() if a.group == group]
        return float(np.mean([a.selected for a in applicants])) if applicants else np.nan

    def group_rate(self, attribute, group):
        applicants = [a for a in self.cohort() if a.group == group]
        return float(np.mean([getattr(a, attribute) for a in applicants])) if applicants else np.nan

    def eligible_admit_rate(self, group):
        applicants = [a for a in self.cohort() if a.group == group and a.eligible]
        return float(np.mean([a.selected for a in applicants])) if applicants else np.nan

    def eligible_incurred_cost(self, group):
        applicants = [a for a in self.cohort() if a.group == group and a.eligible]
        return float(np.mean([a.incurred_gaming_cost for a in applicants])) if applicants else np.nan

    def eligible_required_cost(self, group):
        applicants = [a for a in self.cohort() if a.group == group and a.eligible]
        return float(np.mean([a.required_disclosed_threshold_cost for a in applicants])) if applicants else np.nan

    def eligible_burden(self, group):
        """Backward-compatible name for mean incurred gaming cost."""
        return self.eligible_incurred_cost(group)

    def training_rows(self):
        _, y = self.jury.datastore.training_set("eligible")
        return len(y)

    def _release_labels_due(self):
        """Reveal queued outcomes against the observation where they arose."""
        due, waiting = [], []
        for item in self._pending_labels:
            (due if item[0] <= self.steps else waiting).append(item)
        self._pending_labels = waiting
        self._labels_matured = sum(len(values) for _, _, values in due)
        for _, decision_step, values in due:
            self.jury.datastore.reveal("eligible", values, step=decision_step)

    def reveal_due(self) -> None:
        """Queue labels under the configured selection and delay regimes.

        The default (`admitted_only`, delay zero) follows the original college
        world exactly: eligibility labels for admitted applicants arrive and
        are used at the end of the same round. Delayed labels retain the
        decision-time feature row, rather than attaching an outcome to a later
        cohort's observation.
        """
        if self.label_regime == "all":
            applicants = self.cohort()
        else:
            applicants = [a for a in self.cohort() if a.selected]
        values = {a.unique_id: a.eligible for a in applicants}
        self._pending_labels.append((self.steps + self.label_delay, self.steps, values))
        self._release_labels_due()

    def step(self):
        # Labels delayed from earlier decisions are visible before this cohort
        # is observed. For the default immediate regime this is a no-op.
        self._release_labels_due()
        self.spawn_cohort()
        cohort_set = self.cohort_agentset()

        # 1. publish: the jury's Stackelberg move is a PARAMETER of its rule,
        #    copied onto the model where subjects can read it. The decision is
        #    not computed yet, let alone applied.
        self.published_threshold = self.jury.policy.threshold
        self._used_fallback = False

        # 2. subjects react to the published parameter.
        self.before_subjects_act()
        cohort_set.shuffle_do("step")

        # 3. observe the gamed scores, decide.
        _, decisions = self.jury.step(cohort_set)
        if decisions is None:
            # Cold start (ContinuousJury before its first fit): admit above the
            # published threshold directly, and log it so the audit is complete.
            decisions = pd.Series(
                {a.unique_id: int(a.score >= self.published_threshold) for a in self.cohort()}
            )
            self.jury.datastore.record_decision(decisions, step=self.steps)
            self._used_fallback = True

        # 4. apply, reveal (admitted only), learn.
        self.apply(decisions)
        self.reveal_due()
        if self.steps % self.retrain_every == 0:
            self.jury.learn()

        self.datacollector.collect(self)
        self.after_round()


def run(policy=None, steps=50, seed=0, **kw):
    model = AdmissionModel(policy=policy, rng=seed, **kw)
    for _ in range(steps):
        model.step()
    return model


def eligible_burden_gap(model) -> tuple[float, float, float]:
    """The source's social-burden auditor: mean gaming cost paid by the truly
    ELIGIBLE, per group (ineligible applicants' burden is their own problem;
    burden on the eligible is the institution taxing the deserving).
    Subject-level data straight off the DataCollector agent path.
    """
    agents = model.datacollector.get_agenttype_vars_dataframe(Applicant)
    eligible = agents[agents["eligible"] == 1]
    by_group = eligible.groupby("group")["burden"].mean()
    return float(by_group.get(0, 0.0)), float(by_group.get(1, 0.0)), float(
        by_group.get(1, 0.0) - by_group.get(0, 0.0)
    )


if __name__ == "__main__":
    # --- A/B: the fairsim learning jury against the plain-mesa Step 1 ---------
    from fairsim.domains.college_mesa import AdmissionModel as MesaModel

    mesa_model = MesaModel(learn=True, gaming=True, rng=0)
    mesa_traj = []
    for _ in range(50):
        mesa_model.step()
        mesa_traj.append(mesa_model.published_threshold)

    fs_model = run(policy=LearnedThreshold(0.5), steps=50, seed=0)
    fs_traj = fs_model.datacollector.get_model_vars_dataframe()["threshold"].tolist()
    assert fs_traj == mesa_traj, "fairsim and mesa learning juries diverged"
    print("A/B: fairsim learning jury is bit-identical to the mesa version (50 rounds).")
    print("  threshold trajectory: " + " ".join(f"{t:.2f}" for t in fs_traj[::5]))

    # --- success test: what gaming does to the learned threshold --------------
    no_gaming = run(policy=LearnedThreshold(0.5), steps=50, seed=0, gaming=False)
    t_gamed = fs_traj[-1]
    t_clean = no_gaming.published_threshold
    print(f"\nlearned threshold after 50 rounds: {t_clean:.3f} without gaming, {t_gamed:.3f} with.")
    print("Gaming RATCHETS the threshold up: the gamed mass sits exactly at the cutoff and is")
    print("mostly ineligible, so the accuracy-max threshold steps above it, and the gamers follow,")
    print("until the cutoff hits the top of the scale. (The plan predicted drift toward the")
    print("disadvantaged group's mean; the actual equilibrium is a race to the ceiling.)")
    assert t_gamed > t_clean

    # At the ratcheted equilibrium, admission tests AFFORDABILITY of gaming, not
    # quality: group 0 can always pay to the ceiling, group 1 cannot.
    agents = fs_model.datacollector.get_agenttype_vars_dataframe(Applicant)
    late = agents[agents.index.get_level_values("Step") > 25]
    admit_by_group = late.groupby("group")["selected"].mean()
    print(f"\nadmit rate after the ratchet (rounds 26-50): "
          f"group 0 {admit_by_group[0]:.2f}, group 1 {admit_by_group[1]:.2f}")

    # --- metrics (Step 5): DP gap off the datastore, burden off the collector -
    from fairlearn.metrics import selection_rate
    from fairsim.metrics import group_metrics_over_time

    dp = group_metrics_over_time(fs_model.jury.datastore, selection_rate)
    tbl = dp.by_group.unstack()
    print("\nselection rate by (step, group), learning jury with gaming:")
    print(tbl.iloc[[0, 4, 9, 24, 49]].round(2))

    g0, g1, gap = eligible_burden_gap(fs_model)
    sub = run(policy=LearnedThreshold(0.5), steps=50, seed=0, subsidize=True)
    s0, s1, sgap = eligible_burden_gap(sub)
    print(f"\nsocial burden on the truly eligible (mean gaming cost):")
    print(f"  no subsidy:   group 0 {g0:.3f}, group 1 {g1:.3f}, gap {gap:.3f}")
    print(f"  subsidy 0.8:  group 0 {s0:.3f}, group 1 {s1:.3f}, gap {sgap:.3f}")
    assert sgap < gap, "expected the subsidy to narrow the eligible-burden gap"

    # --- the robust jury (Step 4): anticipate gaming, then freeze -------------
    gc = 0.1
    max_change = {g: min(1.0 / c, gc) for g, c in enumerate((0.8, 1.2))}
    robust = run(policy=RobustThreshold(max_change, burnin=10), steps=50, seed=0, gaming_control=gc)
    rt = robust.datacollector.get_model_vars_dataframe()
    r_agents = robust.datacollector.get_agenttype_vars_dataframe(Applicant)
    r_late = r_agents[r_agents.index.get_level_values("Step") > 10]
    fp = float(r_late[(r_late["selected"]) & (r_late["eligible"] == 0)].shape[0]) / max(
        1, int(r_late["selected"].sum())
    )
    print(f"\nrobust jury (gaming capped at {gc}): burnin at 0, then frozen at "
          f"{rt['threshold'].iloc[-1]:.3f}; {fp:.1%} of post-burnin admits are ineligible.")

    # --- the model-based jury (Step 4): implied threshold via inversion -------
    cont = run(policy=ContinuousJury(0.5), steps=50, seed=0)
    ct = cont.datacollector.get_model_vars_dataframe()["threshold"]
    print(f"\nContinuousJury (logistic on the gamed score): implied published threshold")
    print("  " + " ".join(f"{t:.2f}" for t in ct.tolist()[::5]))
