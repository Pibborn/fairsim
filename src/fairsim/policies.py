# Standard policies: the diagonal of the literature matrix, shipped.
#
# Each class here is a Policy (predict / update / is_ready) that WRAPS another
# Policy, so literature mechanisms compose over any base learner:
#
#     SelfAnnotating(SklearnPolicy(LogisticRegression()), ...)   # Xie & Zhang 2024
#     Imputing(SklearnPolicy(...), ...)                           # SELLF-adjacent
#     demographic_parity(LogisticRegression())                    # Liu / fairlearn line
#
# Why wrappers need the store: MLAgent.learn() hands update() only the LABELED
# rows (that is the point of training_set), but both mechanisms below are
# precisely about the rows the firm did NOT get labels for. So a wrapper that
# self-annotates or imputes needs the firm's memory. Wire it after the MLAgent
# exists, since the MLAgent news its own DataStore:
#
#     firm = MLAgent(world, observer, policy, rule, label="repaid")
#     policy.bind(firm.datastore, "repaid")
#
# This is deliberate, not an accident of construction order: the wrapper reads
# the same audit log the metrics read, so what it trained on is auditable too.

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fairsim.core import DataStore, FairlearnPolicy, Policy


class _StoreBound:
    """Shared plumbing for wrappers that need the firm's memory (see module
    docstring for why). Not a Policy by itself."""

    def __init__(self, base: Policy):
        self.base = base
        self.store: DataStore | None = None
        self.label: str | None = None

    def bind(self, store: DataStore, label: str) -> "Any":
        """Point the wrapper at the firm's DataStore and the label column it
        trains on (the same `label` the MLAgent was given). Returns self so
        you can bind inline."""
        self.store = store
        self.label = label
        return self

    def predict(self, X: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        return self.base.predict(X, groups)

    def is_ready(self) -> bool:
        return self.base.is_ready()

    # -- helpers over the audit log ------------------------------------------

    def _unlabeled(self) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
        """Feature rows recorded but never revealed for our label: the rows
        the mechanism is about. Returns (X, groups, decisions) aligned."""
        assert self.store is not None and self.label is not None, (
            "wrapper not bound: call .bind(firm.datastore, label) after "
            "constructing the MLAgent"
        )
        frame = self.store.frame()
        if frame.empty:
            empty = pd.DataFrame()
            return empty, None, None
        feature_cols = list(self.store._features.columns)
        if self.label in frame.columns:
            mask = frame[self.label].isna()
        else:
            mask = pd.Series(True, index=frame.index)
        sub = frame.loc[mask]
        X = sub[feature_cols]
        groups = sub["group"] if "group" in sub.columns else None
        decisions = sub["decision"] if "decision" in sub.columns else None
        return X, groups, decisions


class SelfAnnotating(_StoreBound):
    """Xie & Zhang (NeurIPS 2024): retrain on your own annotations.

    Each update(), the wrapper takes the human-labeled rows it is handed, adds
    up to `ratio` times as many MODEL-annotated rows (unlabeled rows from the
    store, labeled by the current base model), and fits the base on the union.
    `ratio` is their N/K knob: N model-annotated per K human-annotated
    samples; their theory says the acceptance rate ratchets toward 1 when the
    human fraction is too small.

    annotation:
      * "hard"          : pseudo-label = 1[score >= 0.5]. Their ORIGINAL
                          retraining process, the one that ratchets.
      * "probabilistic" : pseudo-label ~ Bernoulli(score). Their REFINED
                          process, which their Thm 4.1 says stabilizes the
                          dynamics.

    The first fit (base not ready) uses human labels only: you cannot
    self-annotate before you have a self.
    """

    def __init__(self, base: Policy, *, ratio: float = 1.0,
                 annotation: str = "hard", seed: int | None = None):
        super().__init__(base)
        if annotation not in ("hard", "probabilistic"):
            raise ValueError("annotation must be 'hard' or 'probabilistic'")
        self.ratio = ratio
        self.annotation = annotation
        self._rng = np.random.default_rng(seed)

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        if not self.base.is_ready() or self.ratio <= 0 or len(y) == 0:
            self.base.update(X, y, groups)
            return

        Xu, gu, _ = self._unlabeled()
        n_model = min(int(self.ratio * len(y)), len(Xu))
        if n_model == 0:
            self.base.update(X, y, groups)
            return

        # Most recent unlabeled rows first: annotation happens on the data the
        # firm just collected, not on ancient history.
        take = Xu.sort_index(level="step").iloc[-n_model:]
        scores = np.asarray(
            self.base.predict(take, gu.reindex(take.index) if gu is not None else None),
            dtype=float,
        )
        if self.annotation == "hard":
            pseudo = (scores >= 0.5).astype(float)
        else:
            pseudo = (self._rng.random(len(scores)) < np.clip(scores, 0, 1)).astype(float)

        X_all = pd.concat([X, take])
        y_all = np.concatenate([np.asarray(y, dtype=float), pseudo])
        g_all = None
        if groups is not None and gu is not None:
            g_all = pd.concat([groups, gu.reindex(take.index)])
        self.base.update(X_all, y_all, g_all)


class Imputing(_StoreBound):
    """SELLF-adjacent (Valdrighi et al. 2026): impute the labels selective
    labeling withheld, optionally with inverse-propensity weighting.

    The selective-labels problem: the firm only learns outcomes for the
    applicants it accepted, so a model fit on labeled rows alone sees a
    censored world (and fairness metrics computed on it can be arbitrarily
    wrong; their Prop 3.1). The SELLF recipe this wrapper transplants: fit a
    separate LABEL predictor on the labeled rows, reweighted by inverse
    propensity of having been labeled at all, then impute labels for the
    rejected rows and fit the base policy on the union of real and imputed.

    * label_model / propensity_model: sklearn classifiers with predict_proba;
      default LogisticRegression. The propensity model fits P(labeled | X) on
      ALL recorded rows; its inverse weights the label predictor (ipw=True).
    * imputation is probabilistic (label ~ Bernoulli(p_hat)) by default,
      "hard" thresholds at 0.5.

    Needs decisions logged in the store (they always are when an MLAgent
    drives), which is what training_set(with_decisions=True) exists for.
    """

    def __init__(self, base: Policy, *, label_model: Any = None,
                 propensity_model: Any = None, ipw: bool = True,
                 imputation: str = "probabilistic", seed: int | None = None):
        super().__init__(base)
        from sklearn.linear_model import LogisticRegression
        self.label_model = label_model or LogisticRegression(max_iter=1000)
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.ipw = ipw
        if imputation not in ("hard", "probabilistic"):
            raise ValueError("imputation must be 'hard' or 'probabilistic'")
        self.imputation = imputation
        self._rng = np.random.default_rng(seed)

    def update(self, X: pd.DataFrame, y: np.ndarray, groups: pd.Series | None = None) -> None:
        Xu, gu, _ = self._unlabeled()
        if len(y) == 0 or len(Xu) == 0 or len(np.unique(y)) < 2:
            # nothing to impute (or the label predictor can't be fit yet)
            self.base.update(X, y, groups)
            return

        weights = None
        if self.ipw:
            # P(labeled | X) on all recorded rows; clip to keep weights sane.
            X_all_rows = pd.concat([X, Xu])
            labeled = np.concatenate([np.ones(len(X)), np.zeros(len(Xu))])
            self.propensity_model.fit(X_all_rows, labeled)
            p = self.propensity_model.predict_proba(X)[:, 1]
            weights = 1.0 / np.clip(p, 0.05, 1.0)

        self.label_model.fit(X, y, sample_weight=weights)
        p_hat = self.label_model.predict_proba(Xu)[:, 1]
        if self.imputation == "hard":
            imputed = (p_hat >= 0.5).astype(float)
        else:
            imputed = (self._rng.random(len(p_hat)) < p_hat).astype(float)

        X_fit = pd.concat([X, Xu])
        y_fit = np.concatenate([np.asarray(y, dtype=float), imputed])
        g_fit = pd.concat([groups, gu]) if groups is not None and gu is not None else None
        self.base.update(X_fit, y_fit, g_fit)


# ============================================================================
# Fairness presets: thin, well-defaulted constructors over FairlearnPolicy.
# Both return a policy whose predict() emits FINAL 0/1 decisions, so pair them
# with the `passthrough` rule (see core.passthrough for why you cannot also
# impose a budget on top). Domain-agnostic on purpose: this is the fix for the
# old "DemographicParity is credit-coupled" problem.
# ============================================================================


def demographic_parity(estimator: Any = None, **kw) -> FairlearnPolicy:
    """A ThresholdOptimizer-based policy enforcing demographic parity.
    `estimator` is any sklearn classifier (default LogisticRegression); extra
    kwargs go to ThresholdOptimizer. Pair with the `passthrough` rule."""
    return _threshold_optimizer(estimator, constraints="demographic_parity", **kw)


def equal_opportunity(estimator: Any = None, **kw) -> FairlearnPolicy:
    """A ThresholdOptimizer-based policy enforcing equal opportunity (equal
    true positive rates). Pair with the `passthrough` rule."""
    return _threshold_optimizer(estimator, constraints="true_positive_rate_parity", **kw)


def _threshold_optimizer(estimator: Any, *, constraints: str, **kw) -> FairlearnPolicy:
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.linear_model import LogisticRegression

    # `is None`, not truthiness: sklearn ensembles define __len__ via their
    # fitted estimators_, so bool(unfitted ensemble) raises.
    mitigator = ThresholdOptimizer(
        estimator=estimator if estimator is not None else LogisticRegression(max_iter=1000),
        constraints=constraints,
        predict_method="predict_proba",
        **kw,
    )
    return FairlearnPolicy(mitigator, groups_at_predict=True, scorer="predict")


if __name__ == "__main__":
    # Self-test on a synthetic selective-labels store: 200 applicants, top
    # half by x accepted, true label ~ sigmoid(2x), outcomes revealed for the
    # accepted only. Run from the repo root:
    #     python -m fairsim.policies
    from sklearn.linear_model import LogisticRegression

    from fairsim.core import Batch, SklearnPolicy

    rng = np.random.default_rng(0)

    def make_store(n=200, accept_frac=0.5):
        ds = DataStore()
        x = rng.normal(0, 1, n)
        g = rng.integers(0, 2, n)
        X = pd.DataFrame({"x": x}, index=pd.Index(range(n), name="agent_id"))
        ds.record(Batch(X=X, groups=pd.Series(g, index=X.index, name="group")), step=1)
        dec = (x > np.quantile(x, 1 - accept_frac)).astype(int)
        ds.record_decision(pd.Series(dec, index=X.index), step=1)
        true_y = (rng.random(n) < 1 / (1 + np.exp(-2 * x))).astype(int)
        ds.reveal("y", {i: true_y[i] for i in range(n) if dec[i]}, step=1)
        return ds, X

    ds, X = make_store()
    Xl, yl, gl = ds.training_set("y", with_groups=True)

    pol = SelfAnnotating(
        SklearnPolicy(LogisticRegression(max_iter=1000)), ratio=2.0, seed=0
    ).bind(ds, "y")
    pol.update(Xl, yl, gl)   # first fit: human labels only
    pol.update(Xl, yl, gl)   # second fit: self-annotates
    alone = SklearnPolicy(LogisticRegression(max_iter=1000))
    alone.update(Xl, yl, gl)
    c_wrap = pol.base.estimator.coef_[0][0]
    c_alone = alone.estimator.coef_[0][0]
    assert abs(c_wrap - c_alone) > 1e-6, "self-annotation had no effect"
    print(f"SelfAnnotating: coef {c_wrap:.2f} with annotation vs {c_alone:.2f} without "
          "(hard pseudo-labels sharpen the model's own boundary)")

    ds2, X2 = make_store()
    Xl2, yl2, gl2 = ds2.training_set("y", with_groups=True)
    imp = Imputing(SklearnPolicy(LogisticRegression(max_iter=1000)), seed=0).bind(ds2, "y")
    imp.update(Xl2, yl2, gl2)
    cens = SklearnPolicy(LogisticRegression(max_iter=1000))
    cens.update(Xl2, yl2, gl2)
    low = X2[X2["x"] < -1]
    assert imp.predict(low).mean() < cens.predict(low).mean(), (
        "imputation should be less optimistic where labels were censored"
    )
    print(f"Imputing: mean score on low-x rows {imp.predict(low).mean():.3f} "
          f"vs censored-only {cens.predict(low).mean():.3f}")

    n = 400
    x = rng.normal(0, 1, n)
    g = rng.integers(0, 2, n)
    Xp = pd.DataFrame({"x": x + 0.8 * g}, index=pd.Index(range(n), name="agent_id"))
    gp = pd.Series(g, index=Xp.index, name="group")
    yp = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    for name, ctor in [("demographic_parity", demographic_parity),
                       ("equal_opportunity", equal_opportunity)]:
        p = ctor()
        p.update(Xp, yp, gp)
        sr = pd.Series(p.predict(Xp, gp)).groupby(g).mean()
        print(f"{name}: selection rates by group {sr.round(3).to_dict()}")
    print("policies self-test: all pass")
