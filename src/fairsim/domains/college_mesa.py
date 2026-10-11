"""
College admission (strategic gaming), Step 1: plain mesa, no fairsim yet.

Ports the dynamics of ml-fairness-gym's CollegeAdmissionsEnv, a two-player
Stackelberg game. Each round a fresh cohort of applicants arrives with true
test scores (construct space). The jury PUBLISHES a score threshold. Applicants
below it can spend effort to raise their observed score exactly to the
threshold, if the cost (proportional to the distance, group-dependent) is
affordable. The jury then admits on the gamed scores and, in the learning
variant, refits its threshold on the gamed scores against true eligibility,
which it sees only for ADMITTED applicants (selective labels).

The ordering is the point: publish, subjects react, decide, apply, learn.
Group 1 is disadvantaged twice over: a lower score distribution AND a higher
cost per point of gaming (group_cost 1.2 vs 0.8), so the same distance to the
threshold burdens it more. A subsidy multiplies group 1's cost by
subsidy_beta < 1.

No fairsim imports on purpose. This is the A side of an A/B with
college_fairsim.py.

Run from the repo root:
    python -m fairsim.domains.college_mesa
"""

import numpy as np
from mesa import Agent, DataCollector, Model


def best_threshold(scores, labels):
    """Accuracy-maximizing single threshold, scanned over the observed scores.
    Equivalent to ml-fairness-gym's single_threshold with its default cost
    matrix (tp=1, tn=1, fp=-1, fn=-1), except deterministic: theirs returns a
    smoothed (randomized) threshold off the ROC hull.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.concatenate([[0.0], np.unique(scores), [1.0 + 1e-9]])
    correct = [(np.int64((scores >= t)) == labels).sum() for t in candidates]
    return float(candidates[int(np.argmax(correct))])


class Applicant(Agent):
    """One applicant, alive for one round. Owns its TRUE quality (construct);
    `score` is what the jury sees, and gaming mutates only `score`.
    """

    def __init__(self, model, quality, group):
        super().__init__(model)
        self.quality = quality                       # construct: true test score
        self.group = group
        self.eligible = int(quality >= model.true_thresholds[group])
        self.score = quality                         # observed; gaming moves this
        self.burden = 0.0                            # cost to reach the threshold
        self.manipulated = False
        self.selected = False

    def game(self, published_threshold):
        """Read the PUBLISHED threshold and move the observed score up to it if
        the group-dependent cost is affordable (net utility of flipping the
        decision stays nonnegative, i.e. cost <= 1). Mirrors the source: the
        burden is recorded for everyone below the threshold, gamed or not.
        """
        t = published_threshold
        if self.score >= t:
            return
        cost = self.model.gaming_cost(self.group) * (t - self.score)
        self.burden = cost
        if self.model.gaming and cost <= 1.0:
            self.score = t
            self.manipulated = True


class Jury(Agent):
    """The institution. Publishes a threshold; in the learning variant, refits
    it each round on (gamed score, true eligibility) of the applicants it
    ADMITTED, accumulated over all rounds so far.
    """

    def __init__(self, model, threshold=0.5, learn=False):
        super().__init__(model)
        self.threshold = threshold
        self.learn = learn
        self._scores: list = []
        self._labels: list = []

    def update(self, admitted):
        self._scores.extend(a.score for a in admitted)
        self._labels.extend(a.eligible for a in admitted)
        # Like the source's _can_train: no refit until both labels were seen.
        if self.learn and set(self._labels) == {0, 1}:
            self.threshold = best_threshold(self._scores, self._labels)


class AdmissionModel(Model):
    """One round: fresh cohort, jury publishes, applicants game, jury admits on
    the gamed scores, jury learns from the admitted only.
    """

    def __init__(
        self,
        num_applicants=20,
        mix_weight=(0.6, 0.4),
        mu=(0.8, 0.6),
        sigma=(0.1, 0.3),
        group_cost=(0.8, 1.2),
        true_threshold_factor=0.1,
        threshold=0.5,
        learn=False,
        gaming=True,
        subsidize=False,
        subsidy_beta=0.8,
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
        self.subsidize = subsidize
        self.subsidy_beta = subsidy_beta
        self.published_threshold = threshold
        self.jury = Jury(self, threshold=threshold, learn=learn)

        self.datacollector = DataCollector(
            model_reporters={
                "threshold": lambda m: m.published_threshold,
                "manipulation_rate": lambda m: float(
                    np.mean([a.manipulated for a in m.cohort()])
                ),
                "admit_rate": lambda m: float(
                    np.mean([a.selected for a in m.cohort()])
                ),
            },
            agenttype_reporters={
                Applicant: {
                    "group": "group",
                    "quality": "quality",
                    "score": "score",
                    "burden": "burden",
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

    def cohort(self):
        # agents_by_type has no Applicant key until the first cohort is born
        return list(self.agents_by_type.get(Applicant, []))

    def spawn_cohort(self):
        for a in self.cohort():
            a.remove()
        for _ in range(self.num_applicants):
            group = int(self.rng.random() > self.mix_weight[0])
            quality = float(
                np.clip(self.rng.normal(self.mu[group], self.sigma[group]), 0, 1)
            )
            Applicant(self, quality, group)

    def step(self):
        self.spawn_cohort()
        cohort = self.cohort()

        # 1. publish: the jury's move in the Stackelberg game is a PARAMETER,
        #    written where the applicants can read it, before any decision.
        self.published_threshold = self.jury.threshold

        # 2. subjects react: gaming reads the published threshold.
        self.agents_by_type[Applicant].shuffle_do("game", self.published_threshold)

        # 3. decide + apply: admit on the gamed scores.
        for a in cohort:
            a.selected = a.score >= self.published_threshold

        # 4. learn: true eligibility is revealed for the admitted only.
        self.jury.update([a for a in cohort if a.selected])

        self.datacollector.collect(self)


if __name__ == "__main__":
    # Under a FIXED threshold, gaming shifts the observed score distribution:
    # sub-threshold mass piles up exactly at the threshold.
    fixed = AdmissionModel(learn=False, gaming=True, rng=0)
    for _ in range(25):
        fixed.step()
    agents = fixed.datacollector.get_agenttype_vars_dataframe(Applicant)
    at_t = (agents["score"] == 0.5).mean()
    print("fixed threshold 0.5, gaming on (25 rounds, 20 applicants each):")
    print(f"  mean true quality {agents['quality'].mean():.3f} vs mean observed score {agents['score'].mean():.3f}")
    print(f"  {at_t:.1%} of observed scores sit EXACTLY at the threshold (gamed mass)")
    assert agents["score"].mean() > agents["quality"].mean() + 0.01
    assert at_t > 0.1

    # The learning jury: threshold trajectory under gaming.
    for gaming in (False, True):
        m = AdmissionModel(learn=True, gaming=gaming, rng=0)
        traj = []
        for _ in range(50):
            m.step()
            traj.append(m.published_threshold)
        print(f"\nlearning jury, gaming={gaming}: threshold trajectory")
        print("  " + " ".join(f"{t:.2f}" for t in traj[::5]))
        print(f"  final threshold {traj[-1]:.3f}")
