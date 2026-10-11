"""
How the Observer wires together, now with REAL mesa agents and a real AgentSet.

Run:
    PYTHONPATH=/home/mattia/projects/fairsim .venv/bin/python examples/wiring_example.py

The point of this version: show that the thing we hand the Observer is a mesa
AgentSet, selected with mesa's own tools in the World. AgentSet in, Batch out.
"""

import mesa

from fairsim.core import DeclarativeObserver


# ---------------------------------------------------------------------------
# 1. CONSTRUCT SPACE: a real mesa.Agent carrying only TRUE, hidden attributes.
#    No `observed_skill` here. The agent never knows it's measured.
# ---------------------------------------------------------------------------
class Candidate(mesa.Agent):
    def __init__(self, model, skill, group, status="candidate"):
        super().__init__(model)      # mesa assigns unique_id and registers us
        self.skill = skill           # the truth
        self.group = group           # protected attribute
        self.status = status         # lifecycle flag the World filters on


# ---------------------------------------------------------------------------
# 2. THE MEASUREMENT MAP: a plain function. The domain author's model of
#    "the firm's test is biased against group 1". Reads construct space.
# ---------------------------------------------------------------------------
def biased_test(skill, group):
    penalty = 0.5 if group == 1 else 0.0
    return skill - penalty


# ---------------------------------------------------------------------------
# 3. THE WORLD owns the population and the Observer. It spawns agents and,
#    crucially, SELECTS whom to observe using mesa's AgentSet tools, not a
#    hand-rolled list comprehension. You may want to have an Observer owned by
#    a Firm/MLAgent instead (see below)
# ---------------------------------------------------------------------------
class HiringWorld(mesa.Model):
    def __init__(self, seed=0):
        super().__init__(rng=seed)   # mesa 3.5: pass rng, not seed

        Candidate(self, skill=1.0, group=0)
        Candidate(self, skill=1.0, group=1)                       # same true skill as #1
        Candidate(self, skill=2.0, group=0)
        Candidate(self, skill=0.3, group=1, status="hired")       # already hired: exclude

        # The Observer lives on the institution side; the World holds it here
        # only because this toy has no separate MLAgent yet.
        self.observer = DeclarativeObserver(
            features=[
                "skill",                                  # str spec: read the truth
                # One callable does it all: biased proxy AND group-dependent noise.
                # group 1 is measured both lower (bias) and noisier (heteroscedastic).
                lambda a, rng: biased_test(a.skill, a.group)
                + rng.normal(0, 0.05 if a.group == 0 else 0.20),
            ],
            feature_names=["skill_true", "test_hat"],
            groups="group",
            seed=seed,
        )

    def candidates(self) -> mesa.agent.AgentSet:
        # THIS is the mesa plumbing we were worried about ignoring. agents_by_type
        # gives an AgentSet; .select filters it. No comprehension in sight.
        return self.agents_by_type[Candidate].select(lambda a: a.status == "candidate")


# ---------------------------------------------------------------------------
# 4. WIRING: World selects an AgentSet -> Observer turns it into a Batch.
# ---------------------------------------------------------------------------
world = HiringWorld(seed=0)
pool = world.candidates()

print("selected pool is a:", type(pool).__name__, "of size", len(pool))
print("  (the 'hired' candidate was filtered out by AgentSet.select)\n")

batch = world.observer(pool, world, step=world.steps)

print("X (observation space):")
print(batch.X)
print("\ngroups (held separate, NOT in X):", batch.groups.tolist())
print("'group' column leaked into X?", "group" in batch.X.columns)

# A little more AgentSet plumbing, just to show it's there: group balance of the
# pool, straight from mesa, no pandas needed.
print("\npool composition by group (AgentSet.groupby):",
      {g: len(s) for g, s in world.agents_by_type[Candidate]
                              .select(lambda a: a.status == "candidate")
                              .groupby("group")})
