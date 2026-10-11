"""
Lending, Step 5: metrics, all post-hoc from the audit log.

Runs the learned bank from Step 4 and reads fairness metrics off the bank's
DataStore, with no per-step, per-group bookkeeping in the domain. There are two
metric families, and the gap between them is the whole lesson of this step.

  1. Label-FREE (selection rate by group, and its demographic-parity gap). These
     need only the decision, so they are defined the moment a decision is logged.
  2. Label-BASED. These need the revealed outcome, which arrives late (delayed) and
     only for approved loans (selective). That selectivity has a sharp consequence,
     shown below: the usual decision confusion matrix (TPR/FPR) is DEGENERATE here,
     and the honest label-based metric is the realized outcome among loans made.

Why post-hoc and not a DataCollector reporter: a per-step reporter runs during
step t, but a loan granted at step t is not labelled until t + repay_lag. At step t
there is nothing to report yet, so the label-based number for step t only becomes
computable several steps later. The natural home is a query over the finished
DataStore, not a per-step reporter.

Run:
    python -m fairsim.domains.lending_metrics
"""

from fairlearn.metrics import false_positive_rate, selection_rate, true_positive_rate

from fairsim.domains.lending_fairsim_learned import LendingModel
from fairsim.metrics import group_metrics_over_time


def main() -> None:
    model = LendingModel(rng=0)
    for _ in range(40):
        model.step()

    store = model.bank.datastore

    # 1. Label-free: selection rate by (step, group) and the DP gap per step.
    sel = group_metrics_over_time(store, selection_rate)
    print("selection rate by (step, group):")
    print(sel.by_group.round(3).unstack().iloc[::10])
    print("\ndemographic-parity gap per step:")
    print(sel.difference().round(3).iloc[::10])

    # 2. The trap. TPR/FPR of the decision against the outcome look computable, but
    #    every labelled row was APPROVED (selective labels: we only see outcomes for
    #    loans we made), so y_pred is 1 on all of them and both rates collapse to 1.0.
    #    This is the selective-labels problem, not a metrics bug: you cannot form a
    #    decision confusion matrix when only one side of the decision is ever labelled.
    err = group_metrics_over_time(
        store,
        {"tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true="repaid",
    )
    print("\nTPR/FPR of decision vs outcome (DEGENERATE under selective labels):")
    print(err.by_group.round(3).unstack(level="group").iloc[::20])

    # 3. The honest label-based metric: of the loans actually MADE to each group,
    #    what fraction repaid. Point BOTH y_pred and y_true at "repaid" so
    #    selection_rate averages the outcome over the approved-and-matured rows. This
    #    is a base rate on the observed population, which is all selective labels let
    #    you measure. A group gap here is disparate realized quality, not disparate
    #    treatment; read it alongside the selection-rate gap from part 1.
    made = group_metrics_over_time(store, selection_rate, y_pred="repaid", y_true="repaid")
    print("\nrepayment rate among loans made, by (step, group):")
    print(made.by_group.round(3).unstack().iloc[::10])

    frame = store.frame()
    print(
        f"\nrows: {len(frame)} logged decisions (label-free basis), "
        f"{int(frame['repaid'].notna().sum())} matured-and-labelled (label-based basis, "
        f"all of them approved)."
    )


if __name__ == "__main__":
    main()
