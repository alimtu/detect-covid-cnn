"""📈 Compare — side-by-side comparison of multiple experiments."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.services import comparison_service, experiment_service
from frontend.ui import charts


def render() -> None:
    st.title("📈 Compare Experiments")

    records = experiment_service.list_experiments()
    completed = [r for r in records if r.status == "completed"]
    if len(completed) < 1:
        st.info("Need at least one completed experiment to compare.")
        return

    ids = [r.experiment_id for r in completed]
    default = ids[: min(3, len(ids))]
    selected = st.multiselect("Select experiments", ids, default=default)
    if len(selected) < 2:
        st.warning("Select at least two experiments for a side-by-side comparison.")
        return

    chosen = comparison_service.records_for(selected)

    _metrics_table(chosen)
    st.divider()
    _metric_bars(chosen)
    st.divider()
    _per_class(chosen)
    st.divider()
    _learning_curves(chosen)
    st.divider()
    _confusion(chosen)
    st.divider()
    _exports(chosen)


def _metrics_table(records) -> None:
    st.subheader("Metrics (best per column highlighted)")
    table = comparison_service.metrics_table(records)
    numeric_cols = [c for c in table.columns if c not in ("Model",)]
    styler = table.style.highlight_max(
        subset=numeric_cols, color="rgba(34,197,94,0.35)", axis=0
    ).format({c: "{:.4f}" for c in numeric_cols if c != "Train time (s)"})
    st.dataframe(styler, width="stretch")


def _metric_bars(records) -> None:
    st.subheader("Headline metrics")
    frame = pd.DataFrame(
        {
            "experiment": [r.experiment_id for r in records],
            "Accuracy": [r.accuracy for r in records],
            "Macro F1": [r.f1_macro for r in records],
            "Precision": [r.precision_macro for r in records],
            "Recall": [r.recall_macro for r in records],
        }
    )
    metric = st.selectbox("Metric", ["Accuracy", "Macro F1", "Precision", "Recall"])
    st.altair_chart(
        charts.metric_over_experiments(frame, "experiment", metric, metric),
        width="stretch",
    )


def _per_class(records) -> None:
    st.subheader("Per-class ROC AUC")
    table = comparison_service.per_class_metric(records, "roc_auc")
    if table.empty:
        st.caption("No per-class ROC data available.")
        return
    long = (
        table.reset_index()
        .melt(id_vars="index", var_name="experiment", value_name="value")
        .rename(columns={"index": "class"})
        .dropna(subset=["value"])
    )
    st.altair_chart(
        charts.grouped_bars(long, "class", "value", "experiment", "ROC AUC by class"),
        width="stretch",
    )


def _learning_curves(records) -> None:
    st.subheader("Learning curves")
    column = st.selectbox(
        "Curve",
        ["val_loss", "train_loss", "val_acc", "train_acc"],
        format_func=lambda c: c.replace("_", " ").title(),
    )
    long = comparison_service.learning_curves(records, column)
    st.altair_chart(
        charts.multi_experiment_curves(long, column.replace("_", " ").title()),
        width="stretch",
    )


def _confusion(records) -> None:
    st.subheader("Confusion matrices")
    matrices = comparison_service.confusion_matrices(records)
    if not matrices:
        st.caption("No confusion matrices available.")
        return
    cols = st.columns(min(len(matrices), 3))
    for i, (exp_id, data) in enumerate(matrices.items()):
        with cols[i % len(cols)]:
            st.altair_chart(
                charts.confusion_heatmap(data["matrix"], data["classes"], exp_id),
                width="stretch",
            )


def _exports(records) -> None:
    st.subheader("Export comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=comparison_service.export_csv(records),
            file_name="comparison.csv",
            mime="text/csv",
            width="stretch",
        )
    with c2:
        st.download_button(
            "⬇️ Download JSON",
            data=comparison_service.export_json(records),
            file_name="comparison.json",
            mime="application/json",
            width="stretch",
        )
