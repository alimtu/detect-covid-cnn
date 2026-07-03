"""🏠 Dashboard — project overview, KPIs, live jobs and trend charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.services import config_service, device_service, experiment_service, job_manager
from frontend.services.job_models import JobState
from frontend.services.settings_service import load_settings
from frontend.ui import charts, components
from frontend.ui.components import fmt_metric


def render() -> None:
    st.title("🏠 Research Dashboard")
    st.caption("COVID-19 chest X-ray classification — experiment overview.")

    settings = load_settings()
    records = experiment_service.list_experiments()
    running_dirs = job_manager.running_experiment_dirs()

    _overview(records, settings)
    st.divider()

    active = [j for j in job_manager.list_jobs() if j.state in JobState.ACTIVE]
    if active:
        st.subheader("🚀 Training in progress")

        @st.fragment(run_every="3s")
        def _live() -> None:
            for job in job_manager.active_jobs():
                components.live_monitor(job, compact=True)
                st.divider()

        _live()

    _trend_charts(records)
    st.divider()
    _recent(records)


def _overview(records, settings) -> None:
    best = experiment_service.best_record(records, "f1_macro")
    best_acc = experiment_service.best_record(records, "accuracy")
    device = device_service.resolve_device(settings.default_device)

    components.kpi_row(
        [
            ("Total experiments", str(len(records)), None),
            ("Best experiment", best.experiment_id if best else "—", None),
            ("Best Macro F1", fmt_metric(best.f1_macro) if best else "—", None),
            ("Best Accuracy", fmt_metric(best_acc.accuracy, pct=True) if best_acc else "—", None),
        ]
    )
    components.kpi_row(
        [
            ("Last experiment", records[0].experiment_id if records else "—", None),
            ("Dataset", str(settings.default_dataset_path), None),
            ("Default model", settings.default_model, None),
            ("Device", device_service.device_label(device), None),
        ]
    )


def _history_frame(records) -> pd.DataFrame:
    """Chronological (oldest→newest) frame for trend charts."""
    ordered = list(reversed(records))  # list_experiments returns newest first
    return pd.DataFrame(
        {
            "experiment": [r.experiment_id for r in ordered],
            "model": [r.model for r in ordered],
            "accuracy": [r.accuracy for r in ordered],
            "f1_macro": [r.f1_macro for r in ordered],
            "training_time": [r.training_time_seconds for r in ordered],
        }
    )


def _trend_charts(records) -> None:
    if not records:
        st.info("No experiments yet. Head to **🚀 Train** to start your first one.")
        return
    frame = _history_frame(records)
    st.subheader("📈 Trends")
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            charts.metric_over_experiments(frame, "experiment", "accuracy", "Accuracy over experiments"),
            width="stretch",
        )
    with c2:
        st.altair_chart(
            charts.metric_over_experiments(frame, "experiment", "f1_macro", "Macro F1 over experiments"),
            width="stretch",
        )
    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(
            charts.metric_over_experiments(frame, "experiment", "training_time", "Training time (s)"),
            width="stretch",
        )
    with c4:
        model_best = (
            frame.dropna(subset=["f1_macro"])
            .groupby("model", as_index=False)["f1_macro"].max()
            .rename(columns={"f1_macro": "best_f1"})
        )
        st.altair_chart(
            charts.metric_over_experiments(model_best, "model", "best_f1", "Best Macro F1 by model"),
            width="stretch",
        )


def _recent(records) -> None:
    st.subheader("🧪 Recent experiments")
    if not records:
        st.caption("Nothing here yet.")
        return
    cols = st.columns(3)
    for i, record in enumerate(records[:6]):
        with cols[i % 3]:
            components.experiment_card(record, on_view_key=f"dash_view_{record.experiment_id}")
