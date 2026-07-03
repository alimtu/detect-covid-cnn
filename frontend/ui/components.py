"""Reusable Streamlit UI components shared across pages.

These render backend/service data; they hold no business logic themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from frontend.services import experiment_service, report_service
from frontend.services.experiment_service import ExperimentRecord
from frontend.services.job_models import JobRecord, JobState
from frontend.services.monitor_service import LiveProgress, compute_progress
from frontend.ui import charts, state, theme


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def kpi_row(items: list[tuple[str, str, Optional[str]]]) -> None:
    """Render a row of ``st.metric`` tiles: ``(label, value, delta)``."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def fmt_metric(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if pct else f"{value:.4f}"


def fmt_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def notifications_panel() -> None:
    """Render the recent-notifications feed (typically in the sidebar)."""
    notes = state.recent_notifications()
    st.markdown("#### 🔔 Notifications")
    if not notes:
        st.caption("No notifications yet.")
        return
    for note in notes:
        st.write(f"{note['icon']} {note['message']}")
    if st.button("Clear", key="clear_notifs", width="stretch"):
        state.clear_notifications()
        st.rerun()


# --------------------------------------------------------------------------- #
# Experiment cards
# --------------------------------------------------------------------------- #
def experiment_card(record: ExperimentRecord, *, on_view_key: str | None = None) -> None:
    """Render a compact experiment summary card with an optional View button."""
    theme.inject()
    st.markdown(
        f"""
<div class="exp-card">
  <div class="title">{record.experiment_id} {theme.status_pill(record.status)}</div>
  <div class="sub">{record.model} · {experiment_service._short_date(record.created_at)}</div>
  <div>Acc <strong>{fmt_metric(record.accuracy)}</strong> ·
       F1 <strong>{fmt_metric(record.f1_macro)}</strong> ·
       {fmt_duration(record.training_time_seconds)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if on_view_key and st.button("View", key=on_view_key, width="stretch"):
        st.session_state["selected_experiment"] = record.experiment_id
        state.request_page("experiments")


# --------------------------------------------------------------------------- #
# Live training monitor
# --------------------------------------------------------------------------- #
def live_monitor(job: JobRecord, *, compact: bool = False) -> LiveProgress:
    """Render the live training monitor for a job and return its progress.

    Displays current epoch, progress bar, key metrics, ETA and live charts. The
    charts read the per-epoch history CSV, so they update as the worker writes
    each epoch.
    """
    progress = compute_progress(job)

    header = f"**{job.experiment_name or job.label}** · {job.model}"
    st.markdown(header)

    if progress.total_epochs:
        st.progress(
            progress.fraction,
            text=f"Epoch {progress.epochs_completed}/{progress.total_epochs}",
        )
    else:
        st.progress(0.0, text="Starting…")

    latest = progress.latest
    kpi_row(
        [
            ("Epoch", f"{progress.epochs_completed}/{progress.total_epochs or '—'}", None),
            ("Train loss", fmt_metric(latest.get("train_loss")), None),
            ("Val loss", fmt_metric(latest.get("val_loss")), None),
            ("Val acc", fmt_metric(latest.get("val_acc"), pct=True), None),
        ]
    )
    kpi_row(
        [
            ("Learning rate", _fmt_lr(latest.get("lr")), None),
            ("Best val acc", fmt_metric(progress.best_val_acc, pct=True), None),
            ("Elapsed", fmt_duration(progress.elapsed_seconds), None),
            ("ETA", fmt_duration(progress.eta_seconds), None),
        ]
    )

    if not compact and not progress.history.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                charts.line_over_epochs(
                    progress.history,
                    {"train_loss": "train", "val_loss": "val"},
                    title="Loss",
                ),
                width="stretch",
            )
        with c2:
            st.altair_chart(
                charts.line_over_epochs(
                    progress.history,
                    {"train_acc": "train", "val_acc": "val"},
                    title="Accuracy",
                ),
                width="stretch",
            )
        st.altair_chart(
            charts.line_over_epochs(progress.history, {"lr": "learning rate"}, title="Learning rate"),
            width="stretch",
        )
    return progress


def _fmt_lr(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2e}"


# --------------------------------------------------------------------------- #
# Experiment detail / completion view
# --------------------------------------------------------------------------- #
def experiment_detail(record: ExperimentRecord, *, key_prefix: str = "") -> None:
    """Render the full detail view: metrics, plots, report, downloads."""
    metrics = experiment_service.load_metrics(record.path)

    st.subheader(f"{record.experiment_id}")
    st.caption(
        f"{record.model} · {experiment_service._short_date(record.created_at)} · "
        f"{(record.device or '—').upper()} · trained in {fmt_duration(record.training_time_seconds)}"
    )

    kpi_row(
        [
            ("Accuracy", fmt_metric(record.accuracy, pct=True), None),
            ("Precision", fmt_metric(record.precision_macro), None),
            ("Recall", fmt_metric(record.recall_macro), None),
            ("Macro F1", fmt_metric(record.f1_macro), None),
        ]
    )

    tab_plots, tab_curves, tab_report, tab_downloads = st.tabs(
        ["Evaluation", "Training history", "Report", "Downloads"]
    )

    with tab_plots:
        _plot_grid(record.path)
        report_text = experiment_service.classification_report(record.path)
        if report_text:
            with st.expander("Classification report"):
                st.code(report_text)

    with tab_curves:
        history = experiment_service.load_history(record.path)
        if history.empty:
            st.info("No training history recorded.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(
                    charts.line_over_epochs(
                        history, {"train_loss": "train", "val_loss": "val"}, title="Loss"
                    ),
                    width="stretch",
                )
            with c2:
                st.altair_chart(
                    charts.line_over_epochs(
                        history, {"train_acc": "train", "val_acc": "val"}, title="Accuracy"
                    ),
                    width="stretch",
                )
            st.dataframe(history, width="stretch", hide_index=True)

    with tab_report:
        _report_section(record.path, key_prefix)

    with tab_downloads:
        _downloads_section(record, key_prefix)


def _plot_grid(exp_dir: Path) -> None:
    grid = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("ROC Curves", "roc_curve.png"),
        ("PR Curves", "pr_curve.png"),
        ("Loss", "loss.png"),
        ("Accuracy", "accuracy.png"),
    ]
    available = [(title, p) for title, fn in grid if (p := experiment_service.plot_path(exp_dir, fn))]
    if not available:
        st.info("No plots available for this experiment.")
        return
    cols = st.columns(2)
    for i, (title, path) in enumerate(available):
        with cols[i % 2]:
            st.image(str(path), caption=title, width="stretch")


def _report_section(exp_dir: Path, key_prefix: str) -> None:
    st.write("Generate a self-contained report (HTML embeds all plots; open the HTML and print to PDF).")
    if st.button("Generate report", key=f"{key_prefix}_gen_report"):
        paths_out = report_service.write_reports(exp_dir)
        st.success(f"Report written to {paths_out['html'].name} and {paths_out['markdown'].name}.")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download HTML",
            data=report_service.build_html(exp_dir),
            file_name=f"{exp_dir.name}_report.html",
            mime="text/html",
            width="stretch",
            key=f"{key_prefix}_dl_html",
        )
    with col2:
        st.download_button(
            "⬇️ Download Markdown",
            data=report_service.build_markdown(exp_dir),
            file_name=f"{exp_dir.name}_report.md",
            mime="text/markdown",
            width="stretch",
            key=f"{key_prefix}_dl_md",
        )


def _downloads_section(record: ExperimentRecord, key_prefix: str) -> None:
    exp_dir = record.path
    ckpt_dir = exp_dir / "checkpoints"
    cols = st.columns(2)

    for i, ckpt in enumerate(experiment_service.list_checkpoints(exp_dir)):
        path = ckpt_dir / ckpt
        with cols[i % 2]:
            st.download_button(
                f"⬇️ {ckpt}",
                data=path.read_bytes(),
                file_name=f"{exp_dir.name}_{ckpt}",
                mime="application/octet-stream",
                width="stretch",
                key=f"{key_prefix}_ckpt_{i}",
            )

    metrics_path = exp_dir / "metrics.json"
    config_path = exp_dir / "config.yaml"
    with cols[0]:
        if metrics_path.is_file():
            st.download_button(
                "⬇️ metrics.json",
                data=metrics_path.read_bytes(),
                file_name=f"{exp_dir.name}_metrics.json",
                mime="application/json",
                width="stretch",
                key=f"{key_prefix}_dl_metrics",
            )
    with cols[1]:
        if config_path.is_file():
            st.download_button(
                "⬇️ config.yaml",
                data=config_path.read_bytes(),
                file_name=f"{exp_dir.name}_config.yaml",
                mime="text/yaml",
                width="stretch",
                key=f"{key_prefix}_dl_config",
            )
