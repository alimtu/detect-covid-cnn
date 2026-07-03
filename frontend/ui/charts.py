"""Altair chart builders for the dashboard (no Plotly dependency).

All helpers return ``alt.Chart`` objects that pages render with
``st.altair_chart(..., width="stretch")``.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def _empty(message: str) -> alt.Chart:
    """A tiny placeholder chart shown when there is no data."""
    frame = pd.DataFrame({"x": [0], "y": [0], "text": [message]})
    return (
        alt.Chart(frame)
        .mark_text(color="#9ca3af", size=13)
        .encode(x=alt.X("x", axis=None), y=alt.Y("y", axis=None), text="text")
        .properties(height=220)
    )


def line_over_epochs(
    history: pd.DataFrame,
    columns: dict[str, str],
    title: str = "",
) -> alt.Chart:
    """Multi-series line chart of ``columns`` (col -> legend label) over epochs."""
    present = {col: label for col, label in columns.items() if col in history.columns}
    if history.empty or not present:
        return _empty("no data yet")

    epoch = history["epoch"] if "epoch" in history.columns else pd.Series(
        range(1, len(history) + 1), name="epoch"
    )
    long = pd.DataFrame({"epoch": list(epoch)})
    frames = []
    for col, label in present.items():
        frames.append(
            pd.DataFrame({"epoch": long["epoch"], "value": history[col].values, "series": label})
        )
    data = pd.concat(frames, ignore_index=True)

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=title or "Value"),
            color=alt.Color("series:N", title=""),
            tooltip=["epoch", "series", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=260, title=title)
    )
    return chart


def metric_over_experiments(
    frame: pd.DataFrame, x: str, y: str, title: str = ""
) -> alt.Chart:
    """Bar chart of a metric per experiment."""
    if frame.empty or y not in frame.columns:
        return _empty("no experiments")
    data = frame.dropna(subset=[y])
    if data.empty:
        return _empty("no values")
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", sort=None, title="", axis=alt.Axis(labelAngle=-40)),
            y=alt.Y(f"{y}:Q", title=title or y),
            color=alt.Color(f"{y}:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=[x, alt.Tooltip(f"{y}:Q", format=".4f")],
        )
        .properties(height=260, title=title)
    )


def grouped_bars(
    data: pd.DataFrame, category: str, value: str, group: str, title: str = ""
) -> alt.Chart:
    """Grouped bar chart (e.g. per-class metric grouped by experiment)."""
    if data.empty:
        return _empty("no data")
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{category}:N", title=""),
            y=alt.Y(f"{value}:Q", title=title or value),
            color=alt.Color(f"{group}:N", title=""),
            xOffset=f"{group}:N",
            tooltip=[category, group, alt.Tooltip(f"{value}:Q", format=".4f")],
        )
        .properties(height=280, title=title)
    )


def multi_experiment_curves(long: pd.DataFrame, title: str = "") -> alt.Chart:
    """Learning curves across experiments (long format: epoch/value/experiment)."""
    if long.empty:
        return _empty("no history")
    return (
        alt.Chart(long)
        .mark_line(point=False)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=title or "Value"),
            color=alt.Color("experiment:N", title=""),
            tooltip=["experiment", "epoch", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=280, title=title)
    )


def confusion_heatmap(matrix: list[list[int]], classes: list[str], title: str = "") -> alt.Chart:
    """Heatmap of a confusion matrix."""
    rows = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            rows.append({"true": true_cls, "pred": pred_cls, "count": matrix[i][j]})
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("pred:N", title="Predicted"),
        y=alt.Y("true:N", title="Actual"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["true", "pred", "count"],
    )
    text = base.mark_text(baseline="middle", fontSize=13).encode(
        text="count:Q",
        color=alt.condition(
            alt.datum.count > (max(max(r) for r in matrix) / 2 if matrix else 0),
            alt.value("white"),
            alt.value("black"),
        ),
    )
    return (heat + text).properties(height=260, title=title)
