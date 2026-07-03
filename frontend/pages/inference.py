"""🔍 Inference — predict a chest X-ray class with one or more experiments.

Supports selecting several experiments at once and comparing their predictions
side by side (each experiment rebuilds its exact model from its saved config),
with an agreement note when they concur.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from frontend.services import experiment_service, inference_service
from frontend.services.device_service import resolve_device
from frontend.services.inference_service import PredictionResult
from frontend.services.settings_service import load_settings
from frontend.ui import theme


@st.cache_resource(show_spinner="Loading model…")
def _cached_predictor(experiment_dir: str, checkpoint: str, device_str: str):
    """Cache one predictor per (experiment, checkpoint, device)."""
    return inference_service.load_predictor(experiment_dir, checkpoint, device_str)


def _pretty(label: str) -> str:
    return label.replace("_", " ")


def render() -> None:
    theme.persian_header()
    st.title("🔍 Inference")

    settings = load_settings()
    records = [
        r for r in experiment_service.list_experiments()
        if experiment_service.list_checkpoints(r.path)
    ]
    if not records:
        st.warning("No trained experiments with checkpoints found. Train one first from **🚀 Train**.")
        return

    by_id = {r.experiment_id: r for r in records}
    ids = list(by_id)

    c1, c2 = st.columns([2, 1])
    default = ids[: min(2, len(ids))]
    selected_ids = c1.multiselect("Experiments to compare", ids, default=default)
    checkpoint = c2.selectbox("Checkpoint", ["best_model.pth", "last_model.pth"])

    if not selected_ids:
        st.info("Select at least one experiment.")
        return

    device = resolve_device(settings.default_device)
    with st.sidebar:
        st.markdown("#### Model info")
        for exp_id in selected_ids:
            info = inference_service.model_info(by_id[exp_id].path)
            with st.expander(exp_id, expanded=len(selected_ids) == 1):
                for key, value in info.items():
                    st.write(f"**{key}:** {value if value is not None else '—'}")
        st.write(f"**inference device:** `{device}`")

    uploaded = st.file_uploader(
        "Upload a chest X-ray image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("Awaiting an image upload.")
        return

    image = Image.open(uploaded)
    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Uploaded image", width="stretch")

    with right:
        results: dict[str, PredictionResult] = {}
        for exp_id in selected_ids:
            record = by_id[exp_id]
            available = experiment_service.list_checkpoints(record.path)
            ckpt = checkpoint if checkpoint in available else available[0]
            try:
                predictor = _cached_predictor(str(record.path), ckpt, str(device))
            except FileNotFoundError as exc:
                st.warning(f"**{exp_id}**: {exc}. Skipping.")
                continue
            except Exception as exc:  # noqa: BLE001 - surface load errors in the UI
                st.error(f"Failed to load **{exp_id}**: {exc}")
                continue
            results[exp_id] = inference_service.predict_image(predictor, image)

        if not results:
            return

        cols = st.columns(len(results))
        for col, (exp_id, result) in zip(cols, results.items()):
            with col:
                st.subheader(exp_id)
                _render_result(result)

    _agreement_note(results)


def _render_result(result: PredictionResult) -> None:
    st.metric(
        label="Prediction",
        value=_pretty(result.top_class),
        delta=f"{result.top_confidence * 100:.1f}% confidence",
    )
    st.caption(f"model: `{result.model_name}` · inference {result.inference_ms:.1f} ms")
    st.markdown("**Per-class probability**")
    for class_name, prob in result.probabilities.items():
        st.write(f"{_pretty(class_name)} — {prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))


def _agreement_note(results: dict[str, PredictionResult]) -> None:
    """Show whether the selected models agree on the top class."""
    if len(results) < 2:
        return
    tops = {exp_id: result.top_class for exp_id, result in results.items()}
    if len(set(tops.values())) == 1:
        st.success(f"✅ All {len(results)} models agree: **{_pretty(next(iter(tops.values())))}**.")
    else:
        details = " · ".join(f"{exp_id} → {_pretty(cls)}" for exp_id, cls in tops.items())
        st.warning(f"⚠️ Models disagree: {details}")
