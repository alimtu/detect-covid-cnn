"""⚙️ Settings — edit project defaults used when seeding new experiments."""

from __future__ import annotations

import streamlit as st

from frontend.services import config_service, device_service
from frontend.services.settings_service import AppSettings, load_settings, save_settings


def render() -> None:
    st.title("⚙️ Settings")
    st.caption("These defaults pre-fill a fresh configuration on the Train page.")

    settings = load_settings()
    devices = device_service.available_devices()

    with st.form("settings_form"):
        st.subheader("Data & experiments")
        c1, c2 = st.columns(2)
        dataset_path = c1.text_input("Default dataset path", settings.default_dataset_path)
        output_root = c2.text_input("Default experiment directory", settings.default_output_root)
        image_size = c1.number_input("Default image size", 32, 1024, int(settings.default_image_size), step=32)
        device = c2.selectbox(
            "Default device", devices, index=_index(devices, settings.default_device)
        )

        st.subheader("Model & training defaults")
        c3, c4 = st.columns(2)
        model = c3.selectbox(
            "Default model", config_service.MODELS,
            index=_index(config_service.MODELS, settings.default_model),
            format_func=lambda m: config_service.MODEL_LABELS[m],
        )
        optimizer = c4.selectbox(
            "Default optimizer", config_service.OPTIMIZERS,
            index=_index(config_service.OPTIMIZERS, settings.default_optimizer),
            format_func=lambda o: config_service.OPTIMIZER_LABELS[o],
        )
        scheduler = c3.selectbox(
            "Default scheduler", config_service.SCHEDULERS,
            index=_index(config_service.SCHEDULERS, settings.default_scheduler),
            format_func=lambda s: config_service.SCHEDULER_LABELS[s],
        )
        epochs = c4.number_input("Default epochs", 1, 500, int(settings.default_epochs))
        batch = c3.number_input("Default batch size", 1, 512, int(settings.default_batch_size))

        st.subheader("Interface")
        c5, c6 = st.columns(2)
        theme_choice = c5.selectbox("Theme", ["dark", "light"], index=0 if settings.theme == "dark" else 1)
        auto_save = c6.toggle("Auto-save config on training start", value=settings.auto_save_config)

        submitted = st.form_submit_button("💾 Save settings", type="primary")

    if submitted:
        save_settings(
            AppSettings(
                default_dataset_path=dataset_path,
                default_output_root=output_root,
                default_device=device,
                default_image_size=int(image_size),
                default_optimizer=optimizer,
                default_scheduler=scheduler,
                default_model=model,
                default_epochs=int(epochs),
                default_batch_size=int(batch),
                theme=theme_choice,
                auto_save_config=bool(auto_save),
            )
        )
        st.success("Settings saved.")

    st.divider()
    st.caption(
        "Theme also follows Streamlit's own setting (☰ → Settings). "
        "The value here is stored for reference and future use."
    )


def _index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0
