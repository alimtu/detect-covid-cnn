"""COVID-19 Research Dashboard — the project's primary Streamlit interface.

A multipage app that manages the full experimentation workflow — configure,
train (in a background process), monitor live, evaluate, compare and run
inference — without a terminal or hand-edited YAML. It is a thin frontend over
the existing backend in ``src`` (see ``frontend/``); the ``run.py`` and
``compare.py`` CLIs remain fully functional.

Run with:
    streamlit run app.py

IMPORTANT (background training): the app body is guarded by
``if __name__ == "__main__"``. Streamlit executes this script as ``__main__``,
so the UI builds normally; but the ``spawn`` training child re-imports this
module as ``__mp_main__``, where the guard keeps the UI from being rebuilt inside
the worker. Do not remove the guard.
"""

from __future__ import annotations

import multiprocessing


def _build_app() -> None:
    import streamlit as st

    from frontend.services import paths
    from frontend.ui import components, state, theme

    st.set_page_config(
        page_title="COVID-19 Research Dashboard",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    paths.ensure_runtime_dirs()
    theme.inject()

    # Import page modules lazily so the spawn child never pulls them in.
    from frontend.pages import (
        compare as compare_page,
        dashboard as dashboard_page,
        experiments as experiments_page,
        inference as inference_page,
        settings as settings_page,
        train as train_page,
    )

    pages = {
        "dashboard": st.Page(dashboard_page.render, title="Dashboard", icon="🏠", url_path="dashboard", default=True),
        "train": st.Page(train_page.render, title="Train", icon="🚀", url_path="train"),
        "experiments": st.Page(experiments_page.render, title="Experiments", icon="📊", url_path="experiments"),
        "compare": st.Page(compare_page.render, title="Compare", icon="📈", url_path="compare"),
        "inference": st.Page(inference_page.render, title="Inference", icon="🔍", url_path="inference"),
        "settings": st.Page(settings_page.render, title="Settings", icon="⚙️", url_path="settings"),
    }
    state.register_pages(pages)

    # Reconcile job states and surface notifications once per rerun.
    state.sync_job_notifications()

    with st.sidebar:
        st.markdown("### 🩻 COVID-19 Research")
        st.caption("Chest X-ray classification dashboard")
        st.divider()

    navigation = st.navigation(list(pages.values()), position="sidebar")

    with st.sidebar:
        st.divider()
        components.notifications_panel()

    navigation.run()


if __name__ == "__main__":
    # Spawn is required for torch/MPS safety; setting it here (guarded) means the
    # training child processes inherit a clean, fork-free start method.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. Streamlit reran the script)
    _build_app()
