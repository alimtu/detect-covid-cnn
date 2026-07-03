"""Session-state helpers, notifications and cross-page navigation.

Streamlit keeps ``st.session_state`` alive across reruns within a browser
session, so this is where we stash the Train page's working config, the
inter-page navigation intent, and the notification feed.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.services.job_models import JobState

# --------------------------------------------------------------------------- #
# Navigation registry (populated by app.py) + intent
# --------------------------------------------------------------------------- #
_PAGES: dict[str, Any] = {}


def register_pages(pages: dict[str, Any]) -> None:
    """Register ``{key: StreamlitPage}`` so pages can navigate to one another."""
    _PAGES.update(pages)


def request_page(key: str) -> None:
    """Immediately switch to a registered page by key."""
    page = _PAGES.get(key)
    if page is not None:
        st.switch_page(page)


# --------------------------------------------------------------------------- #
# Working config for the Train page
# --------------------------------------------------------------------------- #
def set_train_config(config: dict[str, Any]) -> None:
    """Load a config dict into the Train page's editable working copy."""
    st.session_state["train_config"] = config


def get_train_config() -> dict[str, Any] | None:
    return st.session_state.get("train_config")


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #
def _feed() -> list[dict[str, str]]:
    return st.session_state.setdefault("notifications", [])


def notify(message: str, icon: str = "🔔", *, toast: bool = True) -> None:
    """Append a notification to the feed and optionally show a toast."""
    _feed().insert(0, {"icon": icon, "message": message})
    del _feed()[50:]  # keep the feed bounded
    if toast:
        st.toast(message, icon=icon)


def recent_notifications(limit: int = 8) -> list[dict[str, str]]:
    return _feed()[:limit]


def clear_notifications() -> None:
    st.session_state["notifications"] = []


_ICONS = {
    JobState.PENDING: ("🟡", "queued"),
    JobState.RUNNING: ("🚀", "training started"),
    JobState.COMPLETED: ("✅", "training completed"),
    JobState.FAILED: ("❌", "training failed"),
    JobState.STOPPED: ("🛑", "training stopped"),
}


def sync_job_notifications() -> None:
    """Diff current job states against last-seen and toast on transitions.

    Called once per rerun (from the app shell). Emits notifications for job
    state changes and for the "model saved" milestone (first epoch checkpoint).
    """
    from frontend.services import job_manager

    seen: dict[str, str] = st.session_state.setdefault("_seen_job_states", {})
    for job in job_manager.list_jobs():
        previous = seen.get(job.job_id)
        if previous == job.state:
            continue
        icon, label = _ICONS.get(job.state, ("🔔", job.state))
        name = job.experiment_name or job.label or job.job_id
        # Suppress a toast on the very first sighting of an already-terminal job
        # (e.g. right after an app reload) to avoid noise.
        first_sight = previous is None
        toast = not (first_sight and job.state in JobState.TERMINAL)
        notify(f"{label.capitalize()}: {name}", icon=icon, toast=toast)
        seen[job.job_id] = job.state
