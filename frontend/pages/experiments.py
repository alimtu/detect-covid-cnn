"""📊 Experiments — browse, filter, act on and inspect every experiment."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.services import config_service, experiment_service, job_manager, paths
from frontend.services.experiment_service import ExperimentRecord
from frontend.ui import components, state


def render() -> None:
    st.title("📊 Experiments")

    selected = st.session_state.get("selected_experiment")
    if selected and (paths.OUTPUTS_ROOT / selected).is_dir():
        _detail_view(selected)
        return

    _list_view()


# --------------------------------------------------------------------------- #
# List view
# --------------------------------------------------------------------------- #
def _list_view() -> None:
    records = experiment_service.list_experiments()
    if not records:
        st.info("No experiments yet. Start one from **🚀 Train**.")
        return

    # Merge in live "running" status from active jobs.
    running = job_manager.running_experiment_dirs()
    for record in records:
        if str(record.path) in running:
            record.status = "running"

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.3])
        query = c1.text_input("Search", placeholder="id or model…").strip().lower()
        models = ["all"] + sorted({r.model for r in records})
        model_filter = c2.selectbox("Model", models)
        statuses = ["all"] + sorted({r.status for r in records})
        status_filter = c3.selectbox("Status", statuses)
        sort_key = c4.selectbox(
            "Sort by", ["Newest", "Macro F1", "Accuracy", "Training time"]
        )

    filtered = _apply_filters(records, query, model_filter, status_filter)
    filtered = _sort(filtered, sort_key)

    if not filtered:
        st.warning("No experiments match the current filters.")
        return

    # Pagination.
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, total_pages, 1, step=1)
    start = (page - 1) * page_size
    page_records = filtered[start : start + page_size]

    df = experiment_service.to_dataframe(page_records)
    event = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(f"{len(filtered)} experiment(s) · page {page}/{total_pages}")

    rows = event.selection.rows if event and event.selection else []
    if rows:
        chosen_id = df.iloc[rows[0]]["Experiment"]
        record = next(r for r in page_records if r.experiment_id == chosen_id)
        _actions(record)


def _apply_filters(records, query, model_filter, status_filter) -> list[ExperimentRecord]:
    out = []
    for record in records:
        if query and query not in record.experiment_id.lower() and query not in record.model.lower():
            continue
        if model_filter != "all" and record.model != model_filter:
            continue
        if status_filter != "all" and record.status != status_filter:
            continue
        out.append(record)
    return out


def _sort(records, sort_key) -> list[ExperimentRecord]:
    if sort_key == "Macro F1":
        return sorted(records, key=lambda r: r.f1_macro or -1, reverse=True)
    if sort_key == "Accuracy":
        return sorted(records, key=lambda r: r.accuracy or -1, reverse=True)
    if sort_key == "Training time":
        return sorted(records, key=lambda r: r.training_time_seconds or -1, reverse=True)
    return records  # already newest-first


def _actions(record: ExperimentRecord) -> None:
    st.markdown(f"#### Actions — `{record.experiment_id}`")
    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button("👁 View", width="stretch", key="act_view"):
        st.session_state["selected_experiment"] = record.experiment_id
        st.rerun()

    if c2.button("📋 Duplicate", width="stretch", key="act_dup"):
        cfg = experiment_service.load_config_dict(record.path)
        if cfg:
            state.set_train_config(config_service.clone(cfg))
            state.notify(f"Loaded config from {record.experiment_id} into Train.", icon="📋")
            state.request_page("train")
        else:
            st.error("No config.yaml found for this experiment.")

    if c3.button("📦 Export", width="stretch", key="act_export"):
        st.session_state["_export_target"] = record.experiment_id

    if c4.button("📂 Open folder", width="stretch", key="act_open"):
        ok = experiment_service.open_in_file_manager(record.path)
        st.toast("Opened in file manager." if ok else "Could not open folder.", icon="📂")

    with c5.popover("✏️ / 🗑", width="stretch"):
        _rename_delete(record)

    if st.session_state.get("_export_target") == record.experiment_id:
        archive = experiment_service.export_archive(
            record.path, paths.OUTPUTS_ROOT / ".exports"
        )
        st.download_button(
            "⬇️ Download archive (.zip)",
            data=archive.read_bytes(),
            file_name=archive.name,
            mime="application/zip",
            key="dl_archive",
        )

    st.caption(f"Path: `{record.path}`")


def _rename_delete(record: ExperimentRecord) -> None:
    st.markdown("**Rename**")
    new_name = st.text_input("New folder name", value=record.experiment_id, key="rename_input")
    if st.button("Apply rename", key="rename_apply"):
        try:
            new_path = experiment_service.rename_experiment(record.path, new_name)
            st.session_state["selected_experiment"] = new_path.name
            state.notify(f"Renamed to {new_path.name}.", icon="✏️")
            st.rerun()
        except (ValueError, FileExistsError) as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("**Delete**")
    st.caption("This permanently removes the experiment folder.")
    confirm = st.checkbox("I understand", key="del_confirm")
    if st.button("🗑 Delete experiment", key="del_apply", disabled=not confirm):
        experiment_service.delete_experiment(record.path)
        st.session_state.pop("selected_experiment", None)
        state.notify(f"Deleted {record.experiment_id}.", icon="🗑")
        st.rerun()


# --------------------------------------------------------------------------- #
# Detail view
# --------------------------------------------------------------------------- #
def _detail_view(experiment_id: str) -> None:
    if st.button("← Back to all experiments"):
        st.session_state.pop("selected_experiment", None)
        st.rerun()

    record = experiment_service.load_record(paths.OUTPUTS_ROOT / experiment_id)
    components.experiment_detail(record, key_prefix="detail")
