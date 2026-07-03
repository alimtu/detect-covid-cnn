"""🚀 Train — configure every hyperparameter graphically and launch training.

Replaces manual YAML editing with widgets. The page edits an in-memory config
dict (seeded from ``base.yaml`` + project defaults), previews the generated YAML,
and submits a background training job via the process-based job manager. A live
monitor tracks the run without freezing the UI.
"""

from __future__ import annotations

import streamlit as st

from frontend.services import config_service, experiment_service, job_manager, paths
from frontend.services.config_service import get_in, set_in
from frontend.services.job_models import JobState
from frontend.services.settings_service import load_settings
from frontend.ui import components, state


def render() -> None:
    st.title("🚀 Train")

    # A job launched from this page takes over the view until it finishes.
    current = st.session_state.get("current_job_id")
    if current:
        job = job_manager.get_job(current)
        if job and job.state in JobState.ACTIVE:
            _monitor_view(current)
            return
        if job and job.state in JobState.TERMINAL:
            _completion_view(job)
            return

    _config_view()


# --------------------------------------------------------------------------- #
# Configuration editor
# --------------------------------------------------------------------------- #
def _config_view() -> None:
    cfg = state.get_train_config()
    if cfg is None:
        cfg = config_service.default_config(load_settings())
        state.set_train_config(cfg)

    _config_toolbar(cfg)
    st.divider()

    _dataset_section(cfg)
    _augmentation_section(cfg)
    _model_section(cfg)
    _optimizer_section(cfg)
    _scheduler_section(cfg)
    _loss_section(cfg)
    _training_section(cfg)
    _logging_eval_section(cfg)

    st.divider()
    _preview_and_launch(cfg)


def _config_toolbar(cfg: dict) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        presets = config_service.list_saved_configs()
        labels = ["— load a config —"] + [str(p.relative_to(paths.PROJECT_ROOT)) for p in presets]
        choice = st.selectbox("Load existing config", labels, key="load_preset")
        if choice != labels[0]:
            selected = presets[labels.index(choice) - 1]
            if st.button("Load selected", key="do_load_preset"):
                state.set_train_config(config_service.load_config_file(selected))
                state.notify(f"Loaded {selected.name}.", icon="📂")
                st.rerun()
    with c2:
        uploaded = st.file_uploader("Upload YAML", type=["yaml", "yml"], key="upload_cfg")
        if uploaded is not None and st.button("Use uploaded", key="do_upload"):
            state.set_train_config(config_service.load_config_bytes(uploaded.getvalue()))
            state.notify("Loaded uploaded config.", icon="📂")
            st.rerun()
    with c3:
        st.write("")
        st.write("")
        if st.button("↺ Reset to defaults", width="stretch", key="reset_cfg"):
            state.set_train_config(config_service.default_config(load_settings()))
            st.rerun()


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _dataset_section(cfg: dict) -> None:
    with st.expander("📁 Dataset", expanded=True):
        c1, c2, c3 = st.columns(3)
        set_in(cfg, "dataset.path", c1.text_input("Dataset path", get_in(cfg, "dataset.path", "DATA")))
        set_in(cfg, "dataset.image_size",
               int(c2.number_input("Image size", 32, 1024, int(get_in(cfg, "dataset.image_size", 224)), step=32)))
        set_in(cfg, "reproducibility.seed",
               int(c3.number_input("Random seed", 0, 10_000, int(get_in(cfg, "reproducibility.seed", 42)))))

        c4, c5, c6 = st.columns(3)
        use_all = c4.toggle("Use all samples", value=(get_in(cfg, "dataset.sample_per_class") in ("all", None)))
        if use_all:
            set_in(cfg, "dataset.sample_per_class", "all")
            c5.caption("Using every image per class.")
        else:
            current = get_in(cfg, "dataset.sample_per_class")
            current = 50 if current in ("all", None) else int(current)
            set_in(cfg, "dataset.sample_per_class",
                   int(c5.number_input("Samples per class", 1, 100_000, current)))
        set_in(cfg, "dataset.shuffle", c6.toggle("Shuffle training data", value=bool(get_in(cfg, "dataset.shuffle", True))))

        st.markdown("**Split ratios** (must sum to 1.0)")
        s1, s2, s3 = st.columns(3)
        set_in(cfg, "dataset.split.train",
               float(s1.number_input("Train", 0.0, 1.0, float(get_in(cfg, "dataset.split.train", 0.7)), step=0.05)))
        set_in(cfg, "dataset.split.val",
               float(s2.number_input("Validation", 0.0, 1.0, float(get_in(cfg, "dataset.split.val", 0.15)), step=0.05)))
        set_in(cfg, "dataset.split.test",
               float(s3.number_input("Test", 0.0, 1.0, float(get_in(cfg, "dataset.split.test", 0.15)), step=0.05)))
        total = sum(float(get_in(cfg, f"dataset.split.{k}", 0)) for k in ("train", "val", "test"))
        (st.success if abs(total - 1.0) < 1e-6 else st.error)(f"Split total: {total:.2f}")

        c7, c8 = st.columns(2)
        set_in(cfg, "dataset.num_workers",
               int(c7.number_input("Data loader workers", 0, 16, int(get_in(cfg, "dataset.num_workers", 4)))))
        set_in(cfg, "dataset.pin_memory", c8.toggle("Pin memory", value=bool(get_in(cfg, "dataset.pin_memory", False))))


def _augmentation_section(cfg: dict) -> None:
    with st.expander("🎛 Data Augmentation"):
        _aug_toggle(cfg, "random_horizontal_flip", "Horizontal Flip", prob=True)
        _aug_toggle(cfg, "random_vertical_flip", "Vertical Flip", prob=True)
        _aug_rotation(cfg)
        _aug_affine(cfg)
        _aug_resized_crop(cfg)
        _aug_random_crop(cfg)
        _aug_color_jitter(cfg)
        _aug_blur(cfg)
        _aug_normalize(cfg)


def _aug_header(cfg: dict, key: str, label: str) -> bool:
    enabled = st.checkbox(label, value=bool(get_in(cfg, f"augmentation.{key}.enabled", False)), key=f"aug_{key}")
    set_in(cfg, f"augmentation.{key}.enabled", enabled)
    return enabled


def _aug_prob(cfg: dict, key: str, col) -> None:
    set_in(cfg, f"augmentation.{key}.p",
           float(col.slider("Probability", 0.0, 1.0, float(get_in(cfg, f"augmentation.{key}.p", 0.5)), step=0.05, key=f"aug_{key}_p")))


def _aug_toggle(cfg: dict, key: str, label: str, prob: bool = False) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        enabled = _aug_header(cfg, key, label)
    if enabled and prob:
        _aug_prob(cfg, key, c2)


def _aug_rotation(cfg: dict) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        enabled = _aug_header(cfg, "random_rotation", "Rotation")
    if enabled:
        set_in(cfg, "augmentation.random_rotation.degrees",
               float(c2.number_input("Degrees", 0.0, 180.0, float(get_in(cfg, "augmentation.random_rotation.degrees", 10)), key="rot_deg")))
        _aug_prob(cfg, "random_rotation", c3)


def _aug_affine(cfg: dict) -> None:
    with st.container():
        enabled = _aug_header(cfg, "random_affine", "Affine")
        if enabled:
            c1, c2, c3, c4 = st.columns(4)
            set_in(cfg, "augmentation.random_affine.degrees",
                   float(c1.number_input("Degrees", 0.0, 180.0, float(get_in(cfg, "augmentation.random_affine.degrees", 0)), key="aff_deg")))
            set_in(cfg, "augmentation.random_affine.shear",
                   float(c2.number_input("Shear", 0.0, 90.0, float(get_in(cfg, "augmentation.random_affine.shear", 0)), key="aff_shear")))
            tr = get_in(cfg, "augmentation.random_affine.translate", [0.1, 0.1])
            set_in(cfg, "augmentation.random_affine.translate",
                   [float(c3.number_input("Translate x", 0.0, 1.0, float(tr[0]), step=0.05, key="aff_tx")),
                    float(c3.number_input("Translate y", 0.0, 1.0, float(tr[1]), step=0.05, key="aff_ty"))])
            sc = get_in(cfg, "augmentation.random_affine.scale", [0.9, 1.1])
            set_in(cfg, "augmentation.random_affine.scale",
                   [float(c4.number_input("Scale min", 0.1, 2.0, float(sc[0]), step=0.05, key="aff_smin")),
                    float(c4.number_input("Scale max", 0.1, 3.0, float(sc[1]), step=0.05, key="aff_smax"))])
            _aug_prob(cfg, "random_affine", c1)


def _aug_resized_crop(cfg: dict) -> None:
    enabled = _aug_header(cfg, "random_resized_crop", "Random Resized Crop")
    if enabled:
        c1, c2 = st.columns(2)
        sc = get_in(cfg, "augmentation.random_resized_crop.scale", [0.8, 1.0])
        set_in(cfg, "augmentation.random_resized_crop.scale",
               [float(c1.number_input("Scale min", 0.05, 1.0, float(sc[0]), step=0.05, key="rrc_smin")),
                float(c2.number_input("Scale max", 0.1, 1.0, float(sc[1]), step=0.05, key="rrc_smax"))])


def _aug_random_crop(cfg: dict) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        enabled = _aug_header(cfg, "random_crop", "Random Crop")
    if enabled:
        set_in(cfg, "augmentation.random_crop.padding",
               int(c2.number_input("Padding", 0, 64, int(get_in(cfg, "augmentation.random_crop.padding", 4)), key="rc_pad")))


def _aug_color_jitter(cfg: dict) -> None:
    enabled = _aug_header(cfg, "color_jitter", "Color Jitter")
    if enabled:
        c1, c2, c3, c4 = st.columns(4)
        for col, param in zip((c1, c2, c3, c4), ("brightness", "contrast", "saturation", "hue")):
            set_in(cfg, f"augmentation.color_jitter.{param}",
                   float(col.number_input(param.title(), 0.0, 1.0, float(get_in(cfg, f"augmentation.color_jitter.{param}", 0.0)), step=0.05, key=f"cj_{param}")))
        _aug_prob(cfg, "color_jitter", c1)


def _aug_blur(cfg: dict) -> None:
    enabled = _aug_header(cfg, "gaussian_blur", "Gaussian Blur")
    if enabled:
        c1, c2, c3 = st.columns(3)
        set_in(cfg, "augmentation.gaussian_blur.kernel_size",
               int(c1.number_input("Kernel size (odd)", 1, 31, int(get_in(cfg, "augmentation.gaussian_blur.kernel_size", 3)), step=2, key="gb_k")))
        sg = get_in(cfg, "augmentation.gaussian_blur.sigma", [0.1, 2.0])
        set_in(cfg, "augmentation.gaussian_blur.sigma",
               [float(c2.number_input("Sigma min", 0.0, 10.0, float(sg[0]), step=0.1, key="gb_smin")),
                float(c3.number_input("Sigma max", 0.0, 10.0, float(sg[1]), step=0.1, key="gb_smax"))])
        _aug_prob(cfg, "gaussian_blur", c1)


def _aug_normalize(cfg: dict) -> None:
    enabled = _aug_header(cfg, "normalize", "Normalize (ImageNet stats)")
    set_in(cfg, "augmentation.normalize.enabled", enabled)
    if enabled:
        st.caption("Uses ImageNet mean/std by default (recommended for pretrained backbones).")


def _model_section(cfg: dict) -> None:
    with st.expander("🧠 Model", expanded=True):
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox(
            "Architecture", config_service.MODELS,
            index=config_service.MODELS.index(get_in(cfg, "model.name", "densenet121")),
            format_func=lambda m: config_service.MODEL_LABELS[m],
        )
        set_in(cfg, "model.name", model)
        set_in(cfg, "model.pretrained", c2.toggle("Pretrained weights", value=bool(get_in(cfg, "model.pretrained", True))))
        set_in(cfg, "model.freeze_backbone", c3.toggle("Freeze backbone", value=bool(get_in(cfg, "model.freeze_backbone", False))))

        c4, c5, c6 = st.columns(3)
        set_in(cfg, "model.dropout",
               float(c4.slider("Dropout", 0.0, 0.9, float(get_in(cfg, "model.dropout", 0.0) or 0.0), step=0.05)))
        _optional_int(cfg, "model.classifier_hidden", c5, "Hidden layer size", 16, 4096, 512)
        _optional_int(cfg, "model.unfreeze_after_epoch", c6, "Unfreeze after epoch", 1, 500, 3)


def _optimizer_section(cfg: dict) -> None:
    with st.expander("⚙️ Optimizer", expanded=True):
        c1, c2, c3 = st.columns(3)
        opt = c1.selectbox(
            "Optimizer", config_service.OPTIMIZERS,
            index=config_service.OPTIMIZERS.index(get_in(cfg, "optimizer.name", "adamw")),
            format_func=lambda o: config_service.OPTIMIZER_LABELS[o],
        )
        set_in(cfg, "optimizer.name", opt)
        set_in(cfg, "optimizer.learning_rate",
               float(c2.number_input("Learning rate", 1e-6, 1.0, float(get_in(cfg, "optimizer.learning_rate", 1e-4)), format="%.6f", step=1e-5)))
        set_in(cfg, "optimizer.weight_decay",
               float(c3.number_input("Weight decay", 0.0, 1.0, float(get_in(cfg, "optimizer.weight_decay", 1e-4)), format="%.6f", step=1e-5)))

        if opt == "sgd":
            c4, c5 = st.columns(2)
            set_in(cfg, "optimizer.momentum",
                   float(c4.number_input("Momentum", 0.0, 1.0, float(get_in(cfg, "optimizer.momentum", 0.9)), step=0.05)))
            set_in(cfg, "optimizer.nesterov", c5.toggle("Nesterov", value=bool(get_in(cfg, "optimizer.nesterov", False))))
        else:
            betas = get_in(cfg, "optimizer.betas", [0.9, 0.999])
            c4, c5 = st.columns(2)
            set_in(cfg, "optimizer.betas",
                   [float(c4.number_input("Beta 1", 0.0, 1.0, float(betas[0]), step=0.01)),
                    float(c5.number_input("Beta 2", 0.0, 1.0, float(betas[1]), format="%.3f", step=0.001))])


def _scheduler_section(cfg: dict) -> None:
    with st.expander("📉 Scheduler"):
        sched = st.selectbox(
            "Scheduler", config_service.SCHEDULERS,
            index=config_service.SCHEDULERS.index(get_in(cfg, "scheduler.name", "reduce_on_plateau")),
            format_func=lambda s: config_service.SCHEDULER_LABELS[s],
        )
        set_in(cfg, "scheduler.name", sched)

        if sched == "reduce_on_plateau":
            c1, c2 = st.columns(2)
            set_in(cfg, "scheduler.reduce_on_plateau.factor",
                   float(c1.number_input("Factor", 0.01, 1.0, float(get_in(cfg, "scheduler.reduce_on_plateau.factor", 0.1)), step=0.05)))
            set_in(cfg, "scheduler.reduce_on_plateau.patience",
                   int(c2.number_input("Patience", 1, 50, int(get_in(cfg, "scheduler.reduce_on_plateau.patience", 2)))))
        elif sched == "cosine":
            c1, c2 = st.columns(2)
            set_in(cfg, "scheduler.cosine.t_max",
                   int(c1.number_input("T max", 1, 500, int(get_in(cfg, "scheduler.cosine.t_max", 50)))))
            set_in(cfg, "scheduler.cosine.eta_min",
                   float(c2.number_input("Eta min", 0.0, 1.0, float(get_in(cfg, "scheduler.cosine.eta_min", 0.0)), format="%.6f", step=1e-5)))
        elif sched == "step":
            c1, c2 = st.columns(2)
            set_in(cfg, "scheduler.step.step_size",
                   int(c1.number_input("Step size", 1, 200, int(get_in(cfg, "scheduler.step.step_size", 10)))))
            set_in(cfg, "scheduler.step.gamma",
                   float(c2.number_input("Gamma", 0.01, 1.0, float(get_in(cfg, "scheduler.step.gamma", 0.1)), step=0.05)))
        elif sched == "onecycle":
            c1, c2 = st.columns(2)
            set_in(cfg, "scheduler.onecycle.max_lr",
                   float(c1.number_input("Max LR", 1e-5, 1.0, float(get_in(cfg, "scheduler.onecycle.max_lr", 0.01)), format="%.5f", step=1e-4)))
            set_in(cfg, "scheduler.onecycle.pct_start",
                   float(c2.number_input("Pct start", 0.0, 1.0, float(get_in(cfg, "scheduler.onecycle.pct_start", 0.3)), step=0.05)))


def _loss_section(cfg: dict) -> None:
    with st.expander("🎯 Loss"):
        loss = st.selectbox(
            "Loss function", config_service.LOSSES,
            index=config_service.LOSSES.index(get_in(cfg, "loss.name", "cross_entropy")),
            format_func=lambda l: config_service.LOSS_LABELS[l],
        )
        set_in(cfg, "loss.name", loss)

        if loss in ("weighted_cross_entropy", "focal"):
            weights = st.selectbox(
                "Class weights", ["none", "balanced"],
                index=0 if get_in(cfg, "loss.class_weights") in (None, "none") else 1,
            )
            set_in(cfg, "loss.class_weights", None if weights == "none" else "balanced")
        if loss == "focal":
            c1, c2 = st.columns(2)
            set_in(cfg, "loss.focal.gamma",
                   float(c1.number_input("Focal gamma", 0.0, 10.0, float(get_in(cfg, "loss.focal.gamma", 2.0)), step=0.5)))


def _training_section(cfg: dict) -> None:
    with st.expander("🏋️ Training", expanded=True):
        c1, c2, c3 = st.columns(3)
        set_in(cfg, "training.epochs",
               int(c1.number_input("Epochs", 1, 500, int(get_in(cfg, "training.epochs", 5)))))
        set_in(cfg, "training.batch_size",
               int(c2.number_input("Batch size", 1, 512, int(get_in(cfg, "training.batch_size", 32)))))
        set_in(cfg, "training.gradient_accumulation_steps",
               int(c3.number_input("Gradient accumulation", 1, 64, int(get_in(cfg, "training.gradient_accumulation_steps", 1)))))

        c4, c5, c6 = st.columns(3)
        _optional_float(cfg, "training.gradient_clip", c4, "Gradient clip (max-norm)", 0.1, 100.0, 1.0)
        set_in(cfg, "training.mixed_precision",
               c5.toggle("Mixed precision (CUDA only)", value=bool(get_in(cfg, "training.mixed_precision", False))))
        _optional_int(cfg, "training.checkpoint.save_every_n_epochs", c6, "Save every N epochs", 1, 100, 5)

        st.markdown("**Early stopping**")
        e1, e2, e3, e4 = st.columns(4)
        set_in(cfg, "training.early_stopping.enabled",
               e1.toggle("Enabled", value=bool(get_in(cfg, "training.early_stopping.enabled", True)), key="es_en"))
        monitor = e2.selectbox("Monitor", config_service.MONITORS,
                               index=config_service.MONITORS.index(get_in(cfg, "training.early_stopping.monitor", "val_loss")))
        set_in(cfg, "training.early_stopping.monitor", monitor)
        set_in(cfg, "training.early_stopping.patience",
               int(e3.number_input("Patience", 1, 100, int(get_in(cfg, "training.early_stopping.patience", 5)), key="es_pat")))
        set_in(cfg, "training.early_stopping.min_delta",
               float(e4.number_input("Min delta", 0.0, 1.0, float(get_in(cfg, "training.early_stopping.min_delta", 0.0)), format="%.4f", step=1e-3, key="es_md")))

        st.markdown("**Checkpoints**")
        k1, k2, k3 = st.columns(3)
        set_in(cfg, "training.checkpoint.save_best", k1.toggle("Save best", value=bool(get_in(cfg, "training.checkpoint.save_best", True))))
        set_in(cfg, "training.checkpoint.save_last", k2.toggle("Save last", value=bool(get_in(cfg, "training.checkpoint.save_last", True))))
        ckpt_monitor = k3.selectbox("Checkpoint monitor", config_service.MONITORS,
                                    index=config_service.MONITORS.index(get_in(cfg, "training.checkpoint.monitor", "val_loss")),
                                    key="ckpt_mon")
        set_in(cfg, "training.checkpoint.monitor", ckpt_monitor)


def _logging_eval_section(cfg: dict) -> None:
    with st.expander("📝 Logging & Evaluation"):
        st.markdown("**Logging sinks**")
        l1, l2, l3 = st.columns(3)
        set_in(cfg, "logging.console", l1.checkbox("Console", value=bool(get_in(cfg, "logging.console", True))))
        set_in(cfg, "logging.csv", l2.checkbox("CSV", value=bool(get_in(cfg, "logging.csv", True))))
        set_in(cfg, "logging.tensorboard", l3.checkbox("TensorBoard", value=bool(get_in(cfg, "logging.tensorboard", False))))

        st.markdown("**Evaluation metrics**")
        metrics = [
            ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
            ("f1", "Macro F1"), ("confusion_matrix", "Confusion Matrix"),
            ("classification_report", "Classification Report"),
            ("roc_curve", "ROC Curve"), ("pr_curve", "PR Curve"),
        ]
        cols = st.columns(4)
        for i, (key, label) in enumerate(metrics):
            with cols[i % 4]:
                set_in(cfg, f"evaluation.metrics.{key}",
                       st.checkbox(label, value=bool(get_in(cfg, f"evaluation.metrics.{key}", True)), key=f"eval_{key}"))


# --------------------------------------------------------------------------- #
# Preview + launch
# --------------------------------------------------------------------------- #
def _preview_and_launch(cfg: dict) -> None:
    st.subheader("🧾 Configuration preview")
    ok, message = config_service.validate(cfg)
    if not ok:
        st.error(f"Invalid configuration: {message}")

    with st.expander("Show generated YAML", expanded=False):
        st.code(config_service.to_yaml(cfg), language="yaml")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Download config",
            data=config_service.to_yaml(cfg),
            file_name="config.yaml",
            mime="text/yaml",
            width="stretch",
        )
    with c2:
        with st.popover("💾 Save config", width="stretch"):
            name = st.text_input("Name", value=f"{get_in(cfg, 'model.name', 'model')}_custom")
            if st.button("Save", key="save_cfg_btn"):
                path = config_service.save_config(cfg, name)
                state.notify(f"Saved config to {path.name}.", icon="💾")
                st.success(f"Saved to {path}")
    with c3:
        if st.button("📋 Duplicate as new", width="stretch"):
            state.set_train_config(config_service.clone(cfg))
            st.toast("Working config duplicated.", icon="📋")

    st.markdown("")
    device = load_settings().default_device
    if st.button("🚀 Start Experiment", type="primary", width="stretch", disabled=not ok):
        settings = load_settings()
        if settings.auto_save_config:
            config_service.save_config(cfg, f"{get_in(cfg, 'model.name', 'model')}_last")
        job_id = job_manager.submit(config_service.clone(cfg), label=get_in(cfg, "model.name", "experiment"), device=device)
        st.session_state["current_job_id"] = job_id
        state.notify(f"Training started ({get_in(cfg, 'model.name')}).", icon="🚀")
        st.rerun()


# --------------------------------------------------------------------------- #
# Live monitor + completion
# --------------------------------------------------------------------------- #
def _monitor_view(job_id: str) -> None:
    st.subheader("🟢 Training in progress")
    top = st.columns([1, 1, 3])
    if top[0].button("🛑 Stop training", type="secondary"):
        job_manager.stop(job_id)
        state.notify("Training stopped.", icon="🛑")
        st.rerun()
    if top[1].button("← New config"):
        st.session_state.pop("current_job_id", None)
        st.rerun()

    @st.fragment(run_every="2s")
    def _live() -> None:
        job = job_manager.get_job(job_id)
        if job is None:
            st.warning("Job record not found.")
            return
        components.live_monitor(job)
        if job.state in JobState.TERMINAL:
            st.rerun()  # break out of the fragment to the completion view

    _live()


def _completion_view(job) -> None:
    if job.state == JobState.COMPLETED:
        st.success(f"✅ Training complete — {job.experiment_name}")
    elif job.state == JobState.STOPPED:
        st.warning(f"🛑 Training stopped — {job.experiment_name or job.label}")
    else:
        st.error(f"❌ Training failed — {job.label}")
        if job.error:
            with st.expander("Error details"):
                st.code(job.error)

    c1, c2 = st.columns(2)
    if c1.button("← Configure a new experiment"):
        st.session_state.pop("current_job_id", None)
        st.rerun()
    if job.experiment_name and c2.button("📊 Open in Experiments"):
        st.session_state["selected_experiment"] = job.experiment_name
        st.session_state.pop("current_job_id", None)
        state.request_page("experiments")

    if job.experiment_dir and (paths.OUTPUTS_ROOT / job.experiment_name).is_dir():
        record = experiment_service.load_record(paths.OUTPUTS_ROOT / job.experiment_name)
        st.divider()
        components.experiment_detail(record, key_prefix="complete")


def _optional_int(cfg: dict, path: str, col, label: str, lo: int, hi: int, default: int) -> None:
    """Render an 'enable + value' pair backing an optional int config field."""
    current = get_in(cfg, path)
    enabled = col.toggle(f"{label}", value=current is not None, key=f"opt_{path}")
    if enabled:
        value = int(current) if isinstance(current, (int, float)) else default
        set_in(cfg, path, int(col.number_input(f"{label} value", lo, hi, value, key=f"optv_{path}", label_visibility="collapsed")))
    else:
        set_in(cfg, path, None)


def _optional_float(cfg: dict, path: str, col, label: str, lo: float, hi: float, default: float) -> None:
    """Render an 'enable + value' pair backing an optional float config field."""
    current = get_in(cfg, path)
    enabled = col.toggle(f"{label}", value=current is not None, key=f"opt_{path}")
    if enabled:
        value = float(current) if isinstance(current, (int, float)) else default
        set_in(cfg, path, float(col.number_input(f"{label} value", lo, hi, value, key=f"optv_{path}", label_visibility="collapsed")))
    else:
        set_in(cfg, path, None)
