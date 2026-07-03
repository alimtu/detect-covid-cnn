# COVID-19 Chest X-ray Classification — Research Framework

A configuration-driven, reproducible deep-learning framework for classifying
chest X-rays from the **COVID-19 Radiography Database** into four classes, built
to compare architectures under an identical pipeline.

Everything that can affect data, augmentation, model, optimization, training,
evaluation, checkpointing, reproducibility or logging is controlled from YAML
config files — **you should never need to edit source code to run a different
experiment.** Every run is saved to a self-contained, reproducible experiment
folder.

> Design rationale for every major choice (why DenseNet121, AdamW, CrossEntropy,
> EarlyStopping, a validation set, ImageNet weights, etc.) is documented in
> [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md).

---

## Classes

`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`
(auto-discovered from the dataset folders, sorted alphabetically).

## Highlights

- **Config-first**: base config + small per-experiment override files.
- **5 models**: DenseNet121, VGG16, ResNet50, EfficientNet-B0, ViT-B/16 — all
  selected from config, easily extended.
- **Fully configurable** augmentation, optimizer (Adam/AdamW/SGD), scheduler
  (ReduceLROnPlateau/Cosine/Step/OneCycle/None), loss (CE/weighted CE/Focal).
- **Training features**: mixed precision (AMP), gradient clipping, gradient
  accumulation, early stopping, freeze/unfreeze backbone, flexible checkpointing.
- **Evaluation**: accuracy, macro precision/recall/F1, confusion matrix,
  classification report, ROC & PR curves (each toggleable).
- **Experiment tracking**: auto-incremented `experiment_NNN_<model>/` folders
  with the resolved config, metrics, per-epoch CSV history, plots and checkpoints.
- **Reproducible**: single seed drives splits, shuffling and initialization.
- **Streamlit Research Dashboard** — a full multi-page app to configure, train
  (in the background), monitor live, evaluate, compare, run inference and
  generate reports, without a terminal or hand-edited YAML. The CLIs still work.

---

## Project Architecture

```
covid-claude/
├── configs/
│   ├── base.yaml                 # every default; the single source of truth
│   └── experiments/              # small override files (inherit base via `base:`)
│       ├── densenet121.yaml
│       ├── vgg16.yaml
│       ├── resnet50.yaml
│       ├── efficientnet_b0.yaml
│       ├── vit_b_16.yaml
│       ├── densenet121_augmented.yaml   # augmentation study
│       └── densenet121_finetune.yaml    # freeze/unfreeze + weighted loss study
├── DATA/                         # dataset root (one folder per class)
├── src/
│   ├── config/                   # layered loader, validation, ExperimentManager
│   ├── datasets/                 # discovery, sampling, transforms, split, loaders
│   ├── models/                   # registry-backed factory + custom heads
│   ├── training/                 # losses, optimizers, schedulers, callbacks, Trainer
│   ├── evaluation/               # config-driven metrics, curves, reports
│   ├── loggers/                  # console / CSV / TensorBoard composite
│   ├── inference/                # single-image Predictor (rebuilds from experiment)
│   ├── pipeline.py               # run_experiment(): shared by the CLI and the UI
│   └── utils/                    # device, seed, logging, plots
├── frontend/                     # Streamlit dashboard (thin layer over src/)
│   ├── services/                 # UI-agnostic orchestration of the backend
│   │   ├── config_service.py     #   build/validate/save configs from widgets
│   │   ├── experiment_service.py #   list/read/delete/rename/export experiments
│   │   ├── job_manager.py        #   background training (spawn process) + queue
│   │   ├── training_worker.py    #   child-process entry → src.pipeline
│   │   ├── monitor_service.py    #   live progress from history CSV + job status
│   │   ├── comparison_service.py #   side-by-side experiment comparison
│   │   ├── inference_service.py  #   Predictor wrapper
│   │   ├── report_service.py     #   HTML/Markdown experiment reports (Jinja2)
│   │   └── settings_service.py   #   persisted project defaults
│   ├── ui/                       # theme, charts (Altair), reusable components
│   └── pages/                    # Dashboard, Train, Experiments, Compare, …
├── docs/DESIGN_DECISIONS.md      # rationale for every design choice
├── outputs/                      # experiment_NNN/ folders (generated)
├── run.py                        # CLI: train + evaluate one experiment
├── compare.py                    # CLI: compare metrics across experiments
├── app.py                        # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

Each module has a single responsibility, so features can be added without
touching unrelated code. See "Extending the framework" below.

---

## Installation

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Device selection is automatic with priority **MPS (Apple Silicon) → CUDA → CPU**.

## Dataset

Point `dataset.path` at a directory with one subfolder per class. Two layouts
are supported automatically:

```
DATA/                          DATA/
├── COVID/                     ├── COVID/images/*.png   (+ optional masks/)
│   └── *.png          OR      ├── Normal/images/*.png
├── Normal/                    ├── Lung_Opacity/images/*.png
...                            └── Viral Pneumonia/images/*.png
```

If a class folder contains `images/`, it is used automatically; otherwise images
are read directly from the class folder. Any `masks/` folder is ignored (mask
support is reserved for future segmentation work).

---

## Configuration

`configs/base.yaml` defines every knob. Experiment files under
`configs/experiments/` inherit it via a top-level `base:` reference and override
only what changes:

```yaml
# configs/experiments/resnet50.yaml
base: ../base.yaml

experiment:
  name: resnet50_baseline

model:
  name: resnet50
```

Overrides are deep-merged, so nested keys can be changed individually. Config
files are validated on load (split ratios, known model/optimizer/scheduler/loss
names, etc.), failing fast with a clear message.

Key sections: `experiment`, `reproducibility`, `dataset`, `augmentation`,
`model`, `loss`, `optimizer`, `scheduler`, `training`, `evaluation`, `logging`.
See `base.yaml` for inline documentation of every field.

---

## The Research Dashboard (recommended)

Launch the full dashboard — the primary interface for the project:

```bash
streamlit run app.py
```

Everything below can be done from the browser, no terminal or YAML editing:

- **🏠 Dashboard** — project overview: totals, best experiment/accuracy/F1, last
  run, detected device, recent experiment cards, and trend charts.
- **🚀 Train** — a graphical control for **every** config field (dataset,
  augmentation, model, optimizer, scheduler, loss, training, logging,
  evaluation). Preview / download / save / load / duplicate the generated config,
  then **Start Experiment**. Training runs in a **background process** (the app
  stays responsive) with a **live monitor** — current epoch, progress bar,
  losses, accuracy, LR, ETA and live charts that update each epoch. On
  completion you get metrics, plots, confusion matrix, ROC/PR and downloads.
- **📊 Experiments** — sortable/searchable/filterable, paginated table with
  per-experiment actions: View, Duplicate (loads its config into Train), Rename,
  Delete, Export (.zip), Open folder.
- **📈 Compare** — pick several experiments for a side-by-side comparison
  (metrics table with best values highlighted, per-class ROC, learning curves,
  confusion matrices) and export the comparison.
- **🔍 Inference** — pick an experiment + checkpoint (`best`/`last`), upload an
  X-ray, and see the predicted class, confidence, per-class probabilities and
  inference time.
- **⚙️ Settings** — project defaults (dataset path, output dir, device, image
  size, default model/optimizer/scheduler, theme, auto-save) used to seed new
  configs.

Reports (HTML with embedded plots + Markdown) can be generated per experiment
from its detail view; open the HTML and print-to-PDF for a PDF.

> Background training uses a dedicated `spawn` child process that calls the
> **same** `src.pipeline.run_experiment` the CLI uses — no shell commands, no
> duplicated logic. Job state is file-based (`outputs/.jobs/`), so it survives
> app reloads and is ready for a future multi-job queue.

---

## Experiment Workflow (CLI)

The command line remains fully supported and produces identical experiment
folders.

### 1. Run an experiment (train + evaluate)

```bash
python run.py --config configs/experiments/densenet121.yaml
```

This creates the next `outputs/experiment_NNN_<model>/`:

```
outputs/experiment_001_densenet121/
├── config.yaml               # fully resolved config (reproduces this run)
├── metrics.json              # final test metrics (+ ROC AUC / AP)
├── summary.json              # hyperparameters + final metrics (for comparison)
├── training_history.csv      # per-epoch loss/accuracy/lr
├── plots/                    # accuracy, loss, confusion_matrix, roc, pr, report
└── checkpoints/              # best_model.pth, last_model.pth (+ periodic)
```

### 2. Run more experiments

```bash
python run.py --config configs/experiments/vgg16.yaml
python run.py --config configs/experiments/resnet50.yaml
```

### 3. Compare them

```bash
python compare.py                       # all experiments in outputs/
python compare.py --experiments experiment_001_densenet121 experiment_002_vgg16
```

Prints a side-by-side table and writes `comparison.txt`/`.csv`/`.json`, and
reports the best experiment by macro F1.

### 4. Or do all of the above in the dashboard

```bash
streamlit run app.py
```

See "The Research Dashboard" above.

---

## Reproducing an Experiment

Every experiment's `config.yaml` is fully resolved and self-contained. To
reproduce a past run:

```bash
python run.py --config outputs/experiment_001_densenet121/config.yaml
```

The fixed `reproducibility.seed` reproduces the same stratified split, shuffling
and initialization (subject to backend determinism limits, e.g. on MPS).

---

## Common Studies (config-only)

- **Full dataset**: set `dataset.sample_per_class: all`.
- **Augmentation study**: `configs/experiments/densenet121_augmented.yaml`.
- **Freeze then fine-tune + class balancing**:
  `configs/experiments/densenet121_finetune.yaml`.
- **Handle imbalance**: `loss.name: weighted_cross_entropy` with
  `loss.class_weights: balanced`, or `loss.name: focal`.
- **Different schedule**: `scheduler.name: cosine | onecycle | step | none`.
- **Mixed precision / larger effective batch** (CUDA): set
  `training.mixed_precision: true` and/or `training.gradient_accumulation_steps`.
- **TensorBoard**: `logging.tensorboard: true`, then
  `tensorboard --logdir outputs/experiment_NNN/tensorboard`.

---

## Extending the Framework

Each extension is localized to one module (minimal changes elsewhere):

- **New model** — add a builder in `src/models/factory.py` and register it:

```python
def _build_resnet101(num_classes, pretrained, dropout, hidden):
    ...
    return ModelBundle(model, head)

_MODEL_REGISTRY["resnet101"] = _build_resnet101
```

Then set `model.name: resnet101`.

- **New dataset** — keep the one-folder-per-class layout and set `dataset.path`
  (and `dataset.class_names` if you want an explicit order). The loader
  auto-discovers classes and images.
- **New augmentation** — add a branch in `src/datasets/transforms.py` reading its
  config block.
- **New optimizer / scheduler / loss** — add a branch in the corresponding
  builder in `src/training/`.
- **Future work** (Grad-CAM, lung segmentation, K-fold cross-validation,
  hyperparameter search, external-dataset evaluation) — the decoupled
  data/model/training layers are designed to accept these as new modules without
  refactoring.

---

## Requirements

PyTorch, torchvision, scikit-learn, matplotlib, pandas, numpy, Pillow, PyYAML,
Streamlit + Altair + Jinja2 (dashboard), and optionally TensorBoard. See
`requirements.txt`.
# detect-covid-cnn
