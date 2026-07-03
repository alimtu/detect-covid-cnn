# Design Decisions

This document explains *why* the framework is built the way it is. It is meant
to support an academic discussion: for each significant choice we state the
decision, the rationale, the trade-offs, and how it is exposed as a
configurable knob (so alternatives can be evaluated experimentally).

---

## Why a configuration-driven framework?

**Decision.** Every parameter affecting data, augmentation, model, optimization,
training, evaluation, checkpointing, reproducibility and logging lives in YAML
config files. Source code is never edited to change an experiment.

**Rationale.** Reproducibility and scientific comparison require that an
experiment be fully described by its configuration. Each run saves its resolved
`config.yaml`, so any result can be regenerated exactly. A base config plus
small override files keeps experiments DRY and their diffs meaningful.

**Trade-off.** More upfront abstraction than a single script, but it pays off as
soon as more than one experiment is run.

---

## Why transfer learning with ImageNet-pretrained weights?

**Decision.** All backbones load ImageNet-pretrained weights by default and only
the classifier head is replaced (`model.pretrained: true`).

**Rationale.** Medical imaging datasets are small relative to the millions of
images used to train ImageNet models. Pretrained convolutional features
(edges, textures, shapes) transfer well to X-rays and dramatically reduce the
data and compute needed to reach good accuracy, while improving convergence
stability. This is standard practice in medical imaging literature.

**Alternative (configurable).** Set `pretrained: false` to train from scratch
and quantify the benefit of transfer learning.

---

## Why compare DenseNet121 and VGG16 (and offer ResNet50 / EfficientNet / ViT)?

**Decision.** DenseNet121 and VGG16 are the two baseline models; ResNet50,
EfficientNet-B0 and ViT-B/16 are also available via `model.name`.

**Rationale.**
- **DenseNet121** — dense connectivity encourages feature reuse and strong
  gradient flow with relatively few parameters (~8M). It is a well-known strong
  performer on chest X-ray tasks (e.g. CheXNet).
- **VGG16** — a simple, uniform, historically important architecture. It is a
  useful, heavier (~138M params) baseline: if a modern compact model beats VGG16,
  that is a meaningful result.
- **ResNet50** — residual connections; a ubiquitous, well-understood baseline.
- **EfficientNet-B0** — compound scaling; strong accuracy-per-FLOP.
- **ViT-B/16** — a transformer baseline to contrast CNN inductive biases with
  attention-based models (typically needs more data / lower LR).

Comparing architectures under an *identical* pipeline isolates the effect of the
architecture itself.

---

## Why AdamW as the default optimizer?

**Decision.** `optimizer.name: adamw` by default (Adam and SGD also available).

**Rationale.** AdamW decouples weight decay from the adaptive gradient update
(unlike Adam's L2-in-the-gradient behaviour), which yields more correct
regularization and typically better generalization. Adaptive methods converge
quickly with little learning-rate tuning — valuable for fine-tuning pretrained
models on small datasets.

**Alternative (configurable).** SGD with momentum often generalizes slightly
better with a carefully tuned LR schedule; switch via `optimizer.name: sgd`.

---

## Why CrossEntropyLoss (with optional weighting / Focal)?

**Decision.** `loss.name: cross_entropy` by default; `weighted_cross_entropy`
and `focal` are available.

**Rationale.** Cross-entropy is the standard, well-calibrated loss for
mutually-exclusive multi-class classification and pairs naturally with a softmax
head. The COVID-19 Radiography Database is **class-imbalanced** (Normal ≫ Viral
Pneumonia), so we expose:
- **Weighted cross-entropy** (`class_weights: balanced`) — inverse-frequency
  weighting so minority classes are not ignored.
- **Focal loss** — down-weights easy examples to focus on hard/rare cases.

Making these configurable lets the imbalance strategy be studied empirically.

---

## Why a validation set (train/val/test = 70/15/15, stratified)?

**Decision.** A three-way **stratified** split with a fixed seed.

**Rationale.**
- The **validation** set drives model selection (best checkpoint), learning-rate
  scheduling and early stopping — *without* touching the test set.
- The **test** set is held out and used only once for final reporting, giving an
  unbiased estimate of generalization.
- **Stratification** preserves class proportions in every split, which is
  essential under class imbalance.
- A **fixed seed** makes the split reproducible across runs, so model
  comparisons use exactly the same data partition.

---

## Why EarlyStopping?

**Decision.** Stop training when the monitored validation metric stops improving
(`training.early_stopping`).

**Rationale.** It prevents overfitting and wasted compute by halting once the
model no longer generalizes better, while `best_model.pth` retains the
best-validation weights. Patience and `min_delta` are configurable so the
stopping sensitivity can be tuned.

---

## Why ReduceLROnPlateau by default (with Cosine / Step / OneCycle available)?

**Decision.** `scheduler.name: reduce_on_plateau` by default.

**Rationale.** It is robust and assumption-light: it lowers the LR only when
validation loss plateaus, which suits fine-tuning where the ideal schedule is
not known in advance. Cosine, Step and OneCycle are provided for experiments
where a predetermined schedule is preferred (e.g. OneCycle for fast convergence).

---

## Why freeze/unfreeze the backbone?

**Decision.** Optional `model.freeze_backbone` with `model.unfreeze_after_epoch`.

**Rationale.** A common transfer-learning recipe: first train only the new head
(fast, stable, avoids destroying pretrained features with large early
gradients), then unfreeze the whole network to fine-tune. When unfreezing, the
optimizer and scheduler are rebuilt so the newly trainable parameters are
optimized correctly.

---

## Why ImageNet normalization and 224×224 inputs?

**Decision.** Resize to 224×224 and normalize with ImageNet mean/std.

**Rationale.** Pretrained torchvision backbones expect inputs distributed like
their training data; matching the normalization statistics and the canonical
224×224 resolution lets the pretrained features behave as intended. Both are
configurable (`dataset.image_size`, `augmentation.normalize`).

---

## Why these evaluation metrics?

**Decision.** Accuracy, macro Precision/Recall/F1, confusion matrix,
classification report, and one-vs-rest ROC and PR curves — each toggleable.

**Rationale.** Accuracy alone is misleading under class imbalance. **Macro**
averaging weights each class equally, exposing poor performance on minority
classes. The **confusion matrix** reveals *which* classes are confused (clinically
important, e.g. COVID vs. Viral Pneumonia). **ROC/PR curves** characterise the
probability-threshold trade-offs per class; PR curves are especially informative
for imbalanced positives.

---

## Why per-experiment folders and experiment tracking?

**Decision.** Each run writes an auto-incremented `experiment_NNN/` folder
containing the resolved config, metrics, per-epoch history, plots and
checkpoints, plus a compact `summary.json`.

**Rationale.** No experiment is ever lost or silently overwritten. Any result is
traceable to the exact configuration that produced it, and `compare.py`
aggregates summaries across experiments for side-by-side analysis — the core of
a reproducible research workflow.

---

## Why reproducibility controls?

**Decision.** A single `reproducibility.seed` seeds Python, NumPy and PyTorch;
`deterministic: true` requests deterministic cuDNN behaviour.

**Rationale.** Deterministic behaviour (where the backend allows) means repeated
runs of the same config yield the same result, which is a prerequisite for
trustworthy comparisons. (Full bitwise determinism is not guaranteed on all
backends, e.g. Apple MPS.)

---

## Extensibility (designed-in, minimal changes required)

- **New model** — register a builder in `src/models/factory.py`; select via config.
- **New augmentation** — add a branch in `src/datasets/transforms.py`.
- **New optimizer/scheduler/loss** — add a branch in the corresponding
  `src/training/*.py` builder.
- **Grad-CAM / segmentation / K-fold / HP search / external-dataset eval** — the
  data, model and training layers are decoupled, so these can be added as new
  modules without refactoring existing code.
