"""Generate a self-contained experiment report (HTML + Markdown).

Uses Jinja2 (already a dependency) to render the config, hyperparameters,
evaluation metrics and the existing plot PNGs into a single portable document.
The HTML embeds images as base64 data URIs so it can be opened or printed to PDF
anywhere with no external files.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from jinja2 import Template

from frontend.services import experiment_service

_PLOTS = [
    ("Training / Validation Loss", "loss.png"),
    ("Training / Validation Accuracy", "accuracy.png"),
    ("Confusion Matrix", "confusion_matrix.png"),
    ("ROC Curves", "roc_curve.png"),
    ("Precision-Recall Curves", "pr_curve.png"),
]

_SCALAR_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "Macro F1"),
]

_HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Report — {{ experiment_id }}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 960px; line-height: 1.5; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  .muted { color: #6b7280; }
  table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }
  th, td { border: 1px solid #d1d5db; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.92rem; }
  th { background: rgba(127,127,127,0.12); }
  .metric-cards { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0 1.5rem; }
  .card { border: 1px solid #d1d5db; border-radius: 10px; padding: 0.75rem 1rem; min-width: 130px; }
  .card .val { font-size: 1.5rem; font-weight: 700; }
  .card .lbl { font-size: 0.8rem; color: #6b7280; }
  img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; margin: 0.5rem 0; }
  pre { background: rgba(127,127,127,0.10); padding: 0.75rem; border-radius: 8px; overflow-x: auto; }
  section { margin-bottom: 2rem; }
</style>
</head>
<body>
  <h1>{{ experiment_id }}</h1>
  <p class="muted">Model <strong>{{ model }}</strong> · {{ created_at }}
     {% if device %}· {{ device|upper }}{% endif %}
     {% if training_time %}· trained in {{ training_time }}{% endif %}</p>

  <section>
    <h2>Evaluation metrics</h2>
    <div class="metric-cards">
      {% for label, value in metric_cards %}
      <div class="card"><div class="val">{{ value }}</div><div class="lbl">{{ label }}</div></div>
      {% endfor %}
    </div>
    {% if per_class_rows %}
    <h3>Per-class ROC AUC / Average Precision</h3>
    <table>
      <tr><th>Class</th><th>ROC AUC</th><th>Avg Precision</th></tr>
      {% for cls, roc, ap in per_class_rows %}
      <tr><td>{{ cls }}</td><td>{{ roc }}</td><td>{{ ap }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
  </section>

  <section>
    <h2>Plots</h2>
    {% for title, uri in plot_images %}
      <h3>{{ title }}</h3>
      <img src="{{ uri }}" alt="{{ title }}"/>
    {% endfor %}
  </section>

  {% if classification_report %}
  <section>
    <h2>Classification report</h2>
    <pre>{{ classification_report }}</pre>
  </section>
  {% endif %}

  <section>
    <h2>Hyperparameters</h2>
    <table>
      <tr><th>Group</th><th>Setting</th><th>Value</th></tr>
      {% for group, key, value in hyperparams %}
      <tr><td>{{ group }}</td><td>{{ key }}</td><td>{{ value }}</td></tr>
      {% endfor %}
    </table>
  </section>

  <section>
    <h2>Full configuration</h2>
    <pre>{{ config_yaml }}</pre>
  </section>
</body>
</html>
"""
)

_MD_TEMPLATE = Template(
    """# Experiment Report — {{ experiment_id }}

**Model:** {{ model }}  |  **Date:** {{ created_at }}{% if device %}  |  **Device:** {{ device|upper }}{% endif %}{% if training_time %}  |  **Training time:** {{ training_time }}{% endif %}

## Evaluation metrics

| Metric | Value |
|---|---|
{% for label, value in metric_cards %}| {{ label }} | {{ value }} |
{% endfor %}
{% if per_class_rows %}
### Per-class ROC AUC / Average Precision

| Class | ROC AUC | Avg Precision |
|---|---|---|
{% for cls, roc, ap in per_class_rows %}| {{ cls }} | {{ roc }} | {{ ap }} |
{% endfor %}
{% endif %}
## Hyperparameters

| Group | Setting | Value |
|---|---|---|
{% for group, key, value in hyperparams %}| {{ group }} | {{ key }} | {{ value }} |
{% endfor %}
## Full configuration

```yaml
{{ config_yaml }}
```

_Plots (loss, accuracy, confusion matrix, ROC, PR) are saved as PNGs under the
experiment's `plots/` folder and embedded in the HTML report._
"""
)


def _img_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "—" if value is None else str(value)


def _build_context(experiment_dir: Path) -> dict[str, Any]:
    from frontend.services import config_service

    record = experiment_service.load_record(experiment_dir)
    metrics = experiment_service.load_metrics(experiment_dir)
    config = experiment_service.load_config_dict(experiment_dir)

    metric_cards = [
        (label, _fmt(metrics.get(key)))
        for key, label in _SCALAR_METRICS
        if metrics.get(key) is not None
    ]

    roc = metrics.get("roc_auc") or {}
    ap = metrics.get("average_precision") or {}
    classes = metrics.get("class_names") or []
    per_class_rows = [
        (cls, _fmt(roc.get(cls)), _fmt(ap.get(cls)))
        for cls in classes
    ]

    hyperparams: list[tuple[str, str, str]] = []
    for group in ("model", "optimizer", "scheduler", "loss", "training", "dataset"):
        section = config.get(group, {})
        if isinstance(section, dict):
            for key, value in section.items():
                if not isinstance(value, (dict, list)):
                    hyperparams.append((group, key, _fmt(value)))

    return {
        "experiment_id": record.experiment_id,
        "model": record.model,
        "created_at": record.created_at or "—",
        "device": record.device,
        "training_time": _duration(record.training_time_seconds),
        "metric_cards": metric_cards,
        "per_class_rows": per_class_rows,
        "hyperparams": hyperparams,
        "config_yaml": config_service.to_yaml(config) if config else "",
        "classification_report": experiment_service.classification_report(experiment_dir),
    }


def _duration(seconds: Any) -> str:
    if not seconds:
        return ""
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def build_html(experiment_dir: str | Path) -> str:
    """Render the full HTML report (images embedded as data URIs)."""
    experiment_dir = Path(experiment_dir)
    context = _build_context(experiment_dir)
    context["plot_images"] = [
        (title, _img_data_uri(p))
        for title, filename in _PLOTS
        if (p := experiment_service.plot_path(experiment_dir, filename)) is not None
    ]
    return _HTML_TEMPLATE.render(**context)


def build_markdown(experiment_dir: str | Path) -> str:
    """Render the Markdown report."""
    context = _build_context(Path(experiment_dir))
    return _MD_TEMPLATE.render(**context)


def write_reports(experiment_dir: str | Path) -> dict[str, Path]:
    """Write ``report.html`` and ``report.md`` into the experiment folder.

    Returns:
        Mapping of format -> written path.
    """
    experiment_dir = Path(experiment_dir)
    html_path = experiment_dir / "report.html"
    md_path = experiment_dir / "report.md"
    html_path.write_text(build_html(experiment_dir), encoding="utf-8")
    md_path.write_text(build_markdown(experiment_dir), encoding="utf-8")
    return {"html": html_path, "markdown": md_path}
