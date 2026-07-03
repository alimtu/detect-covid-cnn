"""Streamlit research-dashboard frontend for the COVID-19 classifier.

This package is a thin orchestration layer over the existing backend in
``src``. It contains **no** training/evaluation business logic of its own:

* ``frontend.services`` wraps the backend modules (config, pipeline, inference,
  experiments) and manages background training jobs.
* ``frontend.ui`` holds reusable presentation helpers (theme, charts, widgets).
* ``frontend.pages`` holds one module per dashboard page.

The multipage app is launched from the project-root ``app.py``.
"""
