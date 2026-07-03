"""Service layer: orchestrates the backend for the Streamlit frontend.

Every module here is UI-agnostic (no ``streamlit`` imports) so the logic can be
unit-tested and reused. Pages import these services; the services import the
backend in ``src``.
"""
