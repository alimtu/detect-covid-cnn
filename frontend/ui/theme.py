"""Global page styling and the Persian classifier header.

Ports the IRANYekanX font loading from the original ``app.py`` and adds a light
set of dashboard styles. Styling is intentionally minimal so it works with both
Streamlit's light and dark themes.
"""

from __future__ import annotations

import base64
from functools import lru_cache

import streamlit as st

from frontend.services import paths

FONT_FAMILY = "IRANYekanX"

FA_TITLE = "طبقه‌بند تصاویر اشعه ایکس قفسه سینه کووید-۱۹"
FA_DESCRIPTION = (
    "یک تصویر اشعه ایکس قفسه سینه بارگذاری کنید و پیش‌بینی مدل‌های آموزش‌دیده را "
    "مقایسه کنید. هر آزمایش، مدل خود را از روی پیکربندی‌ای که با آن آموزش دیده بازسازی می‌کند."
)


def _font_face(weight: int, filename: str) -> str:
    path = paths.FONTS_DIR / filename
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
@font-face {{
    font-family: '{FONT_FAMILY}';
    src: url(data:font/woff;base64,{encoded}) format('woff');
    font-weight: {weight};
    font-style: normal;
    font-display: swap;
}}"""


@lru_cache(maxsize=1)
def _css() -> str:
    faces = "\n".join(
        rule
        for rule in (
            _font_face(400, "IRANYekanXFaNum.woff"),
            _font_face(500, "IRANYekanXFaNum-Medium.woff"),
            _font_face(700, "IRANYekanXFaNum-Bold.woff"),
        )
        if rule
    )
    return f"""
<style>
{faces}
.fa-header {{ font-family: '{FONT_FAMILY}', Tahoma, sans-serif; direction: rtl; text-align: right; }}
.fa-header h1 {{ font-weight: 700; font-size: 1.9rem; margin-bottom: 0.4rem; }}
.fa-header p {{ font-weight: 400; font-size: 1.02rem; line-height: 1.9; color: inherit; }}

/* Experiment / metric cards */
.exp-card {{
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    background: rgba(128,128,128,0.05);
}}
.exp-card .title {{ font-weight: 700; font-size: 1.02rem; }}
.exp-card .sub {{ color: #8b8b8b; font-size: 0.82rem; margin-bottom: 0.4rem; }}
.pill {{
    display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem;
}}
.pill.ok {{ background: rgba(34,197,94,0.18); color: #16a34a; }}
.pill.run {{ background: rgba(59,130,246,0.18); color: #2563eb; }}
.pill.fail {{ background: rgba(239,68,68,0.18); color: #dc2626; }}
.pill.idle {{ background: rgba(128,128,128,0.18); color: #6b7280; }}
</style>
"""


def inject() -> None:
    """Inject the global stylesheet (idempotent per session via caching)."""
    st.markdown(_css(), unsafe_allow_html=True)


def persian_header() -> None:
    """Render the original Persian title + description used by the classifier."""
    inject()
    st.markdown(
        f"""
<div class="fa-header">
  <h1 style="font-family: {FONT_FAMILY};">🩻 {FA_TITLE}</h1>
  <p>{FA_DESCRIPTION}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def status_pill(status: str) -> str:
    """Return an HTML pill span for a status string."""
    mapping = {
        "completed": ("ok", "completed"),
        "running": ("run", "running"),
        "pending": ("run", "pending"),
        "failed": ("fail", "failed"),
        "stopped": ("fail", "stopped"),
        "incomplete": ("idle", "incomplete"),
    }
    css_class, label = mapping.get(status, ("idle", status))
    return f'<span class="pill {css_class}">{label}</span>'
