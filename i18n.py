"""Lightweight localization helpers for the Smartaira Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


LOCALES_DIR = Path(__file__).resolve().parent / "locales"
DEFAULT_LANGUAGE = "ar"
SUPPORTED_LANGUAGES = {
    "ar": {"label": "العربية", "dir": "rtl"},
    "en": {"label": "English", "dir": "ltr"},
    "fr": {"label": "Français", "dir": "ltr"},
    "it": {"label": "Italiano", "dir": "ltr"},
    "de": {"label": "Deutsch", "dir": "ltr"},
}


@st.cache_data(show_spinner=False)
def load_locale(lang_code: str) -> dict[str, Any]:
    """Load a locale file once per Streamlit process."""
    locale_path = LOCALES_DIR / f"{lang_code}.json"
    with locale_path.open("r", encoding="utf-8") as locale_file:
        return json.load(locale_file)


def _valid_language(lang_code: object) -> str:
    return str(lang_code) if str(lang_code) in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def init_i18n() -> str:
    """Resolve and synchronize the active language before rendering the page."""
    query_lang = st.query_params.get("lang")
    session_lang = st.session_state.get("lang")
    active_lang = _valid_language(query_lang or session_lang or DEFAULT_LANGUAGE)
    st.session_state["lang"] = active_lang
    if st.query_params.get("lang") != active_lang:
        st.query_params["lang"] = active_lang
    return active_lang


def get_language() -> str:
    """Return the active language, initializing it when needed."""
    return _valid_language(st.session_state.get("lang", DEFAULT_LANGUAGE))


def set_language(lang_code: str) -> None:
    """Persist a valid language choice in both the session and URL."""
    language = _valid_language(lang_code)
    st.session_state["lang"] = language
    if st.query_params.get("lang") != language:
        st.query_params["lang"] = language


def _lookup(locale: dict[str, Any], key: str) -> Any:
    value: Any = locale
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def t(key: str, **kwargs: Any) -> str:
    """Translate a dot-notated key with safe language fallbacks and formatting."""
    language_chain = (get_language(), "en", "ar")
    value: Any = None
    for lang_code in dict.fromkeys(language_chain):
        try:
            value = _lookup(load_locale(lang_code), key)
        except FileNotFoundError:
            value = None
        if isinstance(value, str) and value:
            break
        value = None

    if value is None:
        return key
    try:
        return value.format(**kwargs)
    except (KeyError, ValueError):
        return value


def apply_direction_styles() -> None:
    """Inject language-aware fonts and explicit bidirectional layout styles."""
    language = get_language()
    direction = SUPPORTED_LANGUAGES[language]["dir"]
    alignment = "right" if direction == "rtl" else "left"
    arrow_scale = "scaleX(-1)" if direction == "rtl" else "none"
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
          .stApp,
          [data-testid="stSidebar"],
          [data-testid="stAppViewContainer"],
          [data-testid="stMainBlockContainer"],
          [data-testid="stVerticalBlock"],
          [data-testid="stHorizontalBlock"],
          [data-testid="stMarkdownContainer"],
          [data-testid="stWidgetLabel"],
          [data-testid="stAlert"],
          [data-testid="stMetricLabel"],
          [data-testid="stExpander"],
          [data-testid="stTabs"],
          [data-testid="stDataFrame"],
          .smartaira-hero,
          .smartaira-tagline-ar,
          .smartaira-sub-ar,
          .login-card,
          .workflow-steps,
          .workflow-step,
          .status-strip,
          .privacy-box,
          .exec-callout,
          .brand-footer,
          .stCaption,
          .stText {{
            direction: {direction} !important;
            text-align: {alignment} !important;
            font-family: 'Cairo', 'Inter', sans-serif !important;
          }}
          input, textarea, select, [role="combobox"], [role="listbox"], [role="radiogroup"] {{
            direction: {direction} !important;
            text-align: {alignment} !important;
            font-family: 'Cairo', 'Inter', sans-serif !important;
          }}
          [data-testid="stMetricValue"] {{
            direction: ltr !important;
            text-align: {alignment} !important;
            font-family: 'Inter', 'Cairo', sans-serif !important;
          }}
          pre, code, kbd, samp, [data-testid="stException"], [data-testid="stCodeBlock"] {{
            direction: ltr !important;
            text-align: left !important;
            font-family: 'SFMono-Regular', Consolas, monospace !important;
          }}
          .i18n-direction-arrow {{ transform: {arrow_scale}; display: inline-block; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
