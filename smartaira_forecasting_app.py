from datetime import datetime
import io
import os
import re
import subprocess
from typing import Optional

import altair as alt
from flaml import AutoML
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st

from i18n import (
    SUPPORTED_LANGUAGES,
    apply_direction_styles,
    init_i18n,
    set_language,
    t,
)


BUILD_ID_FALLBACK = "2e6ef18"


def get_build_id() -> str:
    """Return the deployed commit ID when the host exposes Git metadata."""
    for env_name in ("STREAMLIT_COMMIT_ID", "GIT_COMMIT", "COMMIT_SHA"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip() or BUILD_ID_FALLBACK
    except (OSError, subprocess.SubprocessError):
        return BUILD_ID_FALLBACK


def render_build_id(build_id: str) -> None:
    """Show a compact build marker for deployment verification."""
    st.caption(t("app.build", build_id=build_id))


def on_language_change() -> None:
    """Synchronize the selected language and reset localized forecast output."""
    set_language(st.session_state["language_selector"])
    st.session_state["forecast_result"] = None
    st.session_state["forecast_history"] = []
    st.session_state["language_changed_notice"] = True


DESIGN_OPTIONS = (
    "neumorphism",
    "classic",
    "warm",
)


def get_design_css(design_style: str) -> str:
    """Return lightweight CSS overrides for the user's preferred visual treatment."""
    if design_style == "classic":
        return """
        <style>
            .stApp { background: #F8FAFC; }
            [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stFileUploader"] section, [data-testid="stDataFrame"] {
                background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.07);
            }
            [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
            [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stFileUploader"] section {
                background: #FFFFFF !important; box-shadow: none !important; border-color: #CBD5E1 !important;
            }
            [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button, [data-testid="stFormSubmitButton"] > button {
                background: #FFFFFF; border-radius: 10px; box-shadow: none;
            }
            [data-testid="stButton"] > button:active, [data-testid="stDownloadButton"] > button:active, [data-testid="stFormSubmitButton"] > button:active {
                transform: none; box-shadow: inset 0 1px 3px rgba(15,23,42,0.14);
            }
            [data-testid="stButton"] > button[kind="primary"], [data-testid="stDownloadButton"] > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
                background: #059669; box-shadow: none;
            }
            [data-baseweb="tab"] { background: #FFFFFF; box-shadow: none; border-radius: 9px 9px 0 0; }
            [data-baseweb="tab"][aria-selected="true"] { background: #ECFDF5; box-shadow: inset 0 -3px 0 #10B981; }
        </style>
        """
    if design_style == "warm":
        return """
        <style>
            .stApp { background: #F4EFE6; }
            [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stFileUploader"] section, [data-testid="stDataFrame"] {
                background: #F4EFE6; border-color: rgba(255,255,255,0.72);
                box-shadow: 8px 8px 16px #D8CFC0, -8px -8px 16px #FFFDF8;
            }
            [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
            [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stFileUploader"] section {
                background: #F4EFE6 !important; border-color: #E3D9C8 !important;
                box-shadow: inset 4px 4px 8px #D9D0C1, inset -4px -4px 8px #FFFDF8 !important;
            }
            [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button, [data-testid="stFormSubmitButton"] > button {
                background: #F4EFE6; box-shadow: 6px 6px 12px #D7CEBF, -6px -6px 12px #FFFDF8;
            }
            [data-testid="stButton"] > button[kind="primary"], [data-testid="stDownloadButton"] > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
                background: linear-gradient(145deg, #B7791F, #8B5E18); box-shadow: 6px 6px 12px rgba(112,75,20,0.3), -3px -3px 9px #FFFDF8;
            }
            .smartaira-hero { border-right-color: #C28A2E; }
            .smartaira-tagline-ar { color: #F6C76C; }
            [data-baseweb="tab"] { background: #F4EFE6; box-shadow: 4px 4px 9px #D8CFC0, -4px -4px 9px #FFFDF8; }
            [data-baseweb="tab"][aria-selected="true"] { background: #F5E6C9; color: #744B13; }
        </style>
        """
    return ""


# -----------------------------------------------------------------------------
# Brand CSS Styling (Nile Executive Navy, Smartaira Emerald, Cairo Gold)
# -----------------------------------------------------------------------------
BRAND_CSS = """
<style>
    .stApp {
        background: #E9EEF4;
        direction: rtl;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stAppViewContainer"] {
        direction: rtl;
    }
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stExpander"],
    [data-testid="stAlert"],
    [data-testid="stMetric"],
    [data-testid="stFileUploader"],
    [data-testid="stTextInput"],
    [data-testid="stSelectbox"],
    [data-testid="stRadio"],
    [data-testid="stButton"],
    [data-testid="stDownloadButton"],
    [data-testid="stDataFrame"],
    .stMarkdown,
    .stCaption,
    .stText,
    .stAlert {
        direction: rtl;
        text-align: right;
    }
    input,
    textarea,
    select,
    [role="combobox"],
    [role="radiogroup"],
    [role="listbox"] {
        direction: rtl !important;
        text-align: right !important;
    }
    button {
        direction: rtl;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stDataFrame"] iframe {
        direction: rtl;
    }
    [data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stFileUploader"] section,
    [data-testid="stDataFrame"] {
        background: #E9EEF4;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 16px;
        box-shadow:
            8px 8px 16px #C5CED8,
            -8px -8px 16px #FFFFFF;
    }
    [data-testid="stMetric"] {
        padding: 0.8rem;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stFileUploader"] section {
        background: #E9EEF4 !important;
        border-color: rgba(255, 255, 255, 0.65) !important;
        box-shadow:
            inset 4px 4px 8px #C8D1DC,
            inset -4px -4px 8px #FFFFFF !important;
    }
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.75rem;
        border: 1px solid rgba(255, 255, 255, 0.62);
        border-radius: 14px;
        background: #E9EEF4;
        color: #0F172A;
        box-shadow:
            6px 6px 12px #C4CED9,
            -6px -6px 12px #FFFFFF;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }
    [data-testid="stButton"] > button:active,
    [data-testid="stDownloadButton"] > button:active,
    [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(1px) scale(0.99);
        box-shadow:
            inset 4px 4px 8px #C5CED8,
            inset -4px -4px 8px #FFFFFF;
    }
    [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stDownloadButton"] > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(145deg, #1BC48E, #0A8C68);
        border-color: rgba(255, 255, 255, 0.28);
        color: #FFFFFF;
        box-shadow:
            6px 6px 12px rgba(4, 100, 74, 0.36),
            -3px -3px 9px rgba(255, 255, 255, 0.34),
            inset 1px 1px 0 rgba(255, 255, 255, 0.22);
    }
    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 0;
    }
    [data-baseweb="tab"] {
        background: #E9EEF4;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 12px;
        box-shadow: 4px 4px 9px #C6CFDA, -4px -4px 9px #FFFFFF;
        padding: 0.55rem 0.8rem;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: #DFF4EA;
        color: #065F46;
        box-shadow: inset 3px 3px 7px rgba(5, 111, 81, 0.16), inset -3px -3px 7px rgba(255,255,255,0.88);
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .status-strip {
            align-items: stretch;
            flex-direction: column;
            gap: 0.5rem;
        }
        .smartaira-hero {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
        }
        .smartaira-hero h1 {
            font-size: 1.25rem;
            line-height: 1.35;
            margin-bottom: 0.2rem;
        }
        .smartaira-badge { margin-bottom: 0.35rem; font-size: 0.72rem; }
        .smartaira-tagline-ar {
            font-size: 0.92rem;
            margin-bottom: 0;
        }
        .smartaira-sub-ar {
            display: none;
        }
        .value-card {
            padding: 0.95rem;
        }
        .login-card {
            padding: 1.15rem;
            margin: 0.5rem 0 1rem;
        }
        .workflow-steps {
            gap: 0.2rem;
            padding: 0.45rem;
        }
        .workflow-step {
            font-size: 0.68rem;
        }
        .workflow-line {
            flex-basis: 7px;
        }
        .workflow-step {
            font-size: 0.72rem;
            padding: 0.4rem 0.15rem;
        }
        .stPlotlyChart, .js-plotly-plot { max-width: 100%; }
        [data-testid="stPlotlyChart"] { overflow-x: auto; }
        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
            direction: ltr;
            unicode-bidi: isolate;
            text-align: right;
        }
    }
    .login-card {
        background: #E9EEF4;
        border: 1px solid rgba(255, 255, 255, 0.62);
        border-top: 5px solid #10B981;
        border-radius: 18px;
        padding: 1.5rem;
        max-width: 560px;
        margin: 1rem auto 1.5rem;
        box-shadow:
            10px 10px 20px #C4CDD8,
            -10px -10px 20px #FFFFFF,
            inset 1px 1px 0 rgba(255,255,255,0.9);
        direction: rtl;
        text-align: right;
    }
    .workflow-steps {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0.85rem 0 1rem;
        padding: 0.65rem 0.8rem;
        background: #E9EEF4;
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 14px;
        box-shadow: inset 4px 4px 9px #C7D0DB, inset -4px -4px 9px #FFFFFF;
        direction: rtl;
    }
    .workflow-step {
        flex: 1;
        padding: 0.45rem 0.25rem;
        color: #64748B;
        text-align: center;
        font-size: 0.78rem;
        border-radius: 9px;
    }
    .workflow-step.active {
        color: #065F46;
        background: #DFF4EA;
        font-weight: 700;
        box-shadow: inset 3px 3px 7px rgba(5,111,81,0.15), inset -3px -3px 7px rgba(255,255,255,0.9);
    }
    .workflow-line {
        flex: 0 0 18px;
        height: 1px;
        background: #CBD5E1;
    }
    .smartaira-hero {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.07), transparent 42%),
            linear-gradient(135deg, #0B1323 0%, #24344B 100%);
        color: #FFFFFF;
        padding: 1.1rem 1.4rem;
        border-radius: 18px;
        border-right: 6px solid #10B981;
        margin-bottom: 1.25rem;
        box-shadow:
            8px 8px 18px rgba(15, 23, 42, 0.25),
            -2px -2px 8px rgba(255,255,255,0.24),
            inset 1px 1px 0 rgba(255,255,255,0.15),
            inset -2px -2px 8px rgba(0,0,0,0.18);
    }
    .smartaira-badge {
        display: inline-block;
        background: linear-gradient(145deg, rgba(33, 206, 153, 0.26), rgba(1, 97, 73, 0.38));
        color: #A7F3D0;
        border: 1px solid rgba(167, 243, 208, 0.35);
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.65rem;
    }
    .smartaira-hero h1 {
        color: #FFFFFF;
        font-size: 1.85rem;
        margin: 0 0 0.4rem 0;
        font-weight: 700;
    }
    .smartaira-tagline-ar {
        color: #F59E0B;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
        direction: rtl;
        text-align: right;
    }
    .smartaira-sub-ar {
        color: #E2E8F0;
        font-size: 0.98rem;
        line-height: 1.65;
        direction: rtl;
        text-align: right;
        margin: 0;
    }
    .value-card {
        background: #E9EEF4;
        border: 1px solid rgba(255,255,255,0.62);
        border-top: 4px solid #10B981;
        border-radius: 15px;
        padding: 1.15rem 1.25rem;
        height: 100%;
        box-shadow: 8px 8px 16px #C5CED8, -8px -8px 16px #FFFFFF;
        direction: rtl;
        text-align: right;
    }
    .value-card.gold {
        border-top-color: #F59E0B;
    }
    .value-card.navy {
        border-top-color: #0F172A;
    }
    .value-card-title {
        color: #0F172A;
        font-weight: 700;
        font-size: 1.03rem;
        margin-bottom: 0.4rem;
    }
    .value-card-body {
        color: #475569;
        font-size: 0.89rem;
        line-height: 1.58;
        margin: 0;
    }
    .status-strip {
        background: #E9EEF4;
        border: 1px solid rgba(255,255,255,0.62);
        border-radius: 15px;
        padding: 0.7rem 1rem;
        margin: 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
        color: #0F172A;
        box-shadow: 7px 7px 14px #C7D0DB, -7px -7px 14px #FFFFFF;
    }
    .status-pill-active {
        background: #DFF4EA;
        color: #065F46;
        border: 1px solid #77D9B5;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        box-shadow: inset 2px 2px 5px rgba(5,111,81,0.13), inset -2px -2px 5px rgba(255,255,255,0.9);
    }
    .hint-box {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .privacy-box {
        background: linear-gradient(145deg, #22324A, #0D1729);
        color: #E2E8F0;
        padding: 0.85rem;
        border-radius: 10px;
        font-size: 0.82rem;
        line-height: 1.55;
        border-right: 4px solid #10B981;
        box-shadow: 5px 5px 11px rgba(15,23,42,0.23), inset 1px 1px 0 rgba(255,255,255,0.12);
        direction: rtl;
        text-align: right;
    }
    .exec-callout {
        background: #E1F5EB;
        border: 1px solid rgba(255,255,255,0.64);
        border-right: 5px solid #10B981;
        color: #064E3B;
        padding: 0.9rem 1.15rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        font-size: 0.94rem;
        line-height: 1.6;
        direction: rtl;
        text-align: right;
        box-shadow: 7px 7px 14px rgba(9, 126, 91, 0.14), -7px -7px 14px #FFFFFF, inset 1px 1px 0 rgba(255,255,255,0.88);
    }
    .brand-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.8rem;
        padding-top: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
"""

def get_no_cache_meta(build_id: str) -> str:
    """Version the browser URL and clear stale browser caches on a new deployment."""
    return f"""
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate, max-age=0">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <script>
      (() => {{
        const buildId = "{build_id}";
        const storageKey = "smartaira-active-build";
        const currentUrl = new URL(window.location.href);
        if (currentUrl.searchParams.get("build") !== buildId) {{
          currentUrl.searchParams.set("build", buildId);
          window.location.replace(currentUrl.toString());
          return;
        }}
        if (window.localStorage.getItem(storageKey) !== buildId) {{
          window.localStorage.setItem(storageKey, buildId);
          if ("caches" in window) {{
            window.caches.keys().then((keys) => Promise.all(keys.map((key) => window.caches.delete(key))));
          }}
        }}
      }})();
    </script>
    """


def get_secret(key: str, default: str) -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


def has_configured_login() -> bool:
    """Use demo login only when no complete production credentials are configured."""
    try:
        return bool(st.secrets.get("DEMO_USERNAME")) and bool(st.secrets.get("DEMO_PASSWORD"))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Sample Retail Dataset Generator (720 rows: 180 days x 4 branches)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_sample_retail_csv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2026-05-18", periods=180, freq="D")
    branches = {
        "Nasr_City": {"base": 42000.0, "weekend_mult": 1.38, "promo_lift": 1.38},
        "Sheikh_Zayed": {"base": 38000.0, "weekend_mult": 1.45, "promo_lift": 1.34},
        "Smouha": {"base": 33000.0, "weekend_mult": 1.35, "promo_lift": 1.32},
        "Maadi": {"base": 29000.0, "weekend_mult": 1.30, "promo_lift": 1.28},
    }

    rows = []
    for dt in dates:
        dow = dt.dayofweek
        day = dt.day
        is_weekend = dow in (4, 5)
        is_thursday = dow == 3
        is_tuesday = dow == 1
        is_payday = 1 if (day >= 25 or day <= 3) else 0

        if day in (26, 27, 28):
            is_promo, discount_pct = 1, 20
        elif day in (10, 11):
            is_promo, discount_pct = 1, 15
        else:
            is_promo, discount_pct = 0, 0

        unit_price = 195.00 if dt >= pd.Timestamp("2026-08-15") else 180.00

        for branch_name, cfg in branches.items():
            sales = cfg["base"]
            if is_weekend:
                sales *= cfg["weekend_mult"]
            elif is_thursday:
                sales *= 1.15
            elif is_tuesday:
                sales *= 0.88

            if is_payday:
                sales *= 1.20 if branch_name == "Maadi" else 1.18

            if is_promo:
                sales *= cfg["promo_lift"] if discount_pct == 20 else (cfg["promo_lift"] - 0.08)

            if unit_price > 180.0:
                sales *= 1.03
            noise = rng.normal(1.0, 0.045)
            sales = round(max(sales * noise, 5000.0), 2)

            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "branch": branch_name,
                    "avg_unit_price": unit_price,
                    "is_promo": is_promo,
                    "discount_pct": discount_pct,
                    "is_payday_window": is_payday,
                    "sales_amount": sales,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Date Parsing & CSV Validation Engine
# -----------------------------------------------------------------------------
def parse_flexible_dates(series: pd.Series) -> pd.Series:
    s_clean = series.astype(str).str.strip()
    s_clean = s_clean.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})

    ym_mask = s_clean.str.match(r"^\d{4}-\d{2}$", na=False)
    if ym_mask.mean() > 0.8:
        return pd.to_datetime(s_clean + "-01", format="%Y-%m-%d", errors="coerce")

    iso_mask = s_clean.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.mean() > 0.8:
        return pd.to_datetime(s_clean, errors="coerce").dt.normalize()

    return pd.to_datetime(s_clean, dayfirst=True, errors="coerce").dt.normalize()


def detect_granularity(unique_dates: pd.DatetimeIndex) -> tuple[str, pd.DateOffset]:
    if len(unique_dates) < 2:
        return t("granularity.daily"), pd.DateOffset(days=1)
    diffs = pd.Series(unique_dates).sort_values().diff().dropna().dt.days
    median_days = float(diffs.median())
    if median_days >= 25:
        return t("granularity.monthly"), pd.DateOffset(months=1)
    if median_days >= 5:
        return t("granularity.weekly"), pd.DateOffset(weeks=1)
    return t("granularity.daily"), pd.DateOffset(days=1)


def aggregate_and_complete_sales(
    work: pd.DataFrame, date_col: str, target_col: str, group_col: Optional[str]
) -> pd.DataFrame:
    """Aggregate transactions and add missing calendar periods for forecasting."""
    key_cols = ["_parsed_date"] + ([group_col] if group_col else [])
    aggregations = {}
    for col in work.columns:
        if col in key_cols:
            continue
        if col == target_col:
            aggregations[col] = "sum"
        elif pd.api.types.is_numeric_dtype(work[col]):
            aggregations[col] = "mean"
        else:
            aggregations[col] = "first"

    aggregated = work.groupby(key_cols, as_index=False, dropna=False).agg(aggregations)
    if group_col:
        grouped_items = list(aggregated.groupby(group_col, dropna=False, sort=False))
    else:
        grouped_items = [(None, aggregated)]

    completed_frames = []
    for group_value, series in grouped_items:
        start_date = series["_parsed_date"].min()
        end_date = series["_parsed_date"].max()
        full_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        grid = pd.DataFrame({"_parsed_date": full_dates})
        if group_col:
            grid[group_col] = group_value
        merged = grid.merge(series, on=key_cols, how="left", suffixes=("", "_source"))
        merged[target_col] = merged[target_col].fillna(0.0)
        for col in aggregations:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(work[col]):
                merged[col] = merged[col].ffill().bfill().fillna(0.0)
            else:
                merged[col] = merged[col].ffill().bfill().fillna(t("common.unspecified"))
        merged[date_col] = merged["_parsed_date"].dt.strftime("%Y-%m-%d")
        completed_frames.append(merged)

    return pd.concat(completed_frames, ignore_index=True).sort_values(key_cols).reset_index(drop=True)


def validate_dataset(
    df: pd.DataFrame, date_col: str, target_col: str, group_col: Optional[str]
) -> tuple[bool, str, Optional[pd.DataFrame]]:
    if df is None or df.empty:
        return False, t("validation.empty_file"), None

    if len(df) > 10_000:
        return (
            False,
            t("validation.max_rows", rows=len(df)),
            None,
        )

    if date_col not in df.columns or target_col not in df.columns or date_col == target_col:
        return (
            False,
            t("validation.same_columns"),
            None,
        )

    if group_col and (group_col == date_col or group_col == target_col):
        return (
            False,
            t("validation.invalid_group"),
            None,
        )

    work = df.copy()
    parsed_dates = parse_flexible_dates(work[date_col])
    invalid_ratio = parsed_dates.isna().mean()
    if invalid_ratio > 0.05 or parsed_dates.isna().any():
        bad_idx = int(parsed_dates[parsed_dates.isna()].index[0]) + 2
        return (
            False,
            t("validation.invalid_date", date_col=date_col, row=bad_idx),
            None,
        )
    work["_parsed_date"] = parsed_dates

    raw_target = work[target_col].astype(str).str.strip().str.replace(",", "", regex=False)
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    if numeric_target.isna().any():
        bad_idx = int(numeric_target[numeric_target.isna()].index[0]) + 2
        bad_val = work.loc[numeric_target.isna()].iloc[0][target_col]
        return (
            False,
            t("validation.invalid_value", target_col=target_col, value=bad_val, row=bad_idx),
            None,
        )
    work[target_col] = numeric_target.astype(float)

    if group_col:
        work[group_col] = work[group_col].astype(str).str.strip()

    work = aggregate_and_complete_sales(work, date_col, target_col, group_col)

    if len(work) < 30:
        return (
            False,
            t("validation.insufficient_rows", rows=len(work)),
            None,
        )

    if work[target_col].nunique() <= 1:
        return (
            False,
            t("validation.constant_values", target_col=target_col),
            None,
        )

    if group_col:
        counts = work.groupby(group_col).size()
        if counts.max() < 15:
            return (
                False,
                t("validation.insufficient_group_rows"),
                None,
            )

    return True, t("validation.verified"), work


# -----------------------------------------------------------------------------
# Feature Engineering & Auto-Horizon Forecasting Engine (FLAML + SHAP)
# -----------------------------------------------------------------------------
def build_calendar_features(df: pd.DataFrame, date_series: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["cal_year"] = date_series.dt.year
    out["cal_month"] = date_series.dt.month
    out["cal_day"] = date_series.dt.day
    out["cal_dayofweek"] = date_series.dt.dayofweek
    out["cal_weekofyear"] = date_series.dt.isocalendar().week.astype(int)
    out["cal_is_weekend_fri_sat"] = date_series.dt.dayofweek.isin([4, 5]).astype(int)
    out["cal_is_month_end_payday"] = ((date_series.dt.day >= 25) | (date_series.dt.day <= 3)).astype(int)
    return out


def is_string_or_cat(series: pd.Series) -> bool:
    return (
        pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or series.dtype == object
    )


def run_auto_horizon_forecast(
    validated_df: pd.DataFrame,
    original_cols: list[str],
    date_col: str,
    target_col: str,
    group_col: Optional[str],
    horizon: int,
    approval_status: str,
    scenario_name: Optional[str] = None,
    scenario_multiplier: float = 1.0,
    planned_promo: bool = False,
    planned_discount_pct: float = 0.0,
    planned_price_change_pct: float = 0.0,
) -> dict:
    scenario_name = scenario_name or t("scenario.baseline")
    df = validated_df.sort_values("_parsed_date").reset_index(drop=True)
    unique_dates = pd.DatetimeIndex(df["_parsed_date"].unique()).sort_values()
    granularity, step_offset = detect_granularity(unique_dates)

    cutoff_idx = max(1, int(len(unique_dates) * 0.7))
    recent_start = unique_dates[cutoff_idx]

    if group_col:
        all_groups = df[group_col].dropna().unique().tolist()
        recent_sales = df[df["_parsed_date"] >= recent_start].groupby(group_col)[target_col].sum()
        active_groups = [g for g, val in recent_sales.items() if val > 0]
        if not active_groups:
            active_groups = all_groups
        df_active = df[df[group_col].isin(active_groups)].copy()
        excluded_groups = [g for g in all_groups if g not in active_groups]
    else:
        active_groups = [None]
        excluded_groups = []
        df_active = df.copy()

    feature_cols = [c for c in original_cols if c not in (date_col, target_col)]
    event_pattern = re.compile(r"(promo|discount|campaign|offer|holiday|flash)", re.IGNORECASE)

    future_records = []
    for grp in active_groups:
        sub = df_active[df_active[group_col] == grp] if group_col else df_active
        max_dt = sub["_parsed_date"].max()
        last_row = sub.iloc[-1]

        for step in range(1, horizon + 1):
            f_dt = max_dt + (step_offset * step)
            rec = {
                date_col: f_dt.strftime("%Y-%m-%d"),
                "_parsed_date": f_dt,
                target_col: np.nan,
            }
            for col in feature_cols:
                if group_col and col == group_col:
                    rec[col] = grp
                elif "discount" in col.lower():
                    rec[col] = planned_discount_pct if planned_promo else 0
                elif event_pattern.search(col):
                    rec[col] = 1 if planned_promo else 0
                elif any(token in col.lower() for token in ("price", "cost", "rate")):
                    base_value = last_row[col]
                    rec[col] = base_value * (1 + planned_price_change_pct / 100.0)
                elif "payday" in col.lower():
                    rec[col] = 1 if (f_dt.day >= 25 or f_dt.day <= 3) else 0
                else:
                    rec[col] = last_row[col]
            future_records.append(rec)

    future_df = pd.DataFrame(future_records)

    X_hist_raw = build_calendar_features(df_active[feature_cols], df_active["_parsed_date"])
    X_fut_raw = build_calendar_features(future_df[feature_cols], future_df["_parsed_date"])
    y_hist = df_active[target_col].to_numpy(dtype=float)

    cat_cols = [c for c in X_hist_raw.columns if is_string_or_cat(X_hist_raw[c])]
    for col in cat_cols:
        cats = sorted(X_hist_raw[col].astype(str).unique().tolist())
        X_hist_raw[col] = pd.Categorical(X_hist_raw[col].astype(str), categories=cats)
        X_fut_raw[col] = pd.Categorical(X_fut_raw[col].astype(str), categories=cats)

    for col in X_hist_raw.columns:
        if col not in cat_cols:
            X_hist_raw[col] = pd.to_numeric(X_hist_raw[col], errors="coerce").fillna(0.0)
            X_fut_raw[col] = pd.to_numeric(X_fut_raw[col], errors="coerce").fillna(0.0)

    # 1. Out-of-sample backtest on trailing 20% for honest MAPE benchmark
    split_pos = int(len(df_active) * 0.8)
    non_zero_val = y_hist[split_pos:] > 0
    baseline_mape_pct = None
    wape_pct = None
    mae_value = None
    bias_pct = None
    if len(df_active) >= 40 and non_zero_val.sum() >= 5:
        bt_model = AutoML()
        bt_model.fit(
            X_hist_raw.iloc[:split_pos],
            y_hist[:split_pos],
            task="regression",
            metric="mape",
            time_budget=5,
            estimator_list=["lgbm"],
            verbose=0,
        )
        val_preds = np.maximum(bt_model.predict(X_hist_raw.iloc[split_pos:]), 0.0)
        mape_pct = round(
            float(mean_absolute_percentage_error(y_hist[split_pos:][non_zero_val], val_preds[non_zero_val])) * 100,
            1,
        )
        baseline_value = float(np.mean(y_hist[max(0, split_pos - 7):split_pos]))
        baseline_preds = np.full(len(y_hist[split_pos:]), baseline_value)
        baseline_mape_pct = round(
            float(mean_absolute_percentage_error(y_hist[split_pos:][non_zero_val], baseline_preds[non_zero_val])) * 100,
            1,
        )
        val_actuals = y_hist[split_pos:]
        val_errors = val_preds - val_actuals
        mae_value = round(float(np.mean(np.abs(val_errors))), 2)
        wape_pct = round(float(np.sum(np.abs(val_errors)) / max(np.sum(np.abs(val_actuals)), 1e-9)) * 100, 1)
        bias_pct = round(float(np.sum(val_errors) / max(np.sum(np.abs(val_actuals)), 1e-9)) * 100, 1)
    else:
        mape_pct = None

    accuracy_pct = round(100.0 - mape_pct, 1) if mape_pct is not None else None

    # 2. Full-history FLAML fit for production forecast
    automl = AutoML()
    automl.fit(
        X_hist_raw,
        y_hist,
        task="regression",
        metric="mape",
        time_budget=12,
        estimator_list=["lgbm"],
        verbose=0,
    )

    hist_preds = np.maximum(automl.predict(X_hist_raw), 0.0)
    fut_preds = np.maximum(automl.predict(X_fut_raw), 0.0) * scenario_multiplier
    uncertainty_pct = max(5.0, min(35.0, mape_pct * 1.25)) if mape_pct is not None else 20.0
    fut_low = np.maximum(fut_preds * (1 - uncertainty_pct / 100.0), 0.0)
    fut_high = fut_preds * (1 + uncertainty_pct / 100.0)

    # 3. Compute SHAP explanations on future forecast rows
    estimator = automl.model.estimator
    try:
        expected_features = getattr(estimator, "n_features_in_", None)
        if expected_features is not None and expected_features != X_fut_raw.shape[1]:
            raise ValueError("The selected model does not expose the same feature count for explanations.")
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_fut_raw)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim != 2 or shap_vals.shape[1] != X_fut_raw.shape[1]:
            raise ValueError("The explanation output does not match the forecast feature columns.")
    except Exception:
        # SHAP can fail with some LightGBM/SHAP combinations when categorical
        # columns are present. Forecasting remains usable without explanations.
        shap_vals = np.zeros((len(X_fut_raw), X_fut_raw.shape[1]), dtype=float)

    friendly_names = {
        "cal_is_weekend_fri_sat": t("drivers.weekend"),
        "cal_is_month_end_payday": t("drivers.payday"),
        "cal_dayofweek": t("drivers.day_of_week"),
        "cal_month": t("drivers.month_seasonality"),
        "cal_day": t("drivers.day_of_month"),
        "cal_weekofyear": t("drivers.week_of_year"),
        "cal_year": t("drivers.yearly_trend"),
    }

    top_drivers_per_row = []
    for i in range(len(X_fut_raw)):
        row_s = shap_vals[i]
        top_idx = np.argsort(np.abs(row_s))[::-1][:3]
        parts = []
        for j in top_idx:
            raw_name = X_fut_raw.columns[j]
            label = friendly_names.get(raw_name, raw_name)
            val = X_fut_raw.iloc[i, j]
            impact = row_s[j]
            if abs(impact) >= 1.0:
                parts.append(f"{label}={val} ({impact:+,.0f})")
        top_drivers_per_row.append(", ".join(parts) if parts else t("drivers.base_trend"))

    mean_shap = shap_vals.mean(axis=0)
    driver_summary = pd.DataFrame(
        {
            "Driver": [friendly_names.get(c, c) for c in X_fut_raw.columns],
            "Avg_Impact": mean_shap,
            "Abs_Impact": np.abs(mean_shap),
        }
    )
    driver_summary = (
        driver_summary[driver_summary["Abs_Impact"] > 0.5]
        .sort_values("Abs_Impact", ascending=False)
        .head(8)
        .copy()
    )
    driver_summary["Direction"] = np.where(
        driver_summary["Avg_Impact"] >= 0,
        t("charts.positive_impact"),
        t("charts.negative_impact"),
    )

    top_driver_name = (
        driver_summary.iloc[0]["Driver"] if not driver_summary.empty else t("drivers.historical_trend")
    )

    # Assemble combined output table preserving all original columns
    hist_out = df_active[original_cols].copy()
    hist_out["_sort_date"] = df_active["_parsed_date"]
    hist_out["smartaira_row_type"] = "HISTORICAL"
    hist_out["smartaira_predicted_value"] = np.round(hist_preds, 2)
    hist_out["smartaira_forecast_low"] = np.nan
    hist_out["smartaira_forecast_high"] = np.nan
    hist_out["smartaira_top_drivers"] = t("drivers.actual_history")
    mape_display = f"{mape_pct:.1f}%" if mape_pct is not None else t("common.not_available")
    hist_out["smartaira_model_mape_pct"] = mape_display
    hist_out["smartaira_approval_status"] = approval_status

    fut_out = future_df[original_cols].copy()
    fut_out["_sort_date"] = future_df["_parsed_date"]
    fut_out["smartaira_row_type"] = "FORECAST"
    fut_out["smartaira_predicted_value"] = np.round(fut_preds, 2)
    fut_out["smartaira_forecast_low"] = np.round(fut_low, 2)
    fut_out["smartaira_forecast_high"] = np.round(fut_high, 2)
    fut_out["smartaira_top_drivers"] = top_drivers_per_row
    fut_out["smartaira_model_mape_pct"] = mape_display
    fut_out["smartaira_approval_status"] = approval_status

    history_with_preds = df_active[[date_col] + ([group_col] if group_col else []) + [target_col]].copy()
    history_with_preds["_historical_prediction"] = hist_preds
    series_accuracy_rows = []
    accuracy_groups = active_groups if group_col else [None]
    for accuracy_group in accuracy_groups:
        sample = (
            history_with_preds[history_with_preds[group_col] == accuracy_group]
            if group_col
            else history_with_preds
        )
        actual = sample[target_col].to_numpy(dtype=float)
        predicted = sample["_historical_prediction"].to_numpy(dtype=float)
        non_zero = actual > 0
        series_accuracy_rows.append(
            {
                (group_col or t("common.series")): accuracy_group or t("common.company_total"),
                t("table.historical_error"): round(
                    float(mean_absolute_percentage_error(actual[non_zero], predicted[non_zero])) * 100, 1
                ) if non_zero.any() else None,
                t("table.total_error"): round(
                    float(np.sum(np.abs(predicted - actual)) / max(np.sum(np.abs(actual)), 1e-9)) * 100, 1
                ),
                t("table.average_error"): round(float(np.mean(np.abs(predicted - actual))), 2),
                t("table.bias"): round(float(np.mean(predicted - actual)), 2),
            }
        )
    series_accuracy_df = pd.DataFrame(series_accuracy_rows)

    combined = pd.concat([hist_out, fut_out], ignore_index=True).sort_values("_sort_date")

    total_forecast = float(fut_preds.sum())
    n_series = len(active_groups)
    recent_hist_slice = df_active.sort_values("_parsed_date").tail(horizon * n_series)
    prior_total = float(recent_hist_slice[target_col].sum())
    growth_pct = ((total_forecast - prior_total) / prior_total * 100.0) if prior_total > 0 else 0.0

    # Build Series-Level Executive Summary Table
    previous_sales_col = t("table.previous_sales", horizon=horizon)
    forecast_sales_col = t("table.forecast_sales", horizon=horizon)
    change_col = t("charts.change")
    daily_forecast_col = t("table.daily_forecast")
    forecast_low_col = t("table.forecast_low")
    forecast_high_col = t("table.forecast_high")
    recommended_action_col = t("table.recommended_action")
    peak_day_col = t("table.peak_day")
    if group_col:
        series_rows = []
        for grp in active_groups:
            g_fut = fut_out[fut_out[group_col] == grp]["smartaira_predicted_value"]
            g_hist = (
                df_active[df_active[group_col] == grp]
                .sort_values("_parsed_date")
                .tail(horizon)[target_col]
            )
            fut_sum = float(g_fut.sum())
            hist_sum = float(g_hist.sum())
            g_delta = ((fut_sum - hist_sum) / hist_sum * 100.0) if hist_sum > 0 else 0.0
            series_rows.append(
                {
                    group_col: grp,
                    previous_sales_col: round(hist_sum, 2),
                    forecast_sales_col: round(fut_sum, 2),
                    change_col: round(g_delta, 1),
                    daily_forecast_col: round(float(g_fut.mean()), 2),
                    forecast_low_col: round(float(fut_out[fut_out[group_col] == grp]["smartaira_forecast_low"].sum()), 2),
                    forecast_high_col: round(float(fut_out[fut_out[group_col] == grp]["smartaira_forecast_high"].sum()), 2),
                    recommended_action_col: (
                        t("actions.increase_stock") if g_delta >= 10 else
                        t("actions.review_activation") if g_delta <= -10 else
                        t("actions.continue_monitoring")
                    ),
                    peak_day_col: str(
                        fut_out[fut_out[group_col] == grp]
                        .sort_values("smartaira_predicted_value", ascending=False)
                        .iloc[0][date_col]
                    ),
                }
            )
        series_summary_df = pd.DataFrame(series_rows).sort_values(
            forecast_sales_col, ascending=False
        )
    else:
        series_summary_df = pd.DataFrame(
            [
                {
                    t("common.series"): t("common.company_total"),
                    previous_sales_col: round(prior_total, 2),
                    forecast_sales_col: round(total_forecast, 2),
                    change_col: round(growth_pct, 1),
                    daily_forecast_col: round(float(fut_out["smartaira_predicted_value"].mean()), 2),
                    forecast_low_col: round(float(fut_out["smartaira_forecast_low"].sum()), 2),
                    forecast_high_col: round(float(fut_out["smartaira_forecast_high"].sum()), 2),
                    recommended_action_col: (
                        t("actions.increase_stock") if growth_pct >= 10 else
                        t("actions.review_activation") if growth_pct <= -10 else
                        t("actions.continue_monitoring")
                    ),
                    peak_day_col: str(
                        fut_out.sort_values("smartaira_predicted_value", ascending=False).iloc[0][date_col]
                    ),
                }
            ]
        )

    return {
        "combined_df": combined,
        "series_summary_df": series_summary_df,
        "series_accuracy_df": series_accuracy_df,
        "driver_summary": driver_summary,
        "granularity": granularity,
        "n_series": n_series,
        "active_groups": active_groups,
        "excluded_groups": excluded_groups,
        "mape_pct": mape_pct,
        "baseline_mape_pct": baseline_mape_pct,
        "wape_pct": wape_pct,
        "mae_value": mae_value,
        "bias_pct": bias_pct,
        "accuracy_pct": accuracy_pct,
        "best_estimator": str(automl.best_estimator).upper(),
        "total_forecast": total_forecast,
        "growth_pct": growth_pct,
        "scenario_name": scenario_name,
        "uncertainty_pct": uncertainty_pct,
        "forecast_low_total": float(fut_low.sum()),
        "forecast_high_total": float(fut_high.sum()),
        "top_driver_name": top_driver_name,
        "forecast_start_date": future_df["_parsed_date"].min(),
        "date_col": date_col,
        "target_col": target_col,
        "group_col": group_col,
        "horizon": horizon,
    }


# -----------------------------------------------------------------------------
# Visualization Builders (3D Surface + Executive Accuracy Gauge)
# -----------------------------------------------------------------------------
def build_3d_forecast_figure(res: dict) -> go.Figure:
    combined_df: pd.DataFrame = res["combined_df"]
    fut_df = combined_df[combined_df["smartaira_row_type"] == "FORECAST"].copy()
    g_col = res["group_col"]
    d_col = res["date_col"]

    if g_col and fut_df[g_col].nunique() >= 2:
        pivot = (
            fut_df.pivot_table(
                index=g_col,
                columns=d_col,
                values="smartaira_predicted_value",
                aggfunc="sum",
            )
            .sort_index()
        )
        y_labels = pivot.index.astype(str).tolist()
        x_labels = pivot.columns.astype(str).tolist()
        z_matrix = pivot.to_numpy(dtype=float)
        y_title = g_col.replace("_", " ").title()
    else:
        daily = (
            fut_df.groupby(d_col, as_index=False)["smartaira_predicted_value"]
            .sum()
            .sort_values(d_col)
        )
        x_labels = daily[d_col].astype(str).tolist()
        vals = daily["smartaira_predicted_value"].to_numpy(dtype=float)
        y_labels = [
            t("charts.cautious"),
            t("charts.smartaira_forecast"),
            t("charts.optimistic"),
        ]
        z_matrix = np.vstack([vals * 0.94, vals, vals * 1.06])
        y_title = t("charts.scenario_range")

    x_idx = np.arange(len(x_labels))
    y_idx = np.arange(len(y_labels))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_idx,
            y=y_idx,
            z=z_matrix,
            colorscale=[
                [0.0, "#0F172A"],
                [0.5, "#10B981"],
                [1.0, "#F59E0B"],
            ],
            opacity=0.88,
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "#FFFFFF",
                    "project": {"z": True},
                }
            },
            colorbar={"title": t("charts.forecast_sales"), "thickness": 14},
            hovertemplate=(
                f"<b>{t('charts.hover_date')}</b> %{{text}}<br>"
                f"<b>{y_title}:</b> %{{customdata}}<br>"
                f"<b>{t('charts.hover_forecast')}</b> %{{z:,.0f}}<extra></extra>"
            ),
            text=np.tile(x_labels, (len(y_labels), 1)),
            customdata=np.tile(np.array(y_labels)[:, None], (1, len(x_labels))),
        )
    )

    xx, yy = np.meshgrid(x_idx, y_idx)
    fig.add_trace(
        go.Scatter3d(
            x=xx.ravel(),
            y=yy.ravel(),
            z=z_matrix.ravel(),
            mode="markers",
            marker={
                "size": 4.5,
                "color": "#FFFFFF",
                "line": {"color": "#0F172A", "width": 1.5},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        paper_bgcolor="#FFFFFF",
        scene={
            "xaxis": {
                "title": t("charts.forecast_date"),
                "tickvals": x_idx[:: max(1, len(x_idx) // 6)],
                "ticktext": x_labels[:: max(1, len(x_labels) // 6)],
                "backgroundcolor": "#F8FAFC",
                "gridcolor": "#CBD5E1",
            },
            "yaxis": {
                "title": y_title,
                "tickvals": y_idx,
                "ticktext": y_labels,
                "backgroundcolor": "#F8FAFC",
                "gridcolor": "#CBD5E1",
            },
            "zaxis": {
                "title": t("charts.forecast_sales"),
                "backgroundcolor": "#F8FAFC",
                "gridcolor": "#CBD5E1",
            },
            "camera": {"eye": {"x": 1.65, "y": -1.55, "z": 1.15}},
        },
    )
    return fig


def build_accuracy_gauge(accuracy_pct: float, mape_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=accuracy_pct,
            number={"suffix": "%", "font": {"size": 34, "color": "#0F172A"}},
            delta={"reference": 85.0, "increasing": {"color": "#10B981"}, "suffix": t("charts.accuracy_gauge_delta")},
            title={"text": t("charts.accuracy_gauge_title", error=mape_pct), "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [50, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#10B981"},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [50, 75], "color": "#FEE2E2"},
                    {"range": [75, 88], "color": "#FEF3C7"},
                    {"range": [88, 100], "color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.75, "value": 90},
            },
        )
    )
    fig.update_layout(height=250, margin={"l": 20, "r": 20, "t": 45, "b": 10}, paper_bgcolor="#FFFFFF")
    return fig


# -----------------------------------------------------------------------------
# Main Application Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    active_lang = init_i18n()
    st.set_page_config(
        page_title=t("app.page_title"),
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_direction_styles()
    build_id = get_build_id()
    if st.session_state.get("_active_build_id") != build_id:
        st.cache_data.clear()
        st.session_state["_active_build_id"] = build_id
    if st.query_params.get("build") != build_id:
        st.query_params["build"] = build_id
    st.markdown(get_no_cache_meta(build_id), unsafe_allow_html=True)
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    if st.session_state.get("language_selector") != active_lang:
        st.session_state["language_selector"] = active_lang
    selector_col, design_col = st.columns(2)
    with selector_col:
        st.selectbox(
            t("language.selector"),
            options=list(SUPPORTED_LANGUAGES),
            format_func=lambda code: SUPPORTED_LANGUAGES[code]["label"],
            key="language_selector",
            on_change=on_language_change,
        )
    with design_col:
        design_style = st.selectbox(
            t("design.selector"),
            options=DESIGN_OPTIONS,
            index=0,
            format_func=lambda style: t(f"design.{style}"),
            help=t("design.help"),
            key="design_selector",
        )
    st.markdown(get_design_css(design_style), unsafe_allow_html=True)
    default_state = {
        "authenticated": False,
        "current_user": None,
        "raw_df": None,
        "source_name": None,
        "forecast_result": None,
        "approval_status": "PENDING",
        "approved_at": None,
        "approved_by": None,
        "approval_comment": "",
        "forecast_history": [],
        "language_changed_notice": False,
    }
    for k, v in default_state.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.get("language_changed_notice"):
        st.info(t("language.results_reset"))
        st.session_state["language_changed_notice"] = False

    if not st.session_state["authenticated"]:
        render_build_id(build_id)
        st.markdown(
            f"""
            <style>
                [data-testid="stSidebar"] {{ display: none; }}
            </style>
            <div class="login-card">
                <h2>🔐 {t("auth.title")}</h2>
                <p>{t("auth.subtitle")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        expected_user = get_secret("DEMO_USERNAME", "smartaira").strip()
        expected_password = get_secret("DEMO_PASSWORD", "smartaira").strip()
        configured_login = has_configured_login()
        with st.form("mobile_login_form"):
            username_input = st.text_input(t("auth.username"), value="smartaira")
            password_input = st.text_input(t("auth.password"), value="smartaira", type="password")
            submitted = st.form_submit_button(t("auth.sign_in"), type="primary", width="stretch")
            if submitted:
                if configured_login:
                    credentials_valid = (
                        username_input.strip().lower() == expected_user.lower()
                        and password_input.strip() == expected_password
                    )
                else:
                    credentials_valid = (
                        username_input.strip().lower() == "smartaira"
                        and password_input.strip().lower() == "smartaira"
                    )
                if credentials_valid:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = username_input.strip()
                    st.rerun()
                else:
                    st.error(t("auth.invalid"))
        if configured_login:
            st.info(t("auth.configured_info"))
        else:
            st.info(t("auth.demo_info"))
        st.stop()

    if st.session_state["forecast_result"] is None:
        render_build_id(build_id)

    # Compact product header and visible workflow progress.
    st.markdown(
        f"""
        <div class="smartaira-hero">
            <span class="smartaira-badge">{t("header.badge")}</span>
            <h1>{t("header.title")}</h1>
            <div class="smartaira-tagline-ar">{t("header.subtitle")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_step = 3 if st.session_state["forecast_result"] is not None else (2 if st.session_state["raw_df"] is not None else 1)
    step_labels = [t("workflow.data"), t("workflow.setup"), t("workflow.result")]
    step_markup = "".join(
        f'<div class="workflow-step {"active" if i + 1 == active_step else ""}">{label}</div>'
        + ('<span class="workflow-line"></span>' if i < 2 else '')
        for i, label in enumerate(step_labels)
    )
    st.markdown(f'<div class="workflow-steps">{step_markup}</div>', unsafe_allow_html=True)

    # Main-page workflow controls: no sidebar required.
    st.markdown(f"### {t('data.heading')}")
    data_mode = st.radio(
        t("data.source"),
        [t("data.sample_mode"), t("data.upload_mode")],
        horizontal=True,
    )

    if data_mode == t("data.sample_mode"):
        if st.button(t("data.sample_button", rows=720), type="primary", width="stretch"):
            st.session_state["raw_df"] = generate_sample_retail_csv()
            st.session_state["source_name"] = "smartaira_sample_retail_sales.csv"
            st.session_state["forecast_result"] = None
            st.session_state["approval_status"] = "PENDING"
            st.session_state["approved_at"] = None
    else:
        template_df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "sales_amount": [12500, 13800],
                "branch": ["Cairo Branch", "Cairo Branch"],
            }
        )
        st.download_button(
            t("data.template_download"),
            data=template_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartaira_sales_template.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded = st.file_uploader(t("data.upload_label"), type=["csv"])
        if uploaded is not None:
            try:
                raw_bytes = uploaded.getvalue()
                df_up = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
                st.session_state["raw_df"] = df_up
                st.session_state["source_name"] = uploaded.name
                st.session_state["forecast_result"] = None
                st.session_state["approval_status"] = "PENDING"
                st.session_state["approved_at"] = None
            except Exception as exc:
                st.error(t("data.upload_error", error=exc))

    df_loaded: Optional[pd.DataFrame] = st.session_state["raw_df"]
    date_col = target_col = None
    group_col: Optional[str] = None
    horizon = 30
    scenario_name = t("scenario.baseline")
    scenario_multiplier = 1.0
    planned_promo = False
    planned_discount_pct = 0.0
    planned_price_change_pct = 0.0
    sales_target = 0.0

    if df_loaded is not None:
        st.divider()
        st.subheader(t("settings.heading"))
        cols = df_loaded.columns.tolist()

        default_date_idx = next((i for i, c in enumerate(cols) if "date" in c.lower()), 0)
        default_target_idx = next(
            (i for i, c in enumerate(cols) if any(k in c.lower() for k in ("sales", "revenue", "amount", "qty"))),
            len(cols) - 1,
        )
        no_group_label = t("settings.no_group")
        default_group_choices = [no_group_label] + cols
        default_group_idx = next(
            (i for i, c in enumerate(default_group_choices) if any(k in c.lower() for k in ("branch", "sku", "store", "region"))),
            0,
        )

        date_col = st.selectbox(t("settings.date_column"), options=cols, index=default_date_idx)
        target_col = st.selectbox(t("settings.target_column"), options=cols, index=default_target_idx)
        group_choice = st.selectbox(t("settings.group_column"), options=default_group_choices, index=default_group_idx)
        group_col = None if group_choice == no_group_label else group_choice
        horizon = st.selectbox(t("settings.horizon"), options=[7, 14, 30, 60], index=2, format_func=lambda x: t("settings.horizon_option", days=x))

        with st.expander(t("settings.improve"), expanded=False):
            scenario_options = {
                t("scenario.baseline"): 1.0,
                t("scenario.optimistic"): 1.10,
                t("scenario.cautious"): 0.90,
            }
            scenario_name = st.selectbox(t("settings.scenario"), options=list(scenario_options))
            scenario_multiplier = scenario_options[scenario_name]
            planned_promo = st.checkbox(t("settings.promo"))
            if planned_promo:
                planned_discount_pct = st.number_input(t("settings.discount"), min_value=0.0, max_value=90.0, value=10.0, step=1.0)
            planned_price_change_pct = st.number_input(t("settings.price_change"), min_value=-90.0, max_value=200.0, value=0.0, step=1.0)
            sales_target = st.number_input(t("settings.sales_target"), min_value=0.0, value=0.0, step=1000.0)

        run_clicked = st.button(t("settings.run"), type="primary", width="stretch")
    else:
        run_clicked = False

    with st.expander(t("privacy.title"), expanded=False):
        st.markdown(
            f'<div class="privacy-box">🔒 {t("privacy.body")}</div>',
            unsafe_allow_html=True,
        )
        if st.button(t("privacy.clear_logout"), width="stretch"):
            for k, v in default_state.items():
                st.session_state[k] = v
            st.rerun()

    # Main Canvas: Status Strip & Validation
    if df_loaded is None:
        st.markdown(
            f"""
            <div class="status-strip">
                <span><b>{t("data.empty_status")}</b></span>
                <span class="status-pill-active">{t("data.waiting_file")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(t("data.start_info"))
        st.markdown(
            f'<div class="brand-footer">{t("app.footer")}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    is_valid, val_msg, validated_df = validate_dataset(df_loaded, date_col, target_col, group_col)

    if not is_valid:
        st.error(f"⚠️ {val_msg}")
        st.stop()

    unique_dates = pd.DatetimeIndex(validated_df["_parsed_date"].unique()).sort_values()
    granularity_lbl, _ = detect_granularity(unique_dates)
    series_count = validated_df[group_col].nunique() if group_col else 1
    date_span = f"{unique_dates.min().strftime('%Y-%m-%d')} → {unique_dates.max().strftime('%Y-%m-%d')}"

    stage_label = (
        t("status.ready")
        if st.session_state["forecast_result"] is not None
        else t("status.validated", granularity=granularity_lbl, series_count=series_count)
    )

    st.markdown(
        f"""
        <div class="status-strip">
            <span><b>{t("data.file_status", file_name=st.session_state['source_name'], rows=len(validated_df), date_span=date_span)}</b></span>
            <span class="status-pill-active">{stage_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(t("data.preview"), expanded=False):
        st.dataframe(df_loaded.head(5), width="stretch", hide_index=True)

    if run_clicked:
        with st.spinner(t("spinner.forecasting")):
            result = run_auto_horizon_forecast(
                validated_df=validated_df,
                original_cols=df_loaded.columns.tolist(),
                date_col=date_col,
                target_col=target_col,
                group_col=group_col,
                horizon=horizon,
                approval_status="PENDING",
                scenario_name=scenario_name,
                scenario_multiplier=scenario_multiplier,
                planned_promo=planned_promo,
                planned_discount_pct=planned_discount_pct,
                planned_price_change_pct=planned_price_change_pct,
            )
            result["sales_target"] = sales_target
            result["target_gap"] = sales_target - result["total_forecast"] if sales_target > 0 else None
            st.session_state["forecast_result"] = result
            st.session_state["forecast_history"].append(
                {
                    t("charts.forecast_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                    t("settings.scenario"): scenario_name,
                    t("kpi.total_forecast", horizon=horizon): result["total_forecast"],
                    t("charts.change"): result["growth_pct"],
                }
            )
            st.session_state["approval_status"] = "PENDING"
            st.session_state["approved_at"] = None
            st.rerun()

    res = st.session_state["forecast_result"]
    if res is None:
        st.info(t("settings.review_columns"))
        st.markdown(
            f'<div class="brand-footer">{t("app.footer")}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if res.get("excluded_groups"):
        excluded_text = ", ".join(str(item) for item in res["excluded_groups"][:8])
        st.warning(
            t("status.excluded", count=len(res["excluded_groups"]), groups=excluded_text)
        )

    st.divider()

    trend_word = t("result.increase") if res["growth_pct"] >= 0 else t("result.decrease")
    target_message = ""
    if res.get("sales_target"):
        gap = res["target_gap"]
        target_message = (
            " " + (t("result.target_above", gap=abs(gap)) if gap > 0 else
            t("result.forecast_above", gap=abs(gap)))
        )
    st.markdown(
        f"""
        <div class="exec-callout">
            <b>{t("result.summary", trend=trend_word, growth=abs(res['growth_pct']), horizon=res['horizon'], low=res['forecast_low_total'], high=res['forecast_high_total'], scenario=res['scenario_name'], target_message=target_message)}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Executive KPI Cards
    k1, k2 = st.columns(2)
    k3, k4 = st.columns(2)
    with k1:
        st.metric(
            label=t("kpi.total_forecast", horizon=res["horizon"]),
            value=f"{res['total_forecast']:,.0f}",
            delta=t("kpi.vs_previous", growth=res["growth_pct"]),
        )
    with k2:
        accuracy_display = f"{res['accuracy_pct']:.1f}%" if res.get("accuracy_pct") is not None else t("common.not_available")
        error_display = t("kpi.error_rate", error=res["mape_pct"]) if res.get("mape_pct") is not None else t("kpi.not_enough_data")
        st.metric(
            label=t("kpi.forecast_accuracy"),
            value=accuracy_display,
            delta=error_display,
            delta_color="off",
        )
    with k3:
        st.metric(
            label=t("kpi.forecast_scope"),
            value=t("kpi.active_series", count=res["n_series"]),
            delta=t("kpi.granularity_data", granularity=res["granularity"]),
            delta_color="off",
        )
    with k4:
        st.metric(
            label=t("kpi.top_driver"),
            value=res["top_driver_name"],
            delta=t("kpi.largest_impact"),
            delta_color="off",
        )

    st.markdown("---")

    # Multi-View Analytics Workspace: 2D Graphics + 3D Terrain + Tabular Plans
    tab_decision, tab_charts, tab_3d = st.tabs(
        [
            f"✅ {t('tabs.decision')}",
            f"📈 {t('tabs.charts')}",
            f"🌐 {t('tabs.branches')}",
        ]
    )

    combined_df: pd.DataFrame = res["combined_df"]
    g_col = res["group_col"]
    t_col = res["target_col"]

    # -------------------------------------------------------------------------
    # TAB 1: 2D Interactive Charts & Executive Gauge
    # -------------------------------------------------------------------------
    with tab_charts:
        render_build_id(build_id)
        chart_col1, chart_col2 = st.columns([1.45, 1.0], gap="large")

        with chart_col1:
            st.subheader(f"📈 {t('charts.history_forecast')}")
            if g_col:
                all_groups_label = t("common.all_groups")
                filter_options = [all_groups_label] + res["active_groups"]
                selected_series = st.selectbox(
                    t("charts.filter_series"),
                    options=filter_options,
                    key="chart_2d_series_filter",
                )
                if selected_series == all_groups_label:
                    plot_base = (
                        combined_df.groupby(["_sort_date", "smartaira_row_type"], as_index=False)
                        .agg({t_col: "sum", "smartaira_predicted_value": "sum"})
                    )
                else:
                    plot_base = combined_df[combined_df[g_col] == selected_series].copy()
            else:
                plot_base = combined_df.copy()

            hist_plot = plot_base[plot_base["smartaira_row_type"] == "HISTORICAL"][["_sort_date", t_col]].copy()
            hist_plot = hist_plot.rename(columns={"_sort_date": "Date", t_col: "Sales"})
            hist_plot["Series"] = t("charts.history_legend")

            fut_plot = plot_base[plot_base["smartaira_row_type"] == "FORECAST"][
                ["_sort_date", "smartaira_predicted_value"]
            ].copy()
            fut_plot = fut_plot.rename(columns={"_sort_date": "Date", "smartaira_predicted_value": "Sales"})
            fut_plot["Series"] = t("charts.forecast_legend")

            line_df = pd.concat([hist_plot, fut_plot], ignore_index=True)

            line_chart = (
                alt.Chart(line_df)
                .mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.6)
                .encode(
                    x=alt.X("Date:T", title=t("charts.date")),
                    y=alt.Y("Sales:Q", title=t("charts.sales_value")),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=[t("charts.history_legend"), t("charts.forecast_legend")],
                            range=["#0F172A", "#10B981"],
                        ),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    strokeDash=alt.StrokeDash(
                        "Series:N",
                        scale=alt.Scale(
                            domain=[t("charts.history_legend"), t("charts.forecast_legend")],
                            range=[[1, 0], [6, 4]],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title=t("charts.date"), format="%Y-%m-%d"),
                        alt.Tooltip("Series:N", title=t("charts.type")),
                        alt.Tooltip("Sales:Q", title=t("charts.sales"), format=",.2f"),
                    ],
                )
                .properties(height=360)
                .interactive()
            )

            cutoff_df = pd.DataFrame({"ForecastStart": [res["forecast_start_date"]]})
            rule_chart = (
                alt.Chart(cutoff_df)
                .mark_rule(color="#F59E0B", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="ForecastStart:T")
            )

            st.altair_chart(line_chart + rule_chart, width="stretch")

        with chart_col2:
            st.subheader(f"🔍 {t('charts.drivers')}")
            driver_df: pd.DataFrame = res["driver_summary"]
            if driver_df.empty:
                st.info(t("charts.usual_trend"))
            else:
                bar_chart = (
                    alt.Chart(driver_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Avg_Impact:Q", title=t("charts.average_impact")),
                        y=alt.Y("Driver:N", sort="-x", title=None),
                        color=alt.Color(
                            "Direction:N",
                            scale=alt.Scale(
                                domain=[t("charts.positive_impact"), t("charts.negative_impact")],
                                range=["#10B981", "#F59E0B"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=[
                            alt.Tooltip("Driver:N"),
                            alt.Tooltip("Avg_Impact:Q", title=t("charts.impact"), format="+,.1f"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(bar_chart, width="stretch")

        # Secondary 2D Row: Branch Forecast Share + Executive Accuracy Gauge
        sub_c1, sub_c2 = st.columns([1.45, 1.0], gap="large")
        with sub_c1:
            st.subheader(f"📊 {t('charts.by_group')}")
            summary_df: pd.DataFrame = res["series_summary_df"]
            series_col_name = summary_df.columns[0]
            forecast_col_name = t("table.forecast_sales", horizon=res["horizon"])

            comp_chart = (
                alt.Chart(summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0F172A")
                .encode(
                    x=alt.X(f"{series_col_name}:N", sort="-y", title=None),
                    y=alt.Y(f"{forecast_col_name}:Q", title=t("charts.total_forecast_sales")),
                    color=alt.Color(
                        f"{series_col_name}:N",
                        scale=alt.Scale(range=["#10B981", "#0F172A", "#F59E0B", "#3B82F6", "#14B8A6"]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip(f"{series_col_name}:N"),
                        alt.Tooltip(f"{forecast_col_name}:Q", format=",.2f"),
                        alt.Tooltip(f"{t('charts.change')}:Q", title=t("charts.change"), format="+.1f"),
                    ],
                )
                .properties(height=250)
            )
            st.altair_chart(comp_chart, width="stretch")

        with sub_c2:
            st.subheader(f"🎯 {t('charts.accuracy')}")
            if res.get("accuracy_pct") is not None and res.get("mape_pct") is not None:
                st.plotly_chart(
                    build_accuracy_gauge(res["accuracy_pct"], res["mape_pct"]),
                    width="stretch",
                )
            else:
                st.info(t("charts.not_enough_accuracy"))

    # -------------------------------------------------------------------------
    # TAB 2: Interactive 3D Multi-Branch Demand Terrain
    # -------------------------------------------------------------------------
    with tab_3d:
        render_build_id(build_id)
        st.subheader(f"🌐 {t('charts.branch_details')}")
        st.caption(
            t("charts.branch_details_caption")
        )
        st.plotly_chart(build_3d_forecast_figure(res), width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: Decision-first summary, alerts, and export
    # -------------------------------------------------------------------------
    with tab_decision:
        render_build_id(build_id)
        st.subheader(f"✅ {t('decision.title')}")
        st.caption(t("decision.caption"))

        with st.expander(t("decision.alerts"), expanded=True):
            exceptions_df = res["series_summary_df"].copy()
            change_col = t("charts.change")
            if change_col in exceptions_df.columns:
                exceptions_df = exceptions_df.sort_values(change_col)
            forecast_column = t("table.forecast_sales", horizon=res["horizon"])
            for _, exception in exceptions_df.head(5).iterrows():
                series_name = exception.get(g_col or t("common.series"), t("common.company_total"))
                change_value = float(exception.get(change_col, 0) or 0)
                change_word = t("result.increase") if change_value >= 0 else t("result.decrease")
                forecast_value = float(exception.get(forecast_column, 0) or 0)
                action = exception.get(t("table.recommended_action"), t("actions.continue_monitoring"))
                with st.container(border=True):
                    st.markdown(f"**{series_name}**")
                    st.write(t("decision.alert_item", change_word=change_word, change_pct=abs(change_value), forecast=forecast_value, action=action))

        with st.expander(t("decision.accuracy_details"), expanded=False):
            metric_cols = st.columns(2)
            with metric_cols[0]:
                mape_display = f"{res['mape_pct']:.1f}%" if res.get("mape_pct") is not None else t("common.not_available")
                st.metric(t("decision.accuracy_close"), mape_display)
            with metric_cols[1]:
                wape_display = f"{res['wape_pct']:.1f}%" if res.get("wape_pct") is not None else t("common.not_available")
                st.metric(t("decision.total_error"), wape_display)
            metric_cols = st.columns(2)
            with metric_cols[0]:
                bias_display = f"{res['bias_pct']:+.1f}%" if res.get("bias_pct") is not None else t("common.not_available")
                st.metric(t("decision.bias"), bias_display)
            with metric_cols[1]:
                baseline_display = f"{res['baseline_mape_pct']:.1f}%" if res.get("baseline_mape_pct") is not None else t("common.not_available")
                st.metric(t("decision.baseline_comparison"), baseline_display)
            st.caption(t("decision.accuracy_caption"))

        with st.expander(t("decision.group_summary"), expanded=False):
            st.dataframe(
                res["series_summary_df"],
                width="stretch",
                hide_index=True,
                column_config={
                    t("table.previous_sales", horizon=res["horizon"]): st.column_config.NumberColumn(format="%,.2f"),
                    t("table.forecast_sales", horizon=res["horizon"]): st.column_config.NumberColumn(format="%,.2f"),
                    t("charts.change"): st.column_config.NumberColumn(format="%+.1f%%"),
                    t("table.daily_forecast"): st.column_config.NumberColumn(format="%,.2f"),
                },
            )

        with st.expander(t("decision.group_accuracy"), expanded=False):
            st.dataframe(res["series_accuracy_df"], width="stretch", hide_index=True)

        if len(st.session_state.get("forecast_history", [])) > 1:
            with st.expander(t("decision.compare_previous"), expanded=False):
                history_df = pd.DataFrame(st.session_state["forecast_history"])
                st.dataframe(history_df, width="stretch", hide_index=True)
                st.download_button(
                    t("decision.download_history"),
                    data=history_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=t("export.history_file"),
                    mime="text/csv",
                    width="stretch",
                )

        st.subheader(f"📋 {t('decision.details_export')}")

        export_df = res["combined_df"].drop(columns=["_sort_date"]).copy()
        export_df["smartaira_approval_status"] = st.session_state["approval_status"]
        download_df = export_df.copy()
        download_df["smartaira_row_type"] = download_df["smartaira_row_type"].map(
            {"HISTORICAL": t("common.historical"), "FORECAST": t("common.forecast")}
        )
        download_df["smartaira_approval_status"] = download_df["smartaira_approval_status"].map(
            {"PENDING": t("approval.pending"), "APPROVED": t("approval.approved")}
        )
        download_df = download_df.rename(
            columns={
                "smartaira_row_type": t("table.row_type"),
                "smartaira_predicted_value": t("table.predicted_sales"),
                "smartaira_forecast_low": t("table.forecast_low"),
                "smartaira_forecast_high": t("table.forecast_high"),
                "smartaira_top_drivers": t("table.top_drivers"),
                "smartaira_model_mape_pct": t("table.model_error"),
                "smartaira_approval_status": t("table.approval_status"),
            }
        )

        tbl_f1, tbl_f2 = st.columns([1.4, 1.0])
        with tbl_f1:
            row_filter = st.radio(
                t("table.display"),
                [t("table.future_only"), t("table.all_rows")],
                horizontal=False,
            )
        with tbl_f2:
            if g_col:
                tbl_group_choice = st.selectbox(
                    t("table.filter_group"),
                    options=[t("common.all_groups")] + res["active_groups"],
                    key="table_group_filter",
                )
            else:
                tbl_group_choice = t("common.all_groups")

        display_df = (
            export_df[export_df["smartaira_row_type"] == "FORECAST"].copy()
            if row_filter == t("table.future_only")
            else export_df.copy()
        )
        if g_col and tbl_group_choice != t("common.all_groups"):
            display_df = display_df[display_df[g_col] == tbl_group_choice]

        display_df = display_df.rename(
            columns={
                t_col: t("table.actual_sales"),
                "smartaira_predicted_value": t("table.predicted_sales"),
                "smartaira_row_type": t("table.row_type"),
                "smartaira_top_drivers": t("table.top_drivers"),
                "smartaira_model_mape_pct": t("table.model_error"),
                "smartaira_approval_status": t("table.approval_status"),
            }
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=420,
            column_config={
                t("table.actual_sales"): st.column_config.NumberColumn(format="%,.2f"),
                t("table.predicted_sales"): st.column_config.NumberColumn(format="%,.2f"),
                t("table.row_type"): st.column_config.TextColumn(),
                t("table.top_drivers"): st.column_config.TextColumn(width="large"),
                t("table.model_error"): st.column_config.TextColumn(),
                t("table.approval_status"): st.column_config.TextColumn(),
            },
        )

    st.markdown("---")

    # Always-Accessible Governance Sign-Off & CSV Export Bar
    st.markdown(
        f"""
        <div class="exec-callout">
            ✅ <b>{t("result.completed", series_count=res['n_series'], accuracy=f"{res['accuracy_pct']:.1f}%" if res.get('accuracy_pct') is not None else t("common.not_available_yet"))}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["approval_status"] != "APPROVED":
        approval_comment = st.text_area(
            t("governance.comment"),
            value=st.session_state.get("approval_comment", ""),
            placeholder=t("governance.comment_placeholder"),
        )
    else:
        approval_comment = st.session_state.get("approval_comment", "")

    gov_c1, gov_c2 = st.columns(2)
    with gov_c1:
        if st.session_state["approval_status"] == "APPROVED":
            st.success(
                f"✅ {t('governance.approved_by', user=st.session_state.get('approved_by', t('common.unspecified')), timestamp=st.session_state['approved_at'])}"
            )
            if approval_comment:
                st.caption(t("governance.approval_note", comment=approval_comment))
        else:
            st.caption(t("governance.approval_hint"))
            if st.button(f"✅ {t('governance.approve')}", width="stretch"):
                st.session_state["approval_status"] = "APPROVED"
                st.session_state["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state["approved_by"] = st.session_state.get("current_user", t("common.unspecified"))
                st.session_state["approval_comment"] = approval_comment
                st.rerun()
    with gov_c2:
        csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"⬇️ {t('export.full_forecast')}",
            data=csv_bytes,
            file_name=t("export.forecast_file", date=datetime.now().strftime("%Y%m%d")),
            mime="text/csv",
            type="primary",
            width="stretch",
        )

    st.markdown(
        f'<div class="brand-footer">{t("app.footer")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

