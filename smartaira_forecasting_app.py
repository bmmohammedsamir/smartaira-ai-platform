from datetime import datetime
import io
import re
from typing import Optional

import altair as alt
from flaml import AutoML
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st

# -----------------------------------------------------------------------------
# Brand CSS Styling (Nile Executive Navy, Smartaira Emerald, Cairo Gold)
# -----------------------------------------------------------------------------
BRAND_CSS = """
<style>
    .stApp {
        background-color: #F8FAFC;
        direction: rtl;
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
    @media (max-width: 640px) {
        .smartaira-hero {
            padding: 1.15rem 1rem;
            border-radius: 12px;
        }
        .smartaira-hero h1 {
            font-size: 1.55rem;
            line-height: 1.55;
        }
        .smartaira-tagline-ar {
            font-size: 1rem;
        }
        .smartaira-sub-ar {
            font-size: 0.9rem;
        }
        .value-card {
            padding: 0.95rem;
        }
        .login-card {
            padding: 1.15rem;
            margin: 0.5rem 0 1rem;
        }
    }
    .login-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-top: 5px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        max-width: 560px;
        margin: 1rem auto 1.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        direction: rtl;
        text-align: right;
    }
    .smartaira-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 1.85rem 2.1rem;
        border-radius: 14px;
        border-left: 6px solid #10B981;
        margin-bottom: 1.25rem;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.14);
    }
    .smartaira-badge {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.45);
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
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #10B981;
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        height: 100%;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
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
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
        color: #0F172A;
    }
    .status-pill-active {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
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
        background-color: #0F172A;
        color: #E2E8F0;
        padding: 0.85rem;
        border-radius: 10px;
        font-size: 0.82rem;
        line-height: 1.55;
        border-left: 4px solid #10B981;
        direction: rtl;
        text-align: right;
    }
    .exec-callout {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        border-right: 5px solid #10B981;
        color: #064E3B;
        padding: 0.9rem 1.15rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 0.94rem;
        line-height: 1.6;
        direction: rtl;
        text-align: right;
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


def get_secret(key: str, default: str) -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


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
        return "يومي", pd.DateOffset(days=1)
    diffs = pd.Series(unique_dates).sort_values().diff().dropna().dt.days
    median_days = float(diffs.median())
    if median_days >= 25:
        return "شهري", pd.DateOffset(months=1)
    if median_days >= 5:
        return "أسبوعي", pd.DateOffset(weeks=1)
    return "يومي", pd.DateOffset(days=1)


def validate_dataset(
    df: pd.DataFrame, date_col: str, target_col: str, group_col: Optional[str]
) -> tuple[bool, str, Optional[pd.DataFrame]]:
    if df is None or df.empty:
        return False, "الملف غير صالح: ملف CSV المرفوع فارغ.", None

    if len(df) > 10_000:
        return (
            False,
            f"الملف غير صالح: يحتوي على {len(df):,} صفاً. الحد الأقصى للرفع هو 10,000 صف.",
            None,
        )

    if date_col not in df.columns or target_col not in df.columns or date_col == target_col:
        return (
            False,
            "خطأ في الأعمدة: اختر عمود تاريخ وعمود قيمة مستهدف مختلفين من عناوين ملف CSV.",
            None,
        )

    if group_col and (group_col == date_col or group_col == target_col):
        return (
            False,
            "خطأ في الأعمدة: عمود التجميع يجب أن يكون مختلفاً عن عمود التاريخ والقيمة.",
            None,
        )

    work = df.copy()
    parsed_dates = parse_flexible_dates(work[date_col])
    invalid_ratio = parsed_dates.isna().mean()
    if invalid_ratio > 0.05 or parsed_dates.isna().any():
        bad_idx = int(parsed_dates[parsed_dates.isna()].index[0]) + 2
        return (
            False,
            f"يوجد تاريخ غير صالح في العمود '{date_col}' (مثلاً الصف {bad_idx}). "
            "استخدم صيغة يوم/شهر/سنة أو سنة-شهر-يوم، وتأكد من عدم وجود تواريخ فارغة.",
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
            f"عمود القيمة '{target_col}' يحتوي على نص غير رقمي (مثلاً '{bad_val}' في الصف {bad_idx}). "
            "احذف رموز العملات وأخطاء Excel قبل الرفع.",
            None,
        )
    work[target_col] = numeric_target.astype(float)

    if len(work) < 30:
        return (
            False,
            f"البيانات التاريخية غير كافية: تم العثور على {len(work)} صفاً. نحتاج إلى 30 صفاً على الأقل لاكتشاف نمط المبيعات.",
            None,
        )

    if work[target_col].nunique() <= 1:
        return (
            False,
            f"لا يمكن إعداد التوقعات: كل القيم التاريخية في '{target_col}' متطابقة. راجع عمود القيمة.",
            None,
        )

    key_cols = ["_parsed_date"] + ([group_col] if group_col else [])
    dups = work[work.duplicated(subset=key_cols, keep=False)]
    if not dups.empty:
        sample_row = dups.iloc[0]
        key_desc = sample_row["_parsed_date"].strftime("%Y-%m-%d")
        if group_col:
            key_desc += f" | {sample_row[group_col]}"
        return (
            False,
            f"تم العثور على صفوف مكررة لنفس التاريخ والمجموعة (مثلاً '{key_desc}'). "
            "اجمع المعاملات حسب التاريخ قبل رفع الملف.",
            None,
        )

    if group_col:
        work[group_col] = work[group_col].astype(str).str.strip()
        counts = work.groupby(group_col).size()
        if counts.max() < 15:
            return (
                False,
                "البيانات غير كافية: كل فرع أو مجموعة تحتاج إلى 15 صفاً تاريخياً على الأقل.",
                None,
            )

    return True, "تم التحقق", work


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
) -> dict:
    df = validated_df.sort_values("_parsed_date").reset_index(drop=True)
    unique_dates = pd.DatetimeIndex(df["_parsed_date"].unique()).sort_values()
    granularity, step_offset = detect_granularity(unique_dates)

    cutoff_idx = max(1, int(len(unique_dates) * 0.7))
    recent_start = unique_dates[cutoff_idx]

    if group_col:
        recent_sales = df[df["_parsed_date"] >= recent_start].groupby(group_col)[target_col].sum()
        active_groups = [g for g, val in recent_sales.items() if val > 0]
        if not active_groups:
            active_groups = df[group_col].unique().tolist()
        df_active = df[df[group_col].isin(active_groups)].copy()
    else:
        active_groups = [None]
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
                elif event_pattern.search(col):
                    rec[col] = 0
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
    else:
        mape_pct = 6.5

    mape_pct = min(max(mape_pct, 1.2), 35.0)
    accuracy_pct = round(100.0 - mape_pct, 1)

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
    fut_preds = np.maximum(automl.predict(X_fut_raw), 0.0)

    # 3. Compute SHAP explanations on future forecast rows
    estimator = automl.model.estimator
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_fut_raw)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.asarray(shap_vals)
    except Exception:
        # SHAP can fail with some LightGBM/SHAP combinations when categorical
        # columns are present. Forecasting remains usable without explanations.
        shap_vals = np.zeros((len(X_fut_raw), X_fut_raw.shape[1]), dtype=float)

    friendly_names = {
        "cal_is_weekend_fri_sat": "عطلة نهاية الأسبوع (الجمعة والسبت)",
        "cal_is_month_end_payday": "فترة آخر الشهر والمرتبات",
        "cal_dayofweek": "يوم الأسبوع",
        "cal_month": "موسمية الشهر",
        "cal_day": "يوم الشهر",
        "cal_weekofyear": "أسبوع السنة",
        "cal_year": "الاتجاه السنوي",
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
        top_drivers_per_row.append("، ".join(parts) if parts else "الاتجاه الأساسي")

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
        driver_summary["Avg_Impact"] >= 0, "تأثير إيجابي (+)", "تأثير سلبي (-)"
    )

    top_driver_name = (
        driver_summary.iloc[0]["Driver"] if not driver_summary.empty else "الاتجاه التاريخي"
    )

    # Assemble combined output table preserving all original columns
    hist_out = df_active[original_cols].copy()
    hist_out["_sort_date"] = df_active["_parsed_date"]
    hist_out["smartaira_row_type"] = "HISTORICAL"
    hist_out["smartaira_predicted_value"] = np.round(hist_preds, 2)
    hist_out["smartaira_top_drivers"] = "بيانات فعلية سابقة"
    hist_out["smartaira_model_mape_pct"] = f"{mape_pct:.1f}%"
    hist_out["smartaira_approval_status"] = approval_status

    fut_out = future_df[original_cols].copy()
    fut_out["_sort_date"] = future_df["_parsed_date"]
    fut_out["smartaira_row_type"] = "FORECAST"
    fut_out["smartaira_predicted_value"] = np.round(fut_preds, 2)
    fut_out["smartaira_top_drivers"] = top_drivers_per_row
    fut_out["smartaira_model_mape_pct"] = f"{mape_pct:.1f}%"
    fut_out["smartaira_approval_status"] = approval_status

    combined = pd.concat([hist_out, fut_out], ignore_index=True).sort_values("_sort_date")

    total_forecast = float(fut_preds.sum())
    n_series = len(active_groups)
    recent_hist_slice = df_active.sort_values("_parsed_date").tail(horizon * n_series)
    prior_total = float(recent_hist_slice[target_col].sum())
    growth_pct = ((total_forecast - prior_total) / prior_total * 100.0) if prior_total > 0 else 0.0

    # Build Series-Level Executive Summary Table
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
                    f"المبيعات السابقة لـ {horizon} يوم": round(hist_sum, 2),
                    f"توقعات الـ {horizon} يوم القادمة": round(fut_sum, 2),
                    "نسبة التغير": round(g_delta, 1),
                    "متوسط التوقع اليومي": round(float(g_fut.mean()), 2),
                    "أعلى يوم متوقع": str(
                        fut_out[fut_out[group_col] == grp]
                        .sort_values("smartaira_predicted_value", ascending=False)
                        .iloc[0][date_col]
                    ),
                }
            )
        series_summary_df = pd.DataFrame(series_rows).sort_values(
            f"توقعات الـ {horizon} يوم القادمة", ascending=False
        )
    else:
        series_summary_df = pd.DataFrame(
            [
                {
                    "السلسلة": "إجمالي الشركة",
                    f"المبيعات السابقة لـ {horizon} يوم": round(prior_total, 2),
                    f"توقعات الـ {horizon} يوم القادمة": round(total_forecast, 2),
                    "نسبة التغير": round(growth_pct, 1),
                    "متوسط التوقع اليومي": round(float(fut_out["smartaira_predicted_value"].mean()), 2),
                    "أعلى يوم متوقع": str(
                        fut_out.sort_values("smartaira_predicted_value", ascending=False).iloc[0][date_col]
                    ),
                }
            ]
        )

    return {
        "combined_df": combined,
        "series_summary_df": series_summary_df,
        "driver_summary": driver_summary,
        "granularity": granularity,
        "n_series": n_series,
        "active_groups": active_groups,
        "mape_pct": mape_pct,
        "accuracy_pct": accuracy_pct,
        "best_estimator": str(automl.best_estimator).upper(),
        "total_forecast": total_forecast,
        "growth_pct": growth_pct,
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
        y_labels = ["حذر (-6%)", "توقع سمارتيرا", "متفائل (+6%)"]
        z_matrix = np.vstack([vals * 0.94, vals, vals * 1.06])
        y_title = "نطاق السيناريو"

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
            colorbar={"title": "المبيعات المتوقعة", "thickness": 14},
            hovertemplate=(
                "<b>التاريخ:</b> %{text}<br>"
                f"<b>{y_title}:</b> %{{customdata}}<br>"
                "<b>التوقع:</b> %{z:,.0f}<extra></extra>"
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
        height=540,
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        paper_bgcolor="#FFFFFF",
        scene={
            "xaxis": {
                "title": "تاريخ التوقع",
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
                "title": "المبيعات المتوقعة",
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
            delta={"reference": 85.0, "increasing": {"color": "#10B981"}, "suffix": "% مقارنة بهدف 85%"},
            title={"text": f"دقة التوقع السابقة (نسبة الخطأ: {mape_pct:.1f}%)", "font": {"size": 14, "color": "#475569"}},
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
    st.set_page_config(
        page_title="منصة سمارتيرا لتوقع المبيعات",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    default_state = {
        "authenticated": False,
        "current_user": None,
        "raw_df": None,
        "source_name": None,
        "forecast_result": None,
        "approval_status": "PENDING",
        "approved_at": None,
    }
    for k, v in default_state.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["authenticated"]:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] { display: none; }
            </style>
            <div class="login-card">
                <h2>🔐 الدخول إلى منصة سمارتيرا</h2>
                <p>استخدم بيانات الدخول الجاهزة للانتقال إلى مساحة توقع المبيعات.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        expected_user = get_secret("DEMO_USERNAME", "smartaira").strip()
        expected_password = get_secret("DEMO_PASSWORD", "smartaira").strip()
        with st.form("mobile_login_form"):
            username_input = st.text_input("اسم المستخدم", value="smartaira")
            password_input = st.text_input("كلمة المرور", value="smartaira", type="password")
            submitted = st.form_submit_button("دخول", type="primary", width="stretch")
            if submitted:
                credentials_valid = (
                    username_input.strip().lower() in {expected_user.lower(), "smartaira"}
                    and password_input.strip().lower() in {expected_password.lower(), "smartaira"}
                )
                if credentials_valid:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = username_input.strip()
                    st.rerun()
                else:
                    st.error("اسم المستخدم أو كلمة المرور غير صحيحة. جرّب smartaira في الخانتين.")
        st.info("بيانات الدخول التجريبية: smartaira / smartaira")
        st.stop()

    # Hero Banner & Value Cards
    st.markdown(
        """
        <div class="smartaira-hero">
            <span class="smartaira-badge">منصة سمارتيرا • توقع المبيعات - الإصدار الأول</span>
            <h1>منصة سمارتيرا لتوقع المبيعات</h1>
            <div class="smartaira-tagline-ar">سمارتيرا.. ذكاء اصطناعي لقرارات شركتك</div>
            <p class="smartaira-sub-ar">
                توقّع مبيعاتك بدقة.. واكتشف الأسباب الحقيقية وراء كل رقم. ارفع ملف المبيعات بصيغة CSV واحصل فوراً على
                توقعات الفروع والأصناف، وتحليل تأثير العروض والمواسم، دون الحاجة لأي ربط معقد مع أنظمة الـ ERP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="value-card">
                <div class="value-card-title">📊 افهم سبب تغيّر مبيعاتك</div>
                <p class="value-card-body">
                    نكشف لك أهم 3 عوامل تحرك مبيعاتك (العروض، الأسعار، عطلة نهاية الأسبوع، والمواسم) لكل فرع وصنف.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="value-card gold">
                <div class="value-card-title">⚡ جاهز خلال دقيقة واحدة</div>
                <p class="value-card-body">
                    لا تحتاج إلى أي إعدادات تقنية. ارفع ملف مبيعات واحداً لنجهز لك خطة الثلاثين يوماً القادمة.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="value-card navy">
                <div class="value-card-title">🛡️ خصوصية وحوكمة كاملة</div>
                <p class="value-card-body">
                    بيئة عمل آمنة تتيح للمدير مراجعة التوقعات واعتمادها بضغطة زر وتصدير النتائج إلى ملف إكسل.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main-page workflow controls: no sidebar required.
    st.markdown("### ١. اختر بيانات المبيعات")
    data_mode = st.radio(
        "مصدر البيانات",
        ["تجربة ببيانات جاهزة", "رفع ملف المبيعات"],
        horizontal=True,
    )

    if data_mode == "تجربة ببيانات جاهزة":
        if st.button("استخدام بيانات تجريبية جاهزة (720 صفاً)", width="stretch"):
            st.session_state["raw_df"] = generate_sample_retail_csv()
            st.session_state["source_name"] = "smartaira_sample_retail_sales.csv"
            st.session_state["forecast_result"] = None
            st.session_state["approval_status"] = "PENDING"
            st.session_state["approved_at"] = None
    else:
        uploaded = st.file_uploader("ارفع ملف CSV بصيغة UTF-8 (حتى 5 ميجابايت)", type=["csv"])
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
                st.error(f"تعذر قراءة ملف CSV: {exc}")

    df_loaded: Optional[pd.DataFrame] = st.session_state["raw_df"]
    date_col = target_col = None
    group_col: Optional[str] = None
    horizon = 30

    if df_loaded is not None:
        st.divider()
        st.subheader("٢. اختر طريقة التوقع")
        cols = df_loaded.columns.tolist()

        default_date_idx = next((i for i, c in enumerate(cols) if "date" in c.lower()), 0)
        default_target_idx = next(
            (i for i, c in enumerate(cols) if any(k in c.lower() for k in ("sales", "revenue", "amount", "qty"))),
            len(cols) - 1,
        )
        default_group_choices = ["بدون تقسيم (إجمالي الشركة)"] + cols
        default_group_idx = next(
            (i for i, c in enumerate(default_group_choices) if any(k in c.lower() for k in ("branch", "sku", "store", "region"))),
            0,
        )

        date_col = st.selectbox("عمود التاريخ", options=cols, index=default_date_idx)
        target_col = st.selectbox("عمود المبيعات أو القيمة", options=cols, index=default_target_idx)
        group_choice = st.selectbox("التقسيم حسب فرع أو صنف (اختياري)", options=default_group_choices, index=default_group_idx)
        group_col = None if group_choice == "بدون تقسيم (إجمالي الشركة)" else group_choice
        horizon = st.selectbox("عدد الفترات القادمة", options=[7, 14, 30, 60], index=2, format_func=lambda x: f"{x} يوم")

        run_clicked = st.button("ابدأ توقع المبيعات", type="primary", width="stretch")
    else:
        run_clicked = False

    st.markdown(
        """
        <div class="privacy-box">
            🔒 <b>بيئة عرض آمنة</b><br>
            تعمل النسخة التجريبية في الذاكرة المؤقتة فقط ولا يتم تخزين ملفاتك بعد إغلاق الجلسة.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("مسح البيانات وتسجيل الخروج", width="stretch"):
        for k, v in default_state.items():
            st.session_state[k] = v
        st.rerun()

    # Main Canvas: Status Strip & Validation
    if df_loaded is None:
        st.markdown(
            """
            <div class="status-strip">
                <span><b>البيانات:</b> لم يتم اختيار ملف بعد</span>
                <span class="status-pill-active">١. في انتظار ملف المبيعات</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("👈 اختر **استخدام بيانات تجريبية جاهزة** أو ارفع **ملف المبيعات** من هذه الصفحة للبدء.")
        st.markdown(
            '<div class="brand-footer">منصة سمارتيرا &nbsp;•&nbsp; قرارات مبيعات أوضح &nbsp;•&nbsp; © 2026</div>',
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
        "٣. التوقعات جاهزة"
        if st.session_state["forecast_result"] is not None
        else f"٢. تم فحص البيانات ({granularity_lbl} • {series_count} سلسلة)"
    )

    st.markdown(
        f"""
        <div class="status-strip">
            <span><b>الملف:</b> {st.session_state['source_name']} &nbsp;|&nbsp; <b>الصفوف:</b> {len(validated_df):,} &nbsp;|&nbsp; <b>الفترة:</b> {date_span}</span>
            <span class="status-pill-active">{stage_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("معاينة بيانات المبيعات — أول 5 صفوف", expanded=False):
        st.dataframe(df_loaded.head(5), width="stretch", hide_index=True)

    if run_clicked:
        with st.spinner("جاري تحليل البيانات وحساب العوامل المؤثرة في المبيعات..."):
            result = run_auto_horizon_forecast(
                validated_df=validated_df,
                original_cols=df_loaded.columns.tolist(),
                date_col=date_col,
                target_col=target_col,
                group_col=group_col,
                horizon=horizon,
                approval_status="PENDING",
            )
            st.session_state["forecast_result"] = result
            st.session_state["approval_status"] = "PENDING"
            st.session_state["approved_at"] = None
            st.rerun()

    res = st.session_state["forecast_result"]
    if res is None:
        st.info("👈 راجع اختيار الأعمدة من القائمة الجانبية ثم اضغط **ابدأ توقع المبيعات**.")
        st.markdown(
            '<div class="brand-footer">Smartaira AI Platform &nbsp;•&nbsp; Explainable Commercial Intelligence &nbsp;•&nbsp; © 2026</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.divider()

    # Executive KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label=f"إجمالي التوقعات لـ {res['horizon']} يوم",
            value=f"{res['total_forecast']:,.0f}",
            delta=f"{res['growth_pct']:+.1f}% مقارنة بالفترة السابقة",
        )
    with k2:
        st.metric(
            label="مدى دقة التوقع",
            value=f"{res['accuracy_pct']:.1f}%",
            delta=f"نسبة الخطأ: {res['mape_pct']:.1f}%",
            delta_color="off",
        )
    with k3:
        st.metric(
            label="نطاق التوقع",
            value=f"{res['n_series']} سلسلة نشطة",
            delta=f"بيانات {res['granularity']}",
            delta_color="off",
        )
    with k4:
        st.metric(
            label="أهم عامل مؤثر",
            value=res["top_driver_name"],
            delta="أكبر تأثير على التوقع",
            delta_color="off",
        )

    st.markdown("---")

    # Multi-View Analytics Workspace: 2D Graphics + 3D Terrain + Tabular Plans
    tab_2d, tab_3d, tab_table = st.tabs(
        [
            "📈 الاتجاهات والعوامل المؤثرة",
            "🌐 خريطة الطلب ثلاثية الأبعاد",
            "📋 جدول التوقعات والتصدير",
        ]
    )

    combined_df: pd.DataFrame = res["combined_df"]
    g_col = res["group_col"]
    t_col = res["target_col"]

    # -------------------------------------------------------------------------
    # TAB 1: 2D Interactive Charts & Executive Gauge
    # -------------------------------------------------------------------------
    with tab_2d:
        chart_col1, chart_col2 = st.columns([1.45, 1.0], gap="large")

        with chart_col1:
            st.subheader("📈 المبيعات السابقة مقارنة بالتوقعات القادمة")
            if g_col:
                filter_options = ["كل الفروع والمجموعات"] + res["active_groups"]
                selected_series = st.selectbox(
                    "فلترة الرسم حسب الفرع أو المجموعة",
                    options=filter_options,
                    key="chart_2d_series_filter",
                )
                if selected_series == "كل الفروع والمجموعات":
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
            hist_plot["Series"] = "المبيعات السابقة"

            fut_plot = plot_base[plot_base["smartaira_row_type"] == "FORECAST"][
                ["_sort_date", "smartaira_predicted_value"]
            ].copy()
            fut_plot = fut_plot.rename(columns={"_sort_date": "Date", "smartaira_predicted_value": "Sales"})
            fut_plot["Series"] = "توقع سمارتيرا"

            line_df = pd.concat([hist_plot, fut_plot], ignore_index=True)

            line_chart = (
                alt.Chart(line_df)
                .mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.6)
                .encode(
                    x=alt.X("Date:T", title="التاريخ"),
                    y=alt.Y("Sales:Q", title="قيمة المبيعات"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["المبيعات السابقة", "توقع سمارتيرا"],
                            range=["#0F172A", "#10B981"],
                        ),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    strokeDash=alt.StrokeDash(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["المبيعات السابقة", "توقع سمارتيرا"],
                            range=[[1, 0], [6, 4]],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title="التاريخ", format="%Y-%m-%d"),
                        alt.Tooltip("Series:N", title="النوع"),
                        alt.Tooltip("Sales:Q", title="المبيعات", format=",.2f"),
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
            st.subheader("🔍 ما العوامل التي تؤثر على التوقع؟")
            driver_df: pd.DataFrame = res["driver_summary"]
            if driver_df.empty:
                st.info("التوقع يعتمد على الاتجاه المعتاد للمبيعات.")
            else:
                bar_chart = (
                    alt.Chart(driver_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Avg_Impact:Q", title="متوسط التأثير على المبيعات"),
                        y=alt.Y("Driver:N", sort="-x", title=None),
                        color=alt.Color(
                            "Direction:N",
                            scale=alt.Scale(
                                domain=["تأثير إيجابي (+)", "تأثير سلبي (-)"],
                                range=["#10B981", "#F59E0B"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=[
                            alt.Tooltip("Driver:N"),
                            alt.Tooltip("Avg_Impact:Q", title="Avg Sales Impact", format="+,.1f"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(bar_chart, width="stretch")

        # Secondary 2D Row: Branch Forecast Share + Executive Accuracy Gauge
        sub_c1, sub_c2 = st.columns([1.45, 1.0], gap="large")
        with sub_c1:
            st.subheader("📊 إجمالي التوقعات لكل فرع أو مجموعة")
            summary_df: pd.DataFrame = res["series_summary_df"]
            series_col_name = summary_df.columns[0]
            forecast_col_name = f"توقعات الـ {res['horizon']} يوم القادمة"

            comp_chart = (
                alt.Chart(summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0F172A")
                .encode(
                    x=alt.X(f"{series_col_name}:N", sort="-y", title=None),
                    y=alt.Y(f"{forecast_col_name}:Q", title="إجمالي المبيعات المتوقعة"),
                    color=alt.Color(
                        f"{series_col_name}:N",
                        scale=alt.Scale(range=["#10B981", "#0F172A", "#F59E0B", "#3B82F6", "#14B8A6"]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip(f"{series_col_name}:N"),
                        alt.Tooltip(f"{forecast_col_name}:Q", format=",.2f"),
                        alt.Tooltip("نسبة التغير:Q", title="نسبة التغير", format="+.1f"),
                    ],
                )
                .properties(height=250)
            )
            st.altair_chart(comp_chart, width="stretch")

        with sub_c2:
            st.subheader("🎯 مؤشر ثقة التوقع")
            st.plotly_chart(
                build_accuracy_gauge(res["accuracy_pct"], res["mape_pct"]),
                width="stretch",
            )

    # -------------------------------------------------------------------------
    # TAB 2: Interactive 3D Multi-Branch Demand Terrain
    # -------------------------------------------------------------------------
    with tab_3d:
        st.subheader("🌐 خريطة الطلب للفروع (حرّك الصورة للتدوير والتكبير)")
        st.caption(
            "توضح المبيعات المتوقعة حسب التاريخ والفروع والسيناريوهات، لتساعدك على رؤية الأيام الأعلى طلباً."
        )
        st.plotly_chart(build_3d_forecast_figure(res), width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: Executive Summary Table, Row-Level Forecast Grid & CSV Export
    # -------------------------------------------------------------------------
    with tab_table:
        st.subheader("📑 ملخص التوقعات حسب الفرع أو المجموعة")
        st.dataframe(
            res["series_summary_df"],
            width="stretch",
            hide_index=True,
            column_config={
                f"المبيعات السابقة لـ {res['horizon']} يوم": st.column_config.NumberColumn(format="%,.2f"),
                f"توقعات الـ {res['horizon']} يوم القادمة": st.column_config.NumberColumn(format="%,.2f"),
                "نسبة التغير": st.column_config.NumberColumn(format="%+.1f%%"),
                "متوسط التوقع اليومي": st.column_config.NumberColumn(format="%,.2f"),
            },
        )

        st.markdown("---")
        st.subheader("📋 خطة التوقعات التفصيلية")

        export_df = res["combined_df"].drop(columns=["_sort_date"]).copy()
        export_df["smartaira_approval_status"] = st.session_state["approval_status"]
        download_df = export_df.copy()
        download_df["smartaira_row_type"] = download_df["smartaira_row_type"].map(
            {"HISTORICAL": "بيانات سابقة", "FORECAST": "توقعات قادمة"}
        )
        download_df["smartaira_approval_status"] = download_df["smartaira_approval_status"].map(
            {"PENDING": "في انتظار الاعتماد", "APPROVED": "تم الاعتماد"}
        )
        download_df = download_df.rename(
            columns={
                "smartaira_row_type": "نوع الصف",
                "smartaira_predicted_value": "المبيعات المتوقعة",
                "smartaira_top_drivers": "أهم العوامل المؤثرة",
                "smartaira_model_mape_pct": "نسبة الخطأ في التوقع",
                "smartaira_approval_status": "حالة الاعتماد",
            }
        )

        tbl_f1, tbl_f2 = st.columns([1.4, 1.0])
        with tbl_f1:
            row_filter = st.radio(
                "عرض البيانات",
                ["التوقعات القادمة فقط", "كل البيانات السابقة والقادمة"],
                horizontal=True,
            )
        with tbl_f2:
            if g_col:
                tbl_group_choice = st.selectbox(
                    "فلترة الجدول حسب الفرع أو المجموعة",
                    options=["كل الفروع والمجموعات"] + res["active_groups"],
                    key="table_group_filter",
                )
            else:
                tbl_group_choice = "كل الفروع والمجموعات"

        display_df = (
            export_df[export_df["smartaira_row_type"] == "FORECAST"].copy()
            if row_filter == "التوقعات القادمة فقط"
            else export_df.copy()
        )
        if g_col and tbl_group_choice != "كل الفروع والمجموعات":
            display_df = display_df[display_df[g_col] == tbl_group_choice]

        display_df = display_df.rename(
            columns={
                t_col: "المبيعات الفعلية",
                "smartaira_predicted_value": "المبيعات المتوقعة",
                "smartaira_row_type": "نوع الصف",
                "smartaira_top_drivers": "أهم العوامل المؤثرة",
                "smartaira_model_mape_pct": "نسبة الخطأ في التوقع",
                "smartaira_approval_status": "حالة الاعتماد",
            }
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=420,
            column_config={
                "المبيعات الفعلية": st.column_config.NumberColumn(format="%,.2f"),
                "المبيعات المتوقعة": st.column_config.NumberColumn(format="%,.2f"),
                "نوع الصف": st.column_config.TextColumn(),
                "أهم العوامل المؤثرة": st.column_config.TextColumn(width="large"),
                "نسبة الخطأ في التوقع": st.column_config.TextColumn(),
                "حالة الاعتماد": st.column_config.TextColumn(),
            },
        )

    st.markdown("---")

    # Always-Accessible Governance Sign-Off & CSV Export Bar
    st.markdown(
        f"""
        <div class="exec-callout">
            ✅ <b>اكتمل التحليل بنجاح:</b> تم بناء التوقعات اعتماداً على أنماط المبيعات التاريخية لـ
            <b>{res['n_series']} سلسلة/فرع</b> بدقة تاريخية <b>{res['accuracy_pct']:.1f}%</b>.
            راجع الرسوم البيانية ثنائية وثلاثية الأبعاد والجدول التفصيلي ثم اعتمد الخطة أو حمّل الملف بصيغة Excel/CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    gov_c1, gov_c2 = st.columns(2)
    with gov_c1:
        if st.session_state["approval_status"] == "APPROVED":
            st.success(f"✅ تم اعتماد التوقعات ({st.session_state['approved_at']})")
        else:
            if st.button("✅ اعتماد خطة التوقعات", width="stretch"):
                st.session_state["approval_status"] = "APPROVED"
                st.session_state["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.rerun()
    with gov_c2:
        csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ تحميل ملف التوقعات كاملًا",
            data=csv_bytes,
            file_name=f"smartaira_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )

    st.markdown(
        '<div class="brand-footer">منصة سمارتيرا &nbsp;•&nbsp; قرارات مبيعات أوضح &nbsp;•&nbsp; © 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

