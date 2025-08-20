# app.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import io
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit/servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# =========================================================
# Helper/analytics module "A" built from your latest code (fixed)
# =========================================================
class A:
    # ---------- parsing / time parts ----------
    @staticmethod
    def _parse_dt(series: pd.Series, tz: Optional[str]) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce", utc=True)
        if tz:
            try:
                s = s.dt.tz_convert(tz)
            except Exception:
                # if some entries are naive, ignore conversion errors
                pass
        return s

    @staticmethod
    def _safe_col(df: pd.DataFrame, name: str) -> bool:
        return name in df.columns and df[name].notna().any()

    @staticmethod
    def derive_email_domain(df: pd.DataFrame) -> pd.DataFrame:
        if "email_address" in df.columns:
            df["email_domain"] = (
                df["email_address"].astype(str).str.extract(r"@(.+)$")[0].str.lower()
            )
        else:
            df["email_domain"] = np.nan
        return df

    @staticmethod
    def derive_session_end(df: pd.DataFrame, session_timeout_min: Optional[int]) -> pd.Series:
        # FIX: check for parsed column and build a Series regardless
        if "logout_date_parsed" in df.columns:
            logout = df["logout_date_parsed"].copy()
        else:
            logout = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

        if session_timeout_min is not None:
            need_fill = logout.isna() & df["logon_date_parsed"].notna()
            logout.loc[need_fill] = df.loc[need_fill, "logon_date_parsed"] + pd.to_timedelta(
                session_timeout_min, unit="m"
            )
        return logout

    @staticmethod
    def _week_floor(d: pd.Series) -> pd.Series:
        return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()

    @staticmethod
    def add_time_parts(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
        df["logon_date_parsed"] = A._parse_dt(df.get("logon_date"), tz)
        df["logout_date_parsed"] = A._parse_dt(df.get("logout_date"), tz)
        if A._safe_col(df, "Date") and df["logon_date_parsed"].isna().all():
            df["logon_date_parsed"] = A._parse_dt(df["Date"], tz)

        dt = df["logon_date_parsed"]
        df["date"] = pd.to_datetime(dt.dt.date)  # naive midday timestamps for grouping
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek
        df["month"] = dt.dt.to_period("M").dt.to_timestamp()
        if "Time of Day" in df.columns and df["Time of Day"].notna().any():
            df["time_of_day_raw"] = df["Time of Day"].astype(str)
        return df

    # ---------- analytics ----------
    @staticmethod
    def compute_session_minutes(df: pd.DataFrame, session_timeout_min: Optional[int]) -> pd.DataFrame:
        df = df.copy()
        end = A.derive_session_end(df, session_timeout_min)
        start = df["logon_date_parsed"]
        mins = (end - start).dt.total_seconds() / 60.0
        df["session_minutes"] = mins.where(mins > 0, np.nan)
        return df

    @staticmethod
    def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
        sessions = df.groupby("date").size().rename("sessions")
        dau = df.groupby("date")["user_id"].nunique().rename("dau")
        mau_monthly = df.groupby("month")["user_id"].nunique().rename("mau")
        mau_for_day = pd.Series(index=dau.index, dtype=float)
        for d in dau.index:
            m = pd.Timestamp(d).to_period("M").to_timestamp()
            mau_for_day.loc[d] = float(mau_monthly.get(m, np.nan))
        stickiness = (dau / mau_for_day).rename("stickiness_dau_over_mau")
        return pd.concat([sessions, dau, stickiness], axis=1).reset_index()

    @staticmethod
    def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["week"] = A._week_floor(df["date"])
        wau = df.groupby("week")["user_id"].nunique().rename("wau")
        sessions = df.groupby("week").size().rename("sessions")
        return pd.concat([wau, sessions], axis=1).reset_index()

    @staticmethod
    def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
        mau = df.groupby("month")["user_id"].nunique().rename("mau")
        sessions = df.groupby("month").size().rename("sessions")
        return pd.concat([mau, sessions], axis=1).reset_index()

    @staticmethod
    def session_duration_stats(df: pd.DataFrame) -> pd.DataFrame:
        def _quantiles(series: pd.Series, qs=(0.5, 0.9, 0.99)) -> dict:
            q = series.dropna().quantile(qs)
            return {f"p{int(p*100)}": q.loc[p] for p in qs}

        s = df["session_minutes"]
        stats = {
            "count_non_null": int(s.notna().sum()),
            "count_null": int(s.isna().sum()),
            "mean_minutes": float(np.nanmean(s)),
            **_quantiles(s, (0.5, 0.9, 0.99)),
        }
        return pd.DataFrame([stats])

    @staticmethod
    def org_activity_last_30d(df: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        today = pd.Timestamp.today().normalize() if today is None else today.normalize()
        start = today - pd.Timedelta(days=30)
        scope = df.loc[(df["date"] >= start) & (df["date"] <= today)].copy()

        for col in ["org_id", "org_name", "Industry"]:
            if col not in scope.columns:
                scope[col] = np.nan

        if scope.empty:
            return pd.DataFrame(
                columns=[
                    "org_id", "org_name", "Industry",
                    "users_30d", "sessions_30d",
                    "median_session_min", "p90_session_min",
                    "last_seen", "health_score",
                ]
            )

        agg = scope.groupby(["org_id", "org_name", "Industry"]).agg(
            users_30d=("user_id", "nunique"),
            sessions_30d=("user_id", "size"),
            median_session_min=("session_minutes", "median"),
            p90_session_min=("session_minutes",
                             lambda x: np.nanpercentile(x.dropna(), 90) if x.notna().any() else np.nan),
            last_seen=("date", "max"),
        ).reset_index()

        u = (agg["users_30d"] / agg["users_30d"].max()).fillna(0)
        s = (agg["sessions_30d"] / agg["sessions_30d"].max()).fillna(0)
        d = 1 - ((today - agg["last_seen"]).dt.days.clip(lower=0) / 30).fillna(1)
        agg["health_score"] = ((0.45*u + 0.45*s + 0.10*d) * 100).round(1)

        sort_cols = [c for c in ["health_score", "users_30d", "sessions_30d"] if c in agg.columns]
        return agg.sort_values(sort_cols, ascending=False) if sort_cols else agg

    @staticmethod
    def browser_os_share(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["browser", "browser_version", "operating_system"]:
            if col not in df.columns:
                df[col] = np.nan
        out = (
            df.groupby(["browser", "browser_version", "operating_system"])
              .size()
              .rename("sessions")
              .reset_index()
              .sort_values("sessions", ascending=False)
        )
        return out

    @staticmethod
    def retention_table(df: pd.DataFrame) -> pd.DataFrame:
        if "user_id" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        tmp = df[["user_id", "date"]].dropna()
        if tmp.empty:
            return pd.DataFrame()

        first_seen = tmp.groupby("user_id")["date"].min()
        cohort_month = first_seen.dt.to_period("M").dt.to_timestamp()
        cohort = pd.DataFrame({
            "user_id": first_seen.index,
            "first_day": first_seen.values,
            "cohort_month": cohort_month.values
        })

        events = tmp.merge(cohort, on="user_id", how="inner")
        events["day_index"] = (events["date"] - events["first_day"]).dt.days
        events = events[events["day_index"] >= 0]
        if events.empty:
            return pd.DataFrame()

        cohort_sizes = cohort.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")
        active_by_day = (
            events.groupby(["cohort_month", "day_index"])["user_id"].nunique()
                  .rename("active_users")
                  .reset_index()
                  .merge(cohort_sizes, on="cohort_month", how="left")
        )
        active_by_day["retention"] = (
            active_by_day["active_users"] / active_by_day["cohort_size"]
        ).fillna(0).round(4)

        if active_by_day.empty:
            return pd.DataFrame()

        ret = active_by_day.pivot_table(
            index="cohort_month", columns="day_index", values="retention", fill_value=0.0
        )
        ret = ret.sort_index()
        return ret

    @staticmethod
    def estimate_daily_peak_concurrency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate daily peak concurrency from login (+1) and logout (-1) events.
        """
        if "logon_date_parsed" not in df.columns:
            return pd.DataFrame(columns=["date", "peak_concurrency"])
        if "logout_date_effective" not in df.columns:
            df = df.copy()
            df["logout_date_effective"] = pd.NaT

        starts = pd.to_datetime(df["logon_date_parsed"], errors="coerce").dropna()
        ends = pd.to_datetime(df["logout_date_effective"], errors="coerce").dropna()
        if starts.empty and ends.empty:
            return pd.DataFrame(columns=["date", "peak_concurrency"])

        start_idx = pd.DatetimeIndex(starts)
        end_idx = pd.DatetimeIndex(ends)

        events = pd.concat([
            pd.Series(1, index=start_idx),
            pd.Series(-1, index=end_idx),
        ]).sort_index()

        events.index = pd.DatetimeIndex(events.index)
        concur = events.cumsum()

        daily_peak = concur.resample("D").max().fillna(0).astype(int)
        daily_peak = daily_peak.rename("peak_concurrency")
        daily_peak.index.name = "date"
        return daily_peak.reset_index()

    # ---------- plotting (PNG bytes for downloads) ----------
    @staticmethod
    def _fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def make_time_series_fig(df: pd.DataFrame, x: str, y: str, title: str) -> Tuple[plt.Figure, bytes]:
        if df.empty or x not in df.columns or y not in df.columns:
            return None, b""
        fig = plt.figure(figsize=(8, 3.5))
        ax = fig.gca()
        ax.plot(df[x], df[y])
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if np.issubdtype(df[x].dtype, np.datetime64):
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.tight_layout()
        return fig, A._fig_to_png_bytes(fig)

    @staticmethod
    def make_hist_fig(series: pd.Series, title: str, xlabel: str, bins: int = 60) -> Tuple[plt.Figure, bytes]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None, b""
        cutoff = np.percentile(s, 99)
        s = s[s <= cutoff]
        fig = plt.figure(figsize=(8, 3.5))
        ax = fig.gca()
        ax.hist(s, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        return fig, A._fig_to_png_bytes(fig)

    @staticmethod
    def make_heatmap_hour_dow_fig(df: pd.DataFrame) -> Tuple[plt.Figure, bytes]:
        scope = df.dropna(subset=["hour", "dow"])
        if scope.empty:
            return None, b""
        mat = (scope.groupby(["dow", "hour"])["user_id"].nunique()
               .unstack(fill_value=0)
               .reindex(index=[0, 1, 2, 3, 4, 5, 6], fill_value=0))
        fig = plt.figure(figsize=(8, 3.5))
        ax = fig.gca()
        im = ax.imshow(mat.values, aspect="auto")
        ax.set_title("Unique Users by Hour x Day of Week")
        ax.set_xlabel("hour")
        ax.set_ylabel("dow (0=Mon)")
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig, A._fig_to_png_bytes(fig)

    @staticmethod
    def make_retention_heatmap_fig(ret_tbl: pd.DataFrame) -> Tuple[plt.Figure, bytes]:
        if ret_tbl.empty or ret_tbl.shape[1] <= 1:
            return None, b""
        m = ret_tbl.drop(columns=[], errors="ignore").copy()
        fig = plt.figure(figsize=(8, 3.5))
        ax = fig.gca()
        im = ax.imshow(m.values, aspect="auto", vmin=0, vmax=1)
        ax.set_title("Cohort Retention (daily index)")
        ax.set_xlabel("days since first seen")
        ax.set_ylabel("cohort month")
        ax.set_xticks(np.arange(m.shape[1]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig, A._fig_to_png_bytes(fig)


# =========================================================
# Streamlit App UI
# =========================================================
st.set_page_config(page_title="Login / Activity Analytics + Gemini", layout="wide")
st.title("🔐 Login/Activity Analytics Toolkit (In-Memory)")
st.caption("Upload CSV/Excel, compute metrics & charts completely in memory, and generate insights with Gemini.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    tz = st.text_input("Timezone (e.g., UTC, US/Eastern)", value="UTC")
    session_timeout = st.number_input("Session timeout (minutes, 0=disable)", min_value=0, value=30, step=5)
    lookback_days = st.slider("Org activity lookback (days)", 7, 365, 30, step=1)

    st.markdown("---")
    st.subheader("🤖 Gemini")
    default_secret = ""
    try:
        default_secret = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    default_env = os.environ.get("GOOGLE_API_KEY", "")
    prefill_key = default_secret or default_env

    gemini_api_key = st.text_input("Gemini API Key (optional)", type="password", value=prefill_key)
    gemini_model = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)
    max_tokens = st.slider("Max output tokens", 200, 2000, 700, 50)
    include_sample_rows = st.checkbox("Include a tiny, anonymized sample in Gemini context", value=False)

# File upload
uploaded = st.file_uploader("Upload CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
sheet = None
if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
    sheet = st.text_input("Excel sheet name or index (e.g., 0 or Sheet1)", value="0")

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.markdown("### 💬 Gemini Copilot")
st.caption("Ask anything about the computed metrics (e.g., 'compare DAU last 14 days vs previous 14').")
user_prompt = st.text_input("Your question", placeholder="e.g., What stands out in the last 30 days?")

# Session for results
if "results" not in st.session_state:
    st.session_state["results"] = None

# ---------- helpers for IO & Gemini ----------
AUTO_MAP = {
    "login_time": "logon_date",
    "logon_time": "logon_date",
    "login": "logon_date",
    "timestamp": "logon_date",
    "time": "logon_date",
    "logout_time": "logout_date",
    "signout_time": "logout_date",
    "userid": "user_id",
    "user": "user_id",
    "email": "email_address",
}

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    lower_map = {c.lower().strip(): c for c in df.columns}
    for k, v in AUTO_MAP.items():
        if k in lower_map and v not in df.columns:
            cols[lower_map[k]] = v
    return df.rename(columns=cols)

def _load_df(uploaded_file, sheet_name: Optional[str]):
    if uploaded_file is None:
        return None, "No file uploaded."
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            sn = int(sheet_name) if isinstance(sheet_name, str) and sheet_name.isdigit() else sheet_name
            df = pd.read_excel(uploaded_file, sheet_name=sn)
        df = _auto_map_columns(df)
        for c in ("logon_date", "logout_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df, None
    except Exception as e:
        return None, f"Failed to read file: {e}"

def _dfs_to_csv_bytes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    out = {}
    for name, d in dfs.items():
        buf = io.StringIO()
        try:
            d.to_csv(buf, index=False)
        except Exception:
            d.to_csv(buf)
        out[f"{name}.csv"] = buf.getvalue().encode("utf-8")
    return out

def _fig_bytes_zip(fig_bytes: Dict[str, bytes]) -> Dict[str, bytes]:
    return {f"{k}.png": v for k, v in fig_bytes.items() if v}

def _gemini_generate(prompt: str, metrics_json: dict, api_key: str, model_name: str, max_tokens: int) -> str:
    if not api_key:
        return "⚠️ Provide a Gemini API key in the sidebar or via Secrets/Env."
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"⚠️ google-generativeai not installed: {e}"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        safe_json = json.dumps(metrics_json, default=str)
        sys_prompt = (
            "You are an analytics copilot. Use the provided metrics to answer the user's question. "
            "Be precise, cite specific values, and keep responses under 250 words.\n\n"
            f"METRICS(JSON): {safe_json[:180000]}"
        )
        resp = model.generate_content(sys_prompt + "\n\nUSER QUERY:\n" + prompt,
                                      generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text
                                               if getattr(resp, "candidates", None) else "")
        return text or "⚠️ Empty response from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

def gemini_summary_and_plan(metrics_json: dict, api_key: str, model_name: str, max_tokens: int) -> str:
    if not api_key:
        return "⚠️ Provide a Gemini API key in the sidebar or via Secrets/Env."
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"⚠️ google-generativeai not installed: {e}"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        safe_json = json.dumps(metrics_json, default=str)
        prompt = (
            "You are a data/PM copilot. Produce:\n"
            "1) Executive summary (<=120 words)\n"
            "2) Key insights (bullet list w/ numbers)\n"
            "3) Risks/Anomalies\n"
            "4) 2-week action plan grouped by Security, Engagement, Product, and Ops. "
            "For each action: Owner role, expected impact, and 1 measurable KPI.\n"
            "Use crisp bullets. Only use the data provided.\n\n"
            f"DATA(JSON): {safe_json[:180000]}"
        )
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text
                                               if getattr(resp, "candidates", None) else "")
        return text or "⚠️ Empty response from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ---------- business rollups ----------
def build_business_insights(df: pd.DataFrame, lookback_days: int) -> dict:
    now = pd.Timestamp.now().normalize()
    start = now - pd.Timedelta(days=lookback_days)
    scope = df.loc[(df["date"] >= start) & (df["date"] <= now)].copy()

    for c in ["Industry", "org_name", "browser", "operating_system", "hour"]:
        if c not in scope.columns:
            scope[c] = pd.NA

    by_industry = (scope.groupby("Industry")["user_id"].nunique()
                   .sort_values(ascending=False).reset_index(name="unique_users"))
    by_org = (scope.groupby("org_name")["user_id"].nunique()
              .sort_values(ascending=False).reset_index(name="unique_users"))
    by_browser = (scope.groupby("browser")["user_id"].nunique()
                  .sort_values(ascending=False).reset_index(name="unique_users"))
    by_os = (scope.groupby("operating_system")["user_id"].nunique()
             .sort_values(ascending=False).reset_index(name="unique_users"))
    by_hour = (scope.groupby("hour")["user_id"].nunique()
               .sort_values(ascending=False).reset_index(name="unique_users"))

    return {
        "lookback_days": lookback_days,
        "window_start": str(start.date()),
        "window_end": str(now.date()),
        "by_industry": by_industry,
        "by_org_top10": by_org.head(10),
        "by_browser": by_browser,
        "by_os": by_os,
        "by_hour": by_hour,
    }

# ---------- main analytics runner ----------
def run_analytics(df: pd.DataFrame, tz: str, session_timeout: int, lookback_days: int, include_sample_rows: bool):
    # Normalize / ensure columns
    df = _auto_map_columns(df)
    for col in ["user_id", "org_id", "org_name", "browser", "browser_version",
                "operating_system", "Industry", "Domain"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Enrich
    df = A.derive_email_domain(df)
    df = A.add_time_parts(df, tz=tz)
    timeout = None if session_timeout == 0 else session_timeout
    df = A.compute_session_minutes(df, session_timeout_min=timeout)
    df["logout_date_effective"] = A.derive_session_end(df, timeout)

    # Metrics
    daily = A.daily_metrics(df)
    weekly = A.weekly_metrics(df)
    monthly = A.monthly_metrics(df)
    duration_stats = A.session_duration_stats(df)
    org_30d = A.org_activity_last_30d(df)
    bshare = A.browser_os_share(df)
    ret_tbl = A.retention_table(df)
    concurrency = A.estimate_daily_peak_concurrency(df)

    # Charts
    figs = {}
    st.markdown("#### Core Trends")
    if not daily.empty:
        fig, png = A.make_time_series_fig(daily, "date", "dau", "Daily Active Users (DAU)")
        if fig: figs["dau_daily"] = png; st.pyplot(fig)
        fig, png = A.make_time_series_fig(daily, "date", "sessions", "Daily Sessions")
        if fig: figs["sessions_daily"] = png; st.pyplot(fig)
        daily_plt = daily.copy()
        daily_plt["stickiness_smoothed"] = daily_plt["stickiness_dau_over_mau"].rolling(7, min_periods=1).mean()
        fig, png = A.make_time_series_fig(daily_plt, "date", "stickiness_smoothed", "Stickiness (DAU/MAU, 7d avg)")
        if fig: figs["stickiness_daily"] = png; st.pyplot(fig)

    if df["session_minutes"].notna().any():
        st.markdown("#### Session Duration")
        fig, png = A.make_hist_fig(df["session_minutes"], "Session Duration (trimmed at 99th pct)", "minutes", bins=60)
        if fig: figs["session_duration_hist"] = png; st.pyplot(fig)

    if df["hour"].notna().any() and df["dow"].notna().any():
        st.markdown("#### Hour x Day Heatmap")
        fig, png = A.make_heatmap_hour_dow_fig(df)
        if fig: figs["heatmap_hour_dow"] = png; st.pyplot(fig)

    if not ret_tbl.empty:
        st.markdown("#### Retention")
        fig, png = A.make_retention_heatmap_fig(ret_tbl)
        if fig: figs["retention_heatmap"] = png; st.pyplot(fig)

    if not concurrency.empty:
        st.markdown("#### Estimated Peak Concurrency")
        fig, png = A.make_time_series_fig(concurrency, "date", "peak_concurrency", "Estimated Peak Concurrency (Daily)")
        if fig: figs["concurrency_daily"] = png; st.pyplot(fig)

    # Business rollups & quick bars
    st.markdown("### Business Rollups (Lookback Window)")
    biz = build_business_insights(df, lookback_days=lookback_days)
    if not biz["by_industry"].empty:
        st.markdown("**Top industries (unique users)**")
        st.bar_chart(biz["by_industry"].set_index("Industry"))
    if not biz["by_org_top10"].empty:
        st.markdown("**Top orgs**")
        st.bar_chart(biz["by_org_top10"].set_index("org_name"))
    if not biz["by_browser"].empty:
        st.markdown("**Browser share**")
        st.bar_chart(biz["by_browser"].set_index("browser"))
    if not biz["by_os"].empty:
        st.markdown("**OS share**")
        st.bar_chart(biz["by_os"].set_index("operating_system"))
    if not biz["by_hour"].empty:
        st.markdown("**Active users by hour**")
        st.bar_chart(biz["by_hour"].sort_values("hour").set_index("hour"))

    # Safe JSON for Gemini
    daily2, weekly2, monthly2 = daily.copy(), weekly.copy(), monthly.copy()
    if "date" in daily2.columns: daily2["date"] = daily2["date"].astype(str)
    if "week" in weekly2.columns: weekly2["week"] = weekly2["week"].astype(str)
    if "month" in monthly2.columns: monthly2["month"] = monthly2["month"].astype(str)
    conc2 = concurrency.copy()
    if not conc2.empty and "date" in conc2.columns:
        conc2["date"] = conc2["date"].astype(str)
    org2 = org_30d.copy()
    if not org2.empty and "last_seen" in org2.columns:
        org2["last_seen"] = org2["last_seen"].astype(str)

    metrics_json = {
        "window": {"lookback_days": lookback_days, "timezone": tz},
        "daily_tail": daily2.tail(14).to_dict(orient="list") if not daily2.empty else {},
        "weekly_tail": weekly2.tail(8).to_dict(orient="list") if not weekly2.empty else {},
        "monthly": monthly2.to_dict(orient="list") if not monthly2.empty else {},
        "duration_stats": duration_stats.to_dict(orient="records"),
        "top_orgs_30d": org2.head(10).to_dict(orient="records") if not org2.empty else [],
        "concurrency_tail": conc2.tail(14).to_dict(orient="list") if not conc2.empty else {},
        "biz_rollups": {
            "by_industry": biz["by_industry"].to_dict(orient="records"),
            "by_org_top10": biz["by_org_top10"].to_dict(orient="records"),
            "by_browser": biz["by_browser"].to_dict(orient="records"),
            "by_os": biz["by_os"].to_dict(orient="records"),
            "by_hour": biz["by_hour"].to_dict(orient="records"),
            "window_start": biz["window_start"],
            "window_end": biz["window_end"],
        },
    }

    if include_sample_rows and not df.empty:
        cols = [c for c in df.columns if c not in ["email_address", "first_name", "last_name", "Full Name"]]
        sample = df[cols].head(50).copy()
        for c in ["logon_date_parsed", "logout_date_parsed", "date", "month"]:
            if c in sample.columns:
                sample[c] = sample[c].astype(str)
        metrics_json["sample_rows_head"] = sample.to_dict(orient="records")

    dfs = {
        "metrics_daily": daily, "metrics_weekly": weekly, "metrics_monthly": monthly,
        "session_duration_stats": duration_stats, "org_activity_30d": org_30d,
        "browser_os_share": bshare, "retention_table": ret_tbl, "concurrency_daily_peak": concurrency
    }
    return dfs, figs, metrics_json

# ---------- run button ----------
if run_btn:
    df, err = _load_df(uploaded, sheet)
    if err:
        st.error(err)
    else:
        with st.spinner("Crunching numbers (in memory)..."):
            dfs, figs, metrics_json = run_analytics(
                df, tz=tz, session_timeout=session_timeout, lookback_days=lookback_days, include_sample_rows=include_sample_rows
            )
            st.session_state["results"] = {"dfs": dfs, "figs": figs, "metrics_json": metrics_json}

# ---------- UI tabs ----------
res = st.session_state["results"]
tabs = st.tabs(["Tables", "Downloads", "Gemini Chat", "Insights & Action Plan"])

with tabs[0]:
    st.subheader("Data Tables")
    if not res:
        st.info("Upload a file and click **Run Analysis** to begin.")
    else:
        for name, d in res["dfs"].items():
            st.markdown(f"**{name}**")
            st.dataframe(d)

with tabs[1]:
    st.subheader("Download your results (in memory)")
    if not res:
        st.info("Run analysis first.")
    else:
        csv_bytes = _dfs_to_csv_bytes(res["dfs"])
        fig_bytes = _fig_bytes_zip(res["figs"])
        for fname, b in csv_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="text/csv")
        for fname, b in fig_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="image/png")

with tabs[2]:
    st.subheader("Chat with Gemini")
    ask = st.button("Ask", use_container_width=False)
    if ask and user_prompt and res:
        st.info("Querying Gemini...")
        reply = _gemini_generate(user_prompt, res["metrics_json"], gemini_api_key, gemini_model, max_tokens)
        st.markdown("**Gemini says:**")
        st.write(reply)
    elif not res:
        st.info("Run analysis first.")

with tabs[3]:
    st.subheader("Auto Insights & 2-Week Action Plan")
    if not res:
        st.info("Run analysis first.")
    else:
        if st.button("🧠 Generate executive summary & plan", type="primary", use_container_width=True):
            with st.spinner("Asking Gemini for an executive summary and action plan..."):
                text = gemini_summary_and_plan(res["metrics_json"], gemini_api_key, gemini_model, max_tokens)
                st.markdown(text)
