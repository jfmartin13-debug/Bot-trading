# bot_streamlit.py
# ------------------------------------------------------------
# Core/Satellite Backtester (Streamlit) — SINGLE FILE + MULTI-USER AUTH (SAFE QUOTES)
#
# Features:
# - Upload CSV: wide (Date + tickers) OR many CSVs (1 per ticker)
# - Satellite: Momentum (SINGLE / DUAL) + crash filter (asset MA window days) + Risk-off CASH
# - Fees: bps * turnover, turnover = 0.5 * sum(|Δw|)
# - Core: Buy & Hold (one ticker)
# - Combo: core_weight*core + sat_weight*bot
# - Weight sweep: 10/20/30/50 + current
#
# Security:
# - If no credentials configured -> access blocked (recommended for public)
# - Multi-user auth via Streamlit Secrets "USERS" (JSON dict with hashed passwords)
# - Fallback single password via "APP_PASSWORD"
#
# Runs (audit):
# - 3B: saving is disabled in cloud by default
# - You can enable locally by setting env ENABLE_RUN_SAVE=1
#
# Requirements:
#   pip install streamlit pandas numpy
#
# Run locally:
#   streamlit run bot_streamlit.py
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import math
import hmac
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Core/Satellite Bot (CSV)", page_icon="📈", layout="wide")


# -----------------------------
# Security / Auth
# -----------------------------
BLOCK_IF_NO_CREDENTIALS = True  # 1A
ENABLE_RUN_SAVE = os.getenv("ENABLE_RUN_SAVE", "0").strip() == "1"  # 3B default OFF


def _load_users_from_secrets() -> Optional[Dict[str, str]]:
    """
    Reads USERS from Streamlit secrets.
    Expected in Streamlit Secrets (Settings -> Secrets):
      USERS = '{ "alice":"pbkdf2_sha256$200000$salt_hex$hash_hex" }'
    """
    raw = None
    try:
        raw = st.secrets.get("USERS", None)
    except Exception:
        raw = None

    if not raw:
        return None

    try:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        return None

    return None


def _verify_pbkdf2_sha256(stored: str, password: str) -> bool:
    """
    stored format: pbkdf2_sha256$iters$salt_hex$hash_hex
    """
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


def _require_auth() -> None:
    """
    Auth logic:
    - If USERS exists in secrets -> username+password checked against hashed entries.
    - Else fallback to APP_PASSWORD (single shared password).
    - If neither exists -> block if BLOCK_IF_NO_CREDENTIALS True.
    """
    users = _load_users_from_secrets()

    app_password = None
    try:
        if "APP_PASSWORD" in st.secrets:
            app_password = st.secrets["APP_PASSWORD"]
    except Exception:
        app_password = None
    if not app_password:
        app_password = os.getenv("APP_PASSWORD")

    has_any = bool(users) or bool(app_password)
    if not has_any:
        if BLOCK_IF_NO_CREDENTIALS:
            st.warning("🔐 Aucun identifiant configuré (USERS ou APP_PASSWORD). Accès bloqué.")
            st.stop()
        return

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
        st.session_state.auth_user = None

    if st.session_state.auth_ok:
        return

    st.sidebar.header("🔐 Connexion")

    if users:
        username = st.sidebar.text_input("Utilisateur")
        pwd = st.sidebar.text_input("Mot de passe", type="password")
        if not username or not pwd:
            st.stop()

        stored = users.get(username)
        if stored and _verify_pbkdf2_sha256(stored, pwd):
            st.session_state.auth_ok = True
            st.session_state.auth_user = username
            st.rerun()
        else:
            st.sidebar.error("Identifiants incorrects.")
            st.stop()

    else:
        pwd = st.sidebar.text_input("Mot de passe", type="password")
        if not pwd:
            st.stop()

        if hmac.compare_digest(str(pwd), str(app_password)):
            st.session_state.auth_ok = True
            st.session_state.auth_user = "shared"
            st.rerun()
        else:
            st.sidebar.error("Mot de passe incorrect.")
            st.stop()


_require_auth()


# -----------------------------
# Header
# -----------------------------
st.title("📈 Core/Satellite — Bot Momentum (CSV)")
st.caption("Single-file • CSV upload • Filtre crash (MA) • Frais • Core/Satellite • Auth multi-user")


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_CORE = "SPY"

PRICE_COL_CANDIDATES = [
    "Adj Close", "AdjClose", "adjclose", "adj_close",
    "Close", "close", "PRICE", "Price", "price"
]


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass(frozen=True)
class StrategyConfig:
    rebalance: str                      # "M" or "W"
    top_n: int
    long_only: bool

    momentum_mode: str                  # "SINGLE" or "DUAL"
    lb_single: int
    lb_dual: Tuple[int, int]
    w_dual: Tuple[float, float]

    market_filter_on: bool
    market_filter_asset: str
    market_filter_window_days: int

    risk_off_mode: str                  # "CASH"


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float


# -----------------------------
# Formatting
# -----------------------------
def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


# -----------------------------
# CSV helpers
# -----------------------------
def _detect_date_col(df: pd.DataFrame) -> str:
    if "Date" in df.columns:
        return "Date"
    return df.columns[0]


def _detect_price_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRICE_COL_CANDIDATES:
        if c in df.columns:
            return c
    if df.shape[1] >= 2:
        return df.columns[1]
    return None


def load_wide_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        raise ValueError("CSV invalide : il faut une colonne Date + au moins 1 colonne ticker.")
    date_col = _detect_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df


def infer_ticker_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    name = base.rsplit(".", 1)[0]
    name = name.replace("_us_d", "").replace("_US_D", "")
    name = name.replace(".us", "").replace(".US", "")
    return name.upper()


def load_many_csv(files) -> Tuple[pd.DataFrame, List[str], List[str]]:
    frames: List[pd.DataFrame] = []
    tickers: List[str] = []
    problems: List[str] = []

    for f in files:
        fname = getattr(f, "name", "fichier")
        try:
            df = pd.read_csv(f)
            if df.empty or df.shape[1] < 2:
                problems.append(f"{fname} (vide/invalide)")
                continue

            df.columns = [c.strip() for c in df.columns]
            date_col = _detect_date_col(df)
            price_col = _detect_price_col(df)
            if price_col is None:
                problems.append(f"{fname} (colonne prix introuvable)")
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(by=date_col)

            px = pd.to_numeric(df[price_col], errors="coerce")
            out = pd.DataFrame({"Date": df[date_col], "Price": px}).dropna()
            out = out.drop_duplicates(subset=["Date"]).set_index("Date").sort_index()

            ticker = infer_ticker_from_filename(fname)
            out = out.rename(columns={"Price": ticker})

            frames.append(out)
            tickers.append(ticker)
        except Exception:
            problems.append(f"{fname} (erreur lecture)")

    if not frames:
        raise ValueError("Aucun CSV valide n'a pu être chargé.")

    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    merged = merged.dropna(how="all")
    return merged, tickers, problems


# -----------------------------
# Metrics
# -----------------------------
def resample_prices(close: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    if rebalance == "M":
        return close.resample("M").last()
    if rebalance == "W":
        return close.resample("W-FRI").last()
    raise ValueError("rebalance doit être 'M' ou 'W'")


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).mean()


def periods_per_year(rebalance: str) -> float:
    return 12.0 if rebalance == "M" else 52.0


def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def summarize(eq: pd.Series, ret: pd.Series, ppy: float) -> Dict[str, float]:
    if len(eq) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Calmar": np.nan}
    years = len(ret) / ppy if ppy else np.nan
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = float(ret.std(ddof=0) * math.sqrt(ppy))
    sharpe = safe_div(float(ret.mean() * ppy), vol)
    mdd = max_drawdown(eq)
    calmar = safe_div(float(cagr), abs(mdd)) if not np.isnan(mdd) and mdd != 0 else np.nan
    return {"CAGR": float(cagr), "Vol": vol, "Sharpe": float(sharpe), "MaxDD": float(mdd), "Calmar": float(calmar)}


# -----------------------------
# Strategy
# -----------------------------
def compute_scores(prices_rebal: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    if cfg.momentum_mode == "SINGLE":
        return prices_rebal.pct_change(cfg.lb_single)
    lb1, lb2 = cfg.lb_dual
    w1, w2 = cfg.w_dual
    s1 = prices_rebal.pct_change(lb1)
    s2 = prices_rebal.pct_change(lb2)
    return w1 * s1 + w2 * s2


def compute_risk_on_daily(close_daily: pd.DataFrame, rebal_index: pd.DatetimeIndex, cfg: StrategyConfig) -> pd.Series:
    if not cfg.market_filter_on:
        return pd.Series(True, index=rebal_index)

    a = cfg.market_filter_asset
    if a not in close_daily.columns:
        return pd.Series(True, index=rebal_index)

    px = close_daily[a].dropna()
    ma = sma(px, cfg.market_filter_window_days)
    sig_daily = (px > ma).astype(bool)

    sig_rebal = sig_daily.reindex(rebal_index, method="ffill").fillna(False).astype(bool)
    return sig_rebal


def build_weights(
    prices_rebal: pd.DataFrame,
    risk_on: pd.Series,
    cfg: StrategyConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    scores = compute_scores(prices_rebal, cfg)
    risk_on = risk_on.reindex(prices_rebal.index).fillna(False).astype(bool)

    w = pd.DataFrame(0.0, index=prices_rebal.index, columns=prices_rebal.columns)

    for t in prices_rebal.index:
        if not bool(risk_on.loc[t]):
            continue  # CASH

        row = scores.loc[t].dropna()
        if row.empty:
            continue

        if cfg.long_only:
            row = row[row > 0]
            if row.empty:
                continue

        top = row.sort_values(ascending=False).head(cfg.top_n).index.tolist()
        if top:
            w.loc[t, top] = 1.0 / len(top)

    return w, scores, risk_on


# -----------------------------
# Backtest
# -----------------------------
def backtest(prices_rebal: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, pd.Series]:
    r = prices_rebal.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.fillna(0.0)

    w_prev = weights.shift(1).fillna(0.0)
    turnover = 0.5 * (weights - w_prev).abs().sum(axis=1)

    fee_rate = cfg.fee_bps / 10000.0
    fees = fee_rate * turnover

    ret_gross = (w_prev * r).sum(axis=1).fillna(0.0)
    ret_net = (ret_gross - fees).fillna(0.0)

    eq_net = (1.0 + ret_net).cumprod()

    return {
        "turnover": turnover,
        "fees": fees,
        "ret_net": ret_net,
        "eq_net": eq_net,
    }


# -----------------------------
# Runs / audit saving (optional local)
# -----------------------------
def get_runs_dir() -> Path:
    root = os.getenv("RUNS_DIR", "runs")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_run_dir(prefix: str = "local") -> Path:
    runs = get_runs_dir()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs / f"{stamp}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        return asdict(obj)
    except Exception:
        return str(obj)


def save_run(run_dir: Path, config: Dict[str, Any], stats: Dict[str, Any],
             weights: pd.DataFrame, equity: pd.DataFrame, risk_on: Optional[pd.Series]) -> None:
    (run_dir / "config.json").write_text(json.dumps(_to_jsonable(config), indent=2), encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps(_to_jsonable(stats), indent=2), encoding="utf-8")
    weights.to_csv(run_dir / "weights.csv", index=True)
    equity.to_csv(run_dir / "equity.csv", index=True)
    if risk_on is not None:
        risk_on.rename("risk_on").to_csv(run_dir / "risk_on.csv", index=True)


# -----------------------------
# Combo runner
# -----------------------------
def run_combo(
    close_daily: pd.DataFrame,
    universe: List[str],
    core_ticker: str,
    sat_weight: float,
    strat_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
) -> Dict[str, object]:
    prices_rebal = resample_prices(close_daily, strat_cfg.rebalance).dropna(how="all")
    if prices_rebal.empty:
        raise ValueError("Aucune donnée après resampling (mensuel/hebdo).")

    sat_cols = [c for c in universe if c in prices_rebal.columns]
    if len(sat_cols) < 2:
        raise ValueError("Univers satellite : au moins 2 tickers requis (présents dans les données).")

    risk_on = compute_risk_on_daily(close_daily, prices_rebal.index, strat_cfg)
    weights, scores, risk_on_aligned = build_weights(prices_rebal[sat_cols], risk_on=risk_on, cfg=strat_cfg)
    bt = backtest(prices_rebal[weights.columns], weights, bt_cfg)

    if core_ticker not in prices_rebal.columns:
        raise ValueError(f"Ticker core '{core_ticker}' introuvable dans les données.")
    core_px = prices_rebal[core_ticker].dropna()
    core_ret = core_px.pct_change().fillna(0.0)
    core_eq = (1.0 + core_ret).cumprod()

    bot_ret = bt["ret_net"].reindex(core_ret.index).fillna(0.0)
    core_ret_aligned = core_ret.reindex(bot_ret.index).fillna(0.0)

    core_weight = 1.0 - float(sat_weight)
    combo_ret = core_weight * core_ret_aligned + float(sat_weight) * bot_ret
    combo_eq = (1.0 + combo_ret).cumprod()

    ppy = periods_per_year(strat_cfg.rebalance)
    perf_bot = summarize(bt["eq_net"], bt["ret_net"], ppy)
    perf_core = summarize(core_eq.reindex(combo_eq.index).ffill(), core_ret_aligned, ppy)
    perf_combo = summarize(combo_eq, combo_ret, ppy)

    diag = {
        "pct_risk_on": float(risk_on_aligned.mean()),
        "transitions": int((risk_on_aligned.astype(int).diff().abs() > 0).sum()),
        "avg_turnover": float(bt["turnover"].mean()),
        "sum_fees": float(bt["fees"].sum()),
    }

    equity_df = pd.DataFrame(
        {
            "Combiné": combo_eq,
            "Bot (satellite)": bt["eq_net"].reindex(combo_eq.index).ffill(),
            f"Core ({core_ticker})": core_eq.reindex(combo_eq.index).ffill(),
        }
    ).dropna()

    return {
        "prices_rebal": prices_rebal,
        "weights": weights,
        "scores": scores,
        "risk_on": risk_on_aligned,
        "bt": bt,
        "perf_bot": perf_bot,
        "perf_core": perf_core,
        "perf_combo": perf_combo,
        "diagnostic": diag,
        "equity_df": equity_df,
        "core_weight": core_weight,
        "sat_weight": float(sat_weight),
    }


# -----------------------------
# Sidebar: Data
# -----------------------------
with st.sidebar:
    st.header("1) Données")
    upload_mode = st.radio("Mode d'upload", ["Un seul CSV (wide)", "Plusieurs CSV (un par ticker)"], index=1)

    uploaded_one = None
    uploaded_many = None
    if upload_mode == "Un seul CSV (wide)":
        uploaded_one = st.file_uploader("Upload CSV (Date + colonnes tickers)", type=["csv"], accept_multiple_files=False)
    else:
        uploaded_many = st.file_uploader("Upload plusieurs CSV (1 par ticker)", type=["csv"], accept_multiple_files=True)

try:
    if upload_mode == "Un seul CSV (wide)":
        if uploaded_one is None:
            st.info("➡️ Upload un CSV (wide) pour démarrer.")
            st.stop()
        close = load_wide_csv(uploaded_one)
        problems = []
    else:
        if not uploaded_many:
            st.info("➡️ Upload plusieurs CSV (un par ticker) pour démarrer.")
            st.stop()
        close, _tickers_loaded, problems = load_many_csv(uploaded_many)
except Exception as e:
    st.error(f"Erreur chargement CSV : {e}")
    st.stop()

if close.empty or close.shape[1] < 2:
    st.error("Pas assez de données chargées (au moins 2 tickers requis).")
    st.stop()

if not isinstance(close.index, pd.DatetimeIndex):
    st.error("Index Date invalide après chargement.")
    st.stop()

if problems:
    st.warning("Fichiers ignorés / problèmes:\n- " + "\n- ".join(problems))

available = list(close.columns)


# -----------------------------
# Sidebar: Params
# -----------------------------
with st.sidebar:
    st.divider()
    st.header("2) Satellite (Bot)")

    rebalance = st.selectbox("Fréquence", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)

    max_top = min(8, max(1, len(available)))
    top_n = st.slider("Top N", 1, max_top, 1)
    long_only = st.toggle("Long-only (ignore scores ≤ 0)", value=False)

    st.subheader("Momentum")
    momentum_mode = st.selectbox("Mode", ["DUAL", "SINGLE"], index=0)
    if momentum_mode == "DUAL":
        lb1 = st.slider("Lookback 1 (périodes)", 1, 12, 3)
        lb2 = st.slider("Lookback 2 (périodes)", 2, 18, 12)
        w1 = st.slider("Poids lookback 1", 0.0, 1.0, 0.50, 0.05)
        w2 = 1.0 - float(w1)
        lb_single = 12
    else:
        lb_single = st.slider("Lookback single (périodes)", 1, 18, 12)
        lb1, lb2, w1, w2 = 3, 12, 0.5, 0.5

    st.divider()
    st.header("3) Filtre crash (extincteur)")
    market_filter_on = st.toggle("Activer filtre marché", value=True)

    default_filter_asset = "SPY" if "SPY" in available else available[0]
    market_filter_asset = st.selectbox(
        "Actif filtre (thermomètre)",
        options=available,
        index=available.index(default_filter_asset),
    )
    market_filter_window_days = st.slider("Fenêtre MA (jours)", 50, 400, 300, 10)
    risk_off_mode = st.selectbox("Risk-off", ["CASH"], index=0)

    st.divider()
    st.header("4) Univers satellite")
    default_univ = [t for t in ["SPY", "QQQ", "TLT", "GLD", "IEF", "EFA", "EEM", "VNQ"] if t in available]
    if len(default_univ) < 2:
        default_univ = available[: min(8, len(available))]
    universe = st.multiselect("Univers (satellite)", options=available, default=default_univ)
    if len(universe) < 2:
        st.error("Choisis au moins 2 tickers dans l'univers.")
        st.stop()

    st.divider()
    st.header("5) Core / Combo")
    core_ticker = st.selectbox(
        "Core (buy & hold)",
        options=available,
        index=available.index(DEFAULT_CORE) if DEFAULT_CORE in available else 0
    )

    sat_weight_pct = st.slider("Poids satellite (%)", 0, 80, 50, 5)
    sat_weight = sat_weight_pct / 100.0
    core_weight = 1.0 - sat_weight
    st.caption(f"Combo: {int(core_weight*100)}% Core / {int(sat_weight*100)}% Satellite")

    st.divider()
    st.header("6) Dates")
    start_d = st.date_input("Début", value=pd.to_datetime(close.index.min()).date())
    end_d = st.date_input("Fin", value=pd.to_datetime(close.index.max()).date())


# Data range + needed cols
close = close.loc[(close.index >= pd.to_datetime(str(start_d))) & (close.index <= pd.to_datetime(str(end_d)))].copy()
if close.empty:
    st.error("Aucune donnée dans la plage choisie.")
    st.stop()

needed_cols = sorted(set(universe + [market_filter_asset, core_ticker]))
close = close[needed_cols].copy()

# Build configs
strat_cfg = StrategyConfig(
    rebalance=str(rebalance),
    top_n=int(min(top_n, len(universe))),
    long_only=bool(long_only),
    momentum_mode=str(momentum_mode),
    lb_single=int(lb_single),
    lb_dual=(int(lb1), int(lb2)),
    w_dual=(float(w1), float(w2)),
    market_filter_on=bool(market_filter_on),
    market_filter_asset=str(market_filter_asset),
    market_filter_window_days=int(market_filter_window_days),
    risk_off_mode=str(risk_off_mode),
)
bt_cfg = BacktestConfig(fee_bps=float(fee_bps))

# Run
try:
    out = run_combo(close, universe, core_ticker, sat_weight, strat_cfg, bt_cfg)
except Exception as e:
    st.error(f"Erreur backtest : {e}")
    st.stop()

# Weight sweep
def weight_sweep(weights_list: List[float]) -> pd.DataFrame:
    rows = []
    for w in sorted(set(weights_list)):
        o = run_combo(close, universe, core_ticker, w, strat_cfg, bt_cfg)
        rows.append({
            "Poids bot": f"{int(round(w * 100))}%",
            "CAGR": o["perf_combo"]["CAGR"],
            "Sharpe": o["perf_combo"]["Sharpe"],
            "MaxDD": o["perf_combo"]["MaxDD"],
            "Vol": o["perf_combo"]["Vol"],
        })
    return pd.DataFrame(rows)

sweep_df = weight_sweep([0.10, 0.20, 0.30, 0.50, float(sat_weight)])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Résumé", "🧪 Tests poids", "🔍 Diagnostic", "🧾 Runs (audit)"])

perf_bot = out["perf_bot"]
perf_core = out["perf_core"]
perf_combo = out["perf_combo"]

with tab1:
    st.subheader("Satellite (Bot) — performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR (bot)", fmt_pct(perf_bot["CAGR"]))
    c2.metric("Vol (ann.)", fmt_pct(perf_bot["Vol"]))
    c3.metric("Sharpe", fmt_num(perf_bot["Sharpe"]))
    c4.metric("Max DD", fmt_pct(perf_bot["MaxDD"]))
    c5.metric("Calmar", fmt_num(perf_bot["Calmar"]))

    st.subheader("Core (Buy & Hold) — performance")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("CAGR (core)", fmt_pct(perf_core["CAGR"]))
    d2.metric("Vol (ann.)", fmt_pct(perf_core["Vol"]))
    d3.metric("Sharpe", fmt_num(perf_core["Sharpe"]))
    d4.metric("Max DD", fmt_pct(perf_core["MaxDD"]))
    d5.metric("Calmar", fmt_num(perf_core["Calmar"]))

    st.subheader(f"Portefeuille combiné — {int(out['core_weight'] * 100)}% Core / {int(out['sat_weight'] * 100)}% Bot")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("CAGR (combo)", fmt_pct(perf_combo["CAGR"]))
    e2.metric("Vol (ann.)", fmt_pct(perf_combo["Vol"]))
    e3.metric("Sharpe", fmt_num(perf_combo["Sharpe"]))
    e4.metric("Max DD", fmt_pct(perf_combo["MaxDD"]))
    e5.metric("Calmar", fmt_num(perf_combo["Calmar"]))

    st.subheader("Courbes de capital (net)")
    st.line_chart(out["equity_df"])

    st.subheader("Allocations récentes (cibles bot)")
    w_tail = out["weights"].tail(12).copy()
    w_tail.index = w_tail.index.date
    st.dataframe(w_tail.style.format("{:.0%}"), use_container_width=True)

with tab2:
    st.subheader("Tests 10% / 20% / 30% / 50% + actuel")
    tmp = sweep_df.copy()
    tmp["CAGR"] = tmp["CAGR"].apply(fmt_pct)
    tmp["Vol"] = tmp["Vol"].apply(fmt_pct)
    tmp["Sharpe"] = tmp["Sharpe"].apply(fmt_num)
    tmp["MaxDD"] = tmp["MaxDD"].apply(fmt_pct)
    st.dataframe(tmp, use_container_width=True)

with tab3:
    st.subheader("Diagnostic satellite")
    diag = out["diagnostic"]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("% Risk-on", f"{diag['pct_risk_on'] * 100:.1f}%")
    x2.metric("Transitions on/off", str(diag["transitions"]))
    x3.metric("Turnover moyen", f"{diag['avg_turnover']:.2f}")
    x4.metric("Somme frais (approx)", fmt_pct(diag["sum_fees"]))

with tab4:
    st.subheader("Runs (audit)")
    st.write("3B: la sauvegarde serveur est désactivée par défaut (plus sûr en public).")
    if ENABLE_RUN_SAVE:
        st.info("Sauvegarde activée (ENABLE_RUN_SAVE=1). (Local recommandé)")
    else:
        st.warning("Sauvegarde désactivée. Pour l'activer en local: définir ENABLE_RUN_SAVE=1.")

st.divider()
st.caption("⚠️ Backtest simplifié (pas un conseil financier). Taxes/slippage/exécution non inclus.")
