# bot_streamlit.py
# ------------------------------------------------------------
# Streamlit Backtester (CSV) — Core / Satellite
# - Upload CSVs (wide ou un par ticker)
# - Satellite: Momentum (SINGLE ou DUAL) + filtre crash (SPY/QQQ + MA jours) + Risk-off CASH
# - Frais réalistes: bps * turnover (turnover = 0.5 * sum(|Δw|))
# - Core: Buy & Hold (un ticker)
# - Combo: core_weight + sat_weight
# - Test automatique des poids (10/20/30% + personnalisé)
#
# Dépendances:
#   streamlit, pandas, numpy
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Core/Satellite Bot (CSV)", page_icon="📈", layout="wide")
st.title("📈 Core/Satellite — Bot Momentum (CSV)")
st.caption("Upload CSV • Filtre crash (MA) • Frais réalistes • Core/Satellite • Tests 10/20/30%.")


DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_CORE = "SPY"

PRICE_COL_CANDIDATES = [
    "Adj Close", "AdjClose", "adjclose", "adj_close",
    "Close", "close", "PRICE", "Price", "price"
]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    rebalance: str                      # "M" or "W"
    fee_bps: float                      # e.g. 10 = 0.10%
    top_n: int                          # number of assets to hold
    long_only: bool                     # ignore non-positive momentum

    momentum_mode: str                  # "SINGLE" or "DUAL"
    lb_single: int
    lb_dual: Tuple[int, int]
    w_dual: Tuple[float, float]         # weights for dual

    market_filter_on: bool
    market_filter_asset: str
    market_filter_window_days: int

    risk_off_mode: str                  # "CASH" only in this version


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
        raise ValueError("CSV invalide : il faut Date + au moins 1 colonne ticker.")
    date_col = _detect_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df


def infer_ticker_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    name = base.rsplit(".", 1)[0]  # strip .csv
    # normalize common patterns
    name = name.replace("_us_d", "").replace("_US_D", "")
    name = name.replace(".us", "").replace(".US", "")
    return name.upper()


def load_many_csv(files) -> Tuple[pd.DataFrame, List[str], List[str]]:
    frames = []
    tickers = []
    problems = []

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
# Helpers
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


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


# -----------------------------
# Strategy: scores / filter / weights / backtest
# -----------------------------
def compute_scores(prices_rebal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.momentum_mode == "SINGLE":
        return prices_rebal.pct_change(cfg.lb_single)
    lb1, lb2 = cfg.lb_dual
    w1, w2 = cfg.w_dual
    s1 = prices_rebal.pct_change(lb1)
    s2 = prices_rebal.pct_change(lb2)
    return w1 * s1 + w2 * s2


def compute_risk_on_daily(
    close_daily: pd.DataFrame,
    rebal_index: pd.DatetimeIndex,
    cfg: Config,
) -> pd.Series:
    """
    Signal crash filter basé sur données daily (SPY/QQQ), MA en jours.
    Aligné sur les dates de rebal via ffill (pas de look-ahead).
    """
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
    cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores = compute_scores(prices_rebal, cfg)
    risk_on = risk_on.reindex(prices_rebal.index).fillna(False).astype(bool)

    w = pd.DataFrame(0.0, index=prices_rebal.index, columns=prices_rebal.columns)

    for t in prices_rebal.index:
        # Risk-off => CASH (tout à 0)
        if not bool(risk_on.loc[t]):
            continue

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

    return w, scores


def backtest(prices_rebal: pd.DataFrame, weights: pd.DataFrame, fee_bps: float) -> Dict[str, pd.Series]:
    r = prices_rebal.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.fillna(0.0)

    # Apply previous period weights (no look-ahead)
    w_prev = weights.shift(1).fillna(0.0)

    # Turnover (one-way)
    turnover = 0.5 * (weights - w_prev).abs().sum(axis=1)

    fee_rate = fee_bps / 10000.0
    fees = fee_rate * turnover

    ret_gross = (w_prev * r).sum(axis=1).fillna(0.0)
    ret_net = (ret_gross - fees).fillna(0.0)

    eq_gross = (1.0 + ret_gross).cumprod()
    eq_net = (1.0 + ret_net).cumprod()

    return {
        "turnover": turnover,
        "fees": fees,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
        "eq_gross": eq_gross,
        "eq_net": eq_net,
    }


def run_combo(
    prices_daily: pd.DataFrame,
    universe: List[str],
    core_ticker: str,
    sat_weight: float,
    cfg: Config,
) -> Dict[str, object]:
    # Rebalance prices
    prices_rebal = resample_prices(prices_daily, cfg.rebalance).dropna(how="all")
    if prices_rebal.empty:
        raise ValueError("Aucune donnée après resampling.")

    # Satellite inputs
    cols_for_weights = [c for c in universe if c in prices_rebal.columns]
    if len(cols_for_weights) < 2:
        raise ValueError("Univers satellite : au moins 2 tickers requis (présents dans les données).")

    risk_on = compute_risk_on_daily(prices_daily, prices_rebal.index, cfg)
    weights, scores = build_weights(prices_rebal[cols_for_weights], risk_on=risk_on, cfg=cfg)
    bt = backtest(prices_rebal[weights.columns], weights, cfg.fee_bps)

    # Core buy&hold
    if core_ticker not in prices_rebal.columns:
        raise ValueError("Ticker core introuvable dans les données.")
    core_px = prices_rebal[core_ticker].dropna()
    core_ret = core_px.pct_change().fillna(0.0)
    core_eq = (1.0 + core_ret).cumprod()

    # Align returns
    bot_ret = bt["ret_net"].reindex(core_ret.index).fillna(0.0)
    core_ret_aligned = core_ret.reindex(bot_ret.index).fillna(0.0)

    core_weight = 1.0 - float(sat_weight)
    combo_ret = core_weight * core_ret_aligned + float(sat_weight) * bot_ret
    combo_eq = (1.0 + combo_ret).cumprod()

    ppy = periods_per_year(cfg.rebalance)
    perf_bot = summarize(bt["eq_net"], bt["ret_net"], ppy)
    perf_core = summarize(core_eq.reindex(combo_eq.index).ffill(), core_ret_aligned, ppy)
    perf_combo = summarize(combo_eq, combo_ret, ppy)

    risk_on_aligned = risk_on.reindex(weights.index).fillna(False).astype(bool)
    transitions = int((risk_on_aligned.astype(int).diff().abs() > 0).sum())
    pct_risk_on = float(risk_on_aligned.mean())

    return {
        "prices_rebal": prices_rebal,
        "weights": weights,
        "scores": scores,
        "bt": bt,
        "core_eq": core_eq,
        "core_ret": core_ret_aligned,
        "combo_eq": combo_eq,
        "combo_ret": combo_ret,
        "perf_bot": perf_bot,
        "perf_core": perf_core,
        "perf_combo": perf_combo,
        "risk_on": risk_on_aligned,
        "diagnostic": {
            "pct_risk_on": pct_risk_on,
            "transitions": transitions,
            "avg_turnover": float(bt["turnover"].mean()),
            "sum_fees": float(bt["fees"].sum()),
        },
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

# Load
if upload_mode == "Un seul CSV (wide)":
    if uploaded_one is None:
        st.info("➡️ Upload un CSV (wide) pour démarrer.")
        st.stop()
    try:
        close = load_wide_csv(uploaded_one)
        problems = []
    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")
        st.stop()
else:
    if not uploaded_many:
        st.info("➡️ Upload plusieurs CSV (un par ticker) pour démarrer.")
        st.stop()
    try:
        close, tickers_loaded, problems = load_many_csv(uploaded_many)
    except Exception as e:
        st.error(f"Erreur lecture CSVs : {e}")
        st.stop()

if close.empty or close.shape[1] < 2:
    st.error("Pas assez de données chargées (au moins 2 tickers requis).")
    st.stop()

if problems:
    st.warning("Fichiers ignorés / problèmes:\n- " + "\n- ".join(problems))

available = list(close.columns)


# -----------------------------
# Sidebar: Params
# -----------------------------
with st.sidebar:
    st.divider()
    st.header("2) Paramètres satellite (Bot)")

    rebalance = st.selectbox("Fréquence", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)

    top_n = st.slider("Top N", 1, min(8, max(1, len(available))), 1)

    long_only = st.toggle("Long-only (ignore scores ≤ 0)", value=False)

    st.divider()
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
        help="Actif utilisé pour décider Risk-on / Risk-off."
    )

    market_filter_window_days = st.slider(
        "Fenêtre MA (jours)",
        min_value=50,
        max_value=400,
        value=200,
        step=10,
        help="MA en jours sur l'actif filtre (données daily)."
    )

    risk_off_mode = st.selectbox("Risk-off", ["CASH"], index=0)

    st.divider()
    st.header("4) Univers satellite")
    default_univ = [t for t in DEFAULT_UNIVERSE if t in available]
    if len(default_univ) < 2:
        default_univ = available[: min(8, len(available))]
    universe = st.multiselect("Univers (satellite)", options=available, default=default_univ)

    if len(universe) < 2:
        st.error("Choisis au moins 2 tickers dans l'univers satellite.")
        st.stop()

    st.divider()
    st.header("5) Core / Combo")

    core_ticker = st.selectbox(
        "Core (buy & hold)",
        options=available,
        index=available.index(DEFAULT_CORE) if DEFAULT_CORE in available else 0
    )

    sat_weight_pct = st.slider("Poids satellite (%)", 0, 50, 20, 5)
    sat_weight = sat_weight_pct / 100.0
    core_weight = 1.0 - sat_weight

    st.caption(f"Combo: {int(core_weight*100)}% Core / {int(sat_weight*100)}% Satellite")

    st.divider()
    st.header("6) Dates")
    start_d = st.date_input("Début", value=pd.to_datetime(close.index.min()).date())
    end_d = st.date_input("Fin", value=pd.to_datetime(close.index.max()).date())


# Apply date filter
close = close.loc[(close.index >= pd.to_datetime(str(start_d))) & (close.index <= pd.to_datetime(str(end_d)))].copy()
if close.empty:
    st.error("Aucune donnée dans la plage choisie.")
    st.stop()

# Keep only needed columns (universe + filter + core)
needed_cols = sorted(set(universe + [market_filter_asset, core_ticker]))
close = close[needed_cols].copy()

cfg = Config(
    rebalance=str(rebalance),
    fee_bps=float(fee_bps),
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

# Run main combo
try:
    out = run_combo(
        prices_daily=close,
        universe=universe,
        core_ticker=core_ticker,
        sat_weight=sat_weight,
        cfg=cfg,
    )
except Exception as e:
    st.error(f"Erreur backtest : {e}")
    st.stop()


# -----------------------------
# Weight sweep (10/20/30 + custom)
# -----------------------------
def weight_sweep(weights_list: List[float]) -> pd.DataFrame:
    rows = []
    for w in weights_list:
        o = run_combo(close, universe, core_ticker, w, cfg)
        rows.append({
            "Poids bot": f"{int(round(w*100))}%",
            "CAGR": o["perf_combo"]["CAGR"],
            "Sharpe": o["perf_combo"]["Sharpe"],
            "MaxDD": o["perf_combo"]["MaxDD"],
            "Vol": o["perf_combo"]["Vol"],
        })
    df = pd.DataFrame(rows)
    return df


sweep_weights = sorted(set([0.10, 0.20, 0.30, float(sat_weight)]))
sweep_df = weight_sweep(sweep_weights)


# -----------------------------
# UI
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Résumé", "🧪 Tests poids", "🔍 Diagnostic", "🧾 Message"])

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

    st.subheader(f"Portefeuille combiné — {int(out['core_weight']*100)}% Core / {int(out['sat_weight']*100)}% Bot")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("CAGR (combo)", fmt_pct(perf_combo["CAGR"]))
    e2.metric("Vol (ann.)", fmt_pct(perf_combo["Vol"]))
    e3.metric("Sharpe", fmt_num(perf_combo["Sharpe"]))
    e4.metric("Max DD", fmt_pct(perf_combo["MaxDD"]))
    e5.metric("Calmar", fmt_num(perf_combo["Calmar"]))

    st.caption(
        f"Rebal={cfg.rebalance} | Top {cfg.top_n} | Frais={cfg.fee_bps:.1f} bps | "
        f"Filtre={'ON' if cfg.market_filter_on else 'OFF'} ({cfg.market_filter_asset}, MA{cfg.market_filter_window_days}j) | "
        f"Risk-off={cfg.risk_off_mode} | Momentum={cfg.momentum_mode}"
    )

    st.subheader("Courbes de capital (net)")
    eq_df = pd.DataFrame(
        {
            "Combiné": out["combo_eq"],
            "Bot (satellite)": out["bt"]["eq_net"].reindex(out["combo_eq"].index).ffill(),
            f"Core ({core_ticker})": out["core_eq"].reindex(out["combo_eq"].index).ffill(),
        }
    ).dropna()
    st.line_chart(eq_df)

    st.subheader("Allocations récentes (cibles bot)")
    w_tail = out["weights"].tail(12).copy()
    w_tail.index = w_tail.index.date
    st.dataframe(w_tail.style.format("{:.0%}"), use_container_width=True)

with tab2:
    st.subheader("Tests rapides 10% / 20% / 30% (et ton poids actuel)")
    tmp = sweep_df.copy()
    tmp["CAGR"] = tmp["CAGR"].apply(lambda x: fmt_pct(x))
    tmp["Vol"] = tmp["Vol"].apply(lambda x: fmt_pct(x))
    tmp["Sharpe"] = tmp["Sharpe"].apply(lambda x: fmt_num(x))
    tmp["MaxDD"] = tmp["MaxDD"].apply(lambda x: fmt_pct(x))
    st.dataframe(tmp, use_container_width=True)

    # Best by Sharpe (tie-breaker by MaxDD)
    raw = sweep_df.copy()
    raw["MaxDD_abs"] = raw["MaxDD"].abs()
    best = raw.sort_values(by=["Sharpe", "MaxDD_abs"], ascending=[False, True]).iloc[0]
    st.info(f"✅ Meilleur (dans ce mini-test) : **{best['Poids bot']}** — Sharpe {best['Sharpe']:.2f}, MaxDD {best['MaxDD']*100:.2f}%.")

with tab3:
    st.subheader("Diagnostic Bot (satellite)")
    diag = out["diagnostic"]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("% Risk-on", f"{diag['pct_risk_on']*100:.1f}%")
    x2.metric("Transitions on/off", str(diag["transitions"]))
    x3.metric("Turnover moyen", f"{diag['avg_turnover']:.2f}")
    x4.metric("Somme frais (approx)", fmt_pct(diag["sum_fees"]))

    st.subheader("Dernier signal bot")
    last_t = out["weights"].index[-1]
    is_on = bool(out["risk_on"].loc[last_t])

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Scores momentum (dernier)**")
        last_scores = out["scores"].loc[last_t].dropna().sort_values(ascending=False)
        st.dataframe(last_scores.to_frame("Score").style.format("{:.2%}"), use_container_width=True)
    with colR:
        st.markdown("**Allocation cible (dernier)**")
        last_alloc = out["weights"].loc[last_t]
        last_alloc = last_alloc[last_alloc > 0].sort_values(ascending=False)
        if (not is_on) or last_alloc.empty:
            st.info("CASH (risk-off ou aucun signal).")
        else:
            st.dataframe(last_alloc.to_frame("Poids").style.format("{:.0%}"), use_container_width=True)

with tab4:
    st.subheader("Message automatique (copier/coller)")
    last_t = out["weights"].index[-1]
    last_date = last_t.strftime("%Y-%m-%d")
    is_on = bool(out["risk_on"].loc[last_t])

    alloc = out["weights"].loc[last_t]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual {cfg.lb_dual[0]}+{cfg.lb_dual[1]} (w={cfg.w_dual[0]:.2f}/{cfg.w_dual[1]:.2f})"
    else:
        mom_desc = f"Single {cfg.lb_single}"

    filt_desc = "Filtre OFF" if not cfg.market_filter_on else f"Filtre ON ({cfg.market_filter_asset} > MA{cfg.market_filter_window_days}j)"

    if not is_on:
        pos_line = "Risk-off: CASH (0% exposé)"
    else:
        if alloc.empty:
            pos_line = "Allocation: CASH (aucun score)"
        else:
            parts = [f"{t}: {int(round(w*100))}%" for t, w in alloc.items()]
            pos_line = "Allocation: " + ", ".join(parts)

    msg = "\n".join(
        [
            f"📌 Signal bot — {last_date}",
            f"- Rebal: {'mensuel' if cfg.rebalance == 'M' else 'hebdo'} | Frais: {cfg.fee_bps:.1f} bps",
            f"- Momentum: {mom_desc} | Top {cfg.top_n} | Long-only: {'OUI' if cfg.long_only else 'NON'}",
            f"- {filt_desc} | Risk-on: {'OUI' if is_on else 'NON'}",
            f"- {pos_line}",
            "",
            f"🧩 Combo: {int(out['core_weight']*100)}% Core ({core_ticker}) / {int(out['sat_weight']*100)}% Bot",
            f"📈 Bot (net): CAGR {perf_bot['CAGR']*100:.2f}% | MaxDD {perf_bot['MaxDD']*100:.2f}% | Sharpe {perf_bot['Sharpe']:.2f}",
            f"📊 Core ({core_ticker}): CAGR {perf_core['CAGR']*100:.2f}% | MaxDD {perf_core['MaxDD']*100:.2f}% | Sharpe {perf_core['Sharpe']:.2f}",
            f"✅ Combo: CAGR {perf_combo['CAGR']*100:.2f}% | MaxDD {perf_combo['MaxDD']*100:.2f}% | Sharpe {perf_combo['Sharpe']:.2f}",
        ]
    )
    st.text_area("Message", value=msg, height=280)

st.divider()
st.caption("⚠️ Backtest simplifié (pas un conseil financier). Taxes/slippage/exécution non inclus.")
