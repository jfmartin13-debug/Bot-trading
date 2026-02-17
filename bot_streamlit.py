# bot_streamlit.py
# ------------------------------------------------------------
# Streamlit Trading Bot (Single-asset + Multi-asset)
# Version ULTRA SIMPLE & STABLE: données via upload CSV (aucun provider externe)
#
# Multi-asset :
#   Univers : SPY, QQQ, EFA, EEM, VNQ, TLT, IEF, GLD (ou ce que tu uploades)
#   Filtre marché ON (MA 12 mois sur SPY)
#   Top 2
#   Frais 10 bps sur turnover
#   Message automatique intégré
#
# Format CSV recommandé (wide):
#   Date,SPY,QQQ,EFA,...
#   2006-01-03,123.45,....,...
#   ...
# (prix "Close" ou "Adj Close", peu importe, tant que c'est cohérent)
# ------------------------------------------------------------

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None


st.set_page_config(page_title="Trading Bot — CSV Upload", page_icon="📈", layout="wide")
st.title("📈 Trading Bot — Single + Multi-asset (Dual Momentum) — CSV Upload")
st.caption("Aucune source externe (Stooq/yfinance) : tu uploades tes prix en CSV et le backtest tourne.")


DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_SINGLE = "SPY"


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    rebalance: str                 # "M" ou "W"
    fee_bps: float                 # ex 10 = 0.10%
    top_n: int
    market_filter_on: bool
    market_filter_asset: str
    market_filter_ma_periods: int  # 12 périodes (mensuel) ou 52 (hebdo) selon rebalance
    risk_off_mode: str             # "CASH" ou "DEFENSIVE"
    defensive_asset: str
    momentum_mode: str             # "SINGLE" ou "DUAL"
    mom_single_lb: int
    mom_dual_lb: Tuple[int, int]
    mom_dual_w: Tuple[float, float]
    long_only: bool


def resample_prices(close: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    if rebalance == "M":
        return close.resample("M").last()
    if rebalance == "W":
        return close.resample("W-FRI").last()
    raise ValueError("rebalance doit être 'M' ou 'W'")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def periods_per_year(rebalance: str) -> float:
    return 12.0 if rebalance == "M" else 52.0


def summarize_performance(equity: pd.Series, rets: pd.Series, ppy: float) -> Dict[str, float]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(rets) / ppy if ppy else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = float(rets.std(ddof=0) * math.sqrt(ppy))
    sharpe = safe_div(float(rets.mean() * ppy), vol)
    mdd = max_drawdown(equity)
    calmar = safe_div(cagr, abs(mdd)) if not np.isnan(mdd) and mdd != 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": float(cagr),
        "Vol (ann.)": vol,
        "Sharpe": float(sharpe),
        "Max Drawdown": float(mdd),
        "Calmar": float(calmar),
    }


def compute_momentum(prices: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if cfg.momentum_mode == "SINGLE":
        return prices.pct_change(cfg.mom_single_lb)
    lb1, lb2 = cfg.mom_dual_lb
    w1, w2 = cfg.mom_dual_w
    return w1 * prices.pct_change(lb1) + w2 * prices.pct_change(lb2)


def market_filter(prices: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    a = cfg.market_filter_asset
    if a not in prices.columns:
        return pd.Series(True, index=prices.index)
    ma = sma(prices[a], cfg.market_filter_ma_periods)
    return (prices[a] > ma).fillna(False).astype(bool)


def compute_weights(prices: pd.DataFrame, cfg: BacktestConfig):
    scores = compute_momentum(prices, cfg)
    risk_on = market_filter(prices, cfg) if cfg.market_filter_on else pd.Series(True, index=prices.index)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in prices.index:
        if not bool(risk_on.loc[t]):
            if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in w.columns:
                w.loc[t, cfg.defensive_asset] = 1.0
            continue

        row = scores.loc[t].dropna()
        if row.empty:
            continue

        if cfg.long_only:
            row = row[row > 0]
            if row.empty:
                if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in w.columns:
                    w.loc[t, cfg.defensive_asset] = 1.0
                continue

        top = row.sort_values(ascending=False).head(cfg.top_n).index.tolist()
        if top:
            w.loc[t, top] = 1.0 / len(top)

    return w, scores, risk_on


def backtest(prices: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig):
    r = compute_returns(prices).reindex(prices.index)
    weights = weights.reindex(prices.index).fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)

    turnover = (weights - w_prev).abs().sum(axis=1)
    fee = (cfg.fee_bps / 10000.0) * turnover

    ret_gross = (w_prev * r).sum(axis=1).fillna(0.0)
    ret_net = (ret_gross - fee).fillna(0.0)

    eq_gross = (1.0 + ret_gross).cumprod()
    eq_net = (1.0 + ret_net).cumprod()

    return {
        "weights": weights,
        "scores": None,
        "turnover": turnover,
        "fees": fee,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
        "eq_gross": eq_gross,
        "eq_net": eq_net,
    }


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


def load_wide_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if df.shape[1] < 2:
        raise ValueError("CSV invalide: il faut une colonne Date + au moins 1 ticker.")
    # détecter colonne date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("1) Données (CSV)")
    st.write("Upload un CSV avec colonnes = tickers (SPY, QQQ, …) et 1ère colonne = Date.")
    up = st.file_uploader("Upload CSV (wide)", type=["csv"])

    st.divider()
    st.header("2) Paramètres")

    rebalance = st.selectbox("Fréquence", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)
    top_n = st.slider("Top N", 1, 5, 2)

    momentum_mode = st.selectbox("Momentum", ["DUAL", "SINGLE"], index=0)
    if momentum_mode == "DUAL":
        lb1 = st.slider("Lookback 1 (périodes)", 1, 12, 3)
        lb2 = st.slider("Lookback 2 (périodes)", 2, 18, 6)
        w1 = st.slider("Poids lookback 1", 0.0, 1.0, 0.5, 0.05)
        w2 = 1.0 - w1
        mom_single = 12
    else:
        mom_single = st.slider("Lookback single (périodes)", 1, 18, 12)
        lb1, lb2, w1, w2 = 3, 6, 0.5, 0.5

    market_filter_on = st.toggle("Filtre marché ON", value=True)
    risk_off_mode = st.selectbox("Mode risk-off", ["CASH", "DEFENSIVE"], index=0)
    long_only = st.toggle("Long-only (score > 0)", value=True)

if up is None:
    st.info("➡️ Upload un CSV pour démarrer.")
    st.stop()

try:
    close = load_wide_csv(up)
except Exception as e:
    st.error(f"Erreur lecture CSV: {e}")
    st.stop()

available = list(close.columns)

st.sidebar.divider()
st.sidebar.header("3) Choix tickers")
universe = st.sidebar.multiselect("Univers", options=available, default=[t for t in DEFAULT_UNIVERSE if t in available] or available[: min(8, len(available))])
if len(universe) < 2:
    st.error("Choisis au moins 2 tickers dans l'univers.")
    st.stop()

market_filter_asset = st.sidebar.selectbox("Actif filtre (ex SPY)", options=available, index=available.index("SPY") if "SPY" in available else 0)
defensive_asset = st.sidebar.selectbox("Actif défensif", options=available, index=available.index("IEF") if "IEF" in available else 0)
single_ticker = st.sidebar.selectbox("Single-asset", options=available, index=available.index("SPY") if "SPY" in available else 0)

# MA window en "périodes" aligné sur rebal
ma_default = 12 if rebalance == "M" else 52
ma_periods = st.sidebar.slider("Fenêtre MA (périodes)", 6, 80, ma_default)

# Période dates
st.sidebar.divider()
start = st.sidebar.date_input("Début", value=pd.to_datetime(close.index.min()).date())
end = st.sidebar.date_input("Fin", value=pd.to_datetime(close.index.max()).date())

close = close.loc[(close.index >= pd.to_datetime(str(start))) & (close.index <= pd.to_datetime(str(end)))]
if close.empty:
    st.error("Aucune donnée dans la plage choisie.")
    st.stop()

prices_rebal = resample_prices(close, rebalance).dropna(how="all")
prices_multi = prices_rebal[universe].dropna(how="all")

cfg = BacktestConfig(
    start=str(start),
    end=str(end),
    rebalance=rebalance,
    fee_bps=float(fee_bps),
    top_n=int(min(top_n, len(universe))),
    market_filter_on=bool(market_filter_on),
    market_filter_asset=str(market_filter_asset),
    market_filter_ma_periods=int(ma_periods),
    risk_off_mode=str(risk_off_mode),
    defensive_asset=str(defensive_asset),
    momentum_mode=str(momentum_mode),
    mom_single_lb=int(mom_single),
    mom_dual_lb=(int(lb1), int(lb2)),
    mom_dual_w=(float(w1), float(w2)),
    long_only=bool(long_only),
)

weights, scores, risk_on = compute_weights(prices_multi, cfg)
bt = backtest(prices_multi, weights, cfg)

ppy = periods_per_year(rebalance)
perf = summarize_performance(bt["eq_net"], bt["ret_net"], ppy)

# Single asset
single = prices_rebal[[single_ticker]].dropna()
single_ret = single[single_ticker].pct_change().fillna(0.0)
single_eq = (1.0 + single_ret).cumprod()
perf_single = summarize_performance(single_eq, single_ret, ppy)

# ---------------------------
# UI
# ---------------------------
t1, t2, t3 = st.tabs(["📊 Résumé", "🔍 Diagnostic", "🧾 Message"])

with t1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", fmt_pct(perf["CAGR"]))
    c2.metric("Vol (ann.)", fmt_pct(perf["Vol (ann.)"]))
    c3.metric("Sharpe", f"{perf['Sharpe']:.2f}" if not np.isnan(perf["Sharpe"]) else "—")
    c4.metric("Max DD", fmt_pct(perf["Max Drawdown"]))
    c5.metric("Calmar", f"{perf['Calmar']:.2f}" if not np.isnan(perf["Calmar"]) else "—")

    eq = pd.DataFrame({"Multi (net)": bt["eq_net"], f"Single ({single_ticker})": single_eq.reindex(bt["eq_net"].index).ffill()}).dropna()

    st.subheader("Courbe de capital")
    if px is not None:
        fig = px.line(eq, x=eq.index, y=eq.columns, labels={"x": "Date", "value": "Capital", "variable": "Série"})
        fig.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(eq)

    st.subheader("Allocations récentes")
    tail_w = weights.tail(12).copy()
    tail_w.index = tail_w.index.date
    st.dataframe(tail_w.style.format("{:.0%}"), use_container_width=True)

with t2:
    st.subheader("Diagnostic baisse CAGR (ex: 12p → 3p+6p)")
    risk_on_aligned = risk_on.reindex(bt["eq_net"].index).fillna(False)
    transitions = int((risk_on_aligned.astype(int).diff().abs() > 0).sum())
    pct_risk_on = float(risk_on_aligned.mean())

    w_prev = weights.shift(1).fillna(0.0)
    hhi = (w_prev.pow(2).sum(axis=1)).mean()
    avg_turn = float(bt["turnover"].mean())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("% Risk-on", f"{pct_risk_on*100:.1f}%")
    d2.metric("Transitions", f"{transitions}")
    d3.metric("HHI moyen", f"{hhi:.2f}")
    d4.metric("Turnover moyen", f"{avg_turn:.2f}")

    st.write("➡️ Si le dual momentum baisse le CAGR, c’est souvent turnover↑ + whipsaws↑. Essaie w=0.25/0.75 et/ou DEFENSIVE.")

with t3:
    st.subheader("Message automatique")
    last_dt = prices_multi.index[-1]
    is_on = bool(risk_on.loc[last_dt]) if last_dt in risk_on.index else True
    alloc = weights.loc[last_dt]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual {cfg.mom_dual_lb[0]}+{cfg.mom_dual_lb[1]} (w={cfg.mom_dual_w[0]:.2f}/{cfg.mom_dual_w[1]:.2f})"
    else:
        mom_desc = f"Single {cfg.mom_single_lb}"

    filt_desc = f"Filtre ON ({cfg.market_filter_asset} > MA{cfg.market_filter_ma_periods})" if cfg.market_filter_on else "Filtre OFF"

    if not is_on:
        pos = f"Risk-off: {cfg.risk_off_mode}" + (f" (100% {cfg.defensive_asset})" if cfg.risk_off_mode == "DEFENSIVE" else "")
    else:
        pos = "Allocation: " + (", ".join([f"{t}:{w:.0%}" for t, w in alloc.items()]) if not alloc.empty else "CASH/DEF")

    msg = f"""
📌 **Trading Bot — Signal {last_dt.strftime('%Y-%m-%d')}**
- Univers: {", ".join(universe)}
- Rebal: {"mensuel" if cfg.rebalance=="M" else "hebdo"} | Frais: {cfg.fee_bps:.1f} bps
- Momentum: {mom_desc}
- {filt_desc} | Risk-on: {"OUI" if is_on else "NON"}
- {pos}

📈 Perf (multi, net): CAGR {perf["CAGR"]*100:.2f}% | MaxDD {perf["Max Drawdown"]*100:.2f}% | Sharpe {perf["Sharpe']:.2f if not np.isnan(perf['Sharpe']) else 0:.2f}
"""
    st.text_area("Message", value=textwrap.dedent(msg).strip(), height=220)

st.caption("⚠️ Backtest simplifié (pas un conseil financier).")
