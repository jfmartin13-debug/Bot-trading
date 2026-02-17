# bot_streamlit.py
# ------------------------------------------------------------
# Trading Bot Streamlit — Version SIMPLE & STABLE (CSV Upload)
# - Pas de yfinance / pas de Stooq : tu uploades un CSV de prix
# - Multi-asset : filtre marché (MA 12), Top N, frais (bps), momentum (single/dual)
# - Single-asset : buy&hold comparatif
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Trading Bot (CSV)", page_icon="📈", layout="wide")
st.title("📈 Trading Bot — Single + Multi-asset (Dual Momentum) — CSV")
st.caption("Version ultra stable : pas de data provider externe. Upload un CSV de prix (Date + colonnes tickers).")


DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_SINGLE = "SPY"


@dataclass(frozen=True)
class Config:
    rebalance: str                  # "M" ou "W"
    fee_bps: float                  # ex: 10 = 0.10%
    top_n: int
    market_filter_on: bool
    market_filter_asset: str
    market_filter_window: int       # 12 périodes (mensuel) ou 52 (hebdo)
    risk_off_mode: str              # "CASH" ou "DEFENSIVE"
    defensive_asset: str
    momentum_mode: str              # "SINGLE" ou "DUAL"
    mom_single_lb: int
    mom_dual_lb: Tuple[int, int]
    mom_dual_w: Tuple[float, float]
    long_only: bool


# -----------------------------
# Helpers
# -----------------------------
def load_wide_csv(uploaded_file) -> pd.DataFrame:
    """
    CSV attendu:
      Date,SPY,QQQ,EFA,...
      2006-01-03,123.4,45.6,...
    """
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 2:
        raise ValueError("CSV invalide: il faut une colonne Date + au moins 1 colonne ticker.")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df


def resample_prices(close: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    if rebalance == "M":
        return close.resample("M").last()
    if rebalance == "W":
        return close.resample("W-FRI").last()
    raise ValueError("rebalance doit être 'M' ou 'W'")


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).mean()


def periods_per_year(rebalance: str) -> float:
    return 12.0 if rebalance == "M" else 52.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def summarize(equity: pd.Series, rets: pd.Series, ppy: float) -> Dict[str, float]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(rets) / ppy if ppy else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = float(rets.std(ddof=0) * math.sqrt(ppy))
    sharpe = safe_div(float(rets.mean() * ppy), vol)
    mdd = max_drawdown(equity)
    calmar = safe_div(float(cagr), abs(mdd)) if not np.isnan(mdd) and mdd != 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": float(cagr),
        "Vol": vol,
        "Sharpe": float(sharpe),
        "MaxDD": float(mdd),
        "Calmar": float(calmar),
    }


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


def compute_scores(prices_rebal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.momentum_mode == "SINGLE":
        return prices_rebal.pct_change(cfg.mom_single_lb)
    lb1, lb2 = cfg.mom_dual_lb
    w1, w2 = cfg.mom_dual_w
    s1 = prices_rebal.pct_change(lb1)
    s2 = prices_rebal.pct_change(lb2)
    return w1 * s1 + w2 * s2


def compute_risk_on(prices_rebal: pd.DataFrame, cfg: Config) -> pd.Series:
    if not cfg.market_filter_on:
        return pd.Series(True, index=prices_rebal.index)
    a = cfg.market_filter_asset
    if a not in prices_rebal.columns:
        return pd.Series(True, index=prices_rebal.index)
    ma = sma(prices_rebal[a], cfg.market_filter_window)
    return (prices_rebal[a] > ma).fillna(False).astype(bool)


def compute_weights(prices_rebal: pd.DataFrame, cfg: Config):
    scores = compute_scores(prices_rebal, cfg)
    risk_on = compute_risk_on(prices_rebal, cfg)

    w = pd.DataFrame(0.0, index=prices_rebal.index, columns=prices_rebal.columns)

    for t in prices_rebal.index:
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


def run_backtest(prices_rebal: pd.DataFrame, weights: pd.DataFrame, fee_bps: float):
    r = returns(prices_rebal).fillna(0.0)
    weights = weights.fillna(0.0)

    # poids appliqués sur la période suivante (w_{t-1} * ret_t)
    w_prev = weights.shift(1).fillna(0.0)

    turnover = (weights - w_prev).abs().sum(axis=1)
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


# -----------------------------
# Sidebar: upload + params
# -----------------------------
with st.sidebar:
    st.header("1) Données")
    uploaded = st.file_uploader("Upload CSV (Date + colonnes tickers)", type=["csv"])

    st.divider()
    st.header("2) Paramètres")

    rebalance = st.selectbox("Fréquence", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", 0.0, 200.0, 10.0, 1.0)

    momentum_mode = st.selectbox("Momentum", ["DUAL", "SINGLE"], index=0)
    if momentum_mode == "DUAL":
        lb1 = st.slider("Lookback 1 (périodes)", 1, 12, 3)
        lb2 = st.slider("Lookback 2 (périodes)", 2, 18, 6)
        w1 = st.slider("Poids lookback 1", 0.0, 1.0, 0.5, 0.05)
        w2 = 1.0 - float(w1)
        mom_single_lb = 12
    else:
        mom_single_lb = st.slider("Lookback single (périodes)", 1, 18, 12)
        lb1, lb2, w1, w2 = 3, 6, 0.5, 0.5

    top_n = st.slider("Top N", 1, 5, 2)
    market_filter_on = st.toggle("Filtre marché ON (prix > MA)", value=True)
    risk_off_mode = st.selectbox("Risk-off", ["CASH", "DEFENSIVE"], index=0)
    long_only = st.toggle("Long-only (score > 0)", value=True)

if uploaded is None:
    st.info("➡️ Upload un CSV pour démarrer.")
    st.stop()

try:
    close = load_wide_csv(uploaded)
except Exception as e:
    st.error(f"Erreur lecture CSV: {e}")
    st.stop()

if close.empty:
    st.error("CSV vide ou non lisible.")
    st.stop()

available = list(close.columns)

with st.sidebar:
    st.divider()
    st.header("3) Choix des tickers")

    default_univ = [t for t in DEFAULT_UNIVERSE if t in available]
    if len(default_univ) < 2:
        default_univ = available[: min(8, len(available))]

    universe = st.multiselect("Univers", options=available, default=default_univ)
    if len(universe) < 2:
        st.error("Choisis au moins 2 tickers dans l'univers.")
        st.stop()

    market_filter_asset = st.selectbox(
        "Actif filtre (ex: SPY)", options=available, index=available.index("SPY") if "SPY" in available else 0
    )
    defensive_asset = st.selectbox(
        "Actif défensif", options=available, index=available.index("IEF") if "IEF" in available else 0
    )
    single_ticker = st.selectbox(
        "Single-asset (comparatif)", options=available, index=available.index(DEFAULT_SINGLE) if DEFAULT_SINGLE in available else 0
    )

    ma_default = 12 if rebalance == "M" else 52
    market_filter_window = st.slider("Fenêtre MA (périodes)", 6, 80, ma_default)

    st.divider()
    st.header("4) Dates")
    start_d = st.date_input("Début", value=pd.to_datetime(close.index.min()).date())
    end_d = st.date_input("Fin", value=pd.to_datetime(close.index.max()).date())

# Filtrer dates
close = close.loc[(close.index >= pd.to_datetime(str(start_d))) & (close.index <= pd.to_datetime(str(end_d)))]
if close.empty:
    st.error("Aucune donnée dans la plage choisie.")
    st.stop()

# Rebal
prices_rebal = resample_prices(close, rebalance).dropna(how="all")
prices_multi = prices_rebal[universe].dropna(how="all")

if prices_multi.shape[0] < 20:
    st.warning("Peu de périodes après rebal. Les stats peuvent être instables.")

cfg = Config(
    rebalance=rebalance,
    fee_bps=float(fee_bps),
    top_n=int(min(top_n, len(universe))),
    market_filter_on=bool(market_filter_on),
    market_filter_asset=str(market_filter_asset),
    market_filter_window=int(market_filter_window),
    risk_off_mode=str(risk_off_mode),
    defensive_asset=str(defensive_asset),
    momentum_mode=str(momentum_mode),
    mom_single_lb=int(mom_single_lb),
    mom_dual_lb=(int(lb1), int(lb2)),
    mom_dual_w=(float(w1), float(w2)),
    long_only=bool(long_only),
)

# Compute strategy
weights, scores, risk_on = compute_weights(prices_multi, cfg)
bt = run_backtest(prices_multi, weights, cfg.fee_bps)

ppy = periods_per_year(rebalance)
perf = summarize(bt["eq_net"], bt["ret_net"], ppy)

# Single asset comparatif
single_prices = prices_rebal[[single_ticker]].dropna()
single_ret = single_prices[single_ticker].pct_change().fillna(0.0)
single_eq = (1.0 + single_ret).cumprod()
perf_single = summarize(single_eq, single_ret, ppy)

# -----------------------------
# UI output
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Résumé", "🔍 Diagnostic", "🧾 Message automatique"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", fmt_pct(perf["CAGR"]))
    c2.metric("Vol (ann.)", fmt_pct(perf["Vol"]))
    c3.metric("Sharpe", fmt_num(perf["Sharpe"]))
    c4.metric("Max DD", fmt_pct(perf["MaxDD"]))
    c5.metric("Calmar", fmt_num(perf["Calmar"]))

    st.caption(
        "Multi-asset (net) vs Single-asset buy&hold. Frais appliqués sur turnover à chaque rebal."
    )

    eq_df = pd.DataFrame(
        {
            "Multi (net)": bt["eq_net"],
            "Multi (brut)": bt["eq_gross"],
            f"Single ({single_ticker})": single_eq.reindex(bt["eq_net"].index).ffill(),
        }
    ).dropna()

    st.subheader("Courbe de capital")
    st.line_chart(eq_df)

    st.subheader("Allocations récentes (cibles)")
    w_tail = weights.tail(12).copy()
    w_tail.index = w_tail.index.date
    st.dataframe(w_tail.style.format("{:.0%}"), use_container_width=True)

with tab2:
    st.subheader("Pourquoi le CAGR peut baisser en 3p+6p (dual) ?")
    st.write(
        "- Dual (3+6) réagit plus vite → turnover souvent plus élevé → frais + whipsaws.\n"
        "- Si risk-off = CASH, tu peux rater des périodes favorables (obligations).\n"
        "- Tester: poids 0.25/0.75, risk-off DEFENSIVE, ou rebal mensuel."
    )

    risk_on_aligned = risk_on.reindex(bt["eq_net"].index).fillna(False)
    transitions = int((risk_on_aligned.astype(int).diff().abs() > 0).sum())
    pct_on = float(risk_on_aligned.mean())

    avg_turn = float(bt["turnover"].mean())
    avg_fee = float(bt["fees"].mean())

    w_prev = weights.shift(1).fillna(0.0)
    hhi = float((w_prev.pow(2).sum(axis=1)).mean())  # concentration

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("% Risk-on", f"{pct_on*100:.1f}%")
    d2.metric("Transitions on/off", str(transitions))
    d3.metric("Turnover moyen", f"{avg_turn:.2f}")
    d4.metric("Concentration (HHI)", f"{hhi:.2f}")

    st.caption("HHI proche de 1.0 = très concentré. Turnover élevé = plus de frais et plus de changements.")

    st.subheader("Dernier signal")
    last_t = prices_multi.index[-1]
    last_scores = scores.loc[last_t].dropna().sort_values(ascending=False)
    last_alloc = weights.loc[last_t][weights.loc[last_t] > 0].sort_values(ascending=False)

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Scores momentum (dernier)**")
        st.dataframe(last_scores.to_frame("Score").style.format("{:.2%}"), use_container_width=True)
    with colR:
        st.markdown("**Allocation cible (dernier)**")
        if last_alloc.empty:
            st.info("Aucune position (CASH ou défensif selon paramètres).")
        else:
            st.dataframe(last_alloc.to_frame("Poids").style.format("{:.0%}"), use_container_width=True)

with tab3:
    st.subheader("Message automatique (copier/coller)")
    last_t = prices_multi.index[-1]
    last_date = last_t.strftime("%Y-%m-%d")

    is_on = bool(risk_on.loc[last_t]) if last_t in risk_on.index else True
    alloc = weights.loc[last_t]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual {cfg.mom_dual_lb[0]}+{cfg.mom_dual_lb[1]} (w={cfg.mom_dual_w[0]:.2f}/{cfg.mom_dual_w[1]:.2f})"
    else:
        mom_desc = f"Single {cfg.mom_single_lb}"

    if cfg.market_filter_on:
        filt_desc = f"Filtre ON ({cfg.market_filter_asset} > MA{cfg.market_filter_window})"
    else:
        filt_desc = "Filtre OFF"

    if not is_on:
        if cfg.risk_off_mode == "DEFENSIVE":
            pos_line = f"Risk-off: DEFENSIVE (100% {cfg.defensive_asset})"
        else:
            pos_line = "Risk-off: CASH (0% exposé)"
    else:
        if alloc.empty:
            pos_line = "Allocation: CASH/DEF (aucun score positif)"
        else:
            parts = [f"{t}: {int(round(w*100))}%" for t, w in alloc.items()]
            pos_line = "Allocation: " + ", ".join(parts)

    # Message sans f-string complexe (zéro risque de syntax)
    msg_lines = [
        f"📌 Trading Bot — Signal {last_date}",
        f"- Univers: {', '.join(universe)}",
        f"- Rebal: {'mensuel' if cfg.rebalance == 'M' else 'hebdo'} | Frais: {cfg.fee_bps:.1f} bps",
        f"- Momentum: {mom_desc}",
        f"- {filt_desc} | Risk-on: {'OUI' if is_on else 'NON'}",
        f"- {pos_line}",
        "",
        f"📈 Perf (multi, net): CAGR {perf['CAGR']*100:.2f}% | MaxDD {perf['MaxDD']*100:.2f}% | Sharpe {perf['Sharpe']:.2f}",
        f"📊 Single ({single_ticker}): CAGR {perf_single['CAGR']*100:.2f}% | MaxDD {perf_single['MaxDD']*100:.2f}% | Sharpe {perf_single['Sharpe']:.2f}",
    ]
    msg = "\n".join(msg_lines)

    st.text_area("Message", value=msg, height=240)

st.divider()
st.caption("⚠️ Backtest simplifié (pas un conseil financier). Slippage/taxes/exécution non inclus.")
