# bot_streamlit.py
# ------------------------------------------------------------
# Streamlit Trading Bot (Single-asset + Multi-asset)
# Version STABLE (sans yfinance) : data via Stooq (pandas_datareader)
#
# Multi-asset :
#   Univers : SPY, QQQ, EFA, EEM, VNQ, TLT, IEF, GLD
#   Filtre marché ON (MA 12 mois sur SPY)
#   Top 2
#   Frais 10 bps (0.10%) sur turnover
#   Message automatique intégré
#
# IMPORTANT : fichier complet (pas de patch partiel)
# ------------------------------------------------------------

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plot (optionnel)
try:
    import plotly.express as px
except Exception:
    px = None


# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Trading Bot — Single + Multi-asset", page_icon="📈", layout="wide")
st.title("📈 Trading Bot — Single-asset + Multi-asset (Dual Momentum)")
st.caption(
    "Backtest à fréquence mensuelle/hebdo avec frais (bps) sur turnover. "
    "Data provider: Stooq (via pandas_datareader)."
)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_SINGLE = "SPY"


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    rebalance: str  # "M" ou "W"
    fee_bps: float
    top_n: int
    market_filter_on: bool
    market_filter_asset: str
    market_filter_ma_months: int
    risk_off_mode: str  # "CASH" ou "DEFENSIVE"
    defensive_asset: str
    momentum_mode: str  # "SINGLE" ou "DUAL"
    mom_lookback_single_p: int
    mom_lookback_dual_p: Tuple[int, int]
    mom_dual_weight: Tuple[float, float]
    long_only: bool


# ---------------------------
# Data loading (Stooq)
# ---------------------------
def _ensure_pandas_datareader():
    try:
        import pandas_datareader.data as web  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices_stooq(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Télécharge des prix via Stooq (pandas_datareader).
    Retourne un DataFrame de prix ajustés (Close).
    NOTE: Stooq tickers pour ETFs US fonctionnent souvent tels quels (SPY, QQQ, ...),
    mais si certains ne passent pas, il faut adapter. L'app ne crashe pas : elle affiche les manquants.
    """
    import pandas_datareader.data as web

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    frames = []
    ok_cols = []
    missing = []

    for t in tickers:
        try:
            df = web.DataReader(t, "stooq", start_dt, end_dt)
            # Stooq renvoie souvent index décroissant => trier
            df = df.sort_index()
            if "Close" not in df.columns or df["Close"].dropna().empty:
                missing.append(t)
                continue
            s = df["Close"].rename(t)
            frames.append(s)
            ok_cols.append(t)
        except Exception:
            missing.append(t)

    if not frames:
        return pd.DataFrame()

    close = pd.concat(frames, axis=1).sort_index()
    close.index = pd.to_datetime(close.index)
    close = close.dropna(how="all")

    # stocker la liste des tickers manquants dans l'attribut (pour affichage)
    close.attrs["missing"] = missing
    return close


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


def periods_per_year_from_rebalance(rebalance: str) -> float:
    return 12.0 if rebalance == "M" else 52.0


def summarize_performance(equity: pd.Series, rets: pd.Series, periods_per_year: float) -> Dict[str, float]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(rets) / periods_per_year if periods_per_year else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = float(rets.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = safe_div(float(rets.mean() * periods_per_year), vol)
    mdd = max_drawdown(equity)
    calmar = safe_div(cagr, abs(mdd)) if not np.isnan(mdd) and mdd != 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": float(cagr),
        "Vol (ann.)": vol,
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(mdd),
        "Calmar": float(calmar),
    }


def compute_momentum_scores(prices_rebal: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if cfg.momentum_mode == "SINGLE":
        lb = cfg.mom_lookback_single_p
        return prices_rebal.pct_change(lb)

    if cfg.momentum_mode == "DUAL":
        lb1, lb2 = cfg.mom_lookback_dual_p
        w1, w2 = cfg.mom_dual_weight
        s1 = prices_rebal.pct_change(lb1)
        s2 = prices_rebal.pct_change(lb2)
        return w1 * s1 + w2 * s2

    raise ValueError("momentum_mode invalide (SINGLE/DUAL)")


def market_filter_signal(prices_rebal: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    asset = cfg.market_filter_asset
    if asset not in prices_rebal.columns:
        return pd.Series(True, index=prices_rebal.index)
    ma = sma(prices_rebal[asset], cfg.market_filter_ma_months)
    sig = (prices_rebal[asset] > ma).fillna(False)
    return sig.astype(bool)


def compute_weights_multi_asset(prices_rebal: pd.DataFrame, cfg: BacktestConfig):
    scores = compute_momentum_scores(prices_rebal, cfg)
    risk_on = market_filter_signal(prices_rebal, cfg) if cfg.market_filter_on else pd.Series(True, index=prices_rebal.index)

    weights = pd.DataFrame(0.0, index=prices_rebal.index, columns=prices_rebal.columns)

    for t in prices_rebal.index:
        if not bool(risk_on.loc[t]):
            if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in weights.columns:
                weights.loc[t, cfg.defensive_asset] = 1.0
            continue

        row = scores.loc[t].dropna()
        if row.empty:
            continue

        if cfg.long_only:
            row = row[row > 0]
            if row.empty:
                if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in weights.columns:
                    weights.loc[t, cfg.defensive_asset] = 1.0
                continue

        top = row.sort_values(ascending=False).head(cfg.top_n).index.tolist()
        if not top:
            continue

        w = 1.0 / len(top)
        weights.loc[t, top] = w

    return weights, scores, risk_on


def backtest_from_weights(prices_rebal: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig):
    rets_assets = compute_returns(prices_rebal)

    weights = weights.reindex(prices_rebal.index).fillna(0.0)
    rets_assets = rets_assets.reindex(prices_rebal.index)

    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)

    fee_rate = cfg.fee_bps / 10000.0
    fees = fee_rate * turnover

    port_ret_gross = (w_prev * rets_assets).sum(axis=1).fillna(0.0)
    port_ret_net = (port_ret_gross - fees).fillna(0.0)

    equity_gross = (1.0 + port_ret_gross).cumprod()
    equity_net = (1.0 + port_ret_net).cumprod()

    return {
        "asset_returns": rets_assets,
        "weights": weights,
        "turnover": turnover,
        "fees": fees,
        "port_ret_gross": port_ret_gross,
        "port_ret_net": port_ret_net,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
    }


def backtest_single_asset(prices_rebal: pd.DataFrame, ticker: str):
    if ticker not in prices_rebal.columns:
        raise ValueError(f"{ticker} introuvable dans les prix.")
    rets = prices_rebal[ticker].pct_change().fillna(0.0)
    equity = (1.0 + rets).cumprod()
    return {"returns": rets, "equity": equity}


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:,.2f}%"


def format_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}"


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Début", value=date(2006, 1, 1))
    with c2:
        end = st.date_input("Fin", value=date.today())

    rebalance = st.selectbox("Rebalancement", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

    st.divider()
    st.subheader("Multi-asset")

    universe = st.multiselect("Univers", options=DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)
    top_n = st.slider("Top N", min_value=1, max_value=min(5, max(1, len(universe))), value=2)

    market_filter_on = st.toggle("Filtre marché (MA 12 mois) — ON", value=True)
    market_filter_asset = st.selectbox("Actif du filtre", options=sorted(set(universe + ["SPY"])), index=sorted(set(universe + ["SPY"])).index("SPY"))
    market_filter_ma_months = st.slider("Fenêtre MA (périodes)", 6, 18, 12, 1)

    risk_off_mode = st.selectbox("Mode Risk-off", ["CASH", "DEFENSIVE"], index=0)
    defensive_asset = st.selectbox(
        "Actif défensif (si DEFENSIVE)",
        options=sorted(set(universe)),
        index=sorted(set(universe)).index("IEF") if "IEF" in universe else 0,
    )

    st.divider()
    st.subheader("Momentum")

    momentum_mode = st.selectbox("Mode", ["DUAL", "SINGLE"], index=0)

    if momentum_mode == "SINGLE":
        mom_lookback_single_p = st.slider("Lookback (périodes)", 1, 18, 12, 1)
        lb_dual = (3, 6)
        w_dual = (0.5, 0.5)
    else:
        lb1 = st.slider("Lookback 1 (périodes)", 1, 12, 3, 1)
        lb2 = st.slider("Lookback 2 (périodes)", 2, 18, 6, 1)
        w1 = st.slider("Poids lookback 1", 0.0, 1.0, 0.5, 0.05)
        w2 = 1.0 - float(w1)
        mom_lookback_single_p = 12
        lb_dual = (lb1, lb2)
        w_dual = (w1, w2)

    long_only = st.toggle("Long-only (ignore scores ≤ 0)", value=True)

    st.divider()
    st.subheader("Single-asset")
    single_ticker = st.selectbox("Actif", options=sorted(set(universe + [DEFAULT_SINGLE])), index=sorted(set(universe + [DEFAULT_SINGLE])).index("SPY"))


cfg = BacktestConfig(
    start=str(start),
    end=str(end),
    rebalance=rebalance,
    fee_bps=float(fee_bps),
    top_n=int(top_n),
    market_filter_on=bool(market_filter_on),
    market_filter_asset=str(market_filter_asset),
    market_filter_ma_months=int(market_filter_ma_months),
    risk_off_mode=str(risk_off_mode),
    defensive_asset=str(defensive_asset),
    momentum_mode=str(momentum_mode),
    mom_lookback_single_p=int(mom_lookback_single_p),
    mom_lookback_dual_p=tuple(int(x) for x in lb_dual),
    mom_dual_weight=tuple(float(x) for x in w_dual),
    long_only=bool(long_only),
)

if len(universe) < 2:
    st.error("L'univers doit contenir au moins 2 actifs.")
    st.stop()

if not _ensure_pandas_datareader():
    st.error(
        "Dépendance manquante: **pandas_datareader**.\n\n"
        "✅ Solution (Streamlit Cloud): ajoute un fichier `requirements.txt` à la racine avec:\n"
        "- pandas_datareader\n\n"
        "Optionnel:\n"
        "- plotly\n\n"
        "Puis redeploie l'app."
    )
    st.stop()


# ---------------------------
# Load data
# ---------------------------
with st.spinner("Téléchargement des données via Stooq…"):
    needed = sorted(set(universe + [single_ticker, cfg.market_filter_asset, cfg.defensive_asset]))
    close = download_prices_stooq(needed, cfg.start, cfg.end)

if close.empty:
    st.error(
        "Impossible de récupérer les prix via Stooq. "
        "Vérifie la connectivité ou adapte les tickers au format Stooq."
    )
    st.stop()

missing = close.attrs.get("missing", [])
if missing:
    st.warning("Tickers non trouvés sur Stooq (ignorés): " + ", ".join(missing))

prices_rebal = resample_prices(close, cfg.rebalance).dropna(how="all")

# multi universe only
prices_multi = prices_rebal[[t for t in universe if t in prices_rebal.columns]].dropna(how="all")
if prices_multi.shape[1] < 2:
    st.error("Pas assez d'actifs valides après chargement. (Stooq a peut-être rejeté des tickers)")
    st.stop()


# ---------------------------
# Backtests
# ---------------------------
weights, scores, risk_on = compute_weights_multi_asset(prices_multi, cfg)
bt = backtest_from_weights(prices_multi, weights, cfg)

single_prices = prices_rebal[[c for c in [single_ticker] if c in prices_rebal.columns]].dropna()
single_bt = backtest_single_asset(single_prices, single_ticker) if not single_prices.empty else None

ppyear = periods_per_year_from_rebalance(cfg.rebalance)
perf_net = summarize_performance(bt["equity_net"], bt["port_ret_net"], ppyear)
perf_gross = summarize_performance(bt["equity_gross"], bt["port_ret_gross"], ppyear)
perf_single = summarize_performance(single_bt["equity"], single_bt["returns"], ppyear) if single_bt else None


# ---------------------------
# UI
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Résumé", "🔍 Diagnostic baisse CAGR", "🧾 Message automatique"])

with tab1:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Performance — Multi-asset (net)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CAGR", format_pct(perf_net["CAGR"]))
        k2.metric("Vol (ann.)", format_pct(perf_net["Vol (ann.)"]))
        k3.metric("Sharpe", format_num(perf_net["Sharpe (rf=0)"]))
        k4.metric("Max DD", format_pct(perf_net["Max Drawdown"]))
        k5.metric("Calmar", format_num(perf_net["Calmar"]))

        st.caption(
            f"Univers: {', '.join([t for t in universe if t in prices_multi.columns])} | Top {cfg.top_n} | "
            f"Frais: {cfg.fee_bps:.1f} bps | Rebal: {cfg.rebalance} | "
            f"Momentum: {cfg.momentum_mode} | Filtre: {'ON' if cfg.market_filter_on else 'OFF'}"
        )

        eq = pd.DataFrame({"Multi (net)": bt["equity_net"], "Multi (brut)": bt["equity_gross"]}).dropna()
        if single_bt:
            eq[f"Single ({single_ticker})"] = single_bt["equity"].reindex(eq.index).ffill()

        st.subheader("Courbe de capital")
        if px is not None:
            fig = px.line(eq, x=eq.index, y=eq.columns, labels={"x": "Date", "value": "Capital", "variable": "Série"})
            fig.update_layout(height=420, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(eq)

    with right:
        st.subheader("Trading & frais")
        avg_turnover = float(bt["turnover"].mean())
        avg_fee = float(bt["fees"].mean())
        total_fees = float(bt["fees"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Turnover moyen", f"{avg_turnover:,.2f}")
        c2.metric("Frais moyens", format_pct(avg_fee))
        c3.metric("Somme frais", format_pct(total_fees))

        st.subheader("Allocations récentes")
        w_tail = bt["weights"].tail(12).copy()
        w_tail.index = w_tail.index.date
        st.dataframe(w_tail.style.format("{:.0%}"), use_container_width=True)

with tab2:
    st.subheader("🔍 Analyse baisse de CAGR (ex: 12m → 3m+6m)")
    st.write(
        "- **Turnover**: 3m+6m change plus souvent de Top 2 ⇒ frais + bruit.\n"
        "- **Whipsaws**: momentum court réagit trop vite sur marchés range.\n"
        "- **Filtre MA12**: transitions risk-on/off amplifiées.\n"
        "- **Cash vs Defensive**: si Risk-off=CASH, tu rates les rally oblig.\n"
    )

    risk_on_aligned = risk_on.reindex(bt["equity_net"].index).fillna(False)
    transitions = int((risk_on_aligned.astype(int).diff().abs() > 0).sum())
    pct_risk_on = float(risk_on_aligned.mean())

    w_prev = bt["weights"].shift(1).fillna(0.0)
    hhi = (w_prev.pow(2).sum(axis=1)).replace([np.inf, -np.inf], np.nan)
    n_positions = float((w_prev > 0).sum(axis=1).mean())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("% périodes Risk-on", f"{pct_risk_on*100:,.1f}%")
    d2.metric("Transitions Risk-on/off", f"{transitions}")
    d3.metric("Positions moyennes", f"{n_positions:,.2f}")
    d4.metric("HHI moyen", f"{float(hhi.mean()):,.2f}")

    diag = pd.DataFrame(
        {"ret_net": bt["port_ret_net"], "fees": bt["fees"], "turnover": bt["turnover"], "risk_on": risk_on_aligned.astype(int)}
    ).dropna()

    st.divider()
    st.subheader("Indicateurs (corr simples)")
    st.write(
        f"- Corr(ret, turnover): **{format_num(float(diag['ret_net'].corr(diag['turnover'])))}**\n"
        f"- Corr(ret, fees): **{format_num(float(diag['ret_net'].corr(diag['fees'])))}**"
    )

    st.divider()
    st.subheader("Dernière sélection")
    last_t = prices_multi.index[-1]
    last_scores = scores.loc[last_t].dropna().sort_values(ascending=False)
    last_w = weights.loc[last_t][weights.loc[last_t] > 0].sort_values(ascending=False)

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Scores momentum (dernier)**")
        st.dataframe(last_scores.to_frame("Score").style.format("{:.2%}"), use_container_width=True)
    with cR:
        st.markdown("**Allocation cible (dernier)**")
        if last_w.empty:
            st.info("Aucune position (cash/defensive selon paramètres).")
        else:
            st.dataframe(last_w.to_frame("Poids").style.format("{:.0%}"), use_container_width=True)

    st.info(
        "Optimisations propres (souvent gagnantes) à tester :\n"
        "• Dual moins agressif : w=0.25 (3p) / 0.75 (6p)\n"
        "• Risk-off=DEFENSIVE (IEF/TLT)\n"
        "• Long-only ON (déjà)\n"
        "• Rebal mensuel\n"
        "• Top 3 (si tu acceptes + diversification)"
    )

with tab3:
    st.subheader("🧾 Message automatique")
    last_dt = prices_multi.index[-1]
    last_dt_str = last_dt.strftime("%Y-%m-%d")
    is_risk_on = bool(risk_on.loc[last_dt]) if last_dt in risk_on.index else True

    alloc = weights.loc[last_dt]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual {cfg.mom_lookback_dual_p[0]}+{cfg.mom_lookback_dual_p[1]} (w={cfg.mom_dual_weight[0]:.2f}/{cfg.mom_dual_weight[1]:.2f})"
    else:
        mom_desc = f"Single {cfg.mom_lookback_single_p}"

    filt_desc = (
        f"Filtre ON — {cfg.market_filter_asset} > MA{cfg.market_filter_ma_months}"
        if cfg.market_filter_on
        else "Filtre OFF"
    )

    if not is_risk_on:
        if cfg.risk_off_mode == "DEFENSIVE":
            pos_line = f"Mode risk-off: **DEFENSIVE** (100% {cfg.defensive_asset})"
        else:
            pos_line = "Mode risk-off: **CASH** (0% exposé)"
    else:
        if alloc.empty:
            pos_line = "Aucune position (cash/defensive selon paramètres)."
        else:
            pos_line = "Allocation cible: " + ", ".join([f"{t}: {w:.0%}" for t, w in alloc.items()])

    msg = f"""
📌 **Trading Bot — Signal {last_dt_str}**
- Univers: {", ".join([t for t in universe if t in prices_multi.columns])}
- Rebal: {"mensuel" if cfg.rebalance=="M" else "hebdo"} | Frais: {cfg.fee_bps:.1f} bps
- Momentum: {mom_desc}
- {filt_desc}
- Risk-on: **{"OUI" if is_risk_on else "NON"}**
- {pos_line}

📈 Perf (multi, net): CAGR {perf_net["CAGR"]*100:.2f}% | MaxDD {perf_net["Max Drawdown"]*100:.2f}% | Sharpe {perf_net["Sharpe (rf=0)"]:.2f}
"""
    msg = textwrap.dedent(msg).strip()
    st.text_area("Message", value=msg, height=220)

st.divider()
st.caption("⚠️ Backtest simplifié (pas un conseil financier). Slippage/taxes/exécution non inclus.")
