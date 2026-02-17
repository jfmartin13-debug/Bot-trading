# bot_streamlit.py
# ------------------------------------------------------------
# Streamlit Trading Bot (Single-asset + Multi-asset)
# Multi-asset (univers étendu) :
#   SPY, QQQ, EFA, EEM, VNQ, TLT, IEF, GLD
#   Filtre marché ON (MA 12 mois sur SPY)
#   Top 2
#   Frais 10 bps (0.10%) appliqués sur le turnover (rebal mensuel)
#   Message automatique intégré
#
# IMPORTANT : fichier complet (pas de patch partiel)
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

# Dépendance la plus simple sur Streamlit Cloud
import yfinance as yf

try:
    import plotly.express as px
except Exception:
    px = None


# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(
    page_title="Trading Bot — Single + Multi-asset",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading Bot — Single-asset + Multi-asset (Dual Momentum)")
st.caption(
    "Backtest mensuel (close->close) avec frais (bps) appliqués sur le turnover. "
    "But: analyser pourquoi le CAGR a baissé en double momentum (3m+6m) et optimiser proprement."
)


# ---------------------------
# Paramètres par défaut
# ---------------------------
DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_SINGLE = "SPY"


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    rebalance: str  # "M" ou "W"
    fee_bps: float  # ex: 10 = 10 bps = 0.10%
    top_n: int
    market_filter_on: bool
    market_filter_asset: str  # typiquement "SPY"
    market_filter_ma_months: int  # 12
    risk_off_mode: str  # "CASH" ou "DEFENSIVE"
    defensive_asset: str  # ex: "IEF" ou "TLT"
    momentum_mode: str  # "SINGLE" ou "DUAL"
    mom_lookback_single_m: int  # ex: 12
    mom_lookback_dual_m: Tuple[int, int]  # ex: (3, 6)
    mom_dual_weight: Tuple[float, float]  # ex: (0.5, 0.5)
    long_only: bool  # true


# ---------------------------
# Utils data
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les prix ajustés via yfinance.
    Retourne un DataFrame (DateTimeIndex) avec colonnes tickers.
    """
    # yfinance accepte liste/str
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    # Gestion formats possibles
    if isinstance(data.columns, pd.MultiIndex):
        # auto_adjust=True => colonne "Close" (pas "Adj Close")
        close = data["Close"].copy()
    else:
        # si un seul ticker, data est une série ou df simple
        if "Close" in data.columns:
            close = data[["Close"]].copy()
            close.columns = tickers[:1]
        else:
            close = data.copy()
            close.columns = tickers[:1]

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    # Nettoyage: garder seulement tickers demandés si possible
    close = close[[c for c in tickers if c in close.columns]]

    return close


def resample_prices(close: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    """
    Convertit les prix quotidiens en prix de rebalancement.
    - "M": dernier jour de bourse de chaque mois
    - "W": dernier jour de bourse de chaque semaine
    """
    if rebalance == "M":
        return close.resample("M").last()
    if rebalance == "W":
        return close.resample("W-FRI").last()
    raise ValueError("rebalance doit être 'M' ou 'W'")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def annualize_return(period_ret: float, periods_per_year: float) -> float:
    # (1+R_total)^(1/years)-1
    if period_ret <= -1:
        return np.nan
    return (1 + period_ret) ** (periods_per_year) - 1


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def summarize_performance(equity: pd.Series, rets: pd.Series, periods_per_year: float) -> Dict[str, float]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(rets) / periods_per_year if periods_per_year > 0 else np.nan
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


def compute_momentum_scores(
    prices_rebal: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Momentum à la fréquence de rebalancement (mensuel/hebdo).
    - SINGLE: lookback x mois/semaines
    - DUAL: combinaison de 2 lookbacks (ex 3m + 6m)
    Retourne un DataFrame scores (même index que prices_rebal).
    """
    if cfg.momentum_mode == "SINGLE":
        lb = cfg.mom_lookback_single_m
        scores = prices_rebal.pct_change(lb)
        return scores

    if cfg.momentum_mode == "DUAL":
        lb1, lb2 = cfg.mom_lookback_dual_m
        w1, w2 = cfg.mom_dual_weight
        s1 = prices_rebal.pct_change(lb1)
        s2 = prices_rebal.pct_change(lb2)
        scores = w1 * s1 + w2 * s2
        return scores

    raise ValueError("momentum_mode invalide (SINGLE/DUAL)")


def market_filter_signal(
    prices_rebal: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.Series:
    """
    Filtre marché basé sur MA 12 mois de SPY (ou autre cfg.market_filter_asset),
    à la fréquence de rebalancement.
    Signal True = risk-on ; False = risk-off.
    """
    asset = cfg.market_filter_asset
    if asset not in prices_rebal.columns:
        # si absent, on renvoie tout risk-on pour éviter blocage
        return pd.Series(True, index=prices_rebal.index)

    ma = sma(prices_rebal[asset], cfg.market_filter_ma_months)
    sig = prices_rebal[asset] > ma
    sig = sig.fillna(False)  # pas de MA au début => risk-off
    return sig.astype(bool)


def compute_weights_multi_asset(
    prices_rebal: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Calcule:
    - weights: allocation par période (index rebal)
    - scores: momentum scores
    - risk_on: série bool du filtre marché
    """
    scores = compute_momentum_scores(prices_rebal, cfg)
    risk_on = market_filter_signal(prices_rebal, cfg) if cfg.market_filter_on else pd.Series(True, index=prices_rebal.index)

    weights = pd.DataFrame(0.0, index=prices_rebal.index, columns=prices_rebal.columns)

    for t in prices_rebal.index:
        if not risk_on.loc[t]:
            # Risk-off: CASH ou DEFENSIVE
            if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in weights.columns:
                weights.loc[t, cfg.defensive_asset] = 1.0
            else:
                # CASH => tout à 0 (rendement 0)
                pass
            continue

        # Risk-on: sélectionner top N selon score à la date t
        row = scores.loc[t].dropna()
        if row.empty:
            continue

        # Long-only: ignorer scores négatifs ? (optionnel)
        if cfg.long_only:
            row = row[row > 0]
            if row.empty:
                # aucun score positif => cash/defensive selon mode
                if cfg.risk_off_mode == "DEFENSIVE" and cfg.defensive_asset in weights.columns:
                    weights.loc[t, cfg.defensive_asset] = 1.0
                continue

        top = row.sort_values(ascending=False).head(cfg.top_n).index.tolist()
        if len(top) == 0:
            continue
        w = 1.0 / len(top)
        weights.loc[t, top] = w

    return weights, scores, risk_on


def backtest_from_weights(
    prices_rebal: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: BacktestConfig,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Backtest simple:
    - Rebalancement au timestamp t: on utilise weights[t] pour la période t->t+1
    - Rendement portefeuille = somme(weights[t] * asset_returns[t+1])
    - Frais appliqués sur turnover à t (changement de poids vs t-1) : fee_rate * turnover
    """
    rets_assets = compute_returns(prices_rebal)
    # Align: weights & returns
    weights = weights.reindex(prices_rebal.index).fillna(0.0)
    rets_assets = rets_assets.reindex(prices_rebal.index)

    # turnover à chaque date t: sum(|w_t - w_{t-1}|)
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    fee_rate = cfg.fee_bps / 10000.0
    fees = fee_rate * turnover

    # rendement port: utiliser poids de t sur rendement t (entre t-1 -> t) ?
    # Convention: poids décidés à t-1 s'appliquent sur ret à t.
    port_ret_gross = (w_prev * rets_assets).sum(axis=1).fillna(0.0)
    port_ret_net = (port_ret_gross - fees).fillna(0.0)

    equity_gross = (1.0 + port_ret_gross).cumprod()
    equity_net = (1.0 + port_ret_net).cumprod()

    out = {
        "asset_returns": rets_assets,
        "weights": weights,
        "turnover": turnover,
        "fees": fees,
        "port_ret_gross": port_ret_gross,
        "port_ret_net": port_ret_net,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
    }
    return out


def backtest_single_asset(
    prices_rebal: pd.DataFrame,
    ticker: str,
    cfg: BacktestConfig,
) -> Dict[str, pd.Series]:
    """
    Single-asset buy&hold (avec frais optionnels appliqués uniquement au début si on veut).
    Ici, on met 0 frais pour buy&hold (car turnover ~ 0).
    """
    if ticker not in prices_rebal.columns:
        raise ValueError(f"{ticker} introuvable dans les prix.")

    rets = prices_rebal[ticker].pct_change().fillna(0.0)
    equity = (1.0 + rets).cumprod()
    return {"returns": rets, "equity": equity}


def periods_per_year_from_rebalance(rebalance: str) -> float:
    return 12.0 if rebalance == "M" else 52.0


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:,.2f}%"


def format_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}"


# ---------------------------
# Sidebar — Paramètres
# ---------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    colA, colB = st.columns(2)
    with colA:
        start = st.date_input("Début", value=date(2006, 1, 1))
    with colB:
        end = st.date_input("Fin", value=date.today())

    rebalance = st.selectbox("Fréquence de rebalancement", ["M", "W"], index=0)
    fee_bps = st.number_input("Frais (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

    st.divider()
    st.subheader("Multi-asset")

    universe = st.multiselect("Univers", options=DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)

    top_n = st.slider("Top N", min_value=1, max_value=min(5, max(1, len(universe))), value=2)

    market_filter_on = st.toggle("Filtre marché (MA 12 mois) — ON", value=True)
    market_filter_asset = st.selectbox("Actif du filtre", options=sorted(set(universe + ["SPY"])), index=sorted(set(universe + ["SPY"])).index("SPY"))
    market_filter_ma_months = st.slider("Fenêtre MA (mois)", 6, 18, 12, 1)

    risk_off_mode = st.selectbox("Mode Risk-off", ["CASH", "DEFENSIVE"], index=0)
    defensive_asset = st.selectbox("Actif défensif (si DEFENSIVE)", options=sorted(set(universe)), index=sorted(set(universe)).index("IEF") if "IEF" in universe else 0)

    st.divider()
    st.subheader("Momentum")

    momentum_mode = st.selectbox("Mode", ["DUAL", "SINGLE"], index=0)

    if momentum_mode == "SINGLE":
        mom_lookback_single_m = st.slider("Lookback (périodes)", 1, 18, 12, 1)
        lb_dual = (3, 6)
        w_dual = (0.5, 0.5)
    else:
        lb1 = st.slider("Lookback 1 (périodes)", 1, 12, 3, 1)
        lb2 = st.slider("Lookback 2 (périodes)", 2, 18, 6, 1)
        w1 = st.slider("Poids lookback 1", 0.0, 1.0, 0.5, 0.05)
        w2 = 1.0 - w1
        mom_lookback_single_m = 12
        lb_dual = (lb1, lb2)
        w_dual = (w1, w2)

    long_only = st.toggle("Long-only (ignore scores ≤ 0)", value=True)

    st.divider()
    st.subheader("Single-asset")
    single_ticker = st.selectbox("Actif", options=sorted(set(universe + [DEFAULT_SINGLE])), index=sorted(set(universe + [DEFAULT_SINGLE])).index("SPY"))

    st.caption("💡 Conseil: si ton CAGR a chuté en 3m+6m, regarde le turnover, les périodes risk-off, et la fréquence de rebal.")


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
    mom_lookback_single_m=int(mom_lookback_single_m),
    mom_lookback_dual_m=tuple(int(x) for x in lb_dual),
    mom_dual_weight=tuple(float(x) for x in w_dual),
    long_only=bool(long_only),
)

if len(universe) < 2:
    st.error("L'univers doit contenir au moins 2 actifs pour le mode Top N.")
    st.stop()


# ---------------------------
# Chargement des données
# ---------------------------
with st.spinner("Téléchargement des données (yfinance)…"):
    close = download_prices(sorted(set(universe + [single_ticker, cfg.market_filter_asset, cfg.defensive_asset])), cfg.start, cfg.end)

if close.empty:
    st.error("Aucune donnée récupérée. Vérifie les dates / tickers.")
    st.stop()

prices_rebal = resample_prices(close, cfg.rebalance).dropna(how="all")
prices_rebal = prices_rebal[[c for c in prices_rebal.columns if c in close.columns]]  # garde colonnes valides

# On restreint au seul univers pour les calculs multi-asset
prices_multi = prices_rebal[[t for t in universe if t in prices_rebal.columns]].dropna(how="all")

if prices_multi.shape[0] < 30 and cfg.rebalance == "M":
    st.warning("Peu de périodes mensuelles disponibles. Les stats peuvent être instables.")
if prices_multi.shape[0] < 60 and cfg.rebalance == "W":
    st.warning("Peu de périodes hebdo disponibles. Les stats peuvent être instables.")


# ---------------------------
# Backtests
# ---------------------------
weights, scores, risk_on = compute_weights_multi_asset(prices_multi, cfg)
bt = backtest_from_weights(prices_multi, weights, cfg)

single_prices = prices_rebal[[single_ticker]].dropna()
single_prices_rebal = resample_prices(single_prices, cfg.rebalance).dropna()
bt_single = backtest_single_asset(single_prices_rebal, single_ticker, cfg)

ppyear = periods_per_year_from_rebalance(cfg.rebalance)

perf_net = summarize_performance(bt["equity_net"], bt["port_ret_net"], ppyear)
perf_gross = summarize_performance(bt["equity_gross"], bt["port_ret_gross"], ppyear)
perf_single = summarize_performance(bt_single["equity"], bt_single["returns"], ppyear)


# ---------------------------
# UI — Résultats
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Résumé", "🔍 Diagnostic baisse CAGR", "🧪 Optimisation (grid)", "🧾 Message automatique"])

with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Performance — Multi-asset (net de frais)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CAGR", format_pct(perf_net["CAGR"]))
        k2.metric("Vol (ann.)", format_pct(perf_net["Vol (ann.)"]))
        k3.metric("Sharpe (rf=0)", format_num(perf_net["Sharpe (rf=0)"]))
        k4.metric("Max DD", format_pct(perf_net["Max Drawdown"]))
        k5.metric("Calmar", format_num(perf_net["Calmar"]))

        st.caption(
            f"Univers: {', '.join(universe)} | Top {cfg.top_n} | "
            f"Frais: {cfg.fee_bps:.1f} bps | Rebal: {cfg.rebalance} | "
            f"Momentum: {cfg.momentum_mode} "
            f"({cfg.mom_lookback_dual_m[0]}+{cfg.mom_lookback_dual_m[1]} w={cfg.mom_dual_weight[0]:.2f}/{cfg.mom_dual_weight[1]:.2f}"
            f" ou {cfg.mom_lookback_single_m}) | "
            f"Filtre: {'ON' if cfg.market_filter_on else 'OFF'} (MA {cfg.market_filter_ma_months} sur {cfg.market_filter_asset})"
        )

        st.subheader("Courbe de capital")
        eq = pd.DataFrame(
            {
                "Multi (net)": bt["equity_net"],
                "Multi (brut)": bt["equity_gross"],
                f"Single ({single_ticker})": bt_single["equity"].reindex(bt["equity_net"].index).ffill(),
            }
        ).dropna()

        if px is not None and not eq.empty:
            fig = px.line(eq, x=eq.index, y=eq.columns, labels={"x": "Date", "value": "Capital", "variable": "Série"})
            fig.update_layout(height=420, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(eq)

    with right:
        st.subheader("Comparaison rapide")
        comp = pd.DataFrame(
            [
                ["Multi (net)", perf_net["CAGR"], perf_net["Vol (ann.)"], perf_net["Sharpe (rf=0)"], perf_net["Max Drawdown"]],
                ["Multi (brut)", perf_gross["CAGR"], perf_gross["Vol (ann.)"], perf_gross["Sharpe (rf=0)"], perf_gross["Max Drawdown"]],
                [f"Single ({single_ticker})", perf_single["CAGR"], perf_single["Vol (ann.)"], perf_single["Sharpe (rf=0)"], perf_single["Max Drawdown"]],
            ],
            columns=["Stratégie", "CAGR", "Vol (ann.)", "Sharpe", "Max DD"],
        )
        comp["CAGR"] = comp["CAGR"].map(lambda x: format_pct(x))
        comp["Vol (ann.)"] = comp["Vol (ann.)"].map(lambda x: format_pct(x))
        comp["Max DD"] = comp["Max DD"].map(lambda x: format_pct(x))
        comp["Sharpe"] = comp["Sharpe"].map(lambda x: format_num(x))
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.subheader("Trading & frais")
        avg_turnover = float(bt["turnover"].mean())
        avg_fee = float(bt["fees"].mean())
        total_fees = float(bt["fees"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Turnover moyen / période", f"{avg_turnover:,.2f}")
        c2.metric("Frais moyens / période", format_pct(avg_fee))
        c3.metric("Somme frais (approx.)", format_pct(total_fees))

        st.caption("Turnover = somme(|w_t - w_{t-1}|). Les frais sont appliqués sur ce turnover.")

        st.subheader("Allocations récentes")
        w_tail = bt["weights"].tail(12).copy()
        w_tail.index = w_tail.index.date
        st.dataframe(w_tail.style.format("{:.0%}"), use_container_width=True)

with tab2:
    st.subheader("🔍 Pourquoi le CAGR peut baisser en passant 12m → 3m+6m (dual momentum)")
    st.write(
        "- **Turnover plus élevé**: 3m+6m réagit plus vite, change plus souvent les Top 2 ⇒ plus de frais + plus d’“aller-retour”.\n"
        "- **Plus de whipsaws** (marchés range / faux départs): momentum court capte du bruit.\n"
        "- **Interaction avec le filtre MA12**: le dual momentum peut sortir/rentrer plus fréquemment autour des transitions risk-on/off.\n"
        "- **Biais de fenêtre**: selon la période (ex: post-2009 vs 2022), 12m peut être structurellement meilleur.\n"
        "- **Diversification effective**: Top 2 peut se retrouver souvent concentré sur les mêmes facteurs (ex: equity growth) en dual."
    )

    st.divider()
    st.subheader("Diagnostics chiffrés (sur TON backtest courant)")

    # 1) Exposition risk-off et nombre de transitions
    risk_on_aligned = risk_on.reindex(bt["equity_net"].index).fillna(False)
    transitions = (risk_on_aligned.astype(int).diff().abs() > 0).sum()
    pct_risk_on = float(risk_on_aligned.mean())

    # 2) Concentration (HHI)
    w_prev = bt["weights"].shift(1).fillna(0.0)
    hhi = (w_prev.pow(2).sum(axis=1)).replace([np.inf, -np.inf], np.nan)

    # 3) “Stabilité” des positions
    # nombre moyen d'actifs non nuls
    n_positions = (w_prev > 0).sum(axis=1).mean()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("% périodes Risk-on", f"{pct_risk_on*100:,.1f}%")
    d2.metric("Transitions Risk-on/off", f"{int(transitions)}")
    d3.metric("Positions moyennes", f"{n_positions:,.2f}")
    d4.metric("HHI moyen (concentration)", f"{float(hhi.mean()):,.2f}")

    st.caption("HHI proche de 1.0 = très concentré (ex: 100% un actif).")

    st.divider()
    st.subheader("Décomposition: performance vs turnover vs cash/defensive")

    diag = pd.DataFrame(
        {
            "Port ret net": bt["port_ret_net"],
            "Frais": bt["fees"],
            "Turnover": bt["turnover"],
            "Risk-on": risk_on_aligned.astype(int),
        }
    ).dropna()

    # corr simples (indicatives)
    corr_turn = diag["Port ret net"].corr(diag["Turnover"])
    corr_fee = diag["Port ret net"].corr(diag["Frais"])
    st.write(
        f"- Corr(returns, turnover): **{format_num(float(corr_turn))}**\n"
        f"- Corr(returns, frais): **{format_num(float(corr_fee))}**"
    )

    st.subheader("Table: sélection & scores (dernier point)")
    last_t = prices_multi.index[-1]
    last_scores = scores.loc[last_t].sort_values(ascending=False).dropna()
    last_w = weights.loc[last_t][weights.loc[last_t] > 0].sort_values(ascending=False)

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Scores momentum (dernier)**")
        st.dataframe(last_scores.to_frame("Score").style.format("{:.2%}"), use_container_width=True)
    with cR:
        st.markdown("**Allocation cible (dernier)**")
        if last_w.empty:
            st.info("Aucune position (cash) ou defensive selon paramètres.")
        else:
            st.dataframe(last_w.to_frame("Poids").style.format("{:.0%}"), use_container_width=True)

    st.info(
        "Pistes de fix rapides (souvent efficaces):\n"
        "1) Rebal mensuel (pas hebdo)\n"
        "2) Poids dual moins agressif: ex 25%*3m + 75%*6m\n"
        "3) Exiger score > 0 (long-only ON)\n"
        "4) Risk-off = DEFENSIVE (IEF/TLT) au lieu de CASH\n"
        "5) Tester Top 3 au lieu de Top 2 (si tu acceptes un peu plus de diversification)"
    )

with tab3:
    st.subheader("🧪 Optimisation (grid) — rapide et propre")
    st.caption("On fait un petit grid search (léger) pour éviter de surcharger Streamlit Cloud.")

    col1, col2, col3 = st.columns(3)
    with col1:
        run_grid = st.toggle("Activer le grid search", value=False)
    with col2:
        grid_topn = st.multiselect("Top N à tester", options=[1, 2, 3], default=[2])
    with col3:
        grid_risk_off = st.multiselect("Risk-off à tester", options=["CASH", "DEFENSIVE"], default=[cfg.risk_off_mode])

    st.markdown("**Lookbacks à tester (dual)**")
    g1, g2, g3 = st.columns(3)
    with g1:
        grid_lb1 = st.multiselect("LB1", options=[1, 2, 3, 4, 5, 6], default=[3])
    with g2:
        grid_lb2 = st.multiselect("LB2", options=[3, 4, 5, 6, 9, 12], default=[6])
    with g3:
        grid_w1 = st.multiselect("Poids LB1", options=[0.25, 0.5, 0.75], default=[cfg.mom_dual_weight[0]])

    if run_grid:
        with st.spinner("Grid search en cours…"):
            rows = []
            # bornes raisonnables
            for tn in grid_topn:
                for ro in grid_risk_off:
                    for lb1 in grid_lb1:
                        for lb2 in grid_lb2:
                            if lb2 <= lb1:
                                continue
                            for w1 in grid_w1:
                                w2 = 1.0 - float(w1)
                                cfg2 = BacktestConfig(
                                    start=cfg.start,
                                    end=cfg.end,
                                    rebalance=cfg.rebalance,
                                    fee_bps=cfg.fee_bps,
                                    top_n=int(tn),
                                    market_filter_on=cfg.market_filter_on,
                                    market_filter_asset=cfg.market_filter_asset,
                                    market_filter_ma_months=cfg.market_filter_ma_months,
                                    risk_off_mode=ro,
                                    defensive_asset=cfg.defensive_asset,
                                    momentum_mode="DUAL",
                                    mom_lookback_single_m=cfg.mom_lookback_single_m,
                                    mom_lookback_dual_m=(int(lb1), int(lb2)),
                                    mom_dual_weight=(float(w1), float(w2)),
                                    long_only=cfg.long_only,
                                )
                                w, s, r = compute_weights_multi_asset(prices_multi, cfg2)
                                b = backtest_from_weights(prices_multi, w, cfg2)
                                perf = summarize_performance(b["equity_net"], b["port_ret_net"], ppyear)
                                rows.append(
                                    {
                                        "TopN": int(tn),
                                        "RiskOff": ro,
                                        "LB1": int(lb1),
                                        "LB2": int(lb2),
                                        "W1": float(w1),
                                        "W2": float(w2),
                                        "CAGR": perf["CAGR"],
                                        "Sharpe": perf["Sharpe (rf=0)"],
                                        "MaxDD": perf["Max Drawdown"],
                                        "TurnoverAvg": float(b["turnover"].mean()),
                                    }
                                )

            grid = pd.DataFrame(rows)
            if grid.empty:
                st.warning("Aucun résultat (grid trop restrictif).")
            else:
                grid_sorted = grid.sort_values(by="CAGR", ascending=False).reset_index(drop=True)
                top_show = grid_sorted.head(20).copy()

                top_show["CAGR"] = top_show["CAGR"].map(lambda x: float(x))
                top_show["MaxDD"] = top_show["MaxDD"].map(lambda x: float(x))
                top_show["Sharpe"] = top_show["Sharpe"].map(lambda x: float(x))

                st.subheader("Top 20 (par CAGR)")
                st.dataframe(
                    top_show.style.format(
                        {
                            "CAGR": "{:.2%}",
                            "MaxDD": "{:.2%}",
                            "Sharpe": "{:.2f}",
                            "TurnoverAvg": "{:.2f}",
                            "W1": "{:.2f}",
                            "W2": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

                best = grid_sorted.iloc[0].to_dict()
                st.success(
                    f"Meilleur (CAGR): TopN={best['TopN']} | RiskOff={best['RiskOff']} | "
                    f"LB={best['LB1']}+{best['LB2']} | w={best['W1']:.2f}/{best['W2']:.2f} | "
                    f"CAGR={best['CAGR']*100:.2f}% | Sharpe={best['Sharpe']:.2f} | MaxDD={best['MaxDD']*100:.2f}%"
                )

                csv = grid_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger les résultats (CSV)", data=csv, file_name="grid_search_results.csv", mime="text/csv")
    else:
        st.info("Active le grid search si tu veux tester rapidement des variantes (TopN, lookbacks, poids, risk-off).")

with tab4:
    st.subheader("🧾 Message automatique intégré")
    st.caption("Message prêt à copier/coller (ex: Discord, email, Slack).")

    # Construire le message basé sur la dernière date (rebal)
    last_dt = prices_multi.index[-1]
    last_dt_str = last_dt.strftime("%Y-%m-%d")
    is_risk_on = bool(risk_on.loc[last_dt]) if last_dt in risk_on.index else True

    alloc = weights.loc[last_dt]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    # scores last
    sc = scores.loc[last_dt].dropna().sort_values(ascending=False)

    # Format
    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual momentum {cfg.mom_lookback_dual_m[0]}+{cfg.mom_lookback_dual_m[1]} (w={cfg.mom_dual_weight[0]:.2f}/{cfg.mom_dual_weight[1]:.2f})"
    else:
        mom_desc = f"Single momentum {cfg.mom_lookback_single_m}"

    if cfg.market_filter_on:
        filt_desc = f"Filtre marché ON — {cfg.market_filter_asset} > MA{cfg.market_filter_ma_months}"
    else:
        filt_desc = "Filtre marché OFF"

    if not is_risk_on:
        if cfg.risk_off_mode == "DEFENSIVE":
            pos_line = f"Mode risk-off: **DEFENSIVE** (100% {cfg.defensive_asset})"
        else:
            pos_line = "Mode risk-off: **CASH** (0% exposé)"
    else:
        if alloc.empty:
            pos_line = "Aucune position (cash/defensive selon paramètres)."
        else:
            parts = [f"{t}: {w:.0%}" for t, w in alloc.items()]
            pos_line = "Allocation cible: " + ", ".join(parts)

    msg = f"""
📌 **Trading Bot — Signal {last_dt_str}**
- Univers: {", ".join(universe)}
- Rebal: {"mensuel" if cfg.rebalance=="M" else "hebdo"} | Frais: {cfg.fee_bps:.1f} bps
- Momentum: {mom_desc}
- {filt_desc}
- Risk-on: **{"OUI" if is_risk_on else "NON"}**
- {pos_line}

📈 Perf (multi, net): CAGR {perf_net["CAGR"]*100:.2f}% | MaxDD {perf_net["Max Drawdown"]*100:.2f}% | Sharpe {perf_net["Sharpe (rf=0)"]:.2f}
"""
    msg = textwrap.dedent(msg).strip()

    st.text_area("Message", value=msg, height=220)

    # Détails additionnels: top scores
    st.markdown("**Top scores (dernier point)**")
    st.dataframe(sc.head(8).to_frame("Score").style.format("{:.2%}"), use_container_width=True)

    # Petit “health check”
    if cfg.momentum_mode == "DUAL" and cfg.mom_lookback_dual_m == (3, 6):
        st.warning(
            "Tu es en 3m+6m. Si ton CAGR est tombé ~8–9% alors qu’avant tu étais ~13–14%, "
            "les causes les plus fréquentes sont: turnover↑, whipsaws↑, et un risk-off CASH trop pénalisant sur certaines périodes.\n"
            "Essaye: w=0.25/0.75, ou risk-off=DEFENSIVE (IEF/TLT), ou rebal mensuel + long-only ON."
        )


# ---------------------------
# Footer — Exports
# ---------------------------
st.divider()
st.subheader("Exports")
colx, coly, colz = st.columns(3)

with colx:
    df_eq = pd.DataFrame({"equity_net": bt["equity_net"], "equity_gross": bt["equity_gross"]})
    st.download_button(
        "Télécharger equity (CSV)",
        data=df_eq.to_csv().encode("utf-8"),
        file_name="equity_multi.csv",
        mime="text/csv",
    )

with coly:
    df_w = bt["weights"].copy()
    st.download_button(
        "Télécharger weights (CSV)",
        data=df_w.to_csv().encode("utf-8"),
        file_name="weights_multi.csv",
        mime="text/csv",
    )

with colz:
    df_diag = pd.DataFrame(
        {
            "port_ret_net": bt["port_ret_net"],
            "port_ret_gross": bt["port_ret_gross"],
            "fees": bt["fees"],
            "turnover": bt["turnover"],
            "risk_on": risk_on.reindex(bt["port_ret_net"].index).fillna(False).astype(int),
        }
    )
    st.download_button(
        "Télécharger diagnostics (CSV)",
        data=df_diag.to_csv().encode("utf-8"),
        file_name="diagnostics_multi.csv",
        mime="text/csv",
    )

st.caption("⚠️ Ceci est un backtest simplifié (pas un conseil financier). Les rendements réels peuvent différer (slippage, exécution, taxes, etc.).")
