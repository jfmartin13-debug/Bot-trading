# bot_streamlit.py
# ------------------------------------------------------------
# Trading Bot Streamlit — SIMPLE & STABLE (CSV Upload)
#
# Objectif: zéro dépendance fragile, zéro data provider externe.
# Tu uploades:
#   (A) Un seul CSV "wide" : Date + colonnes tickers (SPY, QQQ, ...)
#   (B) OU plusieurs CSV : 1 fichier par ticker (SPY.csv, QQQ.csv, ...) -> fusion auto
#
# Multi-asset :
#   - Univers (ex: SPY, QQQ, EFA, EEM, VNQ, TLT, IEF, GLD)
#   - Filtre marché ON (prix > MA fenêtre)
#   - Top N (ex: 2)
#   - Frais bps (ex: 10 bps) sur turnover
#   - Momentum SINGLE (ex 12p) ou DUAL (ex 3p+6p pondéré)
#   - Risk-off : CASH ou DEFENSIVE (ex IEF)
#   - Message automatique
#
# requirements.txt minimal:
#   streamlit
#   pandas
#   numpy
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Trading Bot (CSV)", page_icon="📈", layout="wide")
st.title("📈 Trading Bot — Single + Multi-asset (Dual Momentum) — CSV")
st.caption("Ultra stable : pas de yfinance / pas de Stooq côté Python. Upload tes CSV et le bot tourne.")


DEFAULT_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD"]
DEFAULT_SINGLE = "SPY"

PRICE_COL_CANDIDATES = [
    "Adj Close", "AdjClose", "adjclose", "adj_close",
    "Close", "close", "PRICE", "Price", "price"
]


@dataclass(frozen=True)
class Config:
    rebalance: str                  # "M" ou "W"
    fee_bps: float                  # ex 10 = 0.10%
    top_n: int
    market_filter_on: bool
    market_filter_asset: str
    market_filter_window: int       # 12 périodes (mensuel) ou 52 (hebdo), etc.
    risk_off_mode: str              # "CASH" ou "DEFENSIVE"
    defensive_asset: str
    momentum_mode: str              # "SINGLE" ou "DUAL"
    mom_single_lb: int
    mom_dual_lb: Tuple[int, int]
    mom_dual_w: Tuple[float, float]
    long_only: bool


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


def load_wide_csv(uploaded_file) -> pd.DataFrame:
    """
    CSV wide attendu:
      Date,SPY,QQQ,EFA,...
      2006-01-03,123.4,45.6,...
    """
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 2:
        raise ValueError("CSV invalide: il faut une colonne Date + au moins 1 colonne ticker.")

    date_col = _detect_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    return df


def infer_ticker_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    name = base.rsplit(".", 1)[0]  # enlève .csv
    # si fichiers "spy.us.csv" -> ticker "SPY"
    name = name.replace(".us", "").replace(".US", "")
    return name.upper()


def load_many_csv(files) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Charge plusieurs CSV (1 par ticker) et fusionne sur la date.
    Le ticker est déduit du nom de fichier: SPY.csv -> SPY
    Chaque CSV doit contenir une colonne date + une colonne prix (Close/Adj Close).
    """
    frames = []
    tickers_loaded = []
    problems = []

    for f in files:
        fname = getattr(f, "name", "fichier")
        try:
            df = pd.read_csv(f)
            if df.empty or df.shape[1] < 2:
                problems.append(f"{fname} (vide ou invalide)")
                continue

            date_col = _detect_date_col(df)
            price_col = _detect_price_col(df)
            if price_col is None:
                problems.append(f"{fname} (colonne prix introuvable)")
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(by=date_col)

            s = pd.to_numeric(df[price_col], errors="coerce")
            out = pd.DataFrame({"Date": df[date_col], "Price": s}).dropna()
            out = out.drop_duplicates(subset=["Date"]).set_index("Date").sort_index()

            ticker = infer_ticker_from_filename(fname)
            out = out.rename(columns={"Price": ticker})

            frames.append(out)
            tickers_loaded.append(ticker)

        except Exception:
            problems.append(f"{fname} (erreur lecture)")

    if not frames:
        raise ValueError("Aucun CSV valide n'a pu être chargé.")

    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    merged = merged.dropna(how="all")
    return merged, tickers_loaded, problems


# -----------------------------
# Strategy helpers
# -----------------------------
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
    return {"TotalReturn": total_return, "CAGR": float(cagr), "Vol": vol, "Sharpe": float(sharpe), "MaxDD": float(mdd), "Calmar": float(calmar)}


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


def compute_weights(prices_rebal: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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


def run_backtest(prices_rebal: pd.DataFrame, weights: pd.DataFrame, fee_bps: float) -> Dict[str, pd.Series]:
    r = returns(prices_rebal).fillna(0.0)
    weights = weights.fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)

    turnover = (weights - w_prev).abs().sum(axis=1)
    fees = (fee_bps / 10000.0) * turnover

    ret_gross = (w_prev * r).sum(axis=1).fillna(0.0)
    ret_net = (ret_gross - fees).fillna(0.0)

    eq_gross = (1.0 + ret_gross).cumprod()
    eq_net = (1.0 + ret_net).cumprod()

    return {"turnover": turnover, "fees": fees, "ret_gross": ret_gross, "ret_net": ret_net, "eq_gross": eq_gross, "eq_net": eq_net}


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1) Données")
    mode = st.radio("Mode d'upload", ["Un seul CSV (wide)", "Plusieurs CSV (un par ticker)"], index=1)

    uploaded_one = None
    uploaded_many = None

    if mode == "Un seul CSV (wide)":
        uploaded_one = st.file_uploader("Upload CSV (Date + colonnes tickers)", type=["csv"], accept_multiple_files=False)
    else:
        uploaded_many = st.file_uploader("Upload plusieurs CSV (1 par ticker)", type=["csv"], accept_multiple_files=True)

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


# -----------------------------
# Load data
# -----------------------------
if mode == "Un seul CSV (wide)":
    if uploaded_one is None:
        st.info("➡️ Upload un CSV (wide) pour démarrer.")
        st.stop()
    try:
        close = load_wide_csv(uploaded_one)
        problems = []
    except Exception as e:
        st.error("Erreur lecture CSV: " + str(e))
        st.stop()
else:
    if not uploaded_many:
        st.info("➡️ Upload plusieurs CSV (un par ticker) pour démarrer.")
        st.stop()
    try:
        close, tickers_loaded, problems = load_many_csv(uploaded_many)
    except Exception as e:
        st.error("Erreur lecture CSVs: " + str(e))
        st.stop()

if close.empty or close.shape[1] < 2:
    st.error("Pas assez de données chargées (il faut au moins 2 tickers).")
    st.stop()

if problems:
    st.warning("Fichiers ignorés / problèmes:\n- " + "\n- ".join(problems))

available = list(close.columns)

# -----------------------------
# Choix tickers / dates
# -----------------------------
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

    market_filter_asset = st.selectbox("Actif filtre (ex: SPY)", options=available, index=available.index("SPY") if "SPY" in available else 0)
    defensive_asset = st.selectbox("Actif défensif", options=available, index=available.index("IEF") if "IEF" in available else 0)
    single_ticker = st.selectbox("Single-asset (comparatif)", options=available, index=available.index(DEFAULT_SINGLE) if DEFAULT_SINGLE in available else 0)

    ma_default = 12 if rebalance == "M" else 52
    market_filter_window = st.slider("Fenêtre MA (périodes)", 6, 80, ma_default)

    st.divider()
    st.header("4) Dates")
    start_d = st.date_input("Début", value=pd.to_datetime(close.index.min()).date())
    end_d = st.date_input("Fin", value=pd.to_datetime(close.index.max()).date())


# Filter dates
close = close.loc[(close.index >= pd.to_datetime(str(start_d))) & (close.index <= pd.to_datetime(str(end_d)))]
if close.empty:
    st.error("Aucune donnée dans la plage choisie.")
    st.stop()

# Rebalance
prices_rebal = resample_prices(close, rebalance).dropna(how="all")
prices_multi = prices_rebal[universe].dropna(how="all")

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

# Strategy compute
weights, scores, risk_on = compute_weights(prices_multi, cfg)
bt = run_backtest(prices_multi, weights, cfg.fee_bps)

ppy = periods_per_year(rebalance)
perf = summarize(bt["eq_net"], bt["ret_net"], ppy)

# Single asset
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
    risk_on_aligned = risk_on.reindex(bt["eq_net"].index).fillna(False)
    transitions = int((risk_on_aligned.astype(int).diff().abs() > 0).sum())
    pct_on = float(risk_on_aligned.mean())
    avg_turn = float(bt["turnover"].mean())
    hhi = float((weights.shift(1).fillna(0.0).pow(2).sum(axis=1)).mean())

    st.subheader("Diagnostic")
    st.write(
        "- Dual (3+6) réagit plus vite → turnover souvent plus élevé → whipsaws.\n"
        "- Tester: poids 0.25/0.75, risk-off DEFENSIVE, ou rebal mensuel."
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("% Risk-on", f"{pct_on*100:.1f}%")
    d2.metric("Transitions on/off", str(transitions))
    d3.metric("Turnover moyen", f"{avg_turn:.2f}")
    d4.metric("Concentration (HHI)", f"{hhi:.2f}")

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
    last_t = prices_multi.index[-1]
    last_date = last_t.strftime("%Y-%m-%d")
    is_on = bool(risk_on.loc[last_t]) if last_t in risk_on.index else True

    alloc = weights.loc[last_t]
    alloc = alloc[alloc > 0].sort_values(ascending=False)

    if cfg.momentum_mode == "DUAL":
        mom_desc = f"Dual {cfg.mom_dual_lb[0]}+{cfg.mom_dual_lb[1]} (w={cfg.mom_dual_w[0]:.2f}/{cfg.mom_dual_w[1]:.2f})"
    else:
        mom_desc = f"Single {cfg.mom_single_lb}"

    filt_desc = f"Filtre ON ({cfg.market_filter_asset} > MA{cfg.market_filter_window})" if cfg.market_filter_on else "Filtre OFF"

    if not is_on:
        pos_line = "Risk-off: CASH (0% exposé)" if cfg.risk_off_mode == "CASH" else f"Risk-off: DEFENSIVE (100% {cfg.defensive_asset})"
    else:
        if alloc.empty:
            pos_line = "Allocation: CASH/DEF (aucun score positif)"
        else:
            parts = [f"{t}: {int(round(w*100))}%" for t, w in alloc.items()]
            pos_line = "Allocation: " + ", ".join(parts)

    msg = "\n".join(
        [
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
    )

    st.subheader("Message automatique (copier/coller)")
    st.text_area("Message", value=msg, height=240)

st.divider()
st.caption("⚠️ Backtest simplifié (pas un conseil financier). Slippage/taxes/exécution non inclus.")
