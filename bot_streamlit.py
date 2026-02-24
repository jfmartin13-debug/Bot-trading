# bot_streamlit.py
# Streamlit app: Core/Satellite + Momentum (DUAL/SINGLE) + Crash filter (MA days) + Risk-off CASH
# Data source: Yahoo Finance (yfinance) by default with cache
# CSV option preserved: wide CSV + multi CSV upload
# Auth: APP_PASSWORD (fallback) + USERS (multi-user) with compare_digest on bytes
#
# Run:
#   streamlit run bot_streamlit.py
#
# Env/secrets auth options (priority):
#  1) USERS (JSON) -> multi-user:
#       export USERS='{"alice":"secret1","bob":"secret2"}'
#     or in .streamlit/secrets.toml:
#       USERS = '{"alice":"secret1","bob":"secret2"}'
#  2) APP_PASSWORD -> single password:
#       export APP_PASSWORD="my_password"
#     or in secrets.toml:
#       APP_PASSWORD = "my_password"
#
# Notes:
# - "CASH" is treated as a synthetic asset with flat price=1 (0% return).
# - Fees (bps) apply on rebalance days using turnover * fee_bps/10000.
# - Rebalancing is monthly by default (end-of-month).
# - "Sweep 10/20/30/50" is implemented as a momentum lookback-days sweep option.

from __future__ import annotations

import json
import os
import hmac
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt


# ----------------------------
# Auth
# ----------------------------

def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit secrets first, then env
    val = None
    try:
        if name in st.secrets:
            val = st.secrets.get(name)
    except Exception:
        val = None
    if val is None:
        val = os.environ.get(name, default)
    return val


def _parse_users(users_raw: Optional[str]) -> Optional[Dict[str, str]]:
    if not users_raw:
        return None
    try:
        data = json.loads(users_raw)
        if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            return data
    except Exception:
        return None
    return None


def _secure_compare(a: str, b: str) -> bool:
    # compare_digest in bytes (requested)
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def require_login() -> None:
    users = _parse_users(_get_secret("USERS"))
    app_password = _get_secret("APP_PASSWORD", "")

    # If neither is set, allow access but warn (so app still runs).
    if not users and not app_password:
        st.warning("⚠️ Auth non configurée (USERS / APP_PASSWORD). Accès libre.")
        st.session_state["auth_ok"] = True
        return

    if st.session_state.get("auth_ok"):
        return

    st.sidebar.markdown("## 🔒 Connexion")
    if users:
        username = st.sidebar.text_input("Utilisateur", key="auth_user")
        password = st.sidebar.text_input("Mot de passe", type="password", key="auth_pass")
        if st.sidebar.button("Se connecter", use_container_width=True):
            if username in users and _secure_compare(password, users[username]):
                st.session_state["auth_ok"] = True
                st.session_state["auth_user_ok"] = username
                st.sidebar.success("Connecté ✅")
            else:
                st.sidebar.error("Identifiants invalides.")
        st.stop()
    else:
        password = st.sidebar.text_input("Mot de passe", type="password", key="auth_pass_single")
        if st.sidebar.button("Se connecter", use_container_width=True):
            if _secure_compare(password, app_password):
                st.session_state["auth_ok"] = True
                st.sidebar.success("Connecté ✅")
            else:
                st.sidebar.error("Mot de passe invalide.")
        st.stop()


# ----------------------------
# Data loading
# ----------------------------

CASH_SYMBOL = "CASH"


def _normalize_ticker_list(raw: str) -> List[str]:
    # Accept comma / space separated tickers
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False, ttl=60 * 60)  # cache 1h
def load_prices_yahoo(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices (business days) for tickers.
    Includes synthetic CASH (flat 1.0).
    """
    tickers_list = list(tickers)
    want_cash = CASH_SYMBOL in tickers_list
    yf_tickers = [t for t in tickers_list if t != CASH_SYMBOL]

    if yf_tickers:
        df = yf.download(
            yf_tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

        # yfinance returns either multiindex columns or normal
        if isinstance(df.columns, pd.MultiIndex):
            # prefer Adj Close if available, else Close
            if ("Adj Close",) in df.columns:
                prices = df["Adj Close"].copy()
            else:
                prices = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else df["Close"]
        else:
            # single ticker case: df is OHLCV columns
            prices = df["Adj Close"].to_frame(name=yf_tickers[0]) if "Adj Close" in df.columns else df["Close"].to_frame(name=yf_tickers[0])

        prices = prices.sort_index()
    else:
        prices = pd.DataFrame()

    if want_cash:
        if prices.empty:
            # build a date index for the requested range
            idx = pd.bdate_range(start=pd.to_datetime(start), end=pd.to_datetime(end))
            prices = pd.DataFrame(index=idx)
        prices[CASH_SYMBOL] = 1.0

    # Forward-fill missing values (common around IPOs / holidays)
    prices = prices.ffill().dropna(how="all")
    return prices


def load_prices_from_wide_csv(file) -> pd.DataFrame:
    """
    Wide CSV: first column = date, remaining columns = tickers (prices).
    """
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        raise ValueError("CSV wide invalide: besoin d'une colonne date + au moins un ticker.")
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").ffill()
    return df


def load_prices_from_multi_csv(files) -> pd.DataFrame:
    """
    Multiple CSVs: each file contains at least [date, close] columns (or date + Adj Close/Close).
    Filename (without extension) used as ticker by default; also supports a 'Ticker' column if present.
    """
    frames = []
    for f in files:
        name = os.path.splitext(f.name)[0].strip().upper()
        df = pd.read_csv(f)
        cols = {c.lower().strip(): c for c in df.columns}
        if "date" not in cols:
            raise ValueError(f"{f.name}: colonne 'Date' manquante.")
        date_col = cols["date"]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)

        # Choose price column
        price_col = None
        for cand in ["adj close", "adj_close", "adjusted close", "close", "price"]:
            if cand in cols:
                price_col = cols[cand]
                break
        if price_col is None:
            # try any numeric column except ticker
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"{f.name}: aucune colonne prix détectée (Close/Adj Close).")
            price_col = numeric_cols[0]

        ticker = name
        if "ticker" in cols:
            # use first non-null ticker if present
            maybe = df[cols["ticker"]].dropna()
            if not maybe.empty:
                ticker = str(maybe.iloc[0]).strip().upper()

        s = pd.to_numeric(df[price_col], errors="coerce").rename(ticker)
        frames.append(s)

    prices = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    return prices


def align_and_clean_prices(prices: pd.DataFrame, tickers_needed: List[str]) -> pd.DataFrame:
    prices = prices.copy()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    # Add synthetic CASH if needed
    if CASH_SYMBOL in tickers_needed and CASH_SYMBOL not in prices.columns:
        prices[CASH_SYMBOL] = 1.0

    missing = [t for t in tickers_needed if t not in prices.columns]
    if missing:
        raise ValueError(f"Tickers manquants dans les données: {missing}")

    prices = prices[tickers_needed].sort_index().ffill()
    prices = prices.dropna(how="all")
    return prices


# ----------------------------
# Strategy
# ----------------------------

@dataclass
class StrategyConfig:
    core_tickers: List[str]
    satellite_tickers: List[str]
    benchmark_ticker: str           # used for crash filter + dual momentum reference
    risk_off_ticker: str = CASH_SYMBOL
    core_weight: float = 0.60       # rest goes to satellite
    top_k: int = 3                  # number of satellite assets selected
    momentum_mode: str = "DUAL"     # DUAL or SINGLE
    lookback_days: int = 126        # momentum lookback
    crash_ma_days: int = 200        # crash filter moving average
    rebalance: str = "M"            # monthly end
    fee_bps: float = 0.0            # transaction fee bps


def _to_month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # pick last available business day each month from existing index
    s = pd.Series(index=idx, data=np.arange(len(idx)))
    month_end = s.resample("M").last().dropna().index
    # Ensure month_end are in idx (they are)
    return pd.DatetimeIndex(month_end)


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rets


def _momentum_score(prices: pd.DataFrame, asof: pd.Timestamp, lookback_days: int, tickers: List[str]) -> pd.Series:
    # total return over lookback_days ending at asof
    # use nearest available date <= asof
    if asof not in prices.index:
        asof = prices.index[prices.index.get_indexer([asof], method="ffill")[0]]
    i = prices.index.get_loc(asof)
    j = max(0, i - lookback_days)
    p0 = prices.iloc[j][tickers]
    p1 = prices.iloc[i][tickers]
    score = (p1 / p0) - 1.0
    return score.replace([np.inf, -np.inf], np.nan).fillna(-np.inf)


def _crash_filter_on(prices: pd.Series, asof: pd.Timestamp, ma_days: int) -> bool:
    # True if price < MA(ma_days) at asof
    s = prices.copy()
    s = s.loc[:asof]
    if len(s) < max(5, ma_days // 5):
        return False
    ma = s.rolling(ma_days).mean().iloc[-1]
    px = s.iloc[-1]
    if pd.isna(ma) or pd.isna(px):
        return False
    return float(px) < float(ma)


def backtest_core_satellite(prices: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - equity_curve: columns [portfolio, benchmark]
      - weights_hist: weights over time (on rebalance dates)
    """
    all_tickers = sorted(set(cfg.core_tickers + cfg.satellite_tickers + [cfg.benchmark_ticker, cfg.risk_off_ticker]))
    prices = align_and_clean_prices(prices, all_tickers)

    rets = _returns_from_prices(prices)
    idx = prices.index

    # rebalance dates
    if cfg.rebalance.upper().startswith("M"):
        rdates = _to_month_end_dates(idx)
    else:
        # fallback: treat as month end
        rdates = _to_month_end_dates(idx)

    # Build weight history on rebalance dates
    weights = pd.DataFrame(index=rdates, columns=all_tickers, data=0.0)

    core_w = float(np.clip(cfg.core_weight, 0.0, 1.0))
    sat_w = 1.0 - core_w
    core_each = core_w / max(1, len(cfg.core_tickers))

    bench_series = prices[cfg.benchmark_ticker]

    for d in rdates:
        # base core allocation
        w = pd.Series(0.0, index=all_tickers, dtype=float)
        for t in cfg.core_tickers:
            w[t] += core_each

        # crash filter: if benchmark below MA => all risk-off (or just satellites risk-off; here: FULL risk-off)
        crash_on = _crash_filter_on(bench_series, d, cfg.crash_ma_days) if cfg.crash_ma_days > 0 else False
        if crash_on:
            w[:] = 0.0
            w[cfg.risk_off_ticker] = 1.0
            weights.loc[d] = w
            continue

        # momentum selection among satellites
        sats = cfg.satellite_tickers[:]
        if len(sats) == 0 or sat_w <= 0:
            # core only
            remaining = 1.0 - w.sum()
            if remaining > 1e-12:
                w[cfg.risk_off_ticker] += remaining  # keep fully invested
            weights.loc[d] = w
            continue

        scores = _momentum_score(prices, d, cfg.lookback_days, sats).sort_values(ascending=False)
        chosen = list(scores.index[: max(1, min(cfg.top_k, len(scores)))])
        chosen_scores = scores.loc[chosen]

        if cfg.momentum_mode.upper() == "DUAL":
            # Dual momentum: compare best satellite momentum vs risk-off momentum (same lookback)
            risk_off = cfg.risk_off_ticker
            if risk_off not in prices.columns:
                # synthetic cash
                prices[risk_off] = 1.0
            ref_score = _momentum_score(prices, d, cfg.lookback_days, [risk_off]).iloc[0]
            best_score = chosen_scores.iloc[0] if len(chosen_scores) else -np.inf
            if (best_score <= ref_score) or (best_score <= 0):
                # go risk-off for satellite sleeve
                w[risk_off] += sat_w
            else:
                # allocate equally across chosen
                each = sat_w / len(chosen)
                for t in chosen:
                    w[t] += each
        else:
            # SINGLE momentum: pick top K, but if all negative, go risk-off
            if chosen_scores.max() <= 0:
                w[cfg.risk_off_ticker] += sat_w
            else:
                each = sat_w / len(chosen)
                for t in chosen:
                    w[t] += each

        # make sure sums to 1
        total = w.sum()
        if total <= 0:
            w[cfg.risk_off_ticker] = 1.0
        else:
            w = w / total

        weights.loc[d] = w

    # Simulate daily portfolio value with drifted weights and monthly rebalance (with fees)
    port = pd.Series(index=idx, dtype=float)
    bench = (1.0 + rets[cfg.benchmark_ticker]).cumprod()
    port.iloc[0] = 1.0

    current_w = weights.iloc[0].reindex(all_tickers).fillna(0.0)
    last_reb = weights.index[0]

    for i in range(1, len(idx)):
        d = idx[i]
        prev = idx[i - 1]
        r = rets.loc[d, all_tickers].fillna(0.0)

        # if rebalance date, apply turnover fee
        if d in weights.index:
            target_w = weights.loc[d].reindex(all_tickers).fillna(0.0)

            # compute turnover based on current_w vs target_w
            turnover = float(np.abs(target_w - current_w).sum()) / 2.0  # 0..1
            fee = turnover * (cfg.fee_bps / 10000.0)

            # apply fee by reducing portfolio value immediately
            port.loc[prev] = port.loc[prev] * (1.0 - fee)

            current_w = target_w
            last_reb = d

        # daily update: portfolio return = sum(w * asset returns)
        pr = float((current_w * r).sum())
        port.loc[d] = port.loc[prev] * (1.0 + pr)

        # drift weights with returns (optional, but more realistic)
        # w_new ~ w_old*(1+r) normalized
        growth = (1.0 + r).replace([np.inf, -np.inf], 1.0).clip(lower=0.0)
        w_g = current_w * growth
        s = float(w_g.sum())
        if s > 0:
            current_w = w_g / s

    equity = pd.DataFrame({"portfolio": port, "benchmark": bench.reindex(idx).fillna(method="ffill")}, index=idx)
    return equity, weights


def perf_stats(equity: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (freq / max(1, len(r))) - 1.0)
    vol = float(r.std() * np.sqrt(freq))
    sharpe = float((r.mean() * freq) / (r.std() * np.sqrt(freq))) if r.std() > 0 else np.nan
    dd = (equity / equity.cummax()) - 1.0
    maxdd = float(dd.min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}


def plot_equity(equity: pd.DataFrame, logy: bool = True) -> plt.Figure:
    fig, ax = plt.subplots()
    equity = equity.dropna()
    ax.plot(equity.index, equity["portfolio"], label="Portfolio")
    ax.plot(equity.index, equity["benchmark"], label="Benchmark")
    ax.set_title("Évolution du capital")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital")
    ax.legend()
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots()
    dd = (equity / equity.cummax()) - 1.0
    ax.plot(dd.index, dd.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Bot Streamlit - Core/Satellite Momentum", layout="wide")

require_login()

st.title("📈 Core/Satellite + Momentum (DUAL/SINGLE) + Crash Filter (MA)")

with st.sidebar:
    st.markdown("## Source de données")
    source = st.radio("Source", ["Yahoo Finance (yfinance)", "CSV (wide)", "CSV (multi)"], index=0)

    st.markdown("## Période")
    col_a, col_b = st.columns(2)
    start = col_a.text_input("Start (YYYY-MM-DD)", value="2015-01-01")
    end = col_b.text_input("End (YYYY-MM-DD)", value="2026-01-01")

    st.markdown("## Allocation")
    core_tickers_raw = st.text_area("Core tickers (séparés par virgules)", value="SPY")
    sat_tickers_raw = st.text_area("Satellite tickers (séparés par virgules)", value="QQQ, IWM, EFA, EEM, GLD")
    benchmark_ticker = st.text_input("Benchmark (crash filter + chart)", value="SPY").strip().upper()

    core_weight = st.slider("Poids Core", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    top_k = st.number_input("Top K satellites", min_value=1, max_value=50, value=3, step=1)

    st.markdown("## Momentum")
    momentum_mode = st.selectbox("Mode", ["DUAL", "SINGLE"], index=0)
    lookback_mode = st.radio("Lookback", ["Custom", "Sweep 10/20/30/50"], index=0)
    if lookback_mode == "Custom":
        lookback_days = st.number_input("Lookback (jours)", min_value=5, max_value=2000, value=126, step=5)
        sweep_days = None
    else:
        lookback_days = 0
        sweep_days = [10, 20, 30, 50]

    st.markdown("## Crash filter / Risk-off")
    crash_ma_days = st.number_input("Crash filter MA (jours) (0=off)", min_value=0, max_value=1000, value=200, step=10)
    risk_off_ticker = st.text_input("Risk-off ticker (CASH recommandé)", value="CASH").strip().upper()

    st.markdown("## Frais")
    fee_bps = st.number_input("Frais (bps) par rebalance", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

    run_btn = st.button("🚀 Lancer", use_container_width=True)


def get_prices_for_run(all_needed: List[str]) -> pd.DataFrame:
    if source == "Yahoo Finance (yfinance)":
        return load_prices_yahoo(tuple(all_needed), start=start, end=end)
    elif source == "CSV (wide)":
        f = st.sidebar.file_uploader("Upload CSV wide", type=["csv"], accept_multiple_files=False)
        if f is None:
            st.info("Upload un CSV wide pour continuer.")
            st.stop()
        return load_prices_from_wide_csv(f)
    else:
        fs = st.sidebar.file_uploader("Upload plusieurs CSV", type=["csv"], accept_multiple_files=True)
        if not fs:
            st.info("Upload plusieurs CSV pour continuer.")
            st.stop()
        return load_prices_from_multi_csv(fs)


def run_once(lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame, StrategyConfig]:
    core_tickers = _normalize_ticker_list(core_tickers_raw)
    sat_tickers = _normalize_ticker_list(sat_tickers_raw)
    if not core_tickers and not sat_tickers:
        st.error("Veuillez renseigner au moins un ticker (core ou satellite).")
        st.stop()

    cfg = StrategyConfig(
        core_tickers=core_tickers,
        satellite_tickers=sat_tickers,
        benchmark_ticker=benchmark_ticker,
        risk_off_ticker=risk_off_ticker if risk_off_ticker else CASH_SYMBOL,
        core_weight=float(core_weight),
        top_k=int(top_k),
        momentum_mode=str(momentum_mode).upper(),
        lookback_days=int(lookback),
        crash_ma_days=int(crash_ma_days),
        fee_bps=float(fee_bps),
        rebalance="M",
    )

    all_needed = sorted(set(cfg.core_tickers + cfg.satellite_tickers + [cfg.benchmark_ticker, cfg.risk_off_ticker]))
    prices = get_prices_for_run(all_needed)

    # If Yahoo source, we already requested needed tickers; for CSV we need to validate/align
    prices = align_and_clean_prices(prices, all_needed)

    equity, weights = backtest_core_satellite(prices, cfg)
    return equity, weights, cfg


if run_btn:
    try:
        if sweep_days:
            st.subheader("Résultats Sweep (lookback jours: 10/20/30/50)")
            rows = []
            eq_map = {}
            w_map = {}
            for lb in sweep_days:
                equity, weights, cfg = run_once(lb)
                stats = perf_stats(equity["portfolio"])
                rows.append({"lookback_days": lb, **stats, "Final": float(equity["portfolio"].iloc[-1])})
                eq_map[lb] = equity
                w_map[lb] = weights

            res = pd.DataFrame(rows).set_index("lookback_days").sort_index()
            st.dataframe(res.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}", "Final": "{:.2f}"}), use_container_width=True)

            pick = st.selectbox("Afficher le détail pour lookback", options=sweep_days, index=0)
            equity = eq_map[pick]
            weights = w_map[pick]

            c1, c2 = st.columns([2, 1])
            with c1:
                st.pyplot(plot_equity(equity, logy=True))
                st.pyplot(plot_drawdown(equity["portfolio"]))
            with c2:
                st.markdown("### Stats")
                p = perf_stats(equity["portfolio"])
                b = perf_stats(equity["benchmark"])
                st.write("**Portfolio**", {k: (f"{v:.2%}" if k in ["CAGR", "Vol", "MaxDD"] else f"{v:.2f}") for k, v in p.items()})
                st.write("**Benchmark**", {k: (f"{v:.2%}" if k in ["CAGR", "Vol", "MaxDD"] else f"{v:.2f}") for k, v in b.items()})

                st.markdown("### Derniers poids (rebalance)")
                st.dataframe(weights.tail(12).style.format("{:.2%}"), use_container_width=True)

            # Downloads
            st.markdown("### Téléchargements")
            st.download_button("⬇️ equity_curve.csv", equity.to_csv().encode("utf-8"), file_name="equity_curve.csv")
            st.download_button("⬇️ weights_rebalance.csv", weights.to_csv().encode("utf-8"), file_name="weights_rebalance.csv")

        else:
            equity, weights, cfg = run_once(int(lookback_days))

            st.subheader("Résultats")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.pyplot(plot_equity(equity, logy=True))
                st.pyplot(plot_drawdown(equity["portfolio"]))
            with c2:
                st.markdown("### Config")
                st.json({
                    "core": cfg.core_tickers,
                    "satellite": cfg.satellite_tickers,
                    "benchmark": cfg.benchmark_ticker,
                    "risk_off": cfg.risk_off_ticker,
                    "core_weight": cfg.core_weight,
                    "top_k": cfg.top_k,
                    "momentum_mode": cfg.momentum_mode,
                    "lookback_days": cfg.lookback_days,
                    "crash_ma_days": cfg.crash_ma_days,
                    "fee_bps": cfg.fee_bps,
                    "rebalance": cfg.rebalance,
                })

                st.markdown("### Stats")
                p = perf_stats(equity["portfolio"])
                b = perf_stats(equity["benchmark"])
                st.write("**Portfolio**", {k: (f"{v:.2%}" if k in ["CAGR", "Vol", "MaxDD"] else f"{v:.2f}") for k, v in p.items()})
                st.write("**Benchmark**", {k: (f"{v:.2%}" if k in ["CAGR", "Vol", "MaxDD"] else f"{v:.2f}") for k, v in b.items()})

                st.markdown("### Derniers poids (rebalance)")
                st.dataframe(weights.tail(12).style.format("{:.2%}"), use_container_width=True)

            st.markdown("### Téléchargements")
            st.download_button("⬇️ equity_curve.csv", equity.to_csv().encode("utf-8"), file_name="equity_curve.csv")
            st.download_button("⬇️ weights_rebalance.csv", weights.to_csv().encode("utf-8"), file_name="weights_rebalance.csv")

    except Exception as e:
        st.error(f"Erreur: {e}")

else:
    st.info("Configure la stratégie à gauche puis clique **Lancer**.")


# ----------------------------
# requirements.txt (contenu exact)
# ----------------------------
# streamlit==1.36.0
# yfinance==0.2.43
# pandas==2.2.2
# numpy==2.0.1
# matplotlib==3.9.0
