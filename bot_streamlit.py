# bot_streamlit.py
# Streamlit app: Core/Satellite + Momentum (DUAL/SINGLE) + Crash filter (MA days) + Risk-off CASH
# + PDF export, multi-strategy comparison, multi-horizon momentum, weekly/quarterly rebal, hashed passwords,
# + core_weight optimization grid-search.
#
# Data source: Yahoo Finance (yfinance) by default with cache (ROBUST: bulk download + per-ticker fallback)
# CSV option preserved: wide CSV + multi CSV upload
# Auth: APP_PASSWORD (fallback plaintext) + option USERS multi-user (plaintext or PBKDF2-hash), compare_digest on bytes
#
# Run:
#   streamlit run bot_streamlit.py
#
# Auth configuration (priority):
#  1) USERS (JSON) -> multi-user:
#       export USERS='{"alice":"secret1","bob":"pbkdf2$260000$SALT_B64$DK_B64"}'
#     or secrets.toml:
#       USERS = '{"alice":"secret1","bob":"pbkdf2$260000$SALT_B64$DK_B64"}'
#  2) APP_PASSWORD -> single password (plaintext fallback):
#       export APP_PASSWORD="my_password"
#  3) APP_PASSWORD_HASH -> single password hashed (recommended):
#       export APP_PASSWORD_HASH="pbkdf2$260000$SALT_B64$DK_B64"
#
# Notes:
# - "CASH" is treated as a synthetic asset with flat price=1 (0% return).
# - Fees (bps) apply on rebalance days using turnover * fee_bps/10000.
# - Rebalancing supported: Weekly (W-FRI), Monthly (M), Quarterly (Q).
# - Sweep 10/20/30/50 is supported as a lookback sweep.

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas


# ----------------------------
# Auth
# ----------------------------

def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
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


def pbkdf2_hash(password: str, *, iterations: int = 260_000, salt_bytes: Optional[bytes] = None) -> str:
    """
    Returns string format: pbkdf2$<iters>$<salt_b64>$<dk_b64>
    """
    if salt_bytes is None:
        salt_bytes = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations, dklen=32)
    salt_b64 = base64.b64encode(salt_bytes).decode("ascii")
    dk_b64 = base64.b64encode(dk).decode("ascii")
    return f"pbkdf2${iterations}${salt_b64}${dk_b64}"


def pbkdf2_verify(password: str, stored: str) -> bool:
    """
    Verifies stored format pbkdf2$iters$salt_b64$dk_b64
    Uses constant-time compare in bytes.
    """
    try:
        parts = stored.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2":
            return False
        iters = int(parts[1])
        salt = base64.b64decode(parts[2].encode("ascii"))
        dk_expected = base64.b64decode(parts[3].encode("ascii"))
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=len(dk_expected))
        return hmac.compare_digest(dk, dk_expected)
    except Exception:
        return False


def _check_password(input_pw: str, stored_pw_or_hash: str) -> bool:
    # If looks like PBKDF2 format, verify hash; else fallback to plaintext compare
    if stored_pw_or_hash.startswith("pbkdf2$"):
        return pbkdf2_verify(input_pw, stored_pw_or_hash)
    return _secure_compare(input_pw, stored_pw_or_hash)


def require_login() -> None:
    users = _parse_users(_get_secret("USERS"))
    app_password = _get_secret("APP_PASSWORD", "")
    app_password_hash = _get_secret("APP_PASSWORD_HASH", "")

    # If neither is set, allow access but warn (so app still runs).
    if not users and not app_password and not app_password_hash:
        st.warning("⚠️ Auth non configurée (USERS / APP_PASSWORD / APP_PASSWORD_HASH). Accès libre.")
        st.session_state["auth_ok"] = True
        return

    if st.session_state.get("auth_ok"):
        return

    st.sidebar.markdown("## 🔒 Connexion")

    if users:
        username = st.sidebar.text_input("Utilisateur", key="auth_user")
        password = st.sidebar.text_input("Mot de passe", type="password", key="auth_pass")
        if st.sidebar.button("Se connecter", use_container_width=True):
            if username in users and _check_password(password, users[username]):
                st.session_state["auth_ok"] = True
                st.session_state["auth_user_ok"] = username
                st.sidebar.success("Connecté ✅")
            else:
                st.sidebar.error("Identifiants invalides.")
        st.stop()

    # single-user
    password = st.sidebar.text_input("Mot de passe", type="password", key="auth_pass_single")
    if st.sidebar.button("Se connecter", use_container_width=True):
        if app_password_hash:
            ok = _check_password(password, app_password_hash)
        else:
            ok = _secure_compare(password, app_password)
        if ok:
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
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False, ttl=60 * 60)  # cache 1h
def load_prices_yahoo(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Returns adjusted close prices for tickers (business days).
    Robust: tries yf.download first, then falls back to per-ticker history.
    Includes synthetic CASH (flat 1.0).
    """
    tickers_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    want_cash = CASH_SYMBOL in tickers_list
    yf_tickers = [t for t in tickers_list if t != CASH_SYMBOL]

    def _finalize(prices: pd.DataFrame) -> pd.DataFrame:
        if want_cash:
            if prices.empty:
                idx = pd.bdate_range(start=pd.to_datetime(start), end=pd.to_datetime(end))
                prices = pd.DataFrame(index=idx)
            prices[CASH_SYMBOL] = 1.0
        prices = prices.sort_index()
        prices.columns = [str(c).strip().upper() for c in prices.columns]
        prices = prices.ffill().dropna(how="all")
        return prices

    prices = pd.DataFrame()

    # 1) Bulk download (fast)
    if yf_tickers:
        try:
            df = yf.download(
                yf_tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )

            if isinstance(df, pd.DataFrame) and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    # group_by="column" => first level usually fields, second tickers
                    if "Adj Close" in df.columns.get_level_values(0):
                        prices = df["Adj Close"].copy()
                    elif "Close" in df.columns.get_level_values(0):
                        prices = df["Close"].copy()
                else:
                    # single ticker -> OHLCV columns
                    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
                    if col:
                        prices = df[col].to_frame(name=yf_tickers[0])
        except Exception:
            prices = pd.DataFrame()

    # 2) Fallback: per-ticker history (more robust)
    if prices.empty and yf_tickers:
        frames = []
        for t in yf_tickers:
            try:
                h = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False)
                if isinstance(h, pd.DataFrame) and not h.empty:
                    col = "Adj Close" if "Adj Close" in h.columns else ("Close" if "Close" in h.columns else None)
                    if col:
                        s = pd.to_numeric(h[col], errors="coerce").rename(t)
                        frames.append(s)
            except Exception:
                continue
        if frames:
            prices = pd.concat(frames, axis=1)

    prices = _finalize(prices)

    # If still empty, fail with clear message (instead of “tickers manquants”)
    if prices.empty:
        raise RuntimeError(
            "yfinance a renvoyé 0 données (réponse vide). "
            "Ca arrive parfois sur Streamlit Cloud (rate-limit / réseau). "
            "Reboot l'app et réessaie, ou utilise la source CSV."
        )

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
    Multiple CSVs: each file contains at least 'Date' and a price column (Adj Close/Close/Price).
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

        price_col = None
        for cand in ["adj close", "adj_close", "adjusted close", "close", "price"]:
            if cand in cols:
                price_col = cols[cand]
                break
        if price_col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"{f.name}: aucune colonne prix détectée (Close/Adj Close).")
            price_col = numeric_cols[0]

        ticker = name
        if "ticker" in cols:
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

@dataclass(frozen=True)
class MomentumConfig:
    mode: str = "DUAL"  # DUAL or SINGLE
    lookback_days: int = 126
    horizons_days: Tuple[int, ...] = ()  # if non-empty => multi-horizon
    horizons_weights: Tuple[float, ...] = ()  # same length as horizons_days


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    core_tickers: Tuple[str, ...]
    satellite_tickers: Tuple[str, ...]
    benchmark_ticker: str                 # crash filter + chart + dual ref
    risk_off_ticker: str = CASH_SYMBOL
    core_weight: float = 0.60             # rest => satellite sleeve
    top_k: int = 3
    momentum: MomentumConfig = MomentumConfig()
    crash_ma_days: int = 200              # 0 => off
    rebalance: str = "M"                  # W, M, Q
    fee_bps: float = 0.0                  # per rebalance, turnover-based


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rets


def _rebalance_dates(idx: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    s = pd.Series(index=idx, data=np.arange(len(idx)))
    mode_u = (mode or "M").upper().strip()

    if mode_u.startswith("W"):
        r = s.resample("W-FRI").last().dropna().index
        return pd.DatetimeIndex(r)

    if mode_u.startswith("Q"):
        r = s.resample("Q").last().dropna().index
        return pd.DatetimeIndex(r)

    r = s.resample("M").last().dropna().index
    return pd.DatetimeIndex(r)


def _asof_index(prices: pd.DataFrame, asof: pd.Timestamp) -> int:
    if asof in prices.index:
        return int(prices.index.get_loc(asof))
    pos = prices.index.get_indexer([asof], method="ffill")[0]
    if pos < 0:
        return 0
    return int(pos)


def _momentum_total_return(prices: pd.DataFrame, asof: pd.Timestamp, lookback_days: int, tickers: Sequence[str]) -> pd.Series:
    i = _asof_index(prices, asof)
    j = max(0, i - int(lookback_days))
    p0 = prices.iloc[j][list(tickers)]
    p1 = prices.iloc[i][list(tickers)]
    score = (p1 / p0) - 1.0
    return score.replace([np.inf, -np.inf], np.nan).fillna(-np.inf)


def _momentum_score(prices: pd.DataFrame, asof: pd.Timestamp, tickers: Sequence[str], mcfg: MomentumConfig) -> pd.Series:
    if not mcfg.horizons_days:
        return _momentum_total_return(prices, asof, mcfg.lookback_days, tickers)

    days = list(mcfg.horizons_days)
    wts = list(mcfg.horizons_weights) if mcfg.horizons_weights else [1.0] * len(days)
    if len(wts) != len(days):
        wts = [1.0] * len(days)
    wts_sum = float(np.sum(wts)) if float(np.sum(wts)) != 0 else 1.0
    wts = [float(w) / wts_sum for w in wts]

    out = pd.Series(0.0, index=list(tickers), dtype=float)
    for d, w in zip(days, wts):
        out = out + w * _momentum_total_return(prices, asof, int(d), tickers)
    return out.replace([np.inf, -np.inf], np.nan).fillna(-np.inf)


def _crash_filter_on(bench_prices: pd.Series, asof: pd.Timestamp, ma_days: int) -> bool:
    if ma_days <= 0:
        return False
    s = bench_prices.loc[:asof].copy()
    if len(s) < max(10, ma_days // 3):
        return False
    ma = s.rolling(int(ma_days)).mean().iloc[-1]
    px = s.iloc[-1]
    if pd.isna(ma) or pd.isna(px):
        return False
    return float(px) < float(ma)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def backtest(prices: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_tickers = sorted(set(cfg.core_tickers + cfg.satellite_tickers + (cfg.benchmark_ticker, cfg.risk_off_ticker)))
    prices = align_and_clean_prices(prices, all_tickers)

    rets = _returns_from_prices(prices)
    idx = prices.index
    rdates = _rebalance_dates(idx, cfg.rebalance)

    weights = pd.DataFrame(index=rdates, columns=all_tickers, data=0.0)

    core_w = float(np.clip(cfg.core_weight, 0.0, 1.0))
    sat_w = 1.0 - core_w
    core_each = core_w / max(1, len(cfg.core_tickers))
    bench_series = prices[cfg.benchmark_ticker]

    for d in rdates:
        w = pd.Series(0.0, index=all_tickers, dtype=float)

        if _crash_filter_on(bench_series, d, int(cfg.crash_ma_days)):
            w[:] = 0.0
            w[cfg.risk_off_ticker] = 1.0
            weights.loc[d] = w
            continue

        for t in cfg.core_tickers:
            w[t] += core_each

        sats = list(cfg.satellite_tickers)
        if sats and sat_w > 0:
            scores = _momentum_score(prices, d, sats, cfg.momentum).sort_values(ascending=False)
            chosen = list(scores.index[: max(1, min(int(cfg.top_k), len(scores)))])
            chosen_scores = scores.loc[chosen]

            if cfg.momentum.mode.upper() == "DUAL":
                risk_off = cfg.risk_off_ticker
                if risk_off not in prices.columns:
                    prices[risk_off] = 1.0
                ref_score = _momentum_score(prices, d, [risk_off], cfg.momentum).iloc[0]
                best_score = float(chosen_scores.iloc[0]) if len(chosen_scores) else -np.inf
                if (best_score <= ref_score) or (best_score <= 0):
                    w[risk_off] += sat_w
                else:
                    each = sat_w / len(chosen)
                    for t in chosen:
                        w[t] += each
            else:
                if float(chosen_scores.max()) <= 0:
                    w[cfg.risk_off_ticker] += sat_w
                else:
                    each = sat_w / len(chosen)
                    for t in chosen:
                        w[t] += each
        else:
            rem = 1.0 - float(w.sum())
            if rem > 1e-12:
                w[cfg.risk_off_ticker] += rem

        total = float(w.sum())
        if total <= 0:
            w[cfg.risk_off_ticker] = 1.0
        else:
            w = w / total

        weights.loc[d] = w

    port = pd.Series(index=idx, dtype=float)
    bench = (1.0 + rets[cfg.benchmark_ticker]).cumprod()
    port.iloc[0] = 1.0

    current_w = weights.iloc[0].reindex(all_tickers).fillna(0.0)

    for i in range(1, len(idx)):
        d = idx[i]
        prev = idx[i - 1]
        r = rets.loc[d, all_tickers].fillna(0.0)

        if d in weights.index:
            target_w = weights.loc[d].reindex(all_tickers).fillna(0.0)
            turnover = float(np.abs(target_w - current_w).sum()) / 2.0
            fee = turnover * (float(cfg.fee_bps) / 10000.0)
            port.loc[prev] = port.loc[prev] * (1.0 - fee)
            current_w = target_w

        pr = float((current_w * r).sum())
        port.loc[d] = port.loc[prev] * (1.0 + pr)

        growth = (1.0 + r).replace([np.inf, -np.inf], 1.0).clip(lower=0.0)
        w_g = current_w * growth
        s = float(w_g.sum())
        if s > 0:
            current_w = w_g / s

    equity = pd.DataFrame(
        {"portfolio": port, "benchmark": bench.reindex(idx).ffill()},
        index=idx,
    )
    return equity, weights


def perf_stats(equity: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Final": np.nan}
    final = float(equity.iloc[-1])
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (freq / max(1, len(r))) - 1.0)
    vol = float(r.std() * np.sqrt(freq))
    sharpe = float((r.mean() * freq) / (r.std() * np.sqrt(freq))) if float(r.std()) > 0 else np.nan
    dd = (equity / equity.cummax()) - 1.0
    maxdd = float(dd.min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "Final": final}


def plot_equity_multi(equities: Dict[str, pd.DataFrame], logy: bool = True) -> plt.Figure:
    fig, ax = plt.subplots()
    for name, eq in equities.items():
        eq = eq.dropna()
        ax.plot(eq.index, eq["portfolio"], label=name)
    if equities:
        any_eq = next(iter(equities.values()))
        ax.plot(any_eq.index, any_eq["benchmark"], label="Benchmark", linestyle="--")
    ax.set_title("Évolution du capital (comparaison)")
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
# PDF Export
# ----------------------------

def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_pdf_report(
    title: str,
    stats_rows: List[Dict[str, str]],
    equity_png: Optional[bytes] = None,
    dd_png: Optional[bytes] = None,
) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 48
    c.setFont("Helvetica-Bold", 16)
    c.drawString(48, y, title)
    y -= 28

    c.setFont("Helvetica", 10)
    for row in stats_rows:
        line = " | ".join(f"{k}: {v}" for k, v in row.items())
        c.drawString(48, y, line[:140])
        y -= 14
        if y < 120:
            c.showPage()
            y = height - 48
            c.setFont("Helvetica", 10)

    def draw_image(png_bytes: bytes, y_top: float, label: str) -> float:
        nonlocal c
        if not png_bytes:
            return y_top
        img_buf = io.BytesIO(png_bytes)
        img_w = width - 96
        img_h = img_w * 0.55
        y_top -= 14
        c.setFont("Helvetica-Bold", 11)
        c.drawString(48, y_top, label)
        y_top -= 10
        c.drawImage(img_buf, 48, max(48, y_top - img_h), width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        return y_top - img_h - 18

    if equity_png:
        if y < 340:
            c.showPage()
            y = height - 48
        y = draw_image(equity_png, y, "Equity curve")
    if dd_png:
        if y < 340:
            c.showPage()
            y = height - 48
        y = draw_image(dd_png, y, "Drawdown")

    c.save()
    return buf.getvalue()


# ----------------------------
# Optimization
# ----------------------------

def optimize_core_weight(
    prices: pd.DataFrame,
    base_cfg: StrategyConfig,
    grid: Sequence[float],
    objective: str = "Sharpe",
) -> Tuple[pd.DataFrame, float]:
    rows = []
    best_w = float(base_cfg.core_weight)
    best_val = -np.inf

    for w in grid:
        cfg = StrategyConfig(
            name=f"{base_cfg.name} (core={w:.2f})",
            core_tickers=base_cfg.core_tickers,
            satellite_tickers=base_cfg.satellite_tickers,
            benchmark_ticker=base_cfg.benchmark_ticker,
            risk_off_ticker=base_cfg.risk_off_ticker,
            core_weight=float(w),
            top_k=base_cfg.top_k,
            momentum=base_cfg.momentum,
            crash_ma_days=base_cfg.crash_ma_days,
            rebalance=base_cfg.rebalance,
            fee_bps=base_cfg.fee_bps,
        )
        eq, _ = backtest(prices, cfg)
        stt = perf_stats(eq["portfolio"])
        val = float(stt.get(objective, np.nan))
        rows.append({"core_weight": w, **stt})
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_w = float(w)

    df = pd.DataFrame(rows).sort_values(by=objective, ascending=False)
    return df, best_w


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

    st.markdown("## Univers")
    core_tickers_raw = st.text_area("Core tickers (virgules)", value="SPY")
    sat_tickers_raw = st.text_area("Satellite tickers (virgules)", value="QQQ, IWM, EFA, EEM, GLD, TLT")
    benchmark_ticker = st.text_input("Benchmark (crash + chart)", value="SPY").strip().upper()
    risk_off_ticker = st.text_input("Risk-off ticker (CASH recommandé)", value="CASH").strip().upper()

    st.markdown("## Rebalancement")
    rebalance = st.selectbox("Fréquence", ["Weekly (W)", "Monthly (M)", "Quarterly (Q)"], index=1)
    reb_map = {"Weekly (W)": "W", "Monthly (M)": "M", "Quarterly (Q)": "Q"}
    reb_mode = reb_map[rebalance]

    st.markdown("## Crash filter")
    crash_ma_days = st.number_input("MA crash (jours) (0=off)", min_value=0, max_value=1000, value=200, step=10)

    st.markdown("## Frais")
    fee_bps = st.number_input("Frais (bps) / rebalance", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

    st.markdown("## Comparaison / Optimisation")
    compare_mode = st.toggle("Mode comparaison multi-stratégies", value=False)
    optimize_mode = st.toggle("Optimiser core_weight (grid)", value=False)

    run_btn = st.button("🚀 Lancer", use_container_width=True)

    with st.expander("🔐 Générer un hash PBKDF2 (optionnel)", expanded=False):
        st.caption("Utilise ça pour stocker tes mots de passe dans USERS / APP_PASSWORD_HASH.")
        pw = st.text_input("Mot de passe à hasher", type="password")
        it = st.number_input("Iterations", min_value=50_000, max_value=1_000_000, value=260_000, step=10_000)
        if st.button("Générer hash", use_container_width=True):
            if not pw:
                st.warning("Entre un mot de passe.")
            else:
                st.code(pbkdf2_hash(pw, iterations=int(it)), language="text")


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


def build_momentum_ui(prefix: str) -> MomentumConfig:
    st.markdown("### Momentum")
    mode = st.selectbox(f"{prefix} Mode", ["DUAL", "SINGLE"], index=0, key=f"{prefix}_mom_mode")
    use_multi = st.toggle(f"{prefix} Multi-horizons", value=False, key=f"{prefix}_mom_multi")

    if not use_multi:
        lookback_mode = st.radio(f"{prefix} Lookback", ["Custom", "Sweep 10/20/30/50"], index=0, key=f"{prefix}_lb_mode")
        if lookback_mode == "Custom":
            lb = st.number_input(f"{prefix} Lookback (jours)", min_value=5, max_value=2000, value=126, step=5, key=f"{prefix}_lb")
            return MomentumConfig(mode=mode, lookback_days=int(lb), horizons_days=(), horizons_weights=())
        # sweep marker
        return MomentumConfig(mode=mode, lookback_days=0, horizons_days=(10, 20, 30, 50), horizons_weights=(1, 1, 1, 1))

    preset = st.selectbox(
        f"{prefix} Preset horizons",
        ["21/63/126 (1/3/6 mois approx.)", "20/60/120", "Custom"],
        index=0,
        key=f"{prefix}_hz_preset",
    )
    if preset == "21/63/126 (1/3/6 mois approx.)":
        days = (21, 63, 126)
    elif preset == "20/60/120":
        days = (20, 60, 120)
    else:
        raw = st.text_input(f"{prefix} Horizons jours (ex: 20,60,120)", value="21,63,126", key=f"{prefix}_hz_raw")
        days_list = []
        for x in raw.replace(" ", "").split(","):
            if x.strip().isdigit():
                days_list.append(int(x))
        days = tuple([d for d in days_list if d >= 5]) or (21, 63, 126)

    wraw = st.text_input(f"{prefix} Poids (ex: 1,1,1)", value="1,1,1", key=f"{prefix}_hz_wraw")
    w_list = []
    for x in wraw.replace(" ", "").split(","):
        try:
            w_list.append(float(x))
        except Exception:
            pass
    if len(w_list) != len(days):
        w_list = [1.0] * len(days)

    return MomentumConfig(mode=mode, lookback_days=0, horizons_days=tuple(days), horizons_weights=tuple(w_list))


def build_strategy_cfg(name: str, mom: MomentumConfig, core_weight: float, top_k: int) -> StrategyConfig:
    core = tuple(_normalize_ticker_list(core_tickers_raw))
    sat = tuple(_normalize_ticker_list(sat_tickers_raw))
    if not core and not sat:
        raise ValueError("Veuillez renseigner au moins un ticker (core ou satellite).")

    return StrategyConfig(
        name=name,
        core_tickers=core,
        satellite_tickers=sat,
        benchmark_ticker=benchmark_ticker,
        risk_off_ticker=risk_off_ticker if risk_off_ticker else CASH_SYMBOL,
        core_weight=float(core_weight),
        top_k=int(top_k),
        momentum=mom,
        crash_ma_days=int(crash_ma_days),
        rebalance=reb_mode,
        fee_bps=float(fee_bps),
    )


def format_stats_dict(d: Dict[str, float]) -> Dict[str, str]:
    return {
        "CAGR": f"{d['CAGR']:.2%}" if np.isfinite(d["CAGR"]) else "NA",
        "Vol": f"{d['Vol']:.2%}" if np.isfinite(d["Vol"]) else "NA",
        "Sharpe": f"{d['Sharpe']:.2f}" if np.isfinite(d["Sharpe"]) else "NA",
        "MaxDD": f"{d['MaxDD']:.2%}" if np.isfinite(d["MaxDD"]) else "NA",
        "Final": f"{d['Final']:.2f}" if np.isfinite(d["Final"]) else "NA",
    }


if not run_btn:
    st.info("Configure la stratégie à gauche puis clique **Lancer**.")
    st.stop()

try:
    core = _normalize_ticker_list(core_tickers_raw)
    sat = _normalize_ticker_list(sat_tickers_raw)
    if not core and not sat:
        st.error("Veuillez renseigner au moins un ticker (core ou satellite).")
        st.stop()

    all_needed = sorted(set(core + sat + [benchmark_ticker, risk_off_ticker if risk_off_ticker else CASH_SYMBOL]))
    prices_raw = get_prices_for_run(all_needed)
    prices = align_and_clean_prices(prices_raw, all_needed)

    if not compare_mode:
        c_left, c_right = st.columns([1.2, 1.0], vertical_alignment="top")

        with c_left:
            st.subheader("Paramètres de la stratégie")
            mom = build_momentum_ui("S1")
            core_weight = st.slider("Poids Core", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="s1_core_w")
            top_k = st.number_input("Top K satellites", min_value=1, max_value=50, value=3, step=1, key="s1_topk")
            cfg = build_strategy_cfg("Stratégie", mom, core_weight, int(top_k))
            do_sweep = (mom.horizons_days == (10, 20, 30, 50)) and (mom.horizons_weights == (1, 1, 1, 1)) and (mom.lookback_days == 0)

        with c_right:
            st.subheader("Options avancées")
            logy = st.toggle("Échelle log", value=True)
            obj = st.selectbox("Objectif optimisation", ["Sharpe", "CAGR"], index=0)
            grid_step = st.selectbox("Grid pas core_weight", ["0.05", "0.10"], index=0)
            core_grid = np.round(np.arange(0.0, 1.0001, float(grid_step)), 2).tolist()

        if optimize_mode and not do_sweep:
            st.subheader("Optimisation core_weight")
            df_opt, best_w = optimize_core_weight(prices, cfg, core_grid, objective=obj)
            st.dataframe(df_opt.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}", "Final": "{:.2f}"}), use_container_width=True)

            st.success(f"Meilleur core_weight ({obj}) = {best_w:.2f}")
            cfg = StrategyConfig(
                name=f"Stratégie (opt {obj})",
                core_tickers=cfg.core_tickers,
                satellite_tickers=cfg.satellite_tickers,
                benchmark_ticker=cfg.benchmark_ticker,
                risk_off_ticker=cfg.risk_off_ticker,
                core_weight=float(best_w),
                top_k=cfg.top_k,
                momentum=cfg.momentum,
                crash_ma_days=cfg.crash_ma_days,
                rebalance=cfg.rebalance,
                fee_bps=cfg.fee_bps,
            )

        if do_sweep:
            st.subheader("Résultats Sweep (lookback jours: 10/20/30/50)")
            rows = []
            eq_map: Dict[int, pd.DataFrame] = {}
            w_map: Dict[int, pd.DataFrame] = {}
            for lb in (10, 20, 30, 50):
                mom_lb = MomentumConfig(mode=cfg.momentum.mode, lookback_days=lb, horizons_days=(), horizons_weights=())
                cfg_lb = StrategyConfig(
                    name=f"Lookback {lb}",
                    core_tickers=cfg.core_tickers,
                    satellite_tickers=cfg.satellite_tickers,
                    benchmark_ticker=cfg.benchmark_ticker,
                    risk_off_ticker=cfg.risk_off_ticker,
                    core_weight=cfg.core_weight,
                    top_k=cfg.top_k,
                    momentum=mom_lb,
                    crash_ma_days=cfg.crash_ma_days,
                    rebalance=cfg.rebalance,
                    fee_bps=cfg.fee_bps,
                )
                eq, w = backtest(prices, cfg_lb)
                stt = perf_stats(eq["portfolio"])
                rows.append({"lookback_days": lb, **stt})
                eq_map[lb] = eq
                w_map[lb] = w

            res = pd.DataFrame(rows).set_index("lookback_days").sort_index()
            st.dataframe(res.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}", "Final": "{:.2f}"}), use_container_width=True)

            pick = st.selectbox("Afficher le détail (lookback)", options=[10, 20, 30, 50], index=0)
            equity = eq_map[int(pick)]
            weights = w_map[int(pick)]

            c1, c2 = st.columns([2, 1], vertical_alignment="top")
            with c1:
                fig_eq = plot_equity_multi({f"Lookback {pick}": equity}, logy=logy)
                st.pyplot(fig_eq)
                fig_dd = plot_drawdown(equity["portfolio"])
                st.pyplot(fig_dd)

            with c2:
                st.markdown("### Stats")
                p = perf_stats(equity["portfolio"])
                b = perf_stats(equity["benchmark"])
                st.write("**Portfolio**", format_stats_dict(p))
                st.write("**Benchmark**", format_stats_dict(b))
                st.markdown("### Derniers poids (rebalance)")
                st.dataframe(weights.tail(12).style.format("{:.2%}"), use_container_width=True)

            st.markdown("### Téléchargements")
            st.download_button("⬇️ equity_curve.csv", equity.to_csv().encode("utf-8"), file_name="equity_curve.csv")
            st.download_button("⬇️ weights_rebalance.csv", weights.to_csv().encode("utf-8"), file_name="weights_rebalance.csv")

            eq_png = _fig_to_png_bytes(plot_equity_multi({f"Lookback {pick}": equity}, logy=logy))
            dd_png = _fig_to_png_bytes(plot_drawdown(equity["portfolio"]))
            pdf = build_pdf_report(
                title=f"Rapport - Lookback {pick}",
                stats_rows=[
                    {"Portfolio": json.dumps(format_stats_dict(perf_stats(equity['portfolio'])))},
                    {"Benchmark": json.dumps(format_stats_dict(perf_stats(equity['benchmark'])))},
                ],
                equity_png=eq_png,
                dd_png=dd_png,
            )
            st.download_button("⬇️ Rapport PDF", pdf, file_name=f"rapport_lookback_{pick}.pdf", mime="application/pdf")

        else:
            equity, weights = backtest(prices, cfg)

            st.subheader("Résultats")
            c1, c2 = st.columns([2, 1], vertical_alignment="top")
            with c1:
                fig_eq = plot_equity_multi({cfg.name: equity}, logy=logy)
                st.pyplot(fig_eq)
                fig_dd = plot_drawdown(equity["portfolio"])
                st.pyplot(fig_dd)

            with c2:
                st.markdown("### Config")
                st.json({
                    "name": cfg.name,
                    "core": list(cfg.core_tickers),
                    "satellite": list(cfg.satellite_tickers),
                    "benchmark": cfg.benchmark_ticker,
                    "risk_off": cfg.risk_off_ticker,
                    "core_weight": cfg.core_weight,
                    "top_k": cfg.top_k,
                    "momentum": {
                        "mode": cfg.momentum.mode,
                        "lookback_days": cfg.momentum.lookback_days,
                        "horizons_days": list(cfg.momentum.horizons_days),
                        "horizons_weights": list(cfg.momentum.horizons_weights),
                    },
                    "crash_ma_days": cfg.crash_ma_days,
                    "rebalance": cfg.rebalance,
                    "fee_bps": cfg.fee_bps,
                })

                st.markdown("### Stats")
                p = perf_stats(equity["portfolio"])
                b = perf_stats(equity["benchmark"])
                st.write("**Portfolio**", format_stats_dict(p))
                st.write("**Benchmark**", format_stats_dict(b))

                st.markdown("### Derniers poids (rebalance)")
                st.dataframe(weights.tail(12).style.format("{:.2%}"), use_container_width=True)

            st.markdown("### Téléchargements")
            st.download_button("⬇️ equity_curve.csv", equity.to_csv().encode("utf-8"), file_name="equity_curve.csv")
            st.download_button("⬇️ weights_rebalance.csv", weights.to_csv().encode("utf-8"), file_name="weights_rebalance.csv")

            eq_png = _fig_to_png_bytes(plot_equity_multi({cfg.name: equity}, logy=logy))
            dd_png = _fig_to_png_bytes(plot_drawdown(equity["portfolio"]))
            pdf = build_pdf_report(
                title=f"Rapport - {cfg.name}",
                stats_rows=[
                    {"Portfolio": json.dumps(format_stats_dict(p))},
                    {"Benchmark": json.dumps(format_stats_dict(b))},
                ],
                equity_png=eq_png,
                dd_png=dd_png,
            )
            st.download_button("⬇️ Rapport PDF", pdf, file_name="rapport_strategie.pdf", mime="application/pdf")

    else:
        st.subheader("Mode comparaison multi-stratégies")
        st.caption("Définis 2 à 4 variantes (core_weight, top_k, momentum multi-horizons, etc.), puis compare leurs courbes.")

        n = st.slider("Nombre de stratégies", min_value=2, max_value=4, value=2, step=1)
        tabs = st.tabs([f"Stratégie {i+1}" for i in range(n)])

        cfgs: List[StrategyConfig] = []
        for i in range(n):
            with tabs[i]:
                mom = build_momentum_ui(f"C{i+1}")
                cw = st.slider(f"Core weight (S{i+1})", 0.0, 1.0, 0.60, 0.05, key=f"c{i+1}_cw")
                tk = st.number_input(f"Top K (S{i+1})", 1, 50, 3, 1, key=f"c{i+1}_tk")
                nm = st.text_input(f"Nom (S{i+1})", value=f"S{i+1}", key=f"c{i+1}_name").strip() or f"S{i+1}"
                cfgs.append(build_strategy_cfg(nm, mom, float(cw), int(tk)))

        logy = st.toggle("Échelle log (comparaison)", value=True)

        equities: Dict[str, pd.DataFrame] = {}
        weights_last: Dict[str, pd.DataFrame] = {}

        for cfg in cfgs:
            eq, w = backtest(prices, cfg)
            equities[cfg.name] = eq
            weights_last[cfg.name] = w

        fig = plot_equity_multi(equities, logy=logy)
        st.pyplot(fig)

        rows = []
        for name, eq in equities.items():
            stt = perf_stats(eq["portfolio"])
            rows.append({"name": name, **stt})
        df = pd.DataFrame(rows).set_index("name").sort_values("Sharpe", ascending=False)
        st.dataframe(df.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}", "Final": "{:.2f}"}), use_container_width=True)

        pick = st.selectbox("Afficher les poids (rebalance) pour", options=list(equities.keys()), index=0)
        st.dataframe(weights_last[pick].tail(12).style.format("{:.2%}"), use_container_width=True)

        st.markdown("### Téléchargements")
        combined = pd.DataFrame({name: eq["portfolio"] for name, eq in equities.items()})
        combined["benchmark"] = next(iter(equities.values()))["benchmark"]
        st.download_button("⬇️ equity_compare.csv", combined.to_csv().encode("utf-8"), file_name="equity_compare.csv")

        eq_png = _fig_to_png_bytes(plot_equity_multi(equities, logy=logy))
        stats_rows = [{"Strategy": name, **format_stats_dict(perf_stats(eq["portfolio"]))} for name, eq in equities.items()]
        pdf = build_pdf_report(title="Rapport - Comparaison", stats_rows=stats_rows, equity_png=eq_png, dd_png=None)
        st.download_button("⬇️ Rapport PDF (comparaison)", pdf, file_name="rapport_comparaison.pdf", mime="application/pdf")

except Exception as e:
    st.error(f"Erreur: {e}")
