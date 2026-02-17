import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bot Trading (V4)", layout="wide")
st.title("🤖 Bot Trading — Démo publique (V4)")
st.caption("Backtests + signaux. Aucun ordre réel. Données: Stooq.")

# -----------------------------
# Data (robuste)
# -----------------------------
@st.cache_data(ttl=60 * 60)
def safe_download_prices(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        df = pd.read_csv(url)
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        return df
    except Exception:
        return pd.DataFrame()

def clip_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    return d[(d.index >= pd.to_datetime(start)) & (d.index <= pd.to_datetime(end))]

def to_monthly_close(df: pd.DataFrame) -> pd.Series:
    return df["Close"].resample("M").last().dropna()

# -----------------------------
# Indicators (single-asset)
# -----------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# -----------------------------
# Backtest helpers (single-asset)
# -----------------------------
def run_backtest_single(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["Close"].pct_change().fillna(0)

    pos = position.fillna(0).astype(float)
    d["position"] = pos.shift(1).fillna(0)  # exécution le lendemain
    d["trade"] = d["position"].diff().abs().fillna(0)

    fee = (fee_bps / 10000.0)
    d["fee"] = d["trade"] * fee
    d["strategy_ret"] = d["position"] * d["ret"] - d["fee"]

    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1 + d["ret"]).cumprod()
    return d

def stats_from_equity(d: pd.DataFrame, ret_col: str, equity_col: str) -> dict:
    eq = d[equity_col].dropna()
    if len(eq) < 2:
        return {}

    total_return = eq.iloc[-1] - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] ** (1 / years)) - 1

    dd = (eq / eq.cummax()) - 1
    max_dd = dd.min()

    daily = d[ret_col].dropna()
    sharpe = 0.0 if daily.std() == 0 else (daily.mean() / daily.std()) * np.sqrt(252)

    trades = int(d.get("trade", pd.Series([0.0])).sum())
    return {
        "Rendement total": total_return,
        "CAGR (approx.)": cagr,
        "Sharpe (approx.)": sharpe,
        "Max drawdown": max_dd,
        "Trades (approx.)": trades,
    }

# -----------------------------
# Strategies (single-asset)
# -----------------------------
def strat_ma(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ma_fast = df["Close"].rolling(fast).mean()
    ma_slow = df["Close"].rolling(slow).mean()
    return (ma_fast > ma_slow).astype(int)

def strat_rsi(df: pd.DataFrame, period: int, low: float, high: float) -> pd.Series:
    r = rsi(df["Close"], period)
    pos = pd.Series(0, index=df.index, dtype=int)
    in_pos = 0
    for i in range(len(df)):
        if np.isnan(r.iloc[i]):
            pos.iloc[i] = in_pos
            continue
        if in_pos == 0 and r.iloc[i] < low:
            in_pos = 1
        elif in_pos == 1 and r.iloc[i] > high:
            in_pos = 0
        pos.iloc[i] = in_pos
    return pos

def strat_macd(df: pd.DataFrame, fast: int, slow: int, sig: int) -> pd.Series:
    macd_line, signal_line, _ = macd(df["Close"], fast, slow, sig)
    return (macd_line > signal_line).astype(int)

# -----------------------------
# Multi-Asset Rotation (monthly)
# -----------------------------
def rotation_monthly_backtest(
    prices_daily: dict,
    start: str,
    end: str,
    lookback_months: int,
    top_k: int,
    risky_assets: list,
    defensive_assets: list,
    market_filter_on: bool,
    market_symbol: str,
    market_ma_months: int,
    fee_bps: float,
):
    monthly = {}
    for sym, df in prices_daily.items():
        d = clip_dates(df, start, end)
        if d.empty or "Close" not in d.columns:
            continue
        monthly[sym] = to_monthly_close(d)

    if len(monthly) < 2:
        raise ValueError("Pas assez de données mensuelles. Vérifie les tickers et les dates.")

   mdf = pd.DataFrame(monthly).dropna(how="any")

# Double momentum 3 mois + 6 mois
mom_3 = mdf.pct_change(3)
mom_6 = mdf.pct_change(6)
mom = (mom_3 + mom_6) / 2
    if market_filter_on:
        if market_symbol not in mdf.columns:
            raise ValueError(f"Market symbol {market_symbol} absent des données.")
        mkt = mdf[market_symbol]
        mkt_ma = mkt.rolling(market_ma_months).mean()
        risk_on = (mkt > mkt_ma).astype(int)
    else:
        risk_on = pd.Series(1, index=mdf.index)

    weights = pd.DataFrame(0.0, index=mdf.index, columns=mdf.columns)

    for t in range(len(mdf.index)):
        date = mdf.index[t]
        if t < lookback_months:
            continue

        universe = risky_assets if int(risk_on.iloc[t]) == 1 else defensive_assets
        universe = [u for u in universe if u in mdf.columns]
        if not universe:
            continue

        ranks = mom.loc[date, universe].sort_values(ascending=False)
        picks = list(ranks.head(top_k).index)
        w = 1.0 / len(picks)
        for p in picks:
            weights.loc[date, p] = w

    weights = weights.fillna(0.0)

    mret = mdf.pct_change().fillna(0.0)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    fee = (fee_bps / 10000.0) * turnover

    strat_mret = (weights.shift(1).fillna(0.0) * mret).sum(axis=1) - fee
    equity_m = (1 + strat_mret).cumprod()

    bh_mret = mret[market_symbol].fillna(0.0) if market_symbol in mret.columns else mret.mean(axis=1)
    bh_equity_m = (1 + bh_mret).cumprod()

    out_m = pd.DataFrame(
        {
            "strategy_ret_m": strat_mret,
            "equity_m": equity_m,
            "buy_hold_m": bh_equity_m,
            "turnover": turnover,
            "fee_m": fee,
            "risk_on": risk_on,
        },
        index=mdf.index,
    )

    daily_index = clip_dates(prices_daily[market_symbol], start, end).index
    out_d = pd.DataFrame(index=daily_index)
    out_d["equity"] = out_m["equity_m"].reindex(daily_index, method="ffill")
    out_d["buy_hold"] = out_m["buy_hold_m"].reindex(daily_index, method="ffill")
    out_d["strategy_ret"] = out_d["equity"].pct_change().fillna(0.0)
    out_d["ret"] = out_d["buy_hold"].pct_change().fillna(0.0)

    return out_m, out_d, weights

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Mode")
mode = st.sidebar.selectbox(
    "Choisis un mode",
    ["Single-asset (MA / RSI / MACD)", "Multi-asset (Rotation mensuelle)"],
)

start = st.sidebar.date_input("Début", value=pd.to_datetime("2010-01-01"))
end = st.sidebar.date_input("Fin", value=pd.to_datetime("2025-12-31"))
fee_bps = st.sidebar.slider("Frais (bps)", 0, 50, 10)

run = st.sidebar.button("Lancer")

if not run:
    st.info("Choisis un mode et clique **Lancer**.")
    st.stop()

# -----------------------------
# Mode 1: Single-asset
# -----------------------------
if mode.startswith("Single-asset"):
    st.sidebar.subheader("Single-asset")
    ticker = st.sidebar.text_input("Ticker (Stooq)", value="aapl.us")

    strategy = st.sidebar.selectbox("Stratégie", ["Moyennes mobiles (MA)", "RSI (mean-reversion)", "MACD (trend)"])

    if strategy.startswith("Moyennes"):
        fast = st.sidebar.slider("MA rapide", 5, 60, 20)
        slow = st.sidebar.slider("MA lente", 20, 200, 100)
    elif strategy.startswith("RSI"):
        rsi_period = st.sidebar.slider("RSI période", 5, 30, 14)
        low = st.sidebar.slider("Seuil bas (entrée)", 5, 45, 30)
        high = st.sidebar.slider("Seuil haut (sortie)", 55, 95, 70)
    else:
        macd_fast = st.sidebar.slider("MACD fast", 5, 20, 12)
        macd_slow = st.sidebar.slider("MACD slow", 15, 60, 26)
        macd_sig = st.sidebar.slider("MACD signal", 5, 20, 9)

    try:
        prices = safe_download_prices(ticker)
        prices = clip_dates(prices, str(start), str(end))
        if prices.empty:
            st.error("Données indisponibles. Essaie un autre ticker.")
            st.stop()

        if strategy.startswith("Moyennes"):
            if fast >= slow:
                st.error("MA rapide doit être < MA lente.")
                st.stop()
            pos = strat_ma(prices, fast, slow)
        elif strategy.startswith("RSI"):
            if low >= high:
                st.error("Seuil bas doit être < seuil haut.")
                st.stop()
            pos = strat_rsi(prices, rsi_period, low, high)
        else:
            if macd_fast >= macd_slow:
                st.error("MACD fast doit être < MACD slow.")
                st.stop()
            pos = strat_macd(prices, macd_fast, macd_slow, macd_sig)

        d = run_backtest_single(prices, pos, fee_bps)
        s = stats_from_equity(d, ret_col="strategy_ret", equity_col="equity")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
        c2.metric("CAGR", f"{s['CAGR (approx.)']*100:.2f}%")
        c3.metric("Sharpe", f"{s['Sharpe (approx.)']:.2f}")
        c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
        c5.metric("Trades", f"{s['Trades (approx.)']}")

        st.subheader("Équité (Stratégie vs Buy & Hold)")
        st.line_chart(pd.DataFrame({"Stratégie": d["equity"], "Buy & Hold": d["buy_hold"]}))

        st.subheader("Dernier signal")
        last = d.dropna().iloc[-1]
        st.info(f"Au {last.name.date()} : {'📈 LONG' if last['position'] == 1 else '📉 CASH'}")

        st.subheader("Données (dernier 250 jours)")
        st.dataframe(d[["Close", "position", "strategy_ret", "equity", "buy_hold"]].tail(250), use_container_width=True)

    except Exception as e:
        st.exception(e)

# -----------------------------
# Mode 2: Multi-asset
# -----------------------------
else:
    st.sidebar.subheader("Multi-asset (Rotation mensuelle)")

    # Univers étendu
    spy = st.sidebar.text_input("SPY (US Large)", value="spy.us")
    qqq = st.sidebar.text_input("QQQ (Tech)", value="qqq.us")
    efa = st.sidebar.text_input("EFA (Intl Dev)", value="efa.us")
    eem = st.sidebar.text_input("EEM (Emerging)", value="eem.us")
    vnq = st.sidebar.text_input("VNQ (Real Estate)", value="vnq.us")

    tlt = st.sidebar.text_input("TLT (Long Bonds)", value="tlt.us")
    ief = st.sidebar.text_input("IEF (Mid Bonds)", value="ief.us")
    gld = st.sidebar.text_input("GLD (Gold)", value="gld.us")

    lookback = st.sidebar.slider("Lookback momentum (mois)", 1, 12, 3)
    top_k = st.sidebar.slider("Nombre d'actifs sélectionnés", 1, 3, 2)

    market_filter_on = st.sidebar.checkbox("Filtre marché (SPY au-dessus MA)", value=True)
    market_ma = st.sidebar.slider("MA marché (mois)", 6, 24, 12)

    try:
        prices = {
            spy: safe_download_prices(spy),
            qqq: safe_download_prices(qqq),
            efa: safe_download_prices(efa),
            eem: safe_download_prices(eem),
            vnq: safe_download_prices(vnq),
            tlt: safe_download_prices(tlt),
            ief: safe_download_prices(ief),
            gld: safe_download_prices(gld),
        }

        # Nettoyage: garder seulement ceux qui ont des données
        clean_prices = {}
        for sym, df in prices.items():
            d = clip_dates(df, str(start), str(end))
            if d is not None and not d.empty and "Close" in d.columns:
                clean_prices[sym] = df
        prices = clean_prices

        if spy not in prices:
            st.error("SPY n'a pas de données. Vérifie le ticker (spy.us) ou ajuste les dates.")
            st.stop()

        risky_assets = [spy, qqq, efa, eem, vnq]
        defensive_assets = [tlt, ief, gld]

        risky_assets = [x for x in risky_assets if x in prices]
        defensive_assets = [x for x in defensive_assets if x in prices]

        out_m, out_d, w = rotation_monthly_backtest(
            prices_daily=prices,
            start=str(start),
            end=str(end),
            lookback_months=lookback,
            top_k=top_k,
            risky_assets=risky_assets,
            defensive_assets=defensive_assets,
            market_filter_on=market_filter_on,
            market_symbol=spy,
            market_ma_months=market_ma,
            fee_bps=fee_bps,
        )

        s = stats_from_equity(out_d, ret_col="strategy_ret", equity_col="equity")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
        c2.metric("CAGR", f"{s['CAGR (approx.)']*100:.2f}%")
        c3.metric("Sharpe", f"{s['Sharpe (approx.)']:.2f}")
        c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
        c5.metric("Trades (approx.)", f"{int(out_m['turnover'].sum()):d}")

        st.subheader("Équité (Rotation vs Buy & Hold SPY)")
        st.line_chart(pd.DataFrame({"Rotation": out_d["equity"], "Buy & Hold (SPY)": out_d["buy_hold"]}))

        st.subheader("Poids mensuels (dernier 18 mois)")
        st.dataframe(w.tail(18), use_container_width=True)

        st.subheader("Journal mensuel (dernier 24 mois)")
        st.dataframe(out_m.tail(24), use_container_width=True)

        # --------- Message automatique (prêt à publier) ----------
        st.subheader("📝 Message automatique (prêt à publier)")

        valid = out_m.dropna()
        if valid.empty:
            st.warning("Pas assez de données pour générer un message (essaie une période plus longue ou ajuste les paramètres).")
        else:
            last_date = valid.index[-1]
            last_row = out_m.loc[last_date]
            last_weights = w.loc[last_date]

            alloc = last_weights[last_weights > 0].sort_values(ascending=False)
            alloc_text = ", ".join([f"{sym.upper()} {int(weight*100)}%" for sym, weight in alloc.items()])
            regime_text = "RISK-ON ✅" if int(last_row["risk_on"]) == 1 else "RISK-OFF 🛡️"

            msg = f"""📊 Rapport stratégie (Rotation Multi-Asset)

Période backtest: {start} → {end}

Performance:
• CAGR: {s['CAGR (approx.)']*100:.2f}%
• Sharpe: {s['Sharpe (approx.)']:.2f}
• Max Drawdown: {s['Max drawdown']*100:.2f}%

État actuel ({pd.to_datetime(last_date).date()}):
• Régime marché: {regime_text}
• Allocation: {alloc_text}
• Filtre marché (MA): {market_ma} mois
• Frais: {fee_bps} bps

Note: Ceci est un backtest (simulation), pas un conseil financier.
"""

            st.text_area("Message", msg, height=240)
            st.download_button("Télécharger le message (.txt)", msg, file_name="rapport_rotation.txt")

    except Exception as e:
        st.exception(e)

