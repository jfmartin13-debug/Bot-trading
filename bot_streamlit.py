import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bot Trading (Démo)", layout="wide")
st.title("🤖 Bot Trading — Démo publique (V3 Agressif)")
st.caption("Backtest + signaux. Aucun ordre réel. Données: Stooq. (Attention: levier = risque accru)")

# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=60 * 60)
def download_prices(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def clip_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = df.copy()
    return d[(d.index >= pd.to_datetime(start)) & (d.index <= pd.to_datetime(end))]

# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# -----------------------------
# Agressive strategy: MA + RSI confirm
# -----------------------------
def aggressive_signal(df: pd.DataFrame, ma_fast: int, ma_slow: int, rsi_period: int, rsi_min: float) -> pd.Series:
    """
    Long si:
      - MA_fast > MA_slow (tendance)
      - RSI > rsi_min (momentum)
    """
    d = df.copy()
    d["ma_fast"] = d["Close"].rolling(ma_fast).mean()
    d["ma_slow"] = d["Close"].rolling(ma_slow).mean()
    d["rsi"] = rsi(d["Close"], rsi_period)
    sig = ((d["ma_fast"] > d["ma_slow"]) & (d["rsi"] > rsi_min)).astype(int)
    return sig

# -----------------------------
# Backtest engine (with leverage + optional stops)
# -----------------------------
def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    fee_bps: float,
    leverage: float,
    use_regime_filter: bool,
    regime_ma: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> pd.DataFrame:
    """
    - long-only
    - leverage applied to daily returns when in position (simple model)
    - optional regime filter: only long if Close > MA(regime_ma)
    - optional stop loss / take profit based on entry price (state machine)
    """
    d = df.copy()
    d["ret"] = d["Close"].pct_change().fillna(0)

    # Regime filter
    if use_regime_filter:
        d["regime_ma"] = d["Close"].rolling(regime_ma).mean()
        regime_ok = (d["Close"] > d["regime_ma"]).astype(int)
    else:
        regime_ok = pd.Series(1, index=d.index)

    raw_sig = (signal.fillna(0).astype(int) * regime_ok.fillna(0).astype(int)).astype(int)

    # State machine for stops
    pos = pd.Series(0.0, index=d.index)
    entry_price = None
    in_pos = 0

    for i in range(len(d)):
        price = float(d["Close"].iloc[i])
        want_long = int(raw_sig.iloc[i])

        if in_pos == 0:
            if want_long == 1:
                in_pos = 1
                entry_price = price
        else:
            # in position -> check stops if enabled
            if entry_price is not None:
                pnl = (price / entry_price) - 1.0

                if stop_loss_pct > 0 and pnl <= -abs(stop_loss_pct):
                    in_pos = 0
                    entry_price = None
                elif take_profit_pct > 0 and pnl >= abs(take_profit_pct):
                    in_pos = 0
                    entry_price = None
                else:
                    # also exit if signal turns off
                    if want_long == 0:
                        in_pos = 0
                        entry_price = None

        pos.iloc[i] = float(in_pos) * float(leverage)

    # Execute next day
    d["position"] = pos.shift(1).fillna(0)

    # trades for fees: fee on changes in absolute exposure
    d["trade"] = d["position"].diff().abs().fillna(0)
    fee = (fee_bps / 10000.0)
    d["fee"] = d["trade"] * fee

    # leveraged return (simple) minus fees
    d["strategy_ret"] = d["position"] * d["ret"] - d["fee"]

    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1 + d["ret"]).cumprod()
    d["signal"] = raw_sig
    return d

def stats(d: pd.DataFrame) -> dict:
    eq = d["equity"].dropna()
    if len(eq) < 2:
        return {}

    total_return = eq.iloc[-1] - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] ** (1 / years)) - 1

    dd = (eq / eq.cummax()) - 1
    max_dd = dd.min()

    daily = d["strategy_ret"].dropna()
    sharpe = 0.0 if daily.std() == 0 else (daily.mean() / daily.std()) * np.sqrt(252)

    trades = int(d["trade"].sum())
    return {
        "Rendement total": total_return,
        "CAGR (approx.)": cagr,
        "Sharpe (approx.)": sharpe,
        "Max drawdown": max_dd,
        "Trades (approx.)": trades,
    }

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Paramètres")
ticker = st.sidebar.text_input("Ticker (Stooq)", value="aapl.us", help="Ex: aapl.us, msft.us, nvda.us, tsla.us, ry.ca")
start = st.sidebar.date_input("Début", value=pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fin", value=pd.to_datetime("2025-12-31"))

st.sidebar.subheader("Stratégie Agressive (MA + RSI)")
ma_fast = st.sidebar.slider("MA rapide", 5, 60, 15)
ma_slow = st.sidebar.slider("MA lente", 20, 250, 80)
rsi_period = st.sidebar.slider("RSI période", 5, 30, 14)
rsi_min = st.sidebar.slider("RSI minimum (confirmation)", 40, 70, 55)

st.sidebar.subheader("Risque / Exécution")
fee_bps = st.sidebar.slider("Frais (bps)", 0, 50, 10)
leverage = st.sidebar.slider("Levier", 1.0, 2.0, 1.5, 0.1)

use_regime_filter = st.sidebar.checkbox("Filtre régime (protection crash)", value=False)
regime_ma = st.sidebar.slider("MA régime", 100, 300, 200) if use_regime_filter else 200

st.sidebar.subheader("Stops (optionnels)")
stop_loss_pct = st.sidebar.slider("Stop-loss (%)", 0.0, 30.0, 0.0, 0.5) / 100.0
take_profit_pct = st.sidebar.slider("Take-profit (%)", 0.0, 80.0, 0.0, 1.0) / 100.0

run = st.sidebar.button("Lancer le backtest")

if not run:
    st.info("Choisis un ticker et clique sur **Lancer le backtest**.")
    st.stop()

try:
    if ma_fast >= ma_slow:
        st.error("MA rapide doit être < MA lente.")
        st.stop()

    prices = download_prices(ticker)
    prices = clip_dates(prices, str(start), str(end))

    if prices.empty or "Close" not in prices.columns:
        st.error("Données indisponibles. Essaie un autre ticker (ex: aapl.us, nvda.us, tsla.us).")
        st.stop()

    sig = aggressive_signal(prices, ma_fast, ma_slow, rsi_period, rsi_min)
    d = run_backtest(
        prices, sig,
        fee_bps=fee_bps,
        leverage=leverage,
        use_regime_filter=use_regime_filter,
        regime_ma=regime_ma,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    s = stats(d)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
    c2.metric("CAGR (approx.)", f"{s['CAGR (approx.)']*100:.2f}%")
    c3.metric("Sharpe", f"{s['Sharpe (approx.)']:.2f}")
    c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
    c5.metric("Trades", f"{s['Trades (approx.)']}")

    st.subheader("Équité (Stratégie vs Buy & Hold)")
    st.line_chart(pd.DataFrame({"Stratégie": d["equity"], "Buy & Hold": d["buy_hold"]}))

    st.subheader("Dernier état")
    last = d.dropna().iloc[-1]
    state = "📈 LONG" if last["position"] > 0 else "📉 CASH"
    st.info(f"Au {last.name.date()} : {state} | Exposition: {last['position']:.1f}x | Signal brut: {int(last['signal'])}")

    st.subheader("Aperçu (dernier 250 jours)")
    view = d[["Close", "signal", "position", "strategy_ret", "equity", "buy_hold"]].tail(250).copy()
    st.dataframe(view, use_container_width=True)

except Exception as e:
    st.exception(e)
