
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bot Trading (Démo)", layout="wide")
st.title("🤖 Bot Trading — Démo publique (V2)")
st.caption("Backtest + signaux. Aucun ordre réel n’est envoyé. (Données: Stooq)")

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
    d = d[(d.index >= pd.to_datetime(start)) & (d.index <= pd.to_datetime(end))]
    return d

# -----------------------------
# Indicators
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
# Backtest engine (simple)
# -----------------------------
def run_backtest(df: pd.DataFrame, position: pd.Series, fee_bps: float) -> pd.DataFrame:
    """
    position: 0 ou 1 (long only).
    fee_bps: frais en basis points (ex: 10 bps = 0.10%)
    """
    d = df.copy()
    d["ret"] = d["Close"].pct_change().fillna(0)

    # trades quand la position change (0->1 ou 1->0)
    pos = position.fillna(0).astype(float)
    d["position"] = pos.shift(1).fillna(0)  # exécuté le lendemain
    d["trade"] = d["position"].diff().abs().fillna(0)

    fee = (fee_bps / 10000.0)  # bps -> fraction
    d["fee"] = d["trade"] * fee  # payé à chaque changement
    d["strategy_ret"] = d["position"] * d["ret"] - d["fee"]

    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1 + d["ret"]).cumprod()
    return d

def stats(d: pd.DataFrame) -> dict:
    eq = d["equity"].dropna()
    if len(eq) < 2:
        return {}

    total_return = eq.iloc[-1] - 1
    # approx annualization
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] ** (1 / years)) - 1

    dd = (eq / eq.cummax()) - 1
    max_dd = dd.min()

    daily = d["strategy_ret"].dropna()
    sharpe = 0.0 if daily.std() == 0 else (daily.mean() / daily.std()) * np.sqrt(252)

    trades = int(d["trade"].sum())  # approx (entrée + sortie comptent)
    return {
        "Rendement total": total_return,
        "CAGR (approx.)": cagr,
        "Sharpe (approx.)": sharpe,
        "Max drawdown": max_dd,
        "Trades (approx.)": trades,
    }

# -----------------------------
# Strategies
# -----------------------------
def strat_ma(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ma_fast = df["Close"].rolling(fast).mean()
    ma_slow = df["Close"].rolling(slow).mean()
    signal = (ma_fast > ma_slow).astype(int)
    return signal

def strat_rsi(df: pd.DataFrame, period: int, low: float, high: float) -> pd.Series:
    r = rsi(df["Close"], period)
    # simple: long si RSI < low (survente) jusqu'à ce que RSI > high
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
    signal = (macd_line > signal_line).astype(int)
    return signal

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Paramètres")
ticker = st.sidebar.text_input("Ticker (Stooq)", value="aapl.us", help="Ex: aapl.us, msft.us, shop.us, ry.ca, td.ca")
start = st.sidebar.date_input("Début", value=pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fin", value=pd.to_datetime("2025-12-31"))

strategy = st.sidebar.selectbox("Stratégie", ["Moyennes mobiles (MA)", "RSI (mean-reversion)", "MACD (trend)"])
fee_bps = st.sidebar.slider("Frais (bps)", 0, 50, 10, help="10 bps = 0.10% par trade (entrée/sortie)")

# params par stratégie
if strategy.startswith("Moyennes"):
    fast = st.sidebar.slider("MA rapide", 5, 60, 20)
    slow = st.sidebar.slider("MA lente", 20, 200, 50)
elif strategy.startswith("RSI"):
    rsi_period = st.sidebar.slider("RSI période", 5, 30, 14)
    low = st.sidebar.slider("Seuil bas (entrée)", 5, 45, 30)
    high = st.sidebar.slider("Seuil haut (sortie)", 55, 95, 70)
else:
    macd_fast = st.sidebar.slider("MACD fast", 5, 20, 12)
    macd_slow = st.sidebar.slider("MACD slow", 15, 60, 26)
    macd_sig = st.sidebar.slider("MACD signal", 5, 20, 9)

run = st.sidebar.button("Lancer le backtest")

if not run:
    st.info("Choisis un ticker et clique sur **Lancer le backtest**. Ex: `aapl.us`, `msft.us`, `ry.ca`")
    st.stop()

# -----------------------------
# Execution
# -----------------------------
try:
    prices = download_prices(ticker)
    prices = clip_dates(prices, str(start), str(end))

    if prices.empty or "Close" not in prices.columns:
        st.error("Données indisponibles. Essaie un autre ticker (ex: aapl.us, msft.us, ry.ca).")
        st.stop()

    if strategy.startswith("Moyennes"):
        if fast >= slow:
            st.error("MA rapide doit être plus petite que MA lente.")
            st.stop()
        pos = strat_ma(prices, fast, slow)
    elif strategy.startswith("RSI"):
        if low >= high:
            st.error("Le seuil bas doit être plus petit que le seuil haut.")
            st.stop()
        pos = strat_rsi(prices, rsi_period, low, high)
    else:
        if macd_fast >= macd_slow:
            st.error("MACD fast doit être < MACD slow.")
            st.stop()
        pos = strat_macd(prices, macd_fast, macd_slow, macd_sig)

    d = run_backtest(prices, pos, fee_bps)
    s = stats(d)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
    c2.metric("CAGR (approx.)", f"{s['CAGR (approx.)']*100:.2f}%")
    c3.metric("Sharpe", f"{s['Sharpe (approx.)']:.2f}")
    c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
    c5.metric("Trades", f"{s['Trades (approx.)']}")

    st.subheader("Équité (Stratégie vs Buy & Hold)")
    st.line_chart(pd.DataFrame({"Stratégie": d["equity"], "Buy & Hold": d["buy_hold"]}))

    st.subheader("Dernier signal")
    last = d.dropna().iloc[-1]
    signal_txt = "📈 LONG (position ON)" if last["position"] == 1 else "📉 CASH (position OFF)"
    st.info(f"Au {last.name.date()} : {signal_txt}")

    st.subheader("Aperçu des signaux & indicateurs")
    show = d[["Close", "position", "strategy_ret", "equity"]].tail(250).copy()
    st.dataframe(show, use_container_width=True)

except Exception as e:
    st.exception(e)
