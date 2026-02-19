import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bot Trading", layout="wide")
st.title("📈 Bot Trading – Version Simplifiée")

# =========================
# Upload CSV
# =========================
st.sidebar.header("1️⃣ Données")

uploaded_files = st.sidebar.file_uploader(
    "Upload plusieurs CSV (1 par ticker)",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload tes fichiers CSV pour commencer.")
    st.stop()

# =========================
# Chargement données
# =========================
def load_many_csv(files):
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        ticker = f.name.split(".")[0].upper()
        frames.append(df[[price_col]].rename(columns={price_col: ticker}))

    return pd.concat(frames, axis=1).dropna(how="all")

close = load_many_csv(uploaded_files)
available = list(close.columns)

# =========================
# Paramètres
# =========================
st.sidebar.header("2️⃣ Paramètres")

rebalance = st.sidebar.selectbox("Fréquence", ["M", "W"], index=0)
lookback = st.sidebar.slider("Momentum (mois)", 1, 12, 6)

st.sidebar.header("3️⃣ Filtre Crash")

market_filter_on = st.sidebar.toggle("Activer filtre marché", True)

default_asset = "SPY" if "SPY" in available else available[0]
market_filter_asset = st.sidebar.selectbox("Actif filtre", available, index=available.index(default_asset))

ma_window = st.sidebar.slider("MA (jours)", 100, 300, 200)

# =========================
# Resample
# =========================
if rebalance == "M":
    prices_rebal = close.resample("M").last()
else:
    prices_rebal = close.resample("W-FRI").last()

returns = prices_rebal.pct_change()

# =========================
# Momentum simple
# =========================
scores = prices_rebal.pct_change(lookback)
weights = pd.DataFrame(0, index=prices_rebal.index, columns=prices_rebal.columns)

for date in scores.index:
    row = scores.loc[date].dropna()
    if not row.empty:
        top = row.idxmax()
        weights.loc[date, top] = 1

# =========================
# FILTRE CRASH (EXTINCTEUR)
# =========================
if market_filter_on and market_filter_asset in close.columns:
    spy = close[market_filter_asset]
    ma = spy.rolling(ma_window).mean()
    risk_on_daily = spy > ma
    risk_on = risk_on_daily.reindex(weights.index, method="ffill").fillna(False)

    for date in weights.index:
        if not risk_on.loc[date]:
            weights.loc[date] = 0  # 100% CASH

# =========================
# Backtest
# =========================
weights_shifted = weights.shift(1).fillna(0)
strategy_returns = (weights_shifted * returns).sum(axis=1)
equity = (1 + strategy_returns).cumprod()

# =========================
# Stats
# =========================
def max_drawdown(eq):
    peak = eq.cummax()
    dd = eq / peak - 1
    return dd.min()

cagr = (equity.iloc[-1]) ** (12 / len(equity)) - 1
vol = strategy_returns.std() * np.sqrt(12)
sharpe = cagr / vol if vol != 0 else 0
mdd = max_drawdown(equity)

# =========================
# Affichage
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", f"{cagr*100:.2f}%")
col2.metric("Volatilité", f"{vol*100:.2f}%")
col3.metric("Sharpe", f"{sharpe:.2f}")
col4.metric("Max Drawdown", f"{mdd*100:.2f}%")

st.line_chart(equity)

st.write("### Dernière allocation")
st.dataframe(weights.tail(5))
