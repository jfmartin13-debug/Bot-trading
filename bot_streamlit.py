import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("📈 Trading Bot — Dual Momentum")

# =========================
# 1️⃣ DONNÉES
# =========================

st.sidebar.header("1) Données")

mode = st.sidebar.radio(
    "Mode d'upload",
    ["Un seul CSV (wide)", "Plusieurs CSV (un par ticker)"]
)

def load_wide(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df

def load_multi(files):
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        ticker = f.name.split(".")[0].replace("_us_d","").replace(".us","").upper()
        price_col = [c for c in df.columns if "Close" in c][0]
        frames.append(df[[price_col]].rename(columns={price_col: ticker}))
    merged = pd.concat(frames, axis=1)
    return merged

if mode == "Un seul CSV (wide)":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.stop()
    close = load_wide(file)
else:
    files = st.sidebar.file_uploader("Upload plusieurs CSV", type=["csv"], accept_multiple_files=True)
    if not files:
        st.stop()
    close = load_multi(files)

# =========================
# 2️⃣ PARAMÈTRES
# =========================

st.sidebar.header("2) Paramètres")

rebalance = st.sidebar.selectbox("Fréquence", ["M","W"])
fee_bps = st.sidebar.number_input("Frais (bps)", value=10.0)

momentum_mode = st.sidebar.selectbox("Momentum", ["DUAL","SINGLE"])

if momentum_mode == "DUAL":
    lb1 = st.sidebar.slider("Lookback 1", 1, 12, 3)
    lb2 = st.sidebar.slider("Lookback 2", 2, 18, 12)
    weight1 = st.sidebar.slider("Poids lookback 1", 0.0, 1.0, 0.25)
    weight2 = 1 - weight1
else:
    lb_single = st.sidebar.slider("Lookback single", 1, 18, 12)

top_n = st.sidebar.slider("Top N", 1, 5, 2)

market_filter = st.sidebar.toggle("Filtre marché ON", value=True)
risk_off = st.sidebar.selectbox("Risk-off", ["CASH","DEFENSIVE"])
long_only = st.sidebar.toggle("Long-only (score > 0)", value=False)

# =========================
# 3️⃣ CHOIX TICKERS
# =========================

st.sidebar.header("3) Choix des tickers")

available = list(close.columns)
universe = st.sidebar.multiselect("Univers", available, default=available)

if len(universe) < 2:
    st.stop()

# =========================
# 4️⃣ DATES
# =========================

st.sidebar.header("4) Dates")

start = st.sidebar.date_input("Début", close.index.min())
end = st.sidebar.date_input("Fin", close.index.max())

close = close[(close.index >= pd.to_datetime(start)) & (close.index <= pd.to_datetime(end))]

# =========================
# BACKTEST
# =========================

if rebalance == "M":
    prices = close.resample("M").last()
else:
    prices = close.resample("W-FRI").last()

returns = prices.pct_change()

if momentum_mode == "DUAL":
    score = weight1 * prices.pct_change(lb1) + weight2 * prices.pct_change(lb2)
else:
    score = prices.pct_change(lb_single)

weights = pd.DataFrame(0, index=prices.index, columns=prices.columns)

for t in prices.index:
    row = score.loc[t].dropna()
    if long_only:
        row = row[row > 0]
    if row.empty:
        continue
    top = row.sort_values(ascending=False).head(top_n).index
    weights.loc[t, top] = 1/len(top)

w_prev = weights.shift(1).fillna(0)
turnover = (weights - w_prev).abs().sum(axis=1)
fees = turnover * (fee_bps/10000)

ret = (w_prev * returns).sum(axis=1) - fees
equity = (1 + ret).cumprod()

# =========================
# STATS
# =========================

years = len(ret)/12 if rebalance=="M" else len(ret)/52
cagr = equity.iloc[-1]**(1/years) - 1 if years>0 else 0
vol = ret.std()*np.sqrt(12 if rebalance=="M" else 52)
sharpe = cagr/vol if vol!=0 else 0
mdd = (equity/equity.cummax()-1).min()

# =========================
# AFFICHAGE
# =========================

col1,col2,col3,col4 = st.columns(4)
col1.metric("CAGR", f"{cagr*100:.2f}%")
col2.metric("Vol", f"{vol*100:.2f}%")
col3.metric("Sharpe", f"{sharpe:.2f}")
col4.metric("Max DD", f"{mdd*100:.2f}%")

st.line_chart(equity)
