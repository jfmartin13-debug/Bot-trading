import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bot Trading (Démo)", layout="wide")

st.title("🤖 Bot Trading — Démo publique")
st.caption("Backtest + signaux. Aucun ordre réel n’est envoyé.")

# --- Helpers ---
def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données via Stooq (gratuit, pas de clé API).
    Marche souvent sans dépendances externes. Format: Date, Open, High, Low, Close, Volume.
    """
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    df = df.set_index("Date")
    return df

def ma_strategy(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    d = df.copy()
    d["ma_fast"] = d["Close"].rolling(fast).mean()
    d["ma_slow"] = d["Close"].rolling(slow).mean()
    d["signal"] = 0
    d.loc[d["ma_fast"] > d["ma_slow"], "signal"] = 1
    d["position"] = d["signal"].shift(1).fillna(0)  # position prise le lendemain
    d["ret"] = d["Close"].pct_change().fillna(0)
    d["strategy_ret"] = d["position"] * d["ret"]
    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1 + d["ret"]).cumprod()
    return d

def perf_stats(d: pd.DataFrame) -> dict:
    daily = d["strategy_ret"]
    if daily.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)

    total_return = d["equity"].iloc[-1] - 1
    max_dd = (d["equity"] / d["equity"].cummax() - 1).min()

    return {
        "Rendement total": total_return,
        "Sharpe (approx.)": sharpe,
        "Max drawdown": max_dd,
    }

# --- Sidebar ---
st.sidebar.header("Paramètres")
ticker = st.sidebar.text_input("Ticker (Stooq)", value="aapl.us", help="Ex: aapl.us, msft.us, shop.us, ry.ca, td.ca")
start = st.sidebar.date_input("Début", value=pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("Fin", value=pd.to_datetime("2025-12-31"))

fast = st.sidebar.slider("MA rapide", 5, 60, 20)
slow = st.sidebar.slider("MA lente", 20, 200, 50)

run = st.sidebar.button("Lancer le backtest")

# --- Main ---
if run:
    try:
        if fast >= slow:
            st.error("MA rapide doit être plus petite que MA lente.")
            st.stop()

        with st.spinner("Téléchargement des données..."):
            prices = download_prices(ticker, str(start), str(end))

        if prices.empty or "Close" not in prices.columns:
            st.error("Impossible de récupérer les données. Essaie un autre ticker (ex: aapl.us, msft.us, ry.ca).")
            st.stop()

        d = ma_strategy(prices, fast, slow)
        stats = perf_stats(d)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rendement total", f"{stats['Rendement total']*100:.2f}%")
        c2.metric("Sharpe", f"{stats['Sharpe (approx.)']:.2f}")
        c3.metric("Max drawdown", f"{stats['Max drawdown']*100:.2f}%")

        st.subheader("Équité (Stratégie vs Buy & Hold)")
        chart = pd.DataFrame({
            "Stratégie": d["equity"],
            "Buy & Hold": d["buy_hold"]
        })
        st.line_chart(chart)

        st.subheader("Dernier signal")
        last = d.dropna().iloc[-1]
        signal = "📈 ACHAT (position ON)" if last["signal"] == 1 else "📉 CASH (position OFF)"
        st.info(f"Au {last.name.date()} : {signal}")

        st.subheader("Données")
        st.dataframe(d.tail(200), use_container_width=True)

    except Exception as e:
        st.exception(e)
else:
    st.info("Choisis un ticker et clique sur **Lancer le backtest**.")
    st.write("Exemples de tickers: `aapl.us`, `msft.us`, `shop.us`, `ry.ca`, `td.ca`")
