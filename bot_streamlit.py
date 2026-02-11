import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bot Trading SMA", layout="wide")

st.title("📈 Bot de Trading - Moyennes Mobiles")

# --- Paramètres utilisateur ---
ticker = st.text_input("Ticker (ex: AAPL, TSLA, MSFT)", "AAPL")
capital_initial = st.number_input("Capital initial ($)", value=10000)
short_window = st.number_input("SMA courte", value=20)
long_window = st.number_input("SMA longue", value=50)
stop_loss_pct = st.number_input("Stop Loss (%)", value=5)

if st.button("Lancer le bot"):

    data = yf.download(ticker, period="1y")

    if data.empty:
        st.error("Ticker invalide ou données indisponibles.")
    else:
        data["SMA_short"] = data["Close"].rolling(int(short_window)).mean()
        data["SMA_long"] = data["Close"].rolling(int(long_window)).mean()

        data["Signal"] = 0
        data["Signal"][short_window:] = \
            (data["SMA_short"][short_window:] > data["SMA_long"][short_window:]).astype(int)

        capital = capital_initial
        position = 0
        entry_price = 0
        trades = []

        for i in range(len(data)):
            price = data["Close"].iloc[i]

            # Achat
            if data["Signal"].iloc[i] == 1 and position == 0:
                position = capital / price
                entry_price = price
                capital = 0
                trades.append((data.index[i], "BUY", price))

            # Stop Loss
            elif position > 0 and price < entry_price * (1 - stop_loss_pct / 100):
                capital = position * price
                position = 0
                trades.append((data.index[i], "STOP LOSS", price))

            # Vente
            elif data["Signal"].iloc[i] == 0 and position > 0:
                capital = position * price
                position = 0
                trades.append((data.index[i], "SELL", price))

        # Valeur finale
        if position > 0:
            capital = position * data["Close"].iloc[-1]

        st.subheader("Résultat final")
        st.success(f"Capital final : ${capital:,.2f}")

        # --- Graphique ---
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Prix")
        ax.plot(data.index, data["SMA_short"], label="SMA courte")
        ax.plot(data.index, data["SMA_long"], label="SMA longue")

        for trade in trades:
            if trade[1] == "BUY":
                ax.scatter(trade[0], trade[2], marker="^")
            else:
                ax.scatter(trade[0], trade[2], marker="v")

        ax.legend()
        st.pyplot(fig)
