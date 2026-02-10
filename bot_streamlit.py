# bot_streamlit.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("Bot de Trading Simple")

# Saisie utilisateur
symbol = st.text_input("Ticker de l'action", "AAPL")
short_window = st.number_input("Moyenne courte", 5, 100, 20)
long_window = st.number_input("Moyenne longue", 10, 200, 50)
stop_loss = st.number_input("Stop loss (%)", 1, 50, 5) / 100
capital_initial = st.number_input("Capital initial", 100, 100000, 10000)

if st.button("Lancer le bot"):
    data = yf.download(symbol, start="2022-01-01", end="2024-01-01")
    data["SMA_short"] = data["Close"].rolling(short_window).mean()
    data["SMA_long"] = data["Close"].rolling(long_window).mean()
    data["Signal"] = 0
    data.loc[data["SMA_short"] > data["SMA_long"], "Signal"] = 1
    data["Position"] = data["Signal"].diff()

    capital = capital_initial
    position = 0
    entry_price = 0
    buy_dates, sell_dates, buy_prices, sell_prices = [], [], [], []

    for i in range(1, len(data)):
        price = float(data["Close"].iloc[i])
        signal = data["Position"].iloc[i]
        if signal == 1 and capital > 0:
            position = capital / price
            entry_price = price
            capital = 0
            buy_dates.append(data.index[i])
            buy_prices.append(price)
        if position > 0 and price < entry_price * (1 - stop_loss):
            capital = position * price
            position = 0
            sell_dates.append(data.index[i])
            sell_prices.append(price)
        if signal == -1 and position > 0:
            capital = position * price
            position = 0
            sell_dates.append(data.index[i])
            sell_prices.append(price)

    st.write(f"Valeur finale du portefeuille : {capital + position * float(data['Close'].iloc[-1]):.2f} $")

    # Graphique
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data.index, data["Close"], label="Close")
    ax.plot(data.index, data["SMA_short"], label=f"SMA {short_window}")
    ax.plot(data.index, data["SMA_long"], label=f"SMA {long_window}")
    ax.scatter(buy_dates, buy_prices, marker="^", color="green", label="BUY")
    ax.scatter(sell_dates, sell_prices, marker="v", color="red", label="SELL")
    ax.legend()
    st.pyplot(fig)