import streamlit as st
import pandas as pd
import numpy as np

# =============================
# Configuration Streamlit
# =============================
st.set_page_config(page_title="Bot Trading (V5)", layout="wide")
st.title("🤖 Bot Trading — Démo publique (V5)")
st.caption("Backtests + signaux. Aucun ordre réel. Données: Stooq.")

# =============================
# Téléchargement données (robuste)
# =============================
@st.cache_data(ttl=60 * 60)
def telecharger_prix_safe(ticker: str) -> pd.DataFrame:
    """
    Télécharge des données journalières via Stooq.
    Retourne DataFrame vide si ticker invalide ou si Stooq ne répond pas comme prévu.
    """
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


def filtrer_dates(df: pd.DataFrame, debut: str, fin: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    return d[(d.index >= pd.to_datetime(debut)) & (d.index <= pd.to_datetime(fin))]


def cloture_mensuelle(df: pd.DataFrame) -> pd.Series:
    # Dernière clôture de chaque mois
    return df["Close"].resample("M").last().dropna()


# =============================
# Indicateurs (Single-asset)
# =============================
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


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =============================
# Backtest Single-asset
# =============================
def backtest_single_asset(df: pd.DataFrame, position: pd.Series, frais_bps: float) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["Close"].pct_change().fillna(0)

    pos = position.fillna(0).astype(float)
    d["position"] = pos.shift(1).fillna(0)  # exécution le lendemain
    d["trade"] = d["position"].diff().abs().fillna(0)

    fee = (frais_bps / 10000.0)
    d["frais"] = d["trade"] * fee
    d["strat_ret"] = d["position"] * d["ret"] - d["frais"]

    d["equity"] = (1 + d["strat_ret"]).cumprod()
    d["buy_hold"] = (1 + d["ret"]).cumprod()
    return d


def stats_depuis_equity(d: pd.DataFrame, ret_col: str, equity_col: str) -> dict:
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
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
        "Trades": trades,
    }


# =============================
# Stratégies Single-asset
# =============================
def strategie_ma(df: pd.DataFrame, rapide: int, lente: int) -> pd.Series:
    ma_rapide = df["Close"].rolling(rapide).mean()
    ma_lente = df["Close"].rolling(lente).mean()
    return (ma_rapide > ma_lente).astype(int)


def strategie_rsi(df: pd.DataFrame, periode: int, seuil_bas: float, seuil_haut: float) -> pd.Series:
    r = rsi(df["Close"], periode)
    pos = pd.Series(0, index=df.index, dtype=int)
    in_pos = 0
    for i in range(len(df)):
        if np.isnan(r.iloc[i]):
            pos.iloc[i] = in_pos
            continue
        if in_pos == 0 and r.iloc[i] < seuil_bas:
            in_pos = 1
        elif in_pos == 1 and r.iloc[i] > seuil_haut:
            in_pos = 0
        pos.iloc[i] = in_pos
    return pos


def strategie_macd(df: pd.DataFrame, fast: int, slow: int, sig: int) -> pd.Series:
    macd_line, signal_line, _ = macd(df["Close"], fast, slow, sig)
    return (macd_line > signal_line).astype(int)


# =============================
# Backtest Multi-asset (rotation mensuelle)
# =============================
def backtest_rotation_mensuelle(
    prix_journaliers: dict,
    debut: str,
    fin: str,
    top_k: int,
    actifs_risques: list,
    actifs_defensifs: list,
    filtre_marche_actif: bool,
    symbole_marche: str,
    ma_marche_mois: int,
    frais_bps: float,
):
    # 1) Clôtures mensuelles
    mensuel = {}
    for sym, df in prix_journaliers.items():
        d = filtrer_dates(df, debut, fin)
        if d.empty or "Close" not in d.columns:
            continue
        mensuel[sym] = cloture_mensuelle(d)

    if len(mensuel) < 2:
        raise ValueError("Pas assez de données mensuelles. Vérifie les tickers et la période.")

    mdf = pd.DataFrame(mensuel).dropna(how="any")

    # 2) Double momentum (3m + 6m)/2
    mom_3 = mdf.pct_change(3)
    mom_6 = mdf.pct_change(6)
    score_mom = (mom_3 + mom_6) / 2

    # 3) Filtre marché (mensuel)
    if filtre_marche_actif:
        if symbole_marche not in mdf.columns:
            raise ValueError("Le symbole marché (SPY) n'est pas présent dans les données.")
        mkt = mdf[symbole_marche]
        mkt_ma = mkt.rolling(ma_marche_mois).mean()
        risk_on = (mkt > mkt_ma).astype(int)
    else:
        risk_on = pd.Series(1, index=mdf.index)

    # 4) Poids mensuels
    poids = pd.DataFrame(0.0, index=mdf.index, columns=mdf.columns)

    for t in range(len(mdf.index)):
        date = mdf.index[t]

        # On attend au moins 6 mois pour que mom_6 soit défini
        if t < 6:
            continue

        univers = actifs_risques if int(risk_on.iloc[t]) == 1 else actifs_defensifs
        univers = [u for u in univers if u in mdf.columns]
        if not univers:
            continue

        ranks = score_mom.loc[date, univers].sort_values(ascending=False)
        picks = list(ranks.head(top_k).index)
        if not picks:
            continue

        w = 1.0 / len(picks)
        for p in picks:
            poids.loc[date, p] = w

    poids = poids.fillna(0.0)

    # 5) Rendements mensuels + frais sur turnover
    mret = mdf.pct_change().fillna(0.0)
    turnover = poids.diff().abs().sum(axis=1).fillna(0.0)
    frais = (frais_bps / 10000.0) * turnover

    strat_mret = (poids.shift(1).fillna(0.0) * mret).sum(axis=1) - frais
    equity_m = (1 + strat_mret).cumprod()

    # Référence buy & hold : SPY
    bh_mret = mret[symbole_marche] if symbole_marche in mret.columns else mret.mean(axis=1)
    bh_equity_m = (1 + bh_mret.fillna(0.0)).cumprod()

    out_m = pd.DataFrame(
        {
            "strat_ret_m": strat_mret,
            "equity_m": equity_m,
            "buy_hold_m": bh_equity_m,
            "turnover": turnover,
            "frais_m": frais,
            "risk_on": risk_on,
        },
        index=mdf.index,
    )

    # 6) Affichage quotidien (forward-fill mensuel)
    daily_index = filtrer_dates(prix_journaliers[symbole_marche], debut, fin).index
    out_d = pd.DataFrame(index=daily_index)
    out_d["equity"] = out_m["equity_m"].reindex(daily_index, method="ffill")
    out_d["buy_hold"] = out_m["buy_hold_m"].reindex(daily_index, method="ffill")
    out_d["strat_ret"] = out_d["equity"].pct_change().fillna(0.0)
    out_d["ret"] = out_d["buy_hold"].pct_change().fillna(0.0)

    return out_m, out_d, poids


# =============================
# UI
# =============================
st.sidebar.header("Mode")
mode = st.sidebar.selectbox(
    "Choisis un mode",
    ["Single-asset (MA / RSI / MACD)", "Multi-asset (Rotation mensuelle)"],
)

debut = st.sidebar.date_input("Début", value=pd.to_datetime("2010-01-01"))
fin = st.sidebar.date_input("Fin", value=pd.to_datetime("2025-12-31"))
frais_bps = st.sidebar.slider("Frais (bps)", 0, 50, 10)

lancer = st.sidebar.button("Lancer")
if not lancer:
    st.info("Choisis un mode et clique **Lancer**.")
    st.stop()

# =============================
# Mode 1: Single-asset
# =============================
if mode.startswith("Single-asset"):
    st.sidebar.subheader("Single-asset")
    ticker = st.sidebar.text_input("Ticker (Stooq)", value="aapl.us")

    strategie = st.sidebar.selectbox("Stratégie", ["Moyennes mobiles (MA)", "RSI (retour à la moyenne)", "MACD (tendance)"])

    if strategie.startswith("Moyennes"):
        rapide = st.sidebar.slider("MA rapide", 5, 60, 20)
        lente = st.sidebar.slider("MA lente", 20, 200, 100)
    elif strategie.startswith("RSI"):
        periode = st.sidebar.slider("RSI période", 5, 30, 14)
        seuil_bas = st.sidebar.slider("Seuil bas (entrée)", 5, 45, 30)
        seuil_haut = st.sidebar.slider("Seuil haut (sortie)", 55, 95, 70)
    else:
        macd_fast = st.sidebar.slider("MACD fast", 5, 20, 12)
        macd_slow = st.sidebar.slider("MACD slow", 15, 60, 26)
        macd_sig = st.sidebar.slider("MACD signal", 5, 20, 9)

    try:
        prix = telecharger_prix_safe(ticker)
        prix = filtrer_dates(prix, str(debut), str(fin))
        if prix.empty:
            st.error("Données indisponibles. Essaie un autre ticker.")
            st.stop()

        if strategie.startswith("Moyennes"):
            if rapide >= lente:
                st.error("MA rapide doit être < MA lente.")
                st.stop()
            pos = strategie_ma(prix, rapide, lente)
        elif strategie.startswith("RSI"):
            if seuil_bas >= seuil_haut:
                st.error("Seuil bas doit être < seuil haut.")
                st.stop()
            pos = strategie_rsi(prix, periode, seuil_bas, seuil_haut)
        else:
            if macd_fast >= macd_slow:
                st.error("MACD fast doit être < MACD slow.")
                st.stop()
            pos = strategie_macd(prix, macd_fast, macd_slow, macd_sig)

        d = backtest_single_asset(prix, pos, frais_bps)
        s = stats_depuis_equity(d, ret_col="strat_ret", equity_col="equity")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
        c2.metric("CAGR", f"{s['CAGR']*100:.2f}%")
        c3.metric("Sharpe", f"{s['Sharpe']:.2f}")
        c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
        c5.metric("Trades", f"{s['Trades']}")

        st.subheader("Équité (Stratégie vs Buy & Hold)")
        st.line_chart(pd.DataFrame({"Stratégie": d["equity"], "Buy & Hold": d["buy_hold"]}))

        st.subheader("Dernier signal")
        last = d.dropna().iloc[-1]
        st.info(f"Au {last.name.date()} : {'📈 LONG' if last['position'] == 1 else '📉 CASH'}")

    except Exception as e:
        st.exception(e)

# =============================
# Mode 2: Multi-asset
# =============================
else:
    st.sidebar.subheader("Multi-asset (Rotation mensuelle)")

    # Tickers (univers étendu)
    spy = st.sidebar.text_input("SPY (US Large)", value="spy.us")
    qqq = st.sidebar.text_input("QQQ (Tech)", value="qqq.us")
    efa = st.sidebar.text_input("EFA (International développé)", value="efa.us")
    eem = st.sidebar.text_input("EEM (Émergents)", value="eem.us")
    vnq = st.sidebar.text_input("VNQ (Immobilier)", value="vnq.us")

    tlt = st.sidebar.text_input("TLT (Obligations longues)", value="tlt.us")
    ief = st.sidebar.text_input("IEF (Obligations moyennes)", value="ief.us")
    gld = st.sidebar.text_input("GLD (Or)", value="gld.us")

    top_k = st.sidebar.slider("Nombre d'actifs sélectionnés (Top)", 1, 3, 2)

    filtre_marche = st.sidebar.checkbox("Filtre marché actif (SPY au-dessus MA)", value=True)
    ma_marche = st.sidebar.slider("MA marché (mois)", 6, 24, 12)

    try:
        # Téléchargement
        prix = {
            spy: telecharger_prix_safe(spy),
            qqq: telecharger_prix_safe(qqq),
            efa: telecharger_prix_safe(efa),
            eem: telecharger_prix_safe(eem),
            vnq: telecharger_prix_safe(vnq),
            tlt: telecharger_prix_safe(tlt),
            ief: telecharger_prix_safe(ief),
            gld: telecharger_prix_safe(gld),
        }

        # Nettoyage: garder seulement ceux qui ont des données sur la période
        prix_ok = {}
        for sym, df in prix.items():
            d = filtrer_dates(df, str(debut), str(fin))
            if d is not None and not d.empty and "Close" in d.columns:
                prix_ok[sym] = df
        prix = prix_ok

        if spy not in prix:
            st.error("SPY n'a pas de données (ticker ou dates). Essaie spy.us et ajuste la période.")
            st.stop()

        actifs_risques = [spy, qqq, efa, eem, vnq]
        actifs_defensifs = [tlt, ief, gld]

        # Ajuste selon ceux qui ont des données
        actifs_risques = [x for x in actifs_risques if x in prix]
        actifs_defensifs = [x for x in actifs_defensifs if x in prix]

        out_m, out_d, poids = backtest_rotation_mensuelle(
            prix_journaliers=prix,
            debut=str(debut),
            fin=str(fin),
            top_k=top_k,
            actifs_risques=actifs_risques,
            actifs_defensifs=actifs_defensifs,
            filtre_marche_actif=filtre_marche,
            symbole_marche=spy,
            ma_marche_mois=ma_marche,
            frais_bps=frais_bps,
        )

        s = stats_depuis_equity(out_d, ret_col="strat_ret", equity_col="equity")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rendement total", f"{s['Rendement total']*100:.2f}%")
        c2.metric("CAGR", f"{s['CAGR']*100:.2f}%")
        c3.metric("Sharpe", f"{s['Sharpe']:.2f}")
        c4.metric("Max DD", f"{s['Max drawdown']*100:.2f}%")
        c5.metric("Trades (approx.)", f"{int(out_m['turnover'].sum()):d}")

        st.subheader("Équité (Rotation vs Buy & Hold SPY)")
        st.line_chart(pd.DataFrame({"Rotation": out_d["equity"], "Buy & Hold (SPY)": out_d["buy_hold"]}))

        st.subheader("Poids mensuels (dernier 18 mois)")
        st.dataframe(poids.tail(18), use_container_width=True)

        st.subheader("Journal mensuel (dernier 24 mois)")
        st.dataframe(out_m.tail(24), use_container_width=True)

        # Message automatique (prêt à publier)
        st.subheader("📝 Message automatique (prêt à publier)")

        valid = out_m.dropna()
        if valid.empty:
            st.warning("Pas assez de données pour générer un message (essaie une période plus longue ou ajuste les paramètres).")
        else:
            last_date = valid.index[-1]
            last_row = out_m.loc[last_date]
            last_weights = poids.loc[last_date]

            alloc = last_weights[last_weights > 0].sort_values(ascending=False)
            alloc_text = ", ".join([f"{sym.upper()} {int(weight*100)}%" for sym, weight in alloc.items()])
            regime_text = "RISK-ON ✅" if int(last_row["risk_on"]) == 1 else "RISK-OFF 🛡️"

            msg = f"""📊 Rapport stratégie — Rotation Multi-Asset (Double Momentum)

Période backtest : {debut} → {fin}

Performance :
• CAGR : {s['CAGR']*100:.2f}%
• Sharpe : {s['Sharpe']:.2f}
• Max Drawdown : {s['Max drawdown']*100:.2f}%

État actuel ({pd.to_datetime(last_date).date()}) :
• Régime marché : {regime_text}
• Allocation : {alloc_text}
• Filtre marché (MA) : {ma_marche} mois
• Frais : {frais_bps} bps

Note : Ceci est un backtest (simulation) et ne constitue pas un conseil financier.
"""

            st.text_area("Message", msg, height=260)
            st.download_button("Télécharger le message (.txt)", msg, file_name="rapport_rotation.txt")

    except Exception as e:
        st.exception(e)
