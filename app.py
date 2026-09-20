from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from engine import (
    Quarter,
    backtest_signals,
    build_issuer_day_signals,
    combine_quarters,
    download_quarter,
    latest_completed_quarter,
    quarter_range,
)

st.set_page_config(page_title="Independent Insider Radar", layout="wide")
st.title("Independent Insider Radar — v0.4")
st.caption("SEC Form 4 • acquisti P • dati ufficiali gratuiti • nessuno score proprietario")

DATA_DIR = Path("data/sec_form345")
DATA_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Dati SEC")
    email = st.text_input("Email per User-Agent SEC", placeholder="nome@email.it")
    st.caption("La SEC richiede che l'accesso automatizzato sia identificabile.")

    start_year = st.number_input("Anno iniziale", 2006, 2100, 2022, 1)
    start_q = st.selectbox("Trimestre iniziale", [1, 2, 3, 4], index=0)
    latest = latest_completed_quarter()
    end_year = st.number_input("Anno finale", 2006, 2100, latest.year, 1)
    end_q = st.selectbox("Trimestre finale", [1, 2, 3, 4], index=latest.quarter - 1)

    st.header("Filtro acquisti")
    min_trade = st.number_input("Valore minimo singolo componente ($)", 0, 10_000_000, 10_000, 5_000)
    managers_only = st.checkbox("Solo Officer/Director", value=True)
    exclude_10b5 = st.checkbox("Escludi filing marcati 10b5-1", value=True)

    st.header("Cluster")
    window_days = st.slider("Finestra transazioni (± giorni)", 1, 30, 10)
    min_insiders = st.slider("Insider distinti minimi", 2, 6, 2)

st.info(
    "Regola v0.4: Form 4 originale, transazione non-derivata con codice P e A (acquired), "
    "common/ordinary shares, prezzo e quantità positivi. I filing con più reporting owner vengono "
    "scartati perché il dataset piatto SEC non attribuisce ogni riga transazione a uno specifico owner."
)

quarters = quarter_range(int(start_year), int(start_q), int(end_year), int(end_q))

c1, c2 = st.columns(2)
with c1:
    if st.button("1. Scarica / aggiorna SEC", type="primary", use_container_width=True):
        if not email or "@" not in email:
            st.error("Inserisci prima una email valida per il User-Agent SEC.")
        else:
            ok, missing, errors = [], [], []
            bar = st.progress(0.0)
            for i, q in enumerate(quarters, start=1):
                try:
                    path = download_quarter(q, DATA_DIR, email)
                    (ok if path else missing).append(q.label)
                except Exception as exc:
                    errors.append(f"{q.label}: {exc}")
                bar.progress(i / len(quarters))
            st.success(f"Disponibili: {len(ok)} trimestri")
            if missing:
                st.warning("Non trovati sui server SEC: " + ", ".join(missing))
            if errors:
                st.error("\n".join(errors))

with c2:
    existing = [DATA_DIR / q.filename for q in quarters if (DATA_DIR / q.filename).exists()]
    st.metric("ZIP SEC locali", len(existing))

if st.button("2. Costruisci segnali", use_container_width=True):
    existing = [DATA_DIR / q.filename for q in quarters if (DATA_DIR / q.filename).exists()]
    if not existing:
        st.error("Nessun dataset SEC locale. Esegui prima il download.")
    else:
        with st.spinner("Pulizia Form 4 e costruzione eventi..."):
            comp = combine_quarters(
                existing,
                managers_only=managers_only,
                drop_joint_filings=True,
                exclude_10b5_1=exclude_10b5,
                min_trade_value=float(min_trade),
            )
            signals = build_issuer_day_signals(
                comp,
                window_days=int(window_days),
                min_insiders=int(min_insiders),
            )
            st.session_state["components"] = comp
            st.session_state["signals"] = signals

signals = st.session_state.get("signals")
components = st.session_state.get("components")

if isinstance(signals, pd.DataFrame) and not signals.empty:
    st.subheader("Risultato")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Componenti P puliti", f"{len(components):,}")
    m2.metric("Issuer-day pubblici", f"{len(signals):,}")
    m3.metric("Cluster", f"{int(signals['cluster'].sum()):,}")
    valid_cluster_tickers = signals.loc[signals.cluster & signals["ticker"].astype(str).str.strip().ne(""), "ticker"].nunique()
    m4.metric("Ticker cluster prezzabili", f"{valid_cluster_tickers:,}")

    q1, q2 = st.columns(2)
    q1.metric("Segnali senza ticker SEC", f"{int((signals['ticker_status'] == 'missing').sum()):,}")
    q2.metric("Valori SEC da rivedere (≥ $1B)", f"{int(signals['value_review'].sum()):,}")
    st.caption("Le righe senza ticker e i controvalori anomali restano nel dataset SEC: non vengono cancellati. Le prime non entrano nel backtest Yahoo; i secondi sono solo marcati per revisione e non ricevono alcun punteggio.")

    show_cluster_only = st.checkbox("Mostra solo cluster", value=True)
    view = signals[signals["cluster"]].copy() if show_cluster_only else signals.copy()
    view["cluster_value"] = view["cluster_value"].round(0)
    view["new_filing_value"] = view["new_filing_value"].round(0)
    st.dataframe(
        view[[
            "signal_date", "ticker", "ticker_status", "issuer_name", "cluster", "n_insiders",
            "cluster_value", "value_review", "new_filing_value", "window_start", "window_end",
            "roles", "owners", "mean_filing_lag_days", "accessions",
        ]],
        use_container_width=True,
        hide_index=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Scarica vista corrente CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="insider_signals_view.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Scarica tutti i segnali CSV",
            signals.to_csv(index=False).encode("utf-8"),
            file_name="insider_signals_all.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Backtest gratuito con Yahoo Finance")
    st.caption(
        "Ingresso conservativo: OPEN della prima seduta successiva alla filing date SEC. "
        "Excess return = rendimento titolo - SPY sullo stesso intervallo."
    )
    if st.button("3. Esegui event study 1/5/21/63 sedute"):
        with st.spinner("Download prezzi e calcolo..."):
            event, summary = backtest_signals(signals)
            st.session_state["event"] = event
            st.session_state["summary"] = summary

summary = st.session_state.get("summary")
event = st.session_state.get("event")
if isinstance(summary, pd.DataFrame) and not summary.empty:
    if isinstance(event, pd.DataFrame) and not event.empty and "price_status" in event.columns:
        st.subheader("Copertura prezzi Yahoo")
        status = event["price_status"].value_counts(dropna=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Eventi prezzati", f"{int(status.get('ok', 0)):,}")
        c2.metric("Senza ticker", f"{int(status.get('missing_ticker', 0)):,}")
        c3.metric("Prezzo Yahoo mancante", f"{int(status.get('missing_price', 0)):,}")
    st.subheader("Sintesi descrittiva")
    pretty = summary.copy()
    for c in ["mean_excess", "median_excess", "win_rate_excess"]:
        pretty[c] = (pretty[c] * 100).round(2)
    pretty = pretty.rename(columns={
        "mean_excess": "mean_excess_%",
        "median_excess": "median_excess_%",
        "win_rate_excess": "win_rate_excess_%",
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.warning(
        "v0.4 riporta statistiche descrittive, non significatività robusta. "
        "La fase successiva deve aggiungere intervalli di confidenza e confronto cluster-vs-solo "
        "con dipendenza per issuer e periodo."
    )
    if isinstance(event, pd.DataFrame):
        st.download_button(
            "Scarica eventi backtest CSV",
            event.to_csv(index=False).encode("utf-8"),
            file_name="insider_event_study.csv",
            mime="text/csv",
        )
