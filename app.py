from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from live_engine import (
    build_live_radar,
    enrich_live_prices,
    export_state_zip,
    import_state_zip,
    live_state_paths,
    load_live_components,
    state_summary,
    sync_live_sec,
)

VERSION = "1.0"
STATE_DIR = Path("data/live_v1")
paths = live_state_paths(STATE_DIR)
paths["root"].mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Independent Insider Radar LIVE", layout="wide")
st.title(f"Independent Insider Radar LIVE — v{VERSION}")
st.caption("SEC Form 4 • acquisti P • radar operativo indipendente • dati gratuiti • regole congelate dalla ricerca")

st.info(
    "Regole operative congelate: **WATCH = primo cluster ≥2 insider**, **CORE = primo cluster ≥3 insider**. "
    "Il tag **VALUE 100–250k (2026$)** è informativo e non elimina gli altri CORE. "
    "L'orizzonte empirico principale osservato nella ricerca è **5 sedute**."
)

with st.sidebar:
    st.header("SEC Live")
    email = st.text_input("Email per User-Agent SEC", placeholder="nome@email.it")
    lookback_days = st.selectbox("Lookback live iniziale (giorni calendario)", [14, 21, 30], index=1)
    max_per_run = st.selectbox("Form 4 massimi per esecuzione", [500, 1000, 1500, 2500], index=2)
    st.caption("Il sync usa gli indici giornalieri ufficiali EDGAR e salva checkpoint locali. Se restano filing pendenti, premi di nuovo Sync.")

    st.divider()
    st.subheader("Backup stato live")
    uploaded_state = st.file_uploader("Ripristina live_state_v1.zip", type=["zip"])
    if uploaded_state is not None and st.button("Ripristina stato"):
        try:
            restored = import_state_zip(uploaded_state.getvalue(), STATE_DIR)
            st.success("Ripristinati: " + ", ".join(restored))
            st.rerun()
        except Exception as exc:
            st.error(f"Ripristino non riuscito: {exc}")

summary = state_summary(STATE_DIR)
cols = st.columns(5)
cols[0].metric("Ultimo indice SEC", str(summary["latest_index_date"] or "—"))
cols[1].metric("Form 4 scoperti", f"{summary['discovered']:,}")
cols[2].metric("Processati", f"{summary['processed']:,}")
cols[3].metric("Pendenti", f"{summary['pending']:,}")
cols[4].metric("Componenti P", f"{summary['components']:,}")

st.subheader("1. Sincronizza SEC fino all'ultima data disponibile")
end_date = date.today()
start_date = end_date - timedelta(days=int(lookback_days))
st.caption(
    f"Intervallo richiesto: **{start_date.isoformat()} → {end_date.isoformat()}**. "
    "Il radar operativo usa solo una finestra recente: è sufficiente per la finestra cluster ±10 giorni e per l'orizzonte osservato di 5 sedute."
)

if st.button("1. Sync / continua SEC Live", type="primary", use_container_width=True):
    if not email or "@" not in email:
        st.error("Inserisci prima una email valida nella sidebar per il User-Agent SEC.")
    else:
        bar = st.progress(0.0)
        status_box = st.empty()
        def progress(stage, done, total, item, status):
            frac = 0.0 if total <= 0 else min(1.0, done / total)
            bar.progress(frac)
            if stage == "index":
                status_box.caption(f"Indici SEC: {done}/{total} • {item} • {status}")
            else:
                status_box.caption(f"Form 4: {done}/{total} • {item} • {status}")
        try:
            result = sync_live_sec(
                start_date=start_date,
                end_date=end_date,
                contact_email=email,
                state_dir=STATE_DIR,
                min_trade_value=10_000,
                max_filings_per_run=int(max_per_run),
                progress_callback=progress,
            )
            if result["complete"]:
                st.success(
                    f"Sync completo fino all'ultimo indice trovato ({result['latest_available_index_date']}). "
                    f"Form 4 processati: {result['processed_form4']:,}; componenti P: {result['components']:,}."
                )
            else:
                st.warning(
                    f"Checkpoint salvato. Restano **{result['pending_form4']:,} Form 4** da processare. "
                    "Premi di nuovo **Sync / continua** per proseguire."
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Sync SEC non riuscito: {type(exc).__name__}: {exc}")

st.divider()
st.subheader("2. Costruisci Radar")
components = load_live_components(STATE_DIR)
if components.empty:
    st.info("Nessun componente P live disponibile. Esegui prima il Sync SEC.")
else:
    if st.button("2. Costruisci / aggiorna Radar", use_container_width=True):
        radar = build_live_radar(components, window_days=10, episode_gap_days=10)
        radar.to_csv(paths["radar"], index=False)
        st.session_state["live_radar"] = radar
        st.success(f"Radar costruito: {len(radar):,} segnali issuer-day.")

if "live_radar" not in st.session_state and paths["radar"].exists():
    try:
        st.session_state["live_radar"] = pd.read_csv(paths["radar"], low_memory=False)
        st.session_state["live_radar"]["signal_date"] = pd.to_datetime(st.session_state["live_radar"]["signal_date"], errors="coerce")
    except Exception:
        pass

radar = st.session_state.get("live_radar", pd.DataFrame())
if not radar.empty:
    priority_counts = radar["priority"].value_counts()
    c = st.columns(4)
    c[0].metric("CORE first ≥3", int(priority_counts.get("CORE", 0)))
    c[1].metric("WATCH first ≥2", int(priority_counts.get("WATCH", 0)))
    c[2].metric("Repeat CORE", int(priority_counts.get("REPEAT_CORE", 0)))
    c[3].metric("Contesto completo", f"{100*radar['context_complete'].fillna(False).mean():.0f}%")

    st.subheader("3. Aggiorna prezzi Yahoo")
    if st.button("3. Aggiorna prezzi / performance", use_container_width=True):
        with st.spinner("Scarico prezzi recenti Yahoo per CORE/WATCH..."):
            priced = enrich_live_prices(radar)
        priced.to_csv(paths["prices"], index=False)
        st.session_state["live_priced"] = priced
        st.success("Prezzi aggiornati.")

if "live_priced" not in st.session_state and paths["prices"].exists():
    try:
        p = pd.read_csv(paths["prices"], low_memory=False)
        for col in ["signal_date", "entry_date", "price_date"]:
            if col in p.columns:
                p[col] = pd.to_datetime(p[col], errors="coerce")
        st.session_state["live_priced"] = p
    except Exception:
        pass

view = st.session_state.get("live_priced", radar)
if not view.empty:
    st.divider()
    st.header("Radar operativo")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.3, 1.3])
    with f1:
        priorities = st.multiselect(
            "Priorità",
            ["CORE", "WATCH", "REPEAT_CORE", "REPEAT", "SOLO"],
            default=["CORE", "WATCH"],
        )
    with f2:
        max_age = st.selectbox("Età massima segnale", [5, 10, 20, 45, 9999], index=2, format_func=lambda x: "Tutti" if x == 9999 else f"{x} giorni")
    with f3:
        value_only = st.checkbox("Solo VALUE 100–250k", value=False)
    with f4:
        context_only = st.checkbox("Solo contesto completo", value=True)

    filt = view.copy()
    filt["signal_date"] = pd.to_datetime(filt["signal_date"], errors="coerce")
    filt = filt[filt["priority"].isin(priorities)] if priorities else filt.iloc[0:0]
    if max_age != 9999:
        filt = filt[pd.to_numeric(filt["calendar_age_days"], errors="coerce").le(max_age)]
    if value_only:
        filt = filt[filt["value_tag"].fillna("").astype(str).ne("")]
    if context_only:
        filt = filt[filt["context_complete"].fillna(False).astype(bool)]

    # Display-friendly numeric columns while keeping raw CSV export below.
    display_cols = [
        "priority", "ticker", "issuer_name", "signal_date", "n_insiders", "cluster_value", "value_tag",
        "owners", "roles", "calendar_age_days",
    ]
    for optional in ["entry_open", "current_close", "sessions_observed", "return_since_entry", "excess_since_entry", "return_5", "excess_5", "operational_status"]:
        if optional in filt.columns:
            display_cols.append(optional)
    display_cols += ["sec_url", "tradingview_url"]
    display_cols = [c for c in display_cols if c in filt.columns]

    cfg = {
        "signal_date": st.column_config.DateColumn("Trigger SEC", format="DD/MM/YYYY"),
        "cluster_value": st.column_config.NumberColumn("Valore cluster", format="$ %.0f"),
        "entry_open": st.column_config.NumberColumn("Entry OPEN", format="$ %.2f"),
        "current_close": st.column_config.NumberColumn("Ultimo close", format="$ %.2f"),
        "return_since_entry": st.column_config.NumberColumn("Da entry", format="%.2f%%"),
        "excess_since_entry": st.column_config.NumberColumn("Excess vs SPY", format="%.2f%%"),
        "return_5": st.column_config.NumberColumn("Ret 5", format="%.2f%%"),
        "excess_5": st.column_config.NumberColumn("Excess 5", format="%.2f%%"),
        "sec_url": st.column_config.LinkColumn("SEC", display_text="SEC"),
        "tradingview_url": st.column_config.LinkColumn("TV", display_text="TV"),
    }
    shown = filt[display_cols].copy()
    for pct_col in ["return_since_entry", "excess_since_entry", "return_5", "excess_5"]:
        if pct_col in shown.columns:
            shown[pct_col] = pd.to_numeric(shown[pct_col], errors="coerce") * 100.0
    st.dataframe(shown, use_container_width=True, hide_index=True, column_config=cfg, height=520)

    st.caption(
        "CORE/WATCH sono classificazioni del pattern studiato, non raccomandazioni di investimento. "
        "TradingView è qui un semplice link di verifica grafica; il plugin TradingView di ChatGPT resta separato dall'app Streamlit."
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "Scarica radar filtrato CSV",
        data=filt.to_csv(index=False).encode("utf-8"),
        file_name="insider_live_radar_v1_0.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Backup stato live ZIP",
        data=export_state_zip(STATE_DIR),
        file_name="insider_live_state_v1.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()
with st.expander("Metodo congelato e limiti"):
    st.markdown(
        """
- **Fonte:** SEC EDGAR Form 4 originali. Il radar mantiene solo acquisti **P** di common/ordinary shares, acquisizione **A**, prezzo e quantità positivi.
- **Attribuzione prudente:** filing con più reporting owner vengono scartati; vengono mantenuti Officer/Director; 10b5-1 viene escluso quando esplicitamente marcato.
- **Componente minimo:** $10.000, come nella ricerca congelata.
- **Cluster:** finestra sulle transaction date di ±10 giorni, usando soltanto filing già pubblici alla data del segnale.
- **WATCH:** primo episodio che raggiunge almeno 2 insider distinti.
- **CORE:** primo episodio che raggiunge almeno 3 insider distinti.
- **VALUE:** $100k–250k in dollari 2026 è un tag, non un filtro obbligatorio.
- **Orizzonte empirico:** 5 sedute è risultato più robusto di 1 seduta nella replica storica; non implica che ogni segnale salirà.
- **Prezzi:** Yahoo è usato soltanto per il contesto operativo; ticker mancanti/delistati possono non essere prezzabili.
- **Live:** per non trasformare Streamlit in un crawler pesante, il sync recente è checkpointed e può richiedere più esecuzioni.
        """
    )
