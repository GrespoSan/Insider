from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from live_engine import (
    add_operational_columns,
    build_live_radar,
    enrich_live_prices,
    export_state_zip,
    import_state_zip,
    forward_registry_summary,
    load_forward_registry,
    load_forward_registry_meta,
    update_forward_registry,
    live_state_paths,
    load_live_components,
    state_summary,
    sync_live_sec,
)

VERSION = "1.3.1"
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

st.success(
    "**Uso quotidiano:** 1) Sync SEC fino a **Pendenti = 0** → 2) **Costruisci Radar** → "
    "3) **Aggiorna prezzi** → 4) concentra l'attenzione su **NUOVI CORE + CORE ATTIVI**. "
    "WATCH è una preallerta; COMPLETATI alimentano soprattutto il Forward Registry."
)

with st.sidebar:
    st.header("SEC Live")
    email = st.text_input("Email per User-Agent SEC", placeholder="nome@email.it")
    lookback_days = st.selectbox("Lookback live iniziale (giorni calendario)", [14, 21, 30], index=1)
    max_per_run = st.selectbox("Form 4 massimi per esecuzione", [500, 1000, 1500, 2500], index=2)
    st.caption("Il sync usa gli indici giornalieri ufficiali EDGAR e salva checkpoint locali. Se restano filing pendenti, premi di nuovo Sync.")

    st.divider()
    st.subheader("Backup stato live")
    uploaded_state = st.file_uploader("Ripristina live_state_v1*.zip", type=["zip"])
    if uploaded_state is not None and st.button("Ripristina stato"):
        try:
            restored = import_state_zip(uploaded_state.getvalue(), STATE_DIR)
            st.success("Ripristinati: " + ", ".join(restored))
            st.rerun()
        except Exception as exc:
            st.error(f"Ripristino non riuscito: {exc}")

    st.divider()
    st.subheader("Uso quotidiano")
    st.markdown(
        """
**1. Sync SEC**  
Continua finché **Pendenti = 0**.

**2. Costruisci Radar**  
Aggiorna CORE e WATCH.

**3. Aggiorna prezzi**  
Aggiorna stato 0/5 → 5/5 e performance.

**4. Guarda solo**  
**NUOVI CORE + CORE ATTIVI**.

Poi usa **TV** per il grafico TradingView o porta il ticker in ChatGPT per l'analisi grafica.
        """
    )
    st.caption("CORE = priorità principale (≥3 insider). WATCH = preallerta (2 insider). COMPLETATI servono soprattutto al Forward Registry.")

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
    "Il radar operativo usa una finestra recente sufficiente per il cluster ±10 giorni e l'orizzonte osservato di 5 sedute."
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
        # A rebuilt radar invalidates a previous price snapshot.
        st.session_state.pop("live_priced", None)
        if paths["prices"].exists():
            try:
                paths["prices"].unlink()
            except Exception:
                pass
        st.success(f"Radar costruito: {len(radar):,} segnali issuer-day.")

if "live_radar" not in st.session_state and paths["radar"].exists():
    try:
        r = pd.read_csv(paths["radar"], low_memory=False)
        r["signal_date"] = pd.to_datetime(r["signal_date"], errors="coerce")
        st.session_state["live_radar"] = r
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
            priced = add_operational_columns(priced)
        priced.to_csv(paths["prices"], index=False)
        st.session_state["live_priced"] = priced
        st.success("Prezzi aggiornati.")

if "live_priced" not in st.session_state and paths["prices"].exists():
    try:
        p = pd.read_csv(paths["prices"], low_memory=False)
        for col in ["signal_date", "entry_date", "price_date"]:
            if col in p.columns:
                p[col] = pd.to_datetime(p[col], errors="coerce")
        p = add_operational_columns(p)
        st.session_state["live_priced"] = p
    except Exception:
        pass

view = st.session_state.get("live_priced", radar)
if not view.empty:
    view = add_operational_columns(view)
    try:
        forward_registry = update_forward_registry(view, STATE_DIR)
    except Exception as exc:
        forward_registry = load_forward_registry(STATE_DIR)
        st.warning(f"Forward registry non aggiornato in questo passaggio: {type(exc).__name__}: {exc}")
    st.divider()
    st.header("Radar operativo")

    # Global filters are presentation-only and never change the frozen signal definition.
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.3, 1.3])
    with f1:
        priorities = st.multiselect("Priorità", ["CORE", "WATCH"], default=["CORE", "WATCH"])
    with f2:
        max_age = st.selectbox(
            "Età massima segnale",
            [5, 10, 20, 45, 9999],
            index=2,
            format_func=lambda x: "Tutti" if x == 9999 else f"{x} giorni",
        )
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

    # Daily focus: only NEW + ACTIVE CORE, i.e. the signals the user should inspect first.
    today_focus = filt[
        filt["priority"].eq("CORE")
        & filt["operational_bucket"].isin(["NUOVO", "ATTIVO"])
    ].copy()
    if not today_focus.empty:
        bucket_rank = {"NUOVO": 0, "ATTIVO": 1}
        today_focus["_bucket_rank"] = today_focus["operational_bucket"].map(bucket_rank).fillna(9)
        today_focus["_sessions_rank"] = pd.to_numeric(today_focus.get("sessions_observed"), errors="coerce").fillna(-1)
        today_focus = today_focus.sort_values(
            ["_bucket_rank", "_sessions_rank", "signal_date", "n_insiders"],
            ascending=[True, True, False, False],
        )

    st.subheader("🎯 DA GUARDARE OGGI")
    st.caption(
        "Questa è la lista principale: **solo NUOVI CORE e CORE ancora nella finestra 1–4/5 sedute**. "
        "**P/L da Entry** parte dall'OPEN della prima seduta successiva al filing SEC; **Vs SPY** mostra l'excess return sullo stesso intervallo. "
        "Per i segnali 0/5 i valori restano vuoti finché non esiste una seduta di ingresso. Usa **TV** per verificare il grafico e il contesto."
    )
    if today_focus.empty:
        st.info("Nessun NUOVO CORE o CORE ATTIVO con i filtri correnti.")
    else:
        focus_cols = [c for c in [
            "ticker", "issuer_name", "n_insiders", "day_5",
            "return_since_entry", "excess_since_entry", "cluster_value",
            "value_flag", "role_tag", "tradingview_url", "sec_url"
        ] if c in today_focus.columns]
        focus_show = today_focus[focus_cols].copy()
        # P/L e excess sono espressi come frazioni nel motore (0.034 = +3.4%).
        # Li convertiamo in punti percentuali solo per la visualizzazione.
        for pct_col in ["return_since_entry", "excess_since_entry"]:
            if pct_col in focus_show.columns:
                focus_show[pct_col] = pd.to_numeric(focus_show[pct_col], errors="coerce") * 100.0
        focus_cfg = {
            "ticker": st.column_config.TextColumn("Ticker"),
            "issuer_name": st.column_config.TextColumn("Società"),
            "n_insiders": st.column_config.NumberColumn("Insider", format="%d"),
            "day_5": st.column_config.TextColumn("Giorno"),
            "return_since_entry": st.column_config.NumberColumn("P/L da Entry %", format="%.2f%%"),
            "excess_since_entry": st.column_config.NumberColumn("Vs SPY %", format="%.2f%%"),
            "cluster_value": st.column_config.NumberColumn("Valore", format="$ %.0f"),
            "value_flag": st.column_config.TextColumn("VALUE"),
            "role_tag": st.column_config.TextColumn("Ruoli"),
            "tradingview_url": st.column_config.LinkColumn("TV", display_text="Apri TV"),
            "sec_url": st.column_config.LinkColumn("SEC", display_text="SEC"),
        }
        st.dataframe(
            focus_show,
            use_container_width=True,
            hide_index=True,
            column_config=focus_cfg,
            height=min(420, 74 + 35 * len(focus_show)),
        )

    st.success(
        "**Questa è la lista principale da controllare oggi.** "
        "Le sezioni successive sono il dettaglio del Radar: NUOVI, ATTIVI, COMPLETATI e anomalie."
    )

    st.markdown("### Dettaglio Radar")
    st.caption(
        "Qui sotto ritrovi gli stessi segnali organizzati per stato operativo. "
        "Usa questa parte per capire dove si trova ogni segnale nel ciclo 0/5 → 5/5, non come seconda lista di priorità."
    )

    # Status summary: these are workflow states, not trading recommendations.
    status_counts = filt["operational_bucket"].value_counts()
    m = st.columns(5)
    m[0].metric("NUOVI", int(status_counts.get("NUOVO", 0)))
    m[1].metric("ATTIVI 1–4/5", int(status_counts.get("ATTIVO", 0)))
    m[2].metric("COMPLETATI 5/5", int(status_counts.get("COMPLETATO", 0)))
    m[3].metric("DA VERIFICARE", int(status_counts.get("DA VERIFICARE", 0)))
    m[4].metric("DA PREZZARE", int(status_counts.get("DA PREZZARE", 0)))

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

    def render_block(title: str, subset: pd.DataFrame, *, completed: bool = False, new: bool = False):
        st.subheader(title)
        if subset.empty:
            st.caption("Nessun segnale in questa sezione con i filtri correnti.")
            return

        base_cols = [
            "priority", "ticker", "issuer_name", "signal_date", "n_insiders", "insider_band",
            "cluster_value", "value_flag", "role_tag", "day_5", "owners", "roles",
        ]
        if not new:
            base_cols += ["entry_open", "current_close", "sessions_observed", "return_since_entry", "excess_since_entry"]
        if completed:
            base_cols += ["return_5", "excess_5"]
        base_cols += ["sec_url", "tradingview_url"]
        cols_show = [c for c in base_cols if c in subset.columns]
        shown = subset[cols_show].copy()
        for pct_col in ["return_since_entry", "excess_since_entry", "return_5", "excess_5"]:
            if pct_col in shown.columns:
                shown[pct_col] = pd.to_numeric(shown[pct_col], errors="coerce") * 100.0
        st.dataframe(shown, use_container_width=True, hide_index=True, column_config=cfg, height=min(460, 74 + 35 * len(shown)))

    new_df = filt[filt["operational_bucket"].eq("NUOVO")].copy()
    active_df = filt[filt["operational_bucket"].eq("ATTIVO")].copy()
    completed_df = filt[filt["operational_bucket"].eq("COMPLETATO")].copy()
    verify_df = filt[filt["operational_bucket"].isin(["DA VERIFICARE", "DA PREZZARE"])].copy()

    render_block("NUOVI — in attesa della prima seduta", new_df, new=True)
    render_block("ATTIVI — finestra empirica 1–4/5 sedute", active_df)
    render_block("COMPLETATI — finestra 5/5 sedute", completed_df, completed=True)

    if not verify_df.empty:
        with st.expander(f"DA VERIFICARE / PREZZARE ({len(verify_df)})"):
            cols_show = [c for c in [
                "priority", "ticker", "issuer_name", "signal_date", "n_insiders", "cluster_value",
                "price_status", "context_complete", "sec_url", "tradingview_url"
            ] if c in verify_df.columns]
            st.dataframe(verify_df[cols_show], use_container_width=True, hide_index=True, column_config=cfg)

    st.caption(
        "CORE/WATCH sono classificazioni del pattern studiato, non raccomandazioni di investimento. "
        "CEO/CFO e VALUE sono informazioni descrittive: non modificano la regola CORE. "
        "TradingView è un link di verifica grafica; il plugin TradingView di ChatGPT resta separato da Streamlit."
    )

    # Exports with explicit scopes to avoid confusing a filtered view with the whole live universe.
    full_export = view[view["priority"].isin(["CORE", "WATCH"])].copy()
    operational_export = full_export[full_export["context_complete"].fillna(False).astype(bool)].copy()

    st.subheader("Export")
    st.caption(
        f"**Radar completo:** {len(full_export):,} CORE/WATCH, incluso contesto iniziale incompleto.  "
        f"**Radar operativo:** {len(operational_export):,} CORE/WATCH con contesto completo."
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Scarica radar completo CSV",
        data=full_export.to_csv(index=False).encode("utf-8"),
        file_name="insider_live_radar_complete_v1_3.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Scarica radar operativo CSV",
        data=operational_export.to_csv(index=False).encode("utf-8"),
        file_name="insider_live_radar_operational_v1_3.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d3.download_button(
        "Scarica vista filtrata CSV",
        data=filt.to_csv(index=False).encode("utf-8"),
        file_name="insider_live_radar_filtered_v1_3.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Backup stato live ZIP",
        data=export_state_zip(STATE_DIR),
        file_name="insider_live_state_v1_3.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()
st.header("Forward Registry — v1.3.2.1")
registry = load_forward_registry(STATE_DIR)
meta = load_forward_registry_meta(STATE_DIR)
if registry.empty:
    st.info("Il registro forward verrà inizializzato automaticamente appena esiste almeno un CORE/WATCH con contesto completo.")
else:
    reg = registry.copy()
    reg["frozen"] = reg["frozen"].fillna(False).astype(bool)
    origin = reg["tracking_origin"].fillna("").astype(str)
    fwd = reg[origin.eq("FORWARD")].copy()
    baseline = reg[origin.eq("BASELINE")].copy()
    fwd_active = fwd[~fwd["frozen"]].copy()
    fwd_done = fwd[fwd["frozen"]].copy()

    rcols = st.columns(5)
    rcols[0].metric("Baseline migrata", f"{len(baseline):,}")
    rcols[1].metric("Forward registrati", f"{len(fwd):,}")
    rcols[2].metric("Forward attivi", f"{len(fwd_active):,}")
    rcols[3].metric("Forward completati 5D", f"{len(fwd_done):,}")
    rcols[4].metric("Avvio registro", str(meta.get("initialized_at", "—"))[:10])

    st.caption(
        "I segnali già presenti al primo avvio della v1.2/v1.3/v1.3.1/v1.3.2 sono marcati **BASELINE** e restano separati dal vero test prospettico. "
        "Solo i segnali comparsi successivamente sono **FORWARD**. Una volta raggiunte 5 sedute, Ret 5 ed Excess 5 vengono congelati e non riscritti dai refresh successivi."
    )

    perf = forward_registry_summary(reg)
    if not perf.empty:
        show_perf = perf.copy()
        st.subheader("Risultati descrittivi del registro")
        st.dataframe(show_perf, use_container_width=True, hide_index=True)

    st.subheader("Ultimi segnali registrati")
    reg_show = reg.sort_values("signal_date", ascending=False).head(80).copy()
    for pct in ["current_return_since_entry", "current_excess_since_entry", "return_5", "excess_5"]:
        if pct in reg_show.columns:
            reg_show[pct] = pd.to_numeric(reg_show[pct], errors="coerce") * 100.0
    reg_cols = [c for c in [
        "tracking_origin", "priority", "ticker", "issuer_name", "signal_date", "n_insiders", "cluster_value",
        "value_flag", "role_tag", "sessions_observed", "operational_bucket", "current_excess_since_entry",
        "frozen", "exit_5_date", "excess_5", "sec_url", "tradingview_url"
    ] if c in reg_show.columns]
    reg_cfg = {
        "signal_date": st.column_config.DateColumn("Trigger SEC", format="DD/MM/YYYY"),
        "exit_5_date": st.column_config.DateColumn("5ª seduta", format="DD/MM/YYYY"),
        "cluster_value": st.column_config.NumberColumn("Valore cluster", format="$ %.0f"),
        "current_excess_since_entry": st.column_config.NumberColumn("Excess corrente", format="%.2f%%"),
        "excess_5": st.column_config.NumberColumn("Excess 5 congelato", format="%.2f%%"),
        "sec_url": st.column_config.LinkColumn("SEC", display_text="SEC"),
        "tradingview_url": st.column_config.LinkColumn("TV", display_text="TV"),
    }
    st.dataframe(reg_show[reg_cols], use_container_width=True, hide_index=True, column_config=reg_cfg)
    st.download_button(
        "Scarica Forward Registry CSV",
        data=reg.to_csv(index=False).encode("utf-8"),
        file_name="insider_forward_registry_v1_3_2.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
with st.expander("Metodo congelato e limiti"):
    st.markdown(
        """
- **Forward Registry v1.3.2:** il primo avvio crea una BASELINE separata; soltanto i segnali successivi sono FORWARD. A 5 sedute il risultato viene congelato e non viene riscritto.
- **Fonte:** SEC EDGAR Form 4 originali. Il radar mantiene solo acquisti **P** di common/ordinary shares, acquisizione **A**, prezzo e quantità positivi.
- **Attribuzione prudente:** filing con più reporting owner vengono scartati; vengono mantenuti Officer/Director; 10b5-1 viene escluso quando esplicitamente marcato.
- **Componente minimo:** $10.000, come nella ricerca congelata.
- **Cluster:** finestra sulle transaction date di ±10 giorni, usando soltanto filing già pubblici alla data del segnale.
- **WATCH:** primo episodio che raggiunge almeno 2 insider distinti.
- **CORE:** primo episodio che raggiunge almeno 3 insider distinti.
- **VALUE:** $100k–250k in dollari 2026 è un tag, non un filtro obbligatorio.
- **Orizzonte empirico:** 5 sedute è risultato più robusto di 1 seduta nella replica storica; non implica che ogni segnale salirà.
- **Stati v1.3.1:** NUOVO = nessuna seduta successiva ancora disponibile; ATTIVO = 1–4 sedute osservate; COMPLETATO = almeno 5 sedute; DA VERIFICARE = ticker/prezzo/storico non risolto; DA PREZZARE = snapshot Yahoo non ancora aggiornato.
- **Prezzi:** Yahoo è usato soltanto per il contesto operativo; ticker mancanti/delistati possono non essere prezzabili.
- **Live:** per non trasformare Streamlit in un crawler pesante, il sync recente è checkpointed e può richiedere più esecuzioni.
        """
    )
