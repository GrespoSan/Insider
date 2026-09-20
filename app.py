from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from engine import (
    Quarter,
    backtest_signals,
    build_issuer_day_signals,
    combine_quarters,
    download_quarter,
    latest_completed_quarter,
    quarter_range,
    normalize_loaded_signals,
)


# v0.9.1 deployment-safe helpers.
# Kept locally so the app can still start if Streamlit Cloud temporarily serves
# an older cached engine.py from v0.8.
def normalize_loaded_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a previously exported insider_event_study.csv for advanced analysis."""
    if df is None or df.empty:
        raise ValueError("Il CSV event study è vuoto.")
    out = df.copy()
    required = {"issuer_cik", "signal_date", "cluster"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError("CSV event study non compatibile. Colonne mancanti: " + ", ".join(missing))
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    if out["signal_date"].isna().all():
        raise ValueError("signal_date non contiene date valide.")
    def _to_bool(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if pd.isna(v):
            return False
        t = str(v).strip().lower()
        return t in {"true", "1", "yes", "y", "si", "sì"}
    out["cluster"] = out["cluster"].map(_to_bool)
    out["issuer_cik"] = out["issuer_cik"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    if "n_insiders" in out.columns:
        out["n_insiders"] = pd.to_numeric(out["n_insiders"], errors="coerce").fillna(1).astype(int)
    else:
        out["n_insiders"] = np.where(out["cluster"], 2, 1)
    return out


def label_cluster_episodes(
    df: pd.DataFrame,
    *,
    min_insiders: int = 2,
    episode_gap_days: int = 10,
) -> pd.DataFrame:
    """Label rows as SOLO, FIRST_CLUSTER or REPEAT_CLUSTER.

    The cluster threshold is applied to n_insiders already computed using the signal file's
    original transaction window. `episode_gap_days` only defines when a later cluster signal
    is considered a new episode; it does NOT change the original transaction-window width.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["issuer_cik"] = out["issuer_cik"].fillna("").astype(str)
    n = pd.to_numeric(out.get("n_insiders", 1), errors="coerce").fillna(1).astype(int)
    qualifies = n >= int(min_insiders)
    out["analysis_group"] = "SOLO"
    out["cluster_episode_id"] = pd.NA

    for issuer, idx in out[qualifies].groupby("issuer_cik", sort=False).groups.items():
        ordered = out.loc[list(idx)].sort_values("signal_date")
        episode = 0
        prev_date = None
        for ridx, row in ordered.iterrows():
            d = pd.Timestamp(row["signal_date"])
            is_first = prev_date is None or (d.normalize() - prev_date.normalize()).days > int(episode_gap_days)
            if is_first:
                episode += 1
                out.at[ridx, "analysis_group"] = "FIRST_CLUSTER"
            else:
                out.at[ridx, "analysis_group"] = "REPEAT_CLUSTER"
            out.at[ridx, "cluster_episode_id"] = f"{issuer}:{episode}"
            prev_date = d
    return out


def advanced_group_summary(
    event: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    *,
    min_insiders: int = 2,
    episode_gap_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return episode-labelled event data and descriptive summary for 3 groups."""
    labelled = label_cluster_episodes(event, min_insiders=min_insiders, episode_gap_days=episode_gap_days)
    rows = []
    group_order = ["SOLO", "FIRST_CLUSTER", "REPEAT_CLUSTER"]
    for group in group_order:
        sub = labelled[labelled["analysis_group"].eq(group)]
        for h in horizons:
            c = f"excess_{h}"
            if c not in sub.columns:
                continue
            x = pd.to_numeric(sub[c], errors="coerce").dropna()
            if x.empty:
                continue
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            trimmed = x[(x >= lo) & (x <= hi)]
            rows.append({
                "group": group,
                "horizon_sessions": h,
                "n": int(x.size),
                "n_issuers": int(sub.loc[x.index, "issuer_cik"].astype(str).nunique()),
                "mean_excess": float(x.mean()),
                "trimmed_mean_excess_1pct": float(trimmed.mean()) if not trimmed.empty else np.nan,
                "median_excess": float(x.median()),
                "win_rate_excess": float((x > 0).mean()),
                "p01_excess": float(lo),
                "p99_excess": float(hi),
            })
    return labelled, pd.DataFrame(rows)


def issuer_cluster_bootstrap_difference(
    event: pd.DataFrame,
    *,
    group_a: str = "FIRST_CLUSTER",
    group_b: str = "SOLO",
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    min_insiders: int = 2,
    episode_gap_days: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Issuer-cluster bootstrap for mean excess difference group_a - group_b.

    Resamples issuer clusters with replacement and aggregates all observations belonging to
    sampled issuers, preserving within-issuer dependence. This is intentionally a robust
    comparison, not a trading recommendation.
    """
    labelled = label_cluster_episodes(event, min_insiders=min_insiders, episode_gap_days=episode_gap_days)
    rng = np.random.default_rng(seed)
    issuer_ids = labelled["issuer_cik"].dropna().astype(str).unique()
    if len(issuer_ids) == 0:
        return pd.DataFrame()
    rows = []
    for h in horizons:
        col = f"excess_{h}"
        if col not in labelled.columns:
            continue
        d = labelled[["issuer_cik", "analysis_group", col]].copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=[col])
        if d.empty:
            continue
        agg = d.groupby(["issuer_cik", "analysis_group"])[col].agg(["sum", "count"]).reset_index()
        issuer_pos = {u:i for i,u in enumerate(issuer_ids)}
        sums_a = np.zeros(len(issuer_ids)); counts_a = np.zeros(len(issuer_ids))
        sums_b = np.zeros(len(issuer_ids)); counts_b = np.zeros(len(issuer_ids))
        for _, r in agg.iterrows():
            i = issuer_pos.get(str(r["issuer_cik"]))
            if i is None:
                continue
            if r["analysis_group"] == group_a:
                sums_a[i] += float(r["sum"]); counts_a[i] += float(r["count"])
            elif r["analysis_group"] == group_b:
                sums_b[i] += float(r["sum"]); counts_b[i] += float(r["count"])
        obs_a = d.loc[d.analysis_group.eq(group_a), col]
        obs_b = d.loc[d.analysis_group.eq(group_b), col]
        if obs_a.empty or obs_b.empty:
            continue
        observed = float(obs_a.mean() - obs_b.mean())
        boots = []
        for _ in range(int(n_boot)):
            sample = rng.integers(0, len(issuer_ids), size=len(issuer_ids))
            ca = counts_a[sample].sum(); cb = counts_b[sample].sum()
            if ca <= 0 or cb <= 0:
                continue
            ma = sums_a[sample].sum() / ca
            mb = sums_b[sample].sum() / cb
            boots.append(ma - mb)
        if not boots:
            continue
        b = np.asarray(boots, dtype=float)
        rows.append({
            "comparison": f"{group_a} - {group_b}",
            "horizon_sessions": h,
            "observed_diff": observed,
            "ci95_low": float(np.quantile(b, 0.025)),
            "ci95_high": float(np.quantile(b, 0.975)),
            "bootstrap_reps": int(len(b)),
            "n_issuers": int(len(issuer_ids)),
            "robustly_above_zero": bool(np.quantile(b, 0.025) > 0),
        })
    return pd.DataFrame(rows)



def _role_flags(series: pd.Series) -> pd.DataFrame:
    """Derive conservative role flags from the SEC role strings already in the signal export."""
    x = series.fillna("").astype(str).str.upper()
    return pd.DataFrame({
        "has_ceo": x.str.contains(r"\\bCEO\\b|CHIEF EXECUTIVE", regex=True, na=False),
        "has_cfo": x.str.contains(r"\\bCFO\\b|CHIEF FINANCIAL", regex=True, na=False),
        "has_director": x.str.contains("DIRECTOR", regex=False, na=False),
        "has_president": x.str.contains(r"\\bPRESIDENT\\b", regex=True, na=False),
        "has_vp": x.str.contains(r"VICE PRESIDENT|\\bVP\\b", regex=True, na=False),
    }, index=series.index)


def enrich_first_cluster_features(labelled: pd.DataFrame) -> pd.DataFrame:
    out = labelled.copy()
    out["cluster_value"] = pd.to_numeric(out.get("cluster_value"), errors="coerce")
    out["n_insiders"] = pd.to_numeric(out.get("n_insiders"), errors="coerce").fillna(1).astype(int)
    if "value_review" in out.columns:
        def _b(v):
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {"true","1","yes","y","si","sì"}
        out["value_review"] = out["value_review"].map(_b)
    else:
        out["value_review"] = False
    roles = out["roles"] if "roles" in out.columns else pd.Series("", index=out.index)
    flags = _role_flags(roles)
    for c in flags.columns:
        out[c] = flags[c]
    out["has_ceo_or_cfo"] = out["has_ceo"] | out["has_cfo"]
    out["has_ceo_and_cfo"] = out["has_ceo"] & out["has_cfo"]
    bins = [-np.inf, 50_000, 100_000, 250_000, 1_000_000, np.inf]
    labels = ["< $50k", "$50k–100k", "$100k–250k", "$250k–1M", "> $1M"]
    out["value_bucket"] = pd.cut(out["cluster_value"], bins=bins, labels=labels, right=False)
    return out


def subset_summary(df: pd.DataFrame, group_name: str, horizons=(1,5,21,63)) -> pd.DataFrame:
    rows=[]
    for h in horizons:
        col=f"excess_{h}"
        if col not in df.columns:
            continue
        x=pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            continue
        lo,hi=x.quantile(.01),x.quantile(.99)
        tr=x[(x>=lo)&(x<=hi)]
        rows.append({
            "segment": group_name,
            "horizon_sessions": h,
            "n": int(x.size),
            "n_issuers": int(df.loc[x.index,"issuer_cik"].astype(str).nunique()),
            "mean_excess": float(x.mean()),
            "trimmed_mean_excess_1pct": float(tr.mean()) if not tr.empty else np.nan,
            "median_excess": float(x.median()),
            "win_rate_excess": float((x>0).mean()),
        })
    return pd.DataFrame(rows)


def filtered_first_cluster_bootstrap(
    labelled: pd.DataFrame,
    mask: pd.Series,
    *,
    horizons=(1,5,21,63),
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Issuer-cluster bootstrap: selected FIRST_CLUSTER segment minus SOLO."""
    d=labelled.copy()
    selected = d[mask & d["analysis_group"].eq("FIRST_CLUSTER")].copy()
    solo = d[d["analysis_group"].eq("SOLO")].copy()
    issuer_ids = pd.Index(pd.concat([selected["issuer_cik"], solo["issuer_cik"]]).dropna().astype(str).unique())
    if selected.empty or solo.empty or len(issuer_ids)==0:
        return pd.DataFrame()
    rng=np.random.default_rng(seed)
    pos={u:i for i,u in enumerate(issuer_ids)}
    rows=[]
    for h in horizons:
        col=f"excess_{h}"
        if col not in d.columns:
            continue
        a=selected[["issuer_cik",col]].copy(); b=solo[["issuer_cik",col]].copy()
        a[col]=pd.to_numeric(a[col],errors="coerce"); b[col]=pd.to_numeric(b[col],errors="coerce")
        a=a.dropna(subset=[col]); b=b.dropna(subset=[col])
        if a.empty or b.empty:
            continue
        sa=np.zeros(len(issuer_ids)); ca=np.zeros(len(issuer_ids)); sb=np.zeros(len(issuer_ids)); cb=np.zeros(len(issuer_ids))
        for u,g in a.groupby(a["issuer_cik"].astype(str)):
            i=pos[u]; sa[i]=g[col].sum(); ca[i]=g[col].count()
        for u,g in b.groupby(b["issuer_cik"].astype(str)):
            i=pos[u]; sb[i]=g[col].sum(); cb[i]=g[col].count()
        obs=float(a[col].mean()-b[col].mean())
        boots=[]
        for _ in range(int(n_boot)):
            ix=rng.integers(0,len(issuer_ids),size=len(issuer_ids))
            na=ca[ix].sum(); nb=cb[ix].sum()
            if na<=0 or nb<=0:
                continue
            boots.append(sa[ix].sum()/na - sb[ix].sum()/nb)
        if not boots:
            continue
        arr=np.asarray(boots)
        low=float(np.quantile(arr,.025)); high=float(np.quantile(arr,.975))
        rows.append({
            "comparison":"SELECTED_FIRST_CLUSTER - SOLO",
            "horizon_sessions":h,
            "observed_diff":obs,
            "ci95_low":low,
            "ci95_high":high,
            "bootstrap_reps":len(arr),
            "n_selected":len(a),
            "n_selected_issuers":a["issuer_cik"].astype(str).nunique(),
            "robustly_above_zero":low>0,
        })
    return pd.DataFrame(rows)

st.set_page_config(page_title="Independent Insider Radar", layout="wide")
st.title("Independent Insider Radar — v0.10")
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

    st.divider()
    st.header("Riprendi da CSV")
    uploaded_signals = st.file_uploader(
        "Carica insider_signals_all.csv",
        type=["csv"],
        help="Consente di saltare il passo 2 e andare direttamente all'Event Study.",
    )
    if uploaded_signals is not None:
        if st.button("Usa CSV segnali caricato", use_container_width=True):
            try:
                loaded = pd.read_csv(uploaded_signals, low_memory=False)
                loaded = normalize_loaded_signals(loaded)
                st.session_state["signals"] = loaded
                st.session_state["components"] = None
                st.session_state["signals_source"] = "csv"
                # A newly loaded signal set invalidates any previous event-study output.
                st.session_state.pop("event", None)
                st.session_state.pop("summary", None)
                st.success(f"Segnali caricati: {len(loaded):,}")
                if loaded["cluster"].all():
                    st.warning(
                        "Il CSV contiene solo righe CLUSTER. Sembra una vista filtrata (es. insider_signals_view.csv): "
                        "l'Event Study può partire, ma non sarà possibile il confronto SOLO vs CLUSTER. "
                        "Per il test completo usa insider_signals_all.csv."
                    )
            except Exception as exc:
                st.error(f"CSV non valido: {exc}")

    uploaded_event = st.file_uploader(
        "Carica insider_event_study.csv (opzionale)",
        type=["csv"],
        help="Consente di saltare anche Yahoo e passare direttamente all'analisi FIRST_CLUSTER.",
        key="event_csv_uploader",
    )
    if uploaded_event is not None:
        if st.button("Usa Event Study caricato", use_container_width=True):
            try:
                ev = pd.read_csv(uploaded_event, low_memory=False)
                ev = normalize_loaded_event_study(ev)
                st.session_state["event"] = ev
                st.session_state["summary"] = pd.DataFrame()
                st.success(f"Eventi backtest caricati: {len(ev):,}. Puoi analizzare subito FIRST_CLUSTER.")
            except Exception as exc:
                st.error(f"Event Study CSV non valido: {exc}")

st.info(
    "Regola v0.9: Form 4 originale, transazione non-derivata con codice P e A (acquired), "
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
            st.session_state["signals_source"] = "sec"
            st.session_state.pop("event", None)
            st.session_state.pop("summary", None)

signals = st.session_state.get("signals")
components = st.session_state.get("components")

if isinstance(signals, pd.DataFrame) and not signals.empty:
    st.subheader("Risultato")
    source = st.session_state.get("signals_source", "sec")
    if source == "csv":
        st.success("Segnali caricati da CSV: il passo 2 è stato saltato. Puoi andare direttamente al punto 3.")
    m1, m2, m3, m4 = st.columns(4)
    component_count = len(components) if isinstance(components, pd.DataFrame) else None
    m1.metric("Componenti P puliti", f"{component_count:,}" if component_count is not None else "—")
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
    st.caption("Guardia anti-ticker riutilizzato/storico incompleto: l'entry Yahoo deve cadere entro 7 giorni di calendario dal filing SEC.")
    if st.button("3. Esegui event study 1/5/21/63 sedute"):
        progress = st.progress(0.0)
        status_box = st.empty()
        def _progress(done, total, rows_done, rows_total):
            progress.progress(min(done / max(total, 1), 1.0))
            status_box.caption(f"Yahoo: blocco {done}/{total} • eventi elaborati {rows_done:,}/{rows_total:,}")
        with st.spinner("Download prezzi a blocchi e calcolo progressivo..."):
            event, summary = backtest_signals(signals, max_entry_lag_days=7, progress_callback=_progress)
            st.session_state["event"] = event
            st.session_state["summary"] = summary
        progress.progress(1.0)
        status_box.success("Event study completato.")

summary = st.session_state.get("summary")
event = st.session_state.get("event")
if isinstance(summary, pd.DataFrame) and not summary.empty:
    if isinstance(event, pd.DataFrame) and not event.empty and "price_status" in event.columns:
        st.subheader("Copertura prezzi Yahoo")
        status = event["price_status"].value_counts(dropna=False)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Eventi prezzati", f"{int(status.get('ok', 0)):,}")
        c2.metric("Senza ticker", f"{int(status.get('missing_ticker', 0)):,}")
        c3.metric("Prezzo Yahoo mancante", f"{int(status.get('missing_price', 0)):,}")
        c4.metric("Ticker/storico incoerente", f"{int(status.get('stale_symbol_or_gap', 0)):,}")
        other_errors = int(status[status.index.astype(str).str.startswith("row_error_")].sum()) if len(status) else 0
        c5.metric("Record anomali", f"{other_errors:,}")
    st.subheader("Sintesi descrittiva")
    pretty = summary.copy()
    for c in ["mean_excess", "trimmed_mean_excess_1pct", "median_excess", "win_rate_excess", "p01_excess", "p99_excess"]:
        if c in pretty.columns:
            pretty[c] = (pretty[c] * 100).round(2)
    pretty = pretty.rename(columns={
        "mean_excess": "mean_excess_%",
        "trimmed_mean_excess_1pct": "trimmed_mean_1pct_%",
        "median_excess": "median_excess_%",
        "win_rate_excess": "win_rate_excess_%",
        "p01_excess": "p01_%",
        "p99_excess": "p99_%",
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.info(
        "v0.9 aggiunge FIRST_CLUSTER vs REPEAT_CLUSTER e bootstrap a livello issuer. "
        "La finestra di episodio sotto non cambia la finestra transazioni SEC originaria: decide solo quando due trigger cluster appartengono allo stesso episodio."
    )
    if isinstance(event, pd.DataFrame):
        st.download_button(
            "Scarica eventi backtest CSV",
            event.to_csv(index=False).encode("utf-8"),
            file_name="insider_event_study.csv",
            mime="text/csv",
        )


# --- Analisi avanzata v0.9: può partire anche da insider_event_study.csv caricato ---
event_adv = st.session_state.get("event")
if isinstance(event_adv, pd.DataFrame) and not event_adv.empty:
    st.divider()
    st.header("Analisi FIRST CLUSTER — v0.10")
    a1, a2 = st.columns(2)
    with a1:
        adv_min_insiders = st.selectbox("Soglia insider per cluster", [2, 3, 4], index=0, key="adv_min_insiders")
    with a2:
        episode_gap = st.selectbox("Nuovo episodio dopo (giorni)", [3, 5, 10, 20], index=2, key="episode_gap")
    st.caption(
        "Importante: la soglia insider può essere ricalcolata dal CSV perché n_insiders è disponibile. "
        "Il parametro giorni qui separa episodi successivi dello stesso issuer; NON sostituisce la finestra transazioni ±10 giorni usata per costruire i segnali originali."
    )
    labelled, adv_summary = advanced_group_summary(
        event_adv, min_insiders=int(adv_min_insiders), episode_gap_days=int(episode_gap)
    )
    if not adv_summary.empty:
        pretty_adv = adv_summary.copy()
        for c in ["mean_excess", "trimmed_mean_excess_1pct", "median_excess", "win_rate_excess", "p01_excess", "p99_excess"]:
            pretty_adv[c] = (pretty_adv[c] * 100).round(2)
        pretty_adv = pretty_adv.rename(columns={
            "mean_excess":"mean_excess_%",
            "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
            "median_excess":"median_excess_%",
            "win_rate_excess":"win_rate_excess_%",
            "p01_excess":"p01_%",
            "p99_excess":"p99_%",
        })
        st.subheader("SOLO vs FIRST_CLUSTER vs REPEAT_CLUSTER")
        st.dataframe(pretty_adv, use_container_width=True, hide_index=True)

        first_count = int((labelled["analysis_group"] == "FIRST_CLUSTER").sum())
        repeat_count = int((labelled["analysis_group"] == "REPEAT_CLUSTER").sum())
        solo_count = int((labelled["analysis_group"] == "SOLO").sum())
        x1, x2, x3 = st.columns(3)
        x1.metric("SOLO", f"{solo_count:,}")
        x2.metric("FIRST_CLUSTER", f"{first_count:,}")
        x3.metric("REPEAT_CLUSTER", f"{repeat_count:,}")

        st.subheader("Bootstrap issuer: FIRST_CLUSTER − SOLO")
        with st.spinner("Bootstrap per issuer (2.000 repliche)..."):
            boot = issuer_cluster_bootstrap_difference(
                event_adv,
                min_insiders=int(adv_min_insiders),
                episode_gap_days=int(episode_gap),
                n_boot=2000,
                seed=42,
            )
        if not boot.empty:
            boot_pretty = boot.copy()
            for c in ["observed_diff", "ci95_low", "ci95_high"]:
                boot_pretty[c] = (boot_pretty[c] * 100).round(2)
            boot_pretty = boot_pretty.rename(columns={
                "observed_diff":"diff_media_%",
                "ci95_low":"CI95_low_%",
                "ci95_high":"CI95_high_%",
                "robustly_above_zero":"CI_interamente_>0",
            })
            st.dataframe(boot_pretty, use_container_width=True, hide_index=True)
            robust = boot[boot["robustly_above_zero"]]
            if robust.empty:
                st.warning("Con questa configurazione nessun orizzonte ha un CI95% interamente sopra zero: il vantaggio FIRST_CLUSTER non è ancora robustamente distinto da SOLO.")
            else:
                hs = ", ".join(str(int(x)) for x in robust["horizon_sessions"].tolist())
                st.success(f"CI95% interamente sopra zero agli orizzonti: {hs} sedute.")

        st.download_button(
            "Scarica Event Study con etichette episodio",
            labelled.to_csv(index=False).encode("utf-8"),
            file_name="insider_event_study_first_cluster.csv",
            mime="text/csv",
        )

        st.subheader("Sensibilità soglia insider (stessa finestra SEC originaria)")
        sens_rows = []
        for thr in [2, 3, 4]:
            _, sm = advanced_group_summary(event_adv, min_insiders=thr, episode_gap_days=int(episode_gap))
            if sm.empty:
                continue
            fc = sm[sm["group"].eq("FIRST_CLUSTER")].copy()
            for _, r in fc.iterrows():
                sens_rows.append({
                    "min_insiders": thr,
                    "horizon_sessions": int(r["horizon_sessions"]),
                    "n": int(r["n"]),
                    "mean_excess_%": round(float(r["mean_excess"])*100,2),
                    "trimmed_mean_1pct_%": round(float(r["trimmed_mean_excess_1pct"])*100,2),
                    "median_excess_%": round(float(r["median_excess"])*100,2),
                    "win_rate_excess_%": round(float(r["win_rate_excess"])*100,2),
                })
        if sens_rows:
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

        st.caption(
            "Per testare davvero finestre transazioni SEC diverse (±3/±5/±10/±20 giorni) serve ricostruire i segnali dai componenti Form 4, "
            "perché il CSV event study conserva il risultato della finestra usata in origine ma non tutte le singole transazioni necessarie a ricalcolarla."
        )


        st.divider()
        st.header("Quali FIRST_CLUSTER contano davvero? — v0.10")
        st.caption(
            "Analisi esplorativa sui FIRST_CLUSTER già definiti. Nessuno score: separiamo il campione per controvalore e ruolo. "
            "Le righe value_review (controvalori SEC estremi marcati in precedenza) sono escluse dalle tabelle per valore."
        )
        enriched = enrich_first_cluster_features(labelled)
        fc_base = enriched[enriched["analysis_group"].eq("FIRST_CLUSTER")].copy()
        fc_clean_value = fc_base[~fc_base["value_review"]].copy()

        # Value buckets
        st.subheader("FIRST_CLUSTER per controvalore aggregato")
        val_rows=[]
        bucket_order=["< $50k", "$50k–100k", "$100k–250k", "$250k–1M", "> $1M"]
        for bucket in bucket_order:
            g=fc_clean_value[fc_clean_value["value_bucket"].astype(str).eq(bucket)]
            sm=subset_summary(g,bucket)
            if not sm.empty:
                val_rows.append(sm)
        if val_rows:
            vt=pd.concat(val_rows,ignore_index=True)
            vp=vt.copy()
            for c in ["mean_excess","trimmed_mean_excess_1pct","median_excess","win_rate_excess"]:
                vp[c]=(vp[c]*100).round(2)
            vp=vp.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.dataframe(vp,use_container_width=True,hide_index=True)

        # Role buckets
        st.subheader("FIRST_CLUSTER per ruolo presente")
        role_defs={
            "CEO presente": fc_base["has_ceo"],
            "CFO presente": fc_base["has_cfo"],
            "CEO + CFO": fc_base["has_ceo_and_cfo"],
            "Director presente": fc_base["has_director"],
            "3+ insider": fc_base["n_insiders"].ge(3),
            "3+ insider + CEO/CFO": fc_base["n_insiders"].ge(3) & fc_base["has_ceo_or_cfo"],
            "3+ insider + CEO+CFO": fc_base["n_insiders"].ge(3) & fc_base["has_ceo_and_cfo"],
        }
        role_rows=[]
        for name, m in role_defs.items():
            sm=subset_summary(fc_base[m],name)
            if not sm.empty:
                role_rows.append(sm)
        if role_rows:
            rt=pd.concat(role_rows,ignore_index=True)
            rp=rt.copy()
            for c in ["mean_excess","trimmed_mean_excess_1pct","median_excess","win_rate_excess"]:
                rp[c]=(rp[c]*100).round(2)
            rp=rp.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.dataframe(rp,use_container_width=True,hide_index=True)

        st.subheader("Filtro combinato + bootstrap vs SOLO")
        f1,f2,f3=st.columns(3)
        with f1:
            focus_min_n=st.selectbox("Insider minimi",[2,3,4],index=1,key="focus_min_n")
        with f2:
            focus_value=st.selectbox("Controvalore cluster",[
                "Tutti","< $50k","$50k–100k","$100k–250k","$250k–1M","> $1M"
            ],index=0,key="focus_value")
        with f3:
            focus_role=st.selectbox("Ruolo",[
                "Qualsiasi","CEO presente","CFO presente","CEO o CFO","CEO + CFO","Director presente"
            ],index=0,key="focus_role")

        mask = enriched["n_insiders"].ge(int(focus_min_n)) & enriched["analysis_group"].eq("FIRST_CLUSTER")
        if focus_value != "Tutti":
            mask &= enriched["value_bucket"].astype(str).eq(focus_value) & ~enriched["value_review"]
        if focus_role == "CEO presente": mask &= enriched["has_ceo"]
        elif focus_role == "CFO presente": mask &= enriched["has_cfo"]
        elif focus_role == "CEO o CFO": mask &= enriched["has_ceo_or_cfo"]
        elif focus_role == "CEO + CFO": mask &= enriched["has_ceo_and_cfo"]
        elif focus_role == "Director presente": mask &= enriched["has_director"]

        selected=enriched[mask].copy()
        st.metric("FIRST_CLUSTER selezionati",f"{len(selected):,}")
        sel_sm=subset_summary(selected,"SELECTED_FIRST_CLUSTER")
        if sel_sm.empty:
            st.warning("Nessun evento prezzato con questi filtri.")
        else:
            sp=sel_sm.copy()
            for c in ["mean_excess","trimmed_mean_excess_1pct","median_excess","win_rate_excess"]:
                sp[c]=(sp[c]*100).round(2)
            sp=sp.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.dataframe(sp,use_container_width=True,hide_index=True)
            with st.spinner("Bootstrap issuer del segmento selezionato vs SOLO..."):
                fb=filtered_first_cluster_bootstrap(enriched,mask,n_boot=2000,seed=42)
            if not fb.empty:
                fbp=fb.copy()
                for c in ["observed_diff","ci95_low","ci95_high"]:
                    fbp[c]=(fbp[c]*100).round(2)
                fbp=fbp.rename(columns={
                    "observed_diff":"diff_media_%",
                    "ci95_low":"CI95_low_%",
                    "ci95_high":"CI95_high_%",
                    "robustly_above_zero":"CI_interamente_>0",
                })
                st.dataframe(fbp,use_container_width=True,hide_index=True)
                if fb["robustly_above_zero"].any():
                    hs=", ".join(str(int(x)) for x in fb.loc[fb["robustly_above_zero"],"horizon_sessions"])
                    st.success(f"Per questo segmento il CI95% è interamente sopra zero a: {hs} sedute.")
                else:
                    st.warning("Per questo segmento il CI95% attraversa ancora zero a tutti gli orizzonti.")

        st.caption(
            "Attenzione al data-mining: queste segmentazioni sono esplorative. Un filtro che sembra migliore sullo stesso campione 2022–2026 "
            "deve essere validato fuori campione o con walk-forward prima di essere considerato un edge."
        )
