from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import engine as engine_mod

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



def _period_summary(df: pd.DataFrame, period_name: str, config_name: str, horizons=(1,5)) -> pd.DataFrame:
    """Descriptive OOS summary for a frozen configuration in one time period."""
    rows=[]
    if df is None or df.empty:
        return pd.DataFrame()
    for h in horizons:
        c=f"excess_{h}"
        if c not in df.columns:
            continue
        x=pd.to_numeric(df[c], errors="coerce").dropna()
        if x.empty:
            continue
        lo,hi=x.quantile(.01),x.quantile(.99)
        trimmed=x[(x>=lo)&(x<=hi)]
        rows.append({
            "config":config_name,
            "period":period_name,
            "horizon_sessions":int(h),
            "n":int(x.size),
            "n_issuers":int(df.loc[x.index,"issuer_cik"].astype(str).nunique()),
            "mean_excess":float(x.mean()),
            "trimmed_mean_excess_1pct":float(trimmed.mean()) if not trimmed.empty else np.nan,
            "median_excess":float(x.median()),
            "win_rate_excess":float((x>0).mean()),
        })
    return pd.DataFrame(rows)


def frozen_oos_config(event: pd.DataFrame, *, config_name: str, min_insiders: int,
                      value_bucket: str|None=None, episode_gap_days: int=10,
                      discovery_start: str="2022-01-01", discovery_end: str="2024-12-31",
                      validation_start: str="2025-01-01", validation_end: str="2026-06-30"):
    """Apply one pre-declared rule to discovery and validation without retuning it."""
    labelled=label_cluster_episodes(event, min_insiders=min_insiders, episode_gap_days=episode_gap_days)
    enriched=enrich_first_cluster_features(labelled)
    mask=enriched["analysis_group"].eq("FIRST_CLUSTER")
    if value_bucket is not None:
        mask &= enriched["value_bucket"].astype(str).eq(value_bucket) & ~enriched["value_review"]
    selected=enriched[mask].copy()
    selected["signal_date"]=pd.to_datetime(selected["signal_date"], errors="coerce")
    disc_start=pd.Timestamp(discovery_start)
    disc_end=pd.Timestamp(discovery_end)
    val_start=pd.Timestamp(validation_start)
    val_end=pd.Timestamp(validation_end)
    discovery=selected[selected["signal_date"].between(disc_start,disc_end,inclusive="both")].copy()
    validation=selected[selected["signal_date"].between(val_start,val_end,inclusive="both")].copy()
    sm=pd.concat([
        _period_summary(discovery,"EARLY 2022–2024",config_name,(1,5)),
        _period_summary(validation,"LATE 2025–2026Q2",config_name,(1,5)),
    ],ignore_index=True)
    return enriched, selected, discovery, validation, sm


def validation_bootstrap_vs_solo(enriched: pd.DataFrame, selected_validation: pd.DataFrame, *,
                                 validation_start: str="2025-01-01", validation_end: str="2026-06-30", horizons=(1,5),
                                 n_boot: int=2000, seed: int=42) -> pd.DataFrame:
    """Issuer bootstrap in the holdout only: frozen selected FIRST_CLUSTER minus contemporaneous SOLO."""
    d=enriched.copy()
    d["signal_date"]=pd.to_datetime(d["signal_date"], errors="coerce")
    val_start=pd.Timestamp(validation_start)
    val_end=pd.Timestamp(validation_end)
    val=d[d["signal_date"].between(val_start,val_end,inclusive="both")].copy()
    selected_idx=set(selected_validation.index.tolist())
    mask=val.index.to_series().isin(selected_idx)
    return filtered_first_cluster_bootstrap(val, mask, horizons=horizons, n_boot=n_boot, seed=seed)


def build_frozen_oos_tables(event: pd.DataFrame, episode_gap_days: int=10):
    """Run the three rules frozen before looking at 2025–2026Q2 holdout."""
    configs=[
        ("A · FIRST_CLUSTER ≥2 insider",2,None),
        ("B · FIRST_CLUSTER ≥3 insider",3,None),
        ("C · ≥3 insider + $100k–250k",3,"$100k–250k"),
    ]
    summaries=[]; boots=[]; counts=[]
    for i,(name,thr,bucket) in enumerate(configs):
        enriched,selected,discovery,validation,sm=frozen_oos_config(
            event,config_name=name,min_insiders=thr,value_bucket=bucket,
            episode_gap_days=episode_gap_days,
        )
        if not sm.empty:
            summaries.append(sm)
        counts.append({
            "config":name,
            "discovery_events":len(discovery),
            "validation_events":len(validation),
            "discovery_issuers":discovery["issuer_cik"].astype(str).nunique() if not discovery.empty else 0,
            "validation_issuers":validation["issuer_cik"].astype(str).nunique() if not validation.empty else 0,
        })
        b=validation_bootstrap_vs_solo(enriched,validation,horizons=(1,5),n_boot=2000,seed=42+i)
        if not b.empty:
            b.insert(0,"config",name)
            boots.append(b)
    return (
        pd.concat(summaries,ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(boots,ignore_index=True) if boots else pd.DataFrame(),
        pd.DataFrame(counts),
    )



# --- v0.12: replica storica indipendente con regole congelate ---
HIST_START = pd.Timestamp("2006-01-01")
HIST_END = pd.Timestamp("2021-12-31")
HIST_CONFIGS = [
    ("A · FIRST_CLUSTER ≥2 insider", 2, None),
    ("B · FIRST_CLUSTER ≥3 insider", 3, None),
    ("C · ≥3 insider + $100k–250k", 3, "$100k–250k"),
]


def build_historical_replication_tables(event: pd.DataFrame, episode_gap_days: int = 10):
    """Apply only the three pre-declared rules to 2006-2021; no retuning allowed."""
    d = normalize_loaded_event_study(event)
    d = d[pd.to_datetime(d["signal_date"], errors="coerce").between(HIST_START, HIST_END, inclusive="both")].copy()
    summaries, boots, counts, eras = [], [], [], []
    era_defs = [
        ("2006–2010", pd.Timestamp("2006-01-01"), pd.Timestamp("2010-12-31")),
        ("2011–2015", pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
        ("2016–2021", pd.Timestamp("2016-01-01"), pd.Timestamp("2021-12-31")),
    ]
    for i, (name, thr, bucket) in enumerate(HIST_CONFIGS):
        labelled = label_cluster_episodes(d, min_insiders=thr, episode_gap_days=episode_gap_days)
        enriched = enrich_first_cluster_features(labelled)
        mask = enriched["analysis_group"].eq("FIRST_CLUSTER")
        if bucket is not None:
            mask &= enriched["value_bucket"].astype(str).eq(bucket) & ~enriched["value_review"]
        selected = enriched[mask].copy()
        sm = _period_summary(selected, "HISTORICAL 2006–2021", name, (1, 5))
        if not sm.empty:
            summaries.append(sm)
        priced1 = int(pd.to_numeric(selected.get("excess_1"), errors="coerce").notna().sum()) if "excess_1" in selected else 0
        priced5 = int(pd.to_numeric(selected.get("excess_5"), errors="coerce").notna().sum()) if "excess_5" in selected else 0
        counts.append({
            "config": name,
            "selected_events": int(len(selected)),
            "selected_issuers": int(selected["issuer_cik"].astype(str).nunique()) if not selected.empty else 0,
            "priced_1d": priced1,
            "coverage_1d_%": round(100 * priced1 / len(selected), 2) if len(selected) else np.nan,
            "priced_5d": priced5,
            "coverage_5d_%": round(100 * priced5 / len(selected), 2) if len(selected) else np.nan,
        })
        b = filtered_first_cluster_bootstrap(enriched, mask, horizons=(1, 5), n_boot=2000, seed=1200 + i)
        if not b.empty:
            b.insert(0, "config", name)
            boots.append(b)
        for era_name, lo, hi in era_defs:
            es = selected[pd.to_datetime(selected["signal_date"], errors="coerce").between(lo, hi, inclusive="both")]
            em = _period_summary(es, era_name, name, (1, 5))
            if not em.empty:
                eras.append(em)
    return (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(boots, ignore_index=True) if boots else pd.DataFrame(),
        pd.DataFrame(counts),
        pd.concat(eras, ignore_index=True) if eras else pd.DataFrame(),
    )


def historical_price_coverage(event: pd.DataFrame) -> pd.DataFrame:
    d = event.copy()
    d["signal_date"] = pd.to_datetime(d["signal_date"], errors="coerce")
    d = d[d["signal_date"].between(HIST_START, HIST_END, inclusive="both")]
    d["year"] = d["signal_date"].dt.year
    d["priced"] = pd.to_numeric(d.get("excess_1"), errors="coerce").notna()
    rows = []
    for y, g in d.groupby("year"):
        rows.append({
            "year": int(y),
            "events": int(len(g)),
            "priced": int(g["priced"].sum()),
            "coverage_%": round(100 * g["priced"].mean(), 2) if len(g) else np.nan,
            "missing_ticker": int((g.get("price_status", pd.Series(index=g.index, dtype=str)) == "missing_ticker").sum()),
            "missing_price": int((g.get("price_status", pd.Series(index=g.index, dtype=str)) == "missing_price").sum()),
            "stale_symbol_or_gap": int((g.get("price_status", pd.Series(index=g.index, dtype=str)) == "stale_symbol_or_gap").sum()),
        })
    return pd.DataFrame(rows)



def _event_summary_checkpoint_local(event: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Descriptive summary for the local deployment-safe checkpoint runner."""
    rows = []
    if event is None or event.empty:
        return pd.DataFrame()
    for cluster_value, subgroup in event.groupby("cluster", dropna=False):
        for h in horizons:
            col = f"excess_{h}"
            if col not in subgroup.columns:
                continue
            x = pd.to_numeric(subgroup[col], errors="coerce").dropna()
            if x.empty:
                continue
            lo, hi = x.quantile(0.01), x.quantile(0.99)
            trimmed = x[(x >= lo) & (x <= hi)]
            rows.append({
                "group": "CLUSTER" if (not pd.isna(cluster_value) and bool(cluster_value)) else "SOLO",
                "horizon_sessions": h,
                "n": int(x.size),
                "n_issuers": int(subgroup.loc[x.index, "issuer_cik"].astype(str).nunique()) if "issuer_cik" in subgroup.columns else 0,
                "mean_excess": float(x.mean()),
                "trimmed_mean_excess_1pct": float(trimmed.mean()) if not trimmed.empty else np.nan,
                "median_excess": float(x.median()),
                "win_rate_excess": float((x > 0).mean()),
                "p01_excess": float(lo),
                "p99_excess": float(hi),
            })
    return pd.DataFrame(rows)


def _signals_signature_checkpoint_local(sig: pd.DataFrame) -> str:
    import hashlib
    cols = [c for c in ["issuer_cik", "ticker", "signal_date", "accessions", "cluster_value", "n_insiders"] if c in sig.columns]
    core = sig[cols].copy() if cols else sig.copy()
    for c in core.columns:
        if "date" in c.lower():
            core[c] = pd.to_datetime(core[c], errors="coerce").astype(str)
    h = pd.util.hash_pandas_object(core.fillna("").astype(str), index=False).values.tobytes()
    return hashlib.sha256(h).hexdigest()


def backtest_signals_checkpointed_local(
    signals: pd.DataFrame,
    checkpoint_path: str | Path,
    horizons: tuple[int, ...] = (1, 5),
    benchmark: str = "SPY",
    batch_size: int = 40,
    max_entry_lag_days: int = 7,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deployment-safe fallback for v0.12 historical Yahoo backtest.

    It is intentionally kept in app.py so H3 still works when Streamlit Cloud serves an
    older engine.py that has the v0.8-v0.11 Yahoo helpers but not the new checkpoint wrapper.
    """
    import json

    if signals is None or signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_helpers = ["_naive_timestamp", "yahoo_symbol", "_download_yahoo_prices", "_safe_text", "_scalar_float"]
    missing = [name for name in required_helpers if not hasattr(engine_mod, name)]
    if missing:
        raise RuntimeError(
            "engine.py è troppo vecchio anche per il fallback v0.12.1. Mancano: " + ", ".join(missing)
        )

    naive_ts = engine_mod._naive_timestamp
    yahoo_symbol_fn = engine_mod.yahoo_symbol
    download_prices = engine_mod._download_yahoo_prices
    safe_text = engine_mod._safe_text
    scalar_float = engine_mod._scalar_float

    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cp.with_suffix(cp.suffix + ".meta.json")

    sig = signals.copy().reset_index(drop=True)
    sig["signal_date"] = sig["signal_date"].map(naive_ts)
    sig["yahoo_symbol"] = sig["ticker"].map(yahoo_symbol_fn)
    sig["_source_row_id"] = np.arange(len(sig), dtype=int)
    signature = _signals_signature_checkpoint_local(sig)

    existing = pd.DataFrame()
    if cp.exists() and cp.stat().st_size > 0:
        if not meta_path.exists():
            raise ValueError("Checkpoint storico presente ma senza metadata. Azzera il checkpoint e riparti.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signals_signature") != signature or tuple(meta.get("horizons", [])) != tuple(horizons):
            raise ValueError("Il checkpoint storico appartiene a un diverso set di segnali/regole. Azzera il checkpoint prima di continuare.")
        existing = pd.read_csv(cp, low_memory=False)
        if "_source_row_id" not in existing.columns:
            raise ValueError("Checkpoint storico non compatibile. Azzera il checkpoint e riparti.")
        existing["_source_row_id"] = pd.to_numeric(existing["_source_row_id"], errors="coerce").astype("Int64")
    else:
        meta_path.write_text(json.dumps({
            "signals_signature": signature,
            "horizons": list(horizons),
            "benchmark": benchmark,
            "max_entry_lag_days": int(max_entry_lag_days),
            "runner": "app-local-v0.12.1",
        }, indent=2), encoding="utf-8")

    completed_ids = set()
    if not existing.empty:
        completed_ids = set(pd.to_numeric(existing["_source_row_id"], errors="coerce").dropna().astype(int).tolist())
    pending = sig[~sig["_source_row_id"].isin(completed_ids)].copy()

    if pending.empty:
        event = existing.sort_values("_source_row_id").reset_index(drop=True)
        return event, _event_summary_checkpoint_local(event, horizons)

    start = sig["signal_date"].min() - pd.Timedelta(days=10)
    end = sig["signal_date"].max() + pd.Timedelta(days=max(horizons) * 2 + 30)
    bench_map = download_prices([benchmark], start, end, batch_size=1)
    bench = bench_map.get(benchmark, pd.DataFrame())
    if not bench.empty:
        bench = bench.copy()
        bench.index = pd.DatetimeIndex([naive_ts(x) for x in bench.index])

    def blank_base(row):
        base = row.to_dict()
        for h in horizons:
            base[f"ret_{h}"] = np.nan
            base[f"spy_{h}"] = np.nan
            base[f"excess_{h}"] = np.nan
        return base

    def persist(add_rows):
        nonlocal existing
        if not add_rows:
            return
        add = pd.DataFrame(add_rows)
        existing = pd.concat([existing, add], ignore_index=True) if not existing.empty else add
        existing = existing.drop_duplicates("_source_row_id", keep="last").sort_values("_source_row_id")
        tmp = cp.with_suffix(cp.suffix + ".tmp")
        existing.to_csv(tmp, index=False)
        tmp.replace(cp)

    missing_ticker = pending[pending["yahoo_symbol"].map(lambda x: not bool(safe_text(x)))]
    missing_rows = []
    for _, row in missing_ticker.iterrows():
        base = blank_base(row)
        base["price_status"] = "missing_ticker"
        missing_rows.append(base)
    persist(missing_rows)

    pending = pending[pending["yahoo_symbol"].map(lambda x: bool(safe_text(x)))].copy()
    valid_symbols = sorted(pending["yahoo_symbol"].astype(str).unique().tolist())
    n_batches = max(1, (len(valid_symbols) + batch_size - 1) // batch_size) if valid_symbols else 0

    for batch_no, pos in enumerate(range(0, len(valid_symbols), batch_size), start=1):
        batch = valid_symbols[pos:pos + batch_size]
        prices = download_prices(batch, start, end, batch_size=len(batch))
        batch_sig = pending[pending["yahoo_symbol"].isin(batch)]
        batch_rows = []

        for _, row in batch_sig.iterrows():
            try:
                sym = safe_text(row.get("yahoo_symbol", ""))
                base = blank_base(row)
                px = prices.get(sym, pd.DataFrame())
                if px.empty or bench.empty:
                    base["price_status"] = "missing_price"
                    batch_rows.append(base)
                    continue
                signal_ts = naive_ts(row.get("signal_date"))
                if pd.isna(signal_ts):
                    base["price_status"] = "invalid_signal_date"
                    batch_rows.append(base)
                    continue
                px = px.copy()
                px.index = pd.DatetimeIndex([naive_ts(x) for x in px.index])
                entry_candidates = px.index[px.index > signal_ts]
                if len(entry_candidates) == 0:
                    base["price_status"] = "no_future_session"
                    batch_rows.append(base)
                    continue
                entry_date = naive_ts(entry_candidates[0])
                entry_lag_days = int((entry_date.normalize() - signal_ts.normalize()).days)
                base["entry_lag_days"] = entry_lag_days
                base["entry_date"] = entry_date
                if entry_lag_days > int(max_entry_lag_days):
                    base["price_status"] = "stale_symbol_or_gap"
                    batch_rows.append(base)
                    continue
                if entry_date not in bench.index:
                    base["price_status"] = "benchmark_missing_entry"
                    batch_rows.append(base)
                    continue
                entry_open = scalar_float(px.loc[entry_date, "Open"])
                bench_open = scalar_float(bench.loc[entry_date, "Open"])
                if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(bench_open) or bench_open <= 0:
                    base["price_status"] = "bad_entry_price"
                    batch_rows.append(base)
                    continue
                loc = px.index.get_loc(entry_date)
                if not isinstance(loc, (int, np.integer)):
                    loc = int(np.flatnonzero(px.index == entry_date)[0])
                base["entry_open"] = entry_open
                base["price_status"] = "ok"
                for h in horizons:
                    exit_pos = int(loc) + h - 1
                    if exit_pos >= len(px.index):
                        continue
                    exit_date = px.index[exit_pos]
                    if exit_date not in bench.index:
                        continue
                    stock_close = scalar_float(px.iloc[exit_pos]["Close"])
                    spy_close = scalar_float(bench.loc[exit_date, "Close"])
                    if not np.isfinite(stock_close) or stock_close <= 0 or not np.isfinite(spy_close) or spy_close <= 0:
                        continue
                    ret = stock_close / entry_open - 1.0
                    spy_ret = spy_close / bench_open - 1.0
                    base[f"ret_{h}"] = ret
                    base[f"spy_{h}"] = spy_ret
                    base[f"excess_{h}"] = ret - spy_ret
                batch_rows.append(base)
            except Exception as exc:
                err = blank_base(row)
                err["price_status"] = f"row_error_{type(exc).__name__}"
                err["price_error"] = str(exc)[:240]
                batch_rows.append(err)

        persist(batch_rows)
        del prices
        if progress_callback is not None:
            try:
                progress_callback(batch_no, n_batches, len(existing), len(sig))
            except Exception:
                pass

    event = existing.sort_values("_source_row_id").reset_index(drop=True)
    return event, _event_summary_checkpoint_local(event, horizons)

st.set_page_config(page_title="Independent Insider Radar", layout="wide")
st.title("Independent Insider Radar — v0.14")
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
    "Regola base: Form 4 originale, transazione non-derivata con codice P e A (acquired), "
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
        "L’analisi distingue FIRST_CLUSTER vs REPEAT_CLUSTER e usa bootstrap a livello issuer. "
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

        st.divider()
        st.header("Stabilità temporale / pseudo-OOS — v0.11")
        st.caption(
            "Split temporale: 2022–2024 vs 2025–2026 Q2, solo 1 e 5 sedute. "
            "Serve a misurare la stabilità nel tempo delle tre configurazioni già emerse."
        )
        st.info(
            "Configurazioni congelate per questo split: A) FIRST_CLUSTER ≥2 insider; "
            "B) FIRST_CLUSTER ≥3 insider; C) FIRST_CLUSTER ≥3 insider con controvalore $100k–250k."
        )
        st.warning(
            "Nota metodologica importante: questo NON è un holdout completamente incontaminato, perché le configurazioni — soprattutto la fascia $100k–250k — "
            "sono emerse dopo aver già osservato il campione 2022–2026. Va quindi letto come test di stabilità temporale, non come conferma definitiva dell'edge. "
            "La conferma realmente indipendente richiederà un periodo mai usato per scegliere le regole (es. 2006–2021 come replica storica separata, oppure dati futuri post-2026Q2)."
        )
        with st.spinner("Validazione temporale + bootstrap issuer sul holdout..."):
            oos_summary,oos_boot,oos_counts=build_frozen_oos_tables(event_adv,episode_gap_days=int(episode_gap))

        if not oos_counts.empty:
            st.subheader("Dimensione campioni congelati")
            st.dataframe(oos_counts,use_container_width=True,hide_index=True)

        if not oos_summary.empty:
            st.subheader("Periodo iniziale vs periodo recente")
            op=oos_summary.copy()
            for c in ["mean_excess","trimmed_mean_excess_1pct","median_excess","win_rate_excess"]:
                op[c]=(op[c]*100).round(2)
            op=op.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.dataframe(op,use_container_width=True,hide_index=True)

            # compact persistence view, descriptive only
            piv=oos_summary.pivot_table(
                index=["config","horizon_sessions"],columns="period",values=["trimmed_mean_excess_1pct","median_excess","win_rate_excess"],aggfunc="first"
            )
            persist=[]
            for (cfg,h),row in piv.iterrows():
                try:
                    dtrim=float(row[("trimmed_mean_excess_1pct","EARLY 2022–2024")])
                    vtrim=float(row[("trimmed_mean_excess_1pct","LATE 2025–2026Q2")])
                    dmed=float(row[("median_excess","EARLY 2022–2024")])
                    vmed=float(row[("median_excess","LATE 2025–2026Q2")])
                    vwin=float(row[("win_rate_excess","LATE 2025–2026Q2")])
                except Exception:
                    continue
                persist.append({
                    "config":cfg,"horizon_sessions":int(h),
                    "discovery_trimmed_%":round(dtrim*100,2),
                    "validation_trimmed_%":round(vtrim*100,2),
                    "discovery_median_%":round(dmed*100,2),
                    "validation_median_%":round(vmed*100,2),
                    "validation_win_rate_%":round(vwin*100,2),
                    "segno_trimmed_preservato":bool((dtrim>0)==(vtrim>0)),
                })
            if persist:
                st.subheader("Persistenza descrittiva nel periodo recente")
                st.dataframe(pd.DataFrame(persist),use_container_width=True,hide_index=True)

        if not oos_boot.empty:
            st.subheader("Periodo recente 2025–2026 Q2: bootstrap issuer vs SOLO")
            bp=oos_boot.copy()
            for c in ["observed_diff","ci95_low","ci95_high"]:
                bp[c]=(bp[c]*100).round(2)
            bp=bp.rename(columns={
                "observed_diff":"diff_media_%",
                "ci95_low":"CI95_low_%",
                "ci95_high":"CI95_high_%",
                "robustly_above_zero":"CI_interamente_>0",
            })
            keep=[c for c in ["config","horizon_sessions","diff_media_%","CI95_low_%","CI95_high_%","bootstrap_reps","n_selected","n_selected_issuers","CI_interamente_>0"] if c in bp.columns]
            st.dataframe(bp[keep],use_container_width=True,hide_index=True)
            robust=oos_boot[oos_boot["robustly_above_zero"]]
            if robust.empty:
                st.warning(
                    "Nel periodo recente nessuna delle tre regole ha un CI95% issuer-level interamente sopra zero a 1 o 5 sedute. "
                    "Questo non annulla il pattern descrittivo, ma non consente di definirlo un edge robusto."
                )
            else:
                st.success(
                    "Nel periodo recente esistono configurazioni con CI95% issuer-level interamente sopra zero. "
                    "È un risultato interessante, ma resta esplorativo perché il periodo 2025–2026 era già stato osservato quando abbiamo scelto le configurazioni."
                )

        if not oos_summary.empty:
            export=oos_summary.copy()
            st.download_button(
                "Scarica validazione OOS CSV",
                export.to_csv(index=False).encode("utf-8"),
                file_name="insider_oos_validation_v0_11.csv",
                mime="text/csv",
            )

        st.caption(
            "Regola metodologica v0.11: da questo punto le tre configurazioni vengono congelate. Non aggiungiamo soglie o ruoli per migliorare il 2025–2026Q2. "
            "La prossima conferma dovrà arrivare da dati separati che non useremo per scegliere ulteriori filtri."
        )



# --- Historical Replication 2006-2021 (frozen rules) ---
st.divider()
st.header("Replica storica indipendente 2006–2021 — v0.13")
st.info(
    "Test realmente separato dal periodo 2022–2026 usato per esplorare le regole. "
    "Le configurazioni sono congelate: A = FIRST_CLUSTER ≥2; B = FIRST_CLUSTER ≥3; "
    "C = FIRST_CLUSTER ≥3 con controvalore aggregato $100k–250k. "
    "Finestra SEC ±10 giorni, nuovo episodio dopo 10 giorni, componenti ≥$10k, solo Officer/Director, esclusione 10b5-1 quando marcato, ingresso OPEN prima seduta successiva, orizzonti 1 e 5 sedute."
)
st.warning(
    "Non modificare le regole in base al risultato 2006–2021. Se una configurazione non replica, la consideriamo non stabile invece di cercare una nuova soglia che funzioni sullo storico."
)

HIST_DIR = Path("data/historical_replication_v0_12")
HIST_DIR.mkdir(parents=True, exist_ok=True)
HIST_COMPONENT_DIR = HIST_DIR / "components"
HIST_COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
HIST_SIGNALS_PATH = HIST_DIR / "insider_signals_2006_2021.csv"
HIST_EVENT_PATH = HIST_DIR / "insider_event_study_2006_2021.csv"
HIST_META_PATH = HIST_EVENT_PATH.with_suffix(HIST_EVENT_PATH.suffix + ".meta.json")
HIST_QUARTERS = quarter_range(2006, 1, 2021, 4)

local_hist_zips = [DATA_DIR / q.filename for q in HIST_QUARTERS if (DATA_DIR / q.filename).exists()]
hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Trimestri SEC storici", f"{len(local_hist_zips)}/64")
hc2.metric("Componenti trimestrali", f"{len(list(HIST_COMPONENT_DIR.glob('*.csv')))}/64")
hc3.metric("Segnali storici salvati", "Sì" if HIST_SIGNALS_PATH.exists() else "No")
hc4.metric("Checkpoint Yahoo", "Sì" if HIST_EVENT_PATH.exists() else "No")

h1, h2 = st.columns(2)
with h1:
    if st.button("H1. Scarica / aggiorna SEC 2006–2021", use_container_width=True):
        if not email or "@" not in email:
            st.error("Inserisci prima una email valida nel campo User-Agent SEC.")
        else:
            bar = st.progress(0.0)
            status = st.empty()
            errors = []
            for i, q in enumerate(HIST_QUARTERS, start=1):
                try:
                    download_quarter(q, DATA_DIR, email)
                except Exception as exc:
                    errors.append(f"{q.label}: {exc}")
                bar.progress(i / len(HIST_QUARTERS))
                status.caption(f"SEC storico: {i}/{len(HIST_QUARTERS)} • {q.label}")
            if errors:
                st.error("Alcuni trimestri non sono stati scaricati:\n" + "\n".join(errors[:12]))
            else:
                st.success("Tutti i 64 trimestri SEC 2006–2021 sono disponibili.")

with h2:
    if st.button("H2. Costruisci segnali storici congelati", use_container_width=True):
        paths = [DATA_DIR / q.filename for q in HIST_QUARTERS]
        missing = [q.label for q, pth in zip(HIST_QUARTERS, paths) if not pth.exists()]
        if missing:
            st.error(f"Mancano {len(missing)} trimestri SEC. Esegui prima H1. Primi mancanti: {', '.join(missing[:8])}")
        else:
            bar = st.progress(0.0)
            status = st.empty()
            for i, (q, pth) in enumerate(zip(HIST_QUARTERS, paths), start=1):
                comp_path = HIST_COMPONENT_DIR / f"{q.label}.csv"
                if not comp_path.exists() or comp_path.stat().st_size < 20:
                    s_q, o_q, t_q = engine_mod.load_quarter(pth)
                    comp_q = engine_mod.build_components(
                        s_q, o_q, t_q,
                        managers_only=True,
                        drop_joint_filings=True,
                        exclude_10b5_1=True,
                        min_trade_value=10_000,
                    )
                    comp_q.to_csv(comp_path, index=False)
                    del s_q, o_q, t_q, comp_q
                bar.progress(i / len(HIST_QUARTERS))
                status.caption(f"Componenti congelati: {i}/{len(HIST_QUARTERS)} • {q.label}")
            with st.spinner("Unisco i componenti e costruisco i segnali point-in-time..."):
                frames = []
                for q in HIST_QUARTERS:
                    f = pd.read_csv(HIST_COMPONENT_DIR / f"{q.label}.csv", low_memory=False)
                    for dc in ["FILING_DATE", "TRANS_DATE"]:
                        f[dc] = pd.to_datetime(f[dc], errors="coerce")
                    frames.append(f)
                comp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                if not comp.empty:
                    dedupe = ["ISSUERCIK", "RPTOWNERCIK", "TRANS_DATE", "shares", "trade_value", "DIRECT_INDIRECT_OWNERSHIP"]
                    comp = comp.sort_values("FILING_DATE").drop_duplicates(dedupe, keep="first").reset_index(drop=True)
                    hist_signals = build_issuer_day_signals(comp, window_days=10, min_insiders=2)
                    hist_signals = hist_signals[pd.to_datetime(hist_signals["signal_date"], errors="coerce").between(HIST_START, HIST_END, inclusive="both")]
                    hist_signals.to_csv(HIST_SIGNALS_PATH, index=False)
                    st.session_state["hist_signals_v012"] = hist_signals
                    st.success(f"Segnali storici costruiti: {len(hist_signals):,}")
                del frames, comp

st.subheader("Backtest storico Yahoo con checkpoint")
st.caption(
    "Il test 2006–2021 può richiedere tempo e Yahoo non conserva necessariamente tutti i ticker delistati. "
    "La v0.12.1 salva ogni blocco completato: se Streamlit si riavvia, il pulsante H3 riprende dal checkpoint invece di ricominciare da zero."
)
if hasattr(engine_mod, "backtest_signals_checkpointed"):
    st.caption("Runner checkpoint: engine.py v0.12 disponibile.")
else:
    st.info("Runner checkpoint: fallback locale v0.12.1 attivo. H3 può partire anche se Streamlit sta ancora caricando un engine.py precedente.")

uploaded_hist_event = st.file_uploader(
    "Oppure carica un insider_event_study_2006_2021.csv già completato",
    type=["csv"], key="hist_event_upload_v012"
)
if uploaded_hist_event is not None and st.button("Usa Event Study storico caricato", key="use_hist_event_v012"):
    try:
        hev = normalize_loaded_event_study(pd.read_csv(uploaded_hist_event, low_memory=False))
        st.session_state["hist_event_v012"] = hev
        st.success(f"Event Study storico caricato: {len(hev):,} righe.")
    except Exception as exc:
        st.error(f"CSV storico non compatibile: {exc}")

h3a, h3b = st.columns([3, 1])
with h3a:
    if st.button("H3. Esegui / riprendi Event Study storico 1/5 sedute", use_container_width=True):
        if not HIST_SIGNALS_PATH.exists():
            st.error("Prima costruisci i segnali storici con H2.")
        else:
            hist_signals = normalize_loaded_signals(pd.read_csv(HIST_SIGNALS_PATH, low_memory=False))
            pbar = st.progress(0.0)
            pbox = st.empty()
            def _hist_progress(done, total, rows_done, rows_total):
                pbar.progress(min(done / max(total, 1), 1.0))
                pbox.caption(f"Yahoo storico: blocco {done}/{total} • checkpoint {rows_done:,}/{rows_total:,} eventi")
            try:
                with st.spinner("Backtest storico a blocchi; i risultati vengono salvati dopo ogni batch..."):
                    checkpoint_runner = getattr(engine_mod, "backtest_signals_checkpointed", backtest_signals_checkpointed_local)
                    hev, hsum = checkpoint_runner(
                        hist_signals,
                        HIST_EVENT_PATH,
                        horizons=(1, 5),
                        batch_size=40,
                        max_entry_lag_days=7,
                        progress_callback=_hist_progress,
                    )
                st.session_state["hist_event_v012"] = normalize_loaded_event_study(hev)
                pbar.progress(1.0)
                pbox.success("Checkpoint storico aggiornato/completato.")
            except Exception as exc:
                st.error(f"Event Study storico interrotto: {exc}")
with h3b:
    if st.button("Azzera checkpoint Yahoo", use_container_width=True):
        for pp in [HIST_EVENT_PATH, HIST_META_PATH]:
            try:
                if pp.exists(): pp.unlink()
            except Exception:
                pass
        st.session_state.pop("hist_event_v012", None)
        st.success("Checkpoint storico azzerato.")

if HIST_SIGNALS_PATH.exists():
    try:
        _hs_n = sum(1 for _ in open(HIST_SIGNALS_PATH, "rb")) - 1
        st.caption(f"Segnali storici su disco: {_hs_n:,}")
        st.download_button(
            "Scarica segnali storici 2006–2021",
            HIST_SIGNALS_PATH.read_bytes(),
            file_name="insider_signals_2006_2021.csv",
            mime="text/csv",
        )
    except Exception:
        pass
if HIST_EVENT_PATH.exists():
    try:
        st.download_button(
            "Scarica checkpoint/Event Study storico",
            HIST_EVENT_PATH.read_bytes(),
            file_name="insider_event_study_2006_2021.csv",
            mime="text/csv",
        )
    except Exception:
        pass

hist_event = st.session_state.get("hist_event_v012")
if hist_event is None and HIST_EVENT_PATH.exists():
    # Do not auto-load a potentially large file on every rerun; user can explicitly analyze it.
    if st.button("H4. Carica checkpoint e analizza le regole congelate", use_container_width=True):
        try:
            hist_event = normalize_loaded_event_study(pd.read_csv(HIST_EVENT_PATH, low_memory=False))
            st.session_state["hist_event_v012"] = hist_event
        except Exception as exc:
            st.error(f"Impossibile leggere il checkpoint storico: {exc}")

hist_event = st.session_state.get("hist_event_v012")
if isinstance(hist_event, pd.DataFrame) and not hist_event.empty:
    total_expected = None
    if HIST_SIGNALS_PATH.exists():
        try:
            total_expected = sum(1 for _ in open(HIST_SIGNALS_PATH, "rb")) - 1
        except Exception:
            total_expected = None
    st.subheader("Risultato replica storica — regole congelate")
    if total_expected is not None and len(hist_event) < total_expected:
        st.warning(f"Checkpoint ancora parziale: {len(hist_event):,}/{total_expected:,} eventi. Completa H3 prima di interpretare i risultati.")
    else:
        coverage = historical_price_coverage(hist_event)
        if not coverage.empty:
            st.markdown("**Copertura Yahoo per anno**")
            st.dataframe(coverage, use_container_width=True, hide_index=True)
        with st.spinner("Bootstrap issuer-level delle tre configurazioni congelate..."):
            hsum, hboot, hcounts, heras = build_historical_replication_tables(hist_event, episode_gap_days=10)
        if not hcounts.empty:
            st.markdown("**Campioni e copertura delle configurazioni congelate**")
            st.dataframe(hcounts, use_container_width=True, hide_index=True)
        if not hsum.empty:
            hp = hsum.copy()
            for c in ["mean_excess", "trimmed_mean_excess_1pct", "median_excess", "win_rate_excess"]:
                hp[c] = (hp[c] * 100).round(2)
            hp = hp.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.markdown("**Replica 2006–2021: 1 e 5 sedute**")
            st.dataframe(hp, use_container_width=True, hide_index=True)
        if not hboot.empty:
            bp = hboot.copy()
            for c in ["observed_diff", "ci95_low", "ci95_high"]:
                bp[c] = (bp[c] * 100).round(2)
            bp = bp.rename(columns={
                "observed_diff":"diff_media_%",
                "ci95_low":"CI95_low_%",
                "ci95_high":"CI95_high_%",
                "robustly_above_zero":"CI_interamente_>0",
            })
            st.markdown("**Bootstrap issuer-level: configurazione congelata − SOLO**")
            keep = [c for c in ["config","horizon_sessions","diff_media_%","CI95_low_%","CI95_high_%","bootstrap_reps","n_selected","n_selected_issuers","CI_interamente_>0"] if c in bp.columns]
            st.dataframe(bp[keep], use_container_width=True, hide_index=True)
        if not heras.empty:
            ep = heras.copy()
            for c in ["mean_excess", "trimmed_mean_excess_1pct", "median_excess", "win_rate_excess"]:
                ep[c] = (ep[c] * 100).round(2)
            ep = ep.rename(columns={
                "mean_excess":"mean_excess_%",
                "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
                "median_excess":"median_excess_%",
                "win_rate_excess":"win_rate_excess_%",
            })
            st.markdown("**Diagnostica temporale predefinita (non usata per ottimizzare)**")
            st.dataframe(ep, use_container_width=True, hide_index=True)

        if not hsum.empty:
            criteria = []
            boot_lookup = {}
            if not hboot.empty:
                for _, r in hboot.iterrows():
                    boot_lookup[(r["config"], int(r["horizon_sessions"]))] = bool(r["robustly_above_zero"])
            for _, r in hsum.iterrows():
                key = (r["config"], int(r["horizon_sessions"]))
                criteria.append({
                    "config": r["config"],
                    "horizon_sessions": int(r["horizon_sessions"]),
                    "mean_>0": bool(r["mean_excess"] > 0),
                    "trimmed_>0": bool(r["trimmed_mean_excess_1pct"] > 0),
                    "median_>0": bool(r["median_excess"] > 0),
                    "win_rate_>50%": bool(r["win_rate_excess"] > 0.5),
                    "CI95_vs_SOLO_>0": boot_lookup.get(key, False),
                })
            st.markdown("**Criteri pre-dichiarati: direzione e robustezza**")
            st.dataframe(pd.DataFrame(criteria), use_container_width=True, hide_index=True)

        export_parts = []
        if not hsum.empty:
            x = hsum.copy(); x.insert(0, "table", "summary"); export_parts.append(x)
        if not hboot.empty:
            x = hboot.copy(); x.insert(0, "table", "bootstrap"); export_parts.append(x)
        if not hcounts.empty:
            x = hcounts.copy(); x.insert(0, "table", "counts"); export_parts.append(x)
        if export_parts:
            export = pd.concat(export_parts, ignore_index=True, sort=False)
            st.download_button(
                "Scarica replica storica v0.12 CSV",
                export.to_csv(index=False).encode("utf-8"),
                file_name="insider_historical_replication_v0_12.csv",
                mime="text/csv",
            )
        st.caption(
            "Limite strutturale: Yahoo può non avere prezzi per molti ticker delistati/storici. La tabella di copertura è parte integrante del risultato: "
            "una replica positiva con copertura molto bassa non va interpretata come prova definitiva."
        )


# --- v0.13: Robustness Audit of the frozen historical replication ---
ROBUST_AUDIT_CONFIGS = [
    ("A · FIRST_CLUSTER ≥2 insider", 2, None),
    ("B · FIRST_CLUSTER ≥3 insider", 3, None),
    ("C · ≥3 insider + $100k–250k", 3, "$100k–250k"),
]


def _audit_selected_sets(event: pd.DataFrame, episode_gap_days: int = 10) -> dict[str, pd.DataFrame]:
    d = normalize_loaded_event_study(event)
    d = d[pd.to_datetime(d["signal_date"], errors="coerce").between(HIST_START, HIST_END, inclusive="both")].copy()
    out = {}
    for name, thr, bucket in ROBUST_AUDIT_CONFIGS:
        labelled = label_cluster_episodes(d, min_insiders=thr, episode_gap_days=episode_gap_days)
        enriched = enrich_first_cluster_features(labelled)
        mask = enriched["analysis_group"].eq("FIRST_CLUSTER")
        if bucket is not None:
            mask &= enriched["value_bucket"].astype(str).eq(bucket) & ~enriched["value_review"]
        out[name] = enriched.loc[mask].copy()
    return out


def _audit_robust_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "n": 0, "mean_excess": np.nan, "trimmed_mean_1pct": np.nan,
            "winsorized_mean_1pct": np.nan, "median_excess": np.nan,
            "win_rate": np.nan, "p01": np.nan, "p99": np.nan,
            "min": np.nan, "max": np.nan, "mean_without_top1pct_winners": np.nan,
            "top1pct_share_of_positive_sum": np.nan,
        }
    p01, p99 = x.quantile(0.01), x.quantile(0.99)
    trimmed = x[(x >= p01) & (x <= p99)]
    wins = x.clip(lower=p01, upper=p99)
    no_top_winners = x[x <= p99]
    positive_sum = x[x > 0].sum()
    top_pos_sum = x[x > p99].clip(lower=0).sum()
    return {
        "n": int(len(x)),
        "mean_excess": float(x.mean()),
        "trimmed_mean_1pct": float(trimmed.mean()) if len(trimmed) else np.nan,
        "winsorized_mean_1pct": float(wins.mean()),
        "median_excess": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "p01": float(p01),
        "p99": float(p99),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean_without_top1pct_winners": float(no_top_winners.mean()) if len(no_top_winners) else np.nan,
        "top1pct_share_of_positive_sum": float(top_pos_sum / positive_sum) if positive_sum > 0 else np.nan,
    }


def _audit_yearly_tables(selected_sets: dict[str, pd.DataFrame], horizons=(1, 5)) -> pd.DataFrame:
    rows = []
    for config, s0 in selected_sets.items():
        s = s0.copy()
        s["signal_date"] = pd.to_datetime(s["signal_date"], errors="coerce")
        s["year"] = s["signal_date"].dt.year
        for year, g in s.groupby("year"):
            for h in horizons:
                col = f"excess_{h}"
                x = pd.to_numeric(g.get(col), errors="coerce").dropna()
                stats = _audit_robust_stats(x)
                priced_issuers = int(g.loc[x.index, "issuer_cik"].astype(str).nunique()) if len(x) else 0
                rows.append({
                    "config": config,
                    "year": int(year),
                    "horizon_sessions": int(h),
                    "selected_events": int(len(g)),
                    "selected_issuers": int(g["issuer_cik"].astype(str).nunique()),
                    "priced_events": int(len(x)),
                    "priced_issuers": priced_issuers,
                    "coverage_%": float(100 * len(x) / len(g)) if len(g) else np.nan,
                    "mean_excess": stats["mean_excess"],
                    "trimmed_mean_1pct": stats["trimmed_mean_1pct"],
                    "median_excess": stats["median_excess"],
                    "win_rate": stats["win_rate"],
                    "small_sample_flag": bool(len(x) < 30 or priced_issuers < 20),
                })
    return pd.DataFrame(rows)


def _audit_stress_periods(selected_sets: dict[str, pd.DataFrame], horizons=(1, 5)) -> pd.DataFrame:
    periods = [
        ("2008", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31")),
        ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
        ("Esclusi 2008 e 2020", HIST_START, HIST_END),
    ]
    rows = []
    for config, s0 in selected_sets.items():
        s = s0.copy()
        s["signal_date"] = pd.to_datetime(s["signal_date"], errors="coerce")
        for pname, lo, hi in periods:
            if pname == "Esclusi 2008 e 2020":
                g = s[~s["signal_date"].dt.year.isin([2008, 2020])]
            else:
                g = s[s["signal_date"].between(lo, hi, inclusive="both")]
            for h in horizons:
                x = pd.to_numeric(g.get(f"excess_{h}"), errors="coerce").dropna()
                stt = _audit_robust_stats(x)
                rows.append({
                    "config": config,
                    "period": pname,
                    "horizon_sessions": int(h),
                    "n": stt["n"],
                    "n_issuers": int(g.loc[x.index, "issuer_cik"].astype(str).nunique()) if len(x) else 0,
                    "mean_excess": stt["mean_excess"],
                    "trimmed_mean_1pct": stt["trimmed_mean_1pct"],
                    "median_excess": stt["median_excess"],
                    "win_rate": stt["win_rate"],
                })
    return pd.DataFrame(rows)


def _audit_outliers(selected_sets: dict[str, pd.DataFrame], horizons=(1, 5)) -> pd.DataFrame:
    rows = []
    for config, s in selected_sets.items():
        for h in horizons:
            stats = _audit_robust_stats(s.get(f"excess_{h}", pd.Series(dtype=float)))
            rows.append({"config": config, "horizon_sessions": int(h), **stats})
    return pd.DataFrame(rows)


def _audit_missing_data(selected_sets: dict[str, pd.DataFrame], horizons=(1, 5)) -> pd.DataFrame:
    rows = []
    for config, s in selected_sets.items():
        for h in horizons:
            col = f"excess_{h}"
            x = pd.to_numeric(s.get(col), errors="coerce")
            priced = x.notna()
            n_priced = int(priced.sum())
            n_missing = int((~priced).sum())
            observed_sum = float(x[priced].sum()) if n_priced else 0.0
            break_even_missing = (-observed_sum / n_missing) if n_missing else np.nan
            ps = s.get("price_status", pd.Series("", index=s.index)).fillna("").astype(str)
            rows.append({
                "config": config,
                "horizon_sessions": int(h),
                "selected_events": int(len(s)),
                "priced_events": n_priced,
                "missing_events": n_missing,
                "coverage_%": float(100 * n_priced / len(s)) if len(s) else np.nan,
                "observed_mean_excess": float(x[priced].mean()) if n_priced else np.nan,
                "break_even_missing_avg_excess_to_zero": break_even_missing,
                "missing_price": int(((~priced) & ps.eq("missing_price")).sum()),
                "missing_ticker": int(((~priced) & ps.eq("missing_ticker")).sum()),
                "stale_symbol_or_gap": int(((~priced) & ps.eq("stale_symbol_or_gap")).sum()),
                "other_missing": int(((~priced) & ~ps.isin(["missing_price", "missing_ticker", "stale_symbol_or_gap"])).sum()),
            })
    return pd.DataFrame(rows)


def _audit_bootstrap_two_masks(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    *,
    group_a: str,
    group_b: str,
    horizons=(1, 5),
    n_boot: int = 2000,
    seed: int = 1300,
) -> pd.DataFrame:
    rows = []
    base = df.copy()
    base["issuer_cik"] = base["issuer_cik"].fillna("").astype(str)
    ma = pd.Series(mask_a, index=base.index).fillna(False).astype(bool)
    mb = pd.Series(mask_b, index=base.index).fillna(False).astype(bool)
    issuer_ids = base.loc[ma | mb, "issuer_cik"].dropna().astype(str).unique()
    if len(issuer_ids) == 0:
        return pd.DataFrame()
    issuer_pos = {u: i for i, u in enumerate(issuer_ids)}
    rng = np.random.default_rng(seed)
    for h in horizons:
        col = f"excess_{h}"
        if col not in base.columns:
            continue
        vals = pd.to_numeric(base[col], errors="coerce")
        va = ma & vals.notna()
        vb = mb & vals.notna()
        if not va.any() or not vb.any():
            continue
        sums_a = np.zeros(len(issuer_ids)); counts_a = np.zeros(len(issuer_ids))
        sums_b = np.zeros(len(issuer_ids)); counts_b = np.zeros(len(issuer_ids))
        tmp = pd.DataFrame({"issuer_cik": base["issuer_cik"], "v": vals, "a": va, "b": vb})
        for issuer, g in tmp[(tmp["a"] | tmp["b"])].groupby("issuer_cik"):
            i = issuer_pos.get(str(issuer))
            if i is None:
                continue
            xa = g.loc[g["a"], "v"].dropna().to_numpy(dtype=float)
            xb = g.loc[g["b"], "v"].dropna().to_numpy(dtype=float)
            if xa.size:
                sums_a[i] = xa.sum(); counts_a[i] = xa.size
            if xb.size:
                sums_b[i] = xb.sum(); counts_b[i] = xb.size
        observed = float(vals[va].mean() - vals[vb].mean())
        boots = []
        for _ in range(int(n_boot)):
            sample = rng.integers(0, len(issuer_ids), size=len(issuer_ids))
            ca = counts_a[sample].sum(); cb = counts_b[sample].sum()
            if ca <= 0 or cb <= 0:
                continue
            boots.append((sums_a[sample].sum() / ca) - (sums_b[sample].sum() / cb))
        if not boots:
            continue
        b = np.asarray(boots, dtype=float)
        lo, hi = np.quantile(b, [0.025, 0.975])
        rows.append({
            "comparison": f"{group_a} − {group_b}",
            "horizon_sessions": int(h),
            "n_a": int(va.sum()),
            "n_b": int(vb.sum()),
            "observed_diff": observed,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bootstrap_reps": int(len(b)),
            "n_issuers": int(len(issuer_ids)),
            "CI_interamente_>0": bool(lo > 0),
        })
    return pd.DataFrame(rows)


def _audit_c_vs_b_other(event: pd.DataFrame, episode_gap_days: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = normalize_loaded_event_study(event)
    d = d[pd.to_datetime(d["signal_date"], errors="coerce").between(HIST_START, HIST_END, inclusive="both")].copy()
    labelled = label_cluster_episodes(d, min_insiders=3, episode_gap_days=episode_gap_days)
    enriched = enrich_first_cluster_features(labelled)
    bmask = enriched["analysis_group"].eq("FIRST_CLUSTER")
    cmask = bmask & enriched["value_bucket"].astype(str).eq("$100k–250k") & ~enriched["value_review"]
    bother = bmask & ~cmask
    desc = []
    for group_name, mask in [("C · $100k–250k", cmask), ("B restante · ≥3 fuori fascia C", bother)]:
        s = enriched.loc[mask]
        for h in (1, 5):
            stats = _audit_robust_stats(s.get(f"excess_{h}", pd.Series(dtype=float)))
            desc.append({
                "group": group_name,
                "horizon_sessions": h,
                "selected_events": int(len(s)),
                "n": stats["n"],
                "n_issuers": int(s.loc[pd.to_numeric(s.get(f"excess_{h}"), errors="coerce").notna(), "issuer_cik"].astype(str).nunique()) if f"excess_{h}" in s else 0,
                "mean_excess": stats["mean_excess"],
                "trimmed_mean_1pct": stats["trimmed_mean_1pct"],
                "median_excess": stats["median_excess"],
                "win_rate": stats["win_rate"],
            })
    boot = _audit_bootstrap_two_masks(
        enriched, cmask, bother,
        group_a="C $100k–250k", group_b="B restante",
        horizons=(1, 5), n_boot=2000, seed=1313,
    )
    return pd.DataFrame(desc), boot


def _audit_all_price_status_by_year(event: pd.DataFrame) -> pd.DataFrame:
    d = normalize_loaded_event_study(event)
    d["signal_date"] = pd.to_datetime(d["signal_date"], errors="coerce")
    d = d[d["signal_date"].between(HIST_START, HIST_END, inclusive="both")].copy()
    d["year"] = d["signal_date"].dt.year
    d["priced"] = pd.to_numeric(d.get("excess_5"), errors="coerce").notna()
    ps = d.get("price_status", pd.Series("", index=d.index)).fillna("").astype(str)
    d["_price_status"] = ps
    rows = []
    for year, g in d.groupby("year"):
        rows.append({
            "year": int(year),
            "events": int(len(g)),
            "priced": int(g["priced"].sum()),
            "coverage_%": float(100 * g["priced"].mean()) if len(g) else np.nan,
            "missing_price": int(g["_price_status"].eq("missing_price").sum()),
            "missing_ticker": int(g["_price_status"].eq("missing_ticker").sum()),
            "stale_symbol_or_gap": int(g["_price_status"].eq("stale_symbol_or_gap").sum()),
            "bad_entry_or_other": int((~g["priced"] & ~g["_price_status"].isin(["missing_price", "missing_ticker", "stale_symbol_or_gap"])).sum()),
        })
    return pd.DataFrame(rows)


st.divider()
st.header("Robustness Audit — v0.13 (conservato)")
st.caption(
    "Nessun nuovo filtro e nessun retuning. Questa sezione verifica le regole A/B/C già congelate: "
    "stabilità annuale, anni di stress 2008/2020, dipendenza dagli outlier, valore incrementale della fascia $100k–250k "
    "e rischio di bias dovuto ai prezzi Yahoo mancanti."
)
st.warning(
    "Regola metodologica: i risultati di questo audit non devono essere usati per scegliere una nuova soglia. "
    "Se una regola fallisce un controllo, il risultato va registrato come limite della regola congelata."
)

uploaded_audit_event = st.file_uploader(
    "Carica insider_event_study_2006_2021.csv completato (se non è già disponibile nella sessione)",
    type=["csv"], key="robust_audit_upload_v013"
)
if uploaded_audit_event is not None and st.button("Usa Event Study per Robustness Audit", key="use_audit_event_v013"):
    try:
        aud = normalize_loaded_event_study(pd.read_csv(uploaded_audit_event, low_memory=False))
        st.session_state["robust_event_v013"] = aud
        st.success(f"Event Study caricato per audit: {len(aud):,} righe.")
    except Exception as exc:
        st.error(f"CSV non compatibile: {exc}")

if "robust_event_v013" not in st.session_state:
    if isinstance(st.session_state.get("hist_event_v012"), pd.DataFrame) and not st.session_state["hist_event_v012"].empty:
        st.session_state["robust_event_v013"] = st.session_state["hist_event_v012"]

robust_event = st.session_state.get("robust_event_v013")
if isinstance(robust_event, pd.DataFrame) and not robust_event.empty:
    with st.spinner("Calcolo Robustness Audit v0.13..."):
        selected_sets = _audit_selected_sets(robust_event, episode_gap_days=10)
        all_cov = _audit_all_price_status_by_year(robust_event)
        miss = _audit_missing_data(selected_sets, horizons=(1, 5))
        annual = _audit_yearly_tables(selected_sets, horizons=(1, 5))
        stress = _audit_stress_periods(selected_sets, horizons=(1, 5))
        outliers = _audit_outliers(selected_sets, horizons=(1, 5))
        c_desc, c_boot = _audit_c_vs_b_other(robust_event, episode_gap_days=10)

    st.subheader("1. Copertura prezzi e missing-data bias")
    st.markdown("**Copertura Yahoo di tutti i segnali per anno**")
    ac = all_cov.copy()
    if "coverage_%" in ac: ac["coverage_%"] = ac["coverage_%"].round(2)
    st.dataframe(ac, use_container_width=True, hide_index=True)

    mp = miss.copy()
    for c in ["observed_mean_excess", "break_even_missing_avg_excess_to_zero"]:
        if c in mp: mp[c] = (mp[c] * 100).round(2)
    if "coverage_%" in mp: mp["coverage_%"] = mp["coverage_%"].round(2)
    mp = mp.rename(columns={
        "observed_mean_excess":"observed_mean_excess_%",
        "break_even_missing_avg_excess_to_zero":"missing_avg_break_even_to_zero_%",
    })
    st.markdown("**Missing-data stress per configurazione congelata**")
    st.dataframe(mp, use_container_width=True, hide_index=True)
    st.caption(
        "missing_avg_break_even_to_zero_% = rendimento excess medio ipotetico degli eventi non prezzati che, se fosse reale, "
        "porterebbe la media dell'intero campione selezionato a zero. Non è una stima dei rendimenti mancanti: è uno stress test."
    )

    st.subheader("2. Stabilità anno per anno")
    annual_show = annual.copy()
    for c in ["mean_excess", "trimmed_mean_1pct", "median_excess", "win_rate"]:
        if c in annual_show: annual_show[c] = (annual_show[c] * 100).round(2)
    if "coverage_%" in annual_show: annual_show["coverage_%"] = annual_show["coverage_%"].round(2)
    annual_show = annual_show.rename(columns={
        "mean_excess":"mean_excess_%", "trimmed_mean_1pct":"trimmed_mean_1pct_%",
        "median_excess":"median_excess_%", "win_rate":"win_rate_%",
    })
    st.dataframe(annual_show, use_container_width=True, hide_index=True)
    st.caption("small_sample_flag = meno di 30 eventi prezzati oppure meno di 20 issuer: il dato annuale va letto con particolare cautela.")

    st.subheader("3. Anni di stress predefiniti: 2008 e 2020")
    sp = stress.copy()
    for c in ["mean_excess", "trimmed_mean_1pct", "median_excess", "win_rate"]:
        if c in sp: sp[c] = (sp[c] * 100).round(2)
    sp = sp.rename(columns={
        "mean_excess":"mean_excess_%", "trimmed_mean_1pct":"trimmed_mean_1pct_%",
        "median_excess":"median_excess_%", "win_rate":"win_rate_%",
    })
    st.dataframe(sp, use_container_width=True, hide_index=True)

    st.subheader("4. Dipendenza dagli outlier")
    op = outliers.copy()
    for c in ["mean_excess", "trimmed_mean_1pct", "winsorized_mean_1pct", "median_excess", "win_rate", "p01", "p99", "min", "max", "mean_without_top1pct_winners", "top1pct_share_of_positive_sum"]:
        if c in op: op[c] = (op[c] * 100).round(2)
    op = op.rename(columns={
        "mean_excess":"mean_excess_%", "trimmed_mean_1pct":"trimmed_mean_1pct_%",
        "winsorized_mean_1pct":"winsorized_mean_1pct_%", "median_excess":"median_excess_%",
        "win_rate":"win_rate_%", "p01":"p01_%", "p99":"p99_%", "min":"min_%", "max":"max_%",
        "mean_without_top1pct_winners":"mean_without_top1pct_winners_%",
        "top1pct_share_of_positive_sum":"top1pct_share_of_positive_sum_%",
    })
    st.dataframe(op, use_container_width=True, hide_index=True)

    st.subheader("5. La fascia $100k–250k aggiunge davvero qualcosa a B?")
    st.caption(
        "Confronto corretto tra gruppi mutuamente esclusivi: C ($100k–250k) contro il resto dei FIRST_CLUSTER ≥3 insider. "
        "Non confrontiamo C contro B completo, perché C è un sottoinsieme di B."
    )
    cd = c_desc.copy()
    for c in ["mean_excess", "trimmed_mean_1pct", "median_excess", "win_rate"]:
        if c in cd: cd[c] = (cd[c] * 100).round(2)
    cd = cd.rename(columns={
        "mean_excess":"mean_excess_%", "trimmed_mean_1pct":"trimmed_mean_1pct_%",
        "median_excess":"median_excess_%", "win_rate":"win_rate_%",
    })
    st.dataframe(cd, use_container_width=True, hide_index=True)
    cb = c_boot.copy()
    for c in ["observed_diff", "ci95_low", "ci95_high"]:
        if c in cb: cb[c] = (cb[c] * 100).round(2)
    cb = cb.rename(columns={"observed_diff":"diff_media_%", "ci95_low":"CI95_low_%", "ci95_high":"CI95_high_%"})
    st.dataframe(cb, use_container_width=True, hide_index=True)
    if not cb.empty and not bool(cb.get("CI_interamente_>0", pd.Series(False)).any()):
        st.info("Nel test diretto C vs resto di B, nessun orizzonte ha CI95% interamente sopra zero: la fascia $100k–250k può descrivere un sottogruppo forte, ma il suo valore incrementale rispetto a ≥3 insider non è ancora dimostrato.")

    export_parts = []
    for name, frame in [
        ("coverage_all_by_year", all_cov), ("missing_stress", miss), ("annual", annual),
        ("stress_periods", stress), ("outlier_audit", outliers),
        ("C_vs_Bother_descriptive", c_desc), ("C_vs_Bother_bootstrap", c_boot),
    ]:
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            z = frame.copy(); z.insert(0, "table", name); export_parts.append(z)
    if export_parts:
        audit_export = pd.concat(export_parts, ignore_index=True, sort=False)
        st.download_button(
            "Scarica Robustness Audit v0.13 CSV",
            audit_export.to_csv(index=False).encode("utf-8"),
            file_name="insider_robustness_audit_v0_13.csv",
            mime="text/csv",
        )
else:
    st.info("Per il Robustness Audit carica il CSV completo insider_event_study_2006_2021.csv oppure completa/carica H3-H4 sopra.")

# --- v0.14: Value Normalization Audit --------------------------------------
st.divider()
st.header("Value Normalization Audit — v0.14")
st.caption(
    "Controllo dell'osservazione economica: $100k–250k nominali nel 2006 non hanno lo stesso peso di $100k–250k nel 2026. "
    "La regola CORE B (FIRST_CLUSTER ≥3 insider) resta congelata. Confrontiamo soltanto C-NOMINAL con C-REAL corretto per CPI-U."
)
st.warning(
    "Nessuna ottimizzazione della fascia: C-REAL è definita meccanicamente come $100k–250k espressi in dollari 2026. "
    "Per il 2006–2025 usiamo il CPI-U annuale medio; il riferimento 2026 è l'indice CPI-U di agosto 2026 (334.980), ultimo disponibile alla costruzione della v0.14."
)

# CPI-equivalent thresholds are deterministic and visible before looking at returns.
thresholds_v014 = engine_mod.real_band_nominal_thresholds_2026()
tshow = thresholds_v014.copy()
tshow["nominal_lower_equiv"] = tshow["nominal_lower_equiv"].round(0)
tshow["nominal_upper_equiv"] = tshow["nominal_upper_equiv"].round(0)
tshow["inflation_factor_to_2026"] = tshow["inflation_factor_to_2026"].round(3)
st.subheader("Fascia nominale equivalente a $100k–250k in dollari 2026")
st.dataframe(tshow, use_container_width=True, hide_index=True)
st.caption(
    "Esempio: nel 2006 la fascia reale equivalente è molto più bassa in dollari nominali di allora; nel 2025 è già vicina alla fascia 2026. "
    "Questo elimina il vantaggio artificiale che una soglia nominale fissa può dare ai primi anni del campione."
)

u1, u2 = st.columns(2)
with u1:
    hist_norm_upload = st.file_uploader(
        "Storico: insider_event_study_2006_2021.csv",
        type=["csv"], key="value_norm_hist_upload_v014"
    )
    if hist_norm_upload is not None and st.button("Usa storico per Value Audit", key="use_value_norm_hist_v014", use_container_width=True):
        try:
            z = engine_mod.normalize_loaded_event_study(pd.read_csv(hist_norm_upload, low_memory=False))
            st.session_state["value_norm_hist_v014"] = z
            st.success(f"Storico caricato: {len(z):,} righe.")
        except Exception as exc:
            st.error(f"Storico non compatibile: {exc}")
with u2:
    recent_norm_upload = st.file_uploader(
        "Recente: insider_event_study_first_cluster.csv o insider_event_study.csv (2022–2026)",
        type=["csv"], key="value_norm_recent_upload_v014"
    )
    if recent_norm_upload is not None and st.button("Usa recente per Value Audit", key="use_value_norm_recent_v014", use_container_width=True):
        try:
            z = engine_mod.normalize_loaded_event_study(pd.read_csv(recent_norm_upload, low_memory=False))
            st.session_state["value_norm_recent_v014"] = z
            st.success(f"Periodo recente caricato: {len(z):,} righe.")
        except Exception as exc:
            st.error(f"CSV recente non compatibile: {exc}")

# Reuse already-loaded data when available.
if "value_norm_hist_v014" not in st.session_state:
    if isinstance(st.session_state.get("robust_event_v013"), pd.DataFrame) and not st.session_state["robust_event_v013"].empty:
        st.session_state["value_norm_hist_v014"] = st.session_state["robust_event_v013"]
    elif isinstance(st.session_state.get("hist_event_v012"), pd.DataFrame) and not st.session_state["hist_event_v012"].empty:
        st.session_state["value_norm_hist_v014"] = st.session_state["hist_event_v012"]
if "value_norm_recent_v014" not in st.session_state:
    cur = st.session_state.get("event")
    if isinstance(cur, pd.DataFrame) and not cur.empty:
        dd = pd.to_datetime(cur.get("signal_date"), errors="coerce")
        if dd.notna().any() and dd.max() >= pd.Timestamp("2022-01-01"):
            st.session_state["value_norm_recent_v014"] = cur

hist_norm_event = st.session_state.get("value_norm_hist_v014")
recent_norm_event = st.session_state.get("value_norm_recent_v014")

value_audit_parts = []
summary_frames = []
overlap_frames = []
bootstrap_frames = []

if isinstance(hist_norm_event, pd.DataFrame) and not hist_norm_event.empty:
    bh, sh, oh = engine_mod.value_normalization_audit(
        hist_norm_event,
        period_label="2006–2021",
        start_date="2006-01-01", end_date="2021-12-31",
        episode_gap_days=10, horizons=(1,5),
    )
    summary_frames.append(sh); overlap_frames.append(oh)
    for col, label in [
        ("value_band_nominal_100_250", "C-NOMINAL vs resto B"),
        ("value_band_real_2026_100_250", "C-REAL vs resto B"),
    ]:
        bt = engine_mod.value_band_incremental_bootstrap(bh, band_col=col, label=label, horizons=(1,5), n_boot=2000, seed=1414)
        if not bt.empty:
            bt.insert(0, "period", "2006–2021"); bootstrap_frames.append(bt)
    q=bh[[c for c in ["issuer_cik","ticker","signal_date","cluster_value","cluster_value_2026","value_band_nominal_100_250","value_band_real_2026_100_250","excess_1","excess_5"] if c in bh.columns]].copy()
    q.insert(0,"period","2006–2021"); value_audit_parts.append(q)

if isinstance(recent_norm_event, pd.DataFrame) and not recent_norm_event.empty:
    br, sr, orr = engine_mod.value_normalization_audit(
        recent_norm_event,
        period_label="2022–2026",
        start_date="2022-01-01", end_date="2026-12-31",
        episode_gap_days=10, horizons=(1,5),
    )
    summary_frames.append(sr); overlap_frames.append(orr)
    for col, label in [
        ("value_band_nominal_100_250", "C-NOMINAL vs resto B"),
        ("value_band_real_2026_100_250", "C-REAL vs resto B"),
    ]:
        bt = engine_mod.value_band_incremental_bootstrap(br, band_col=col, label=label, horizons=(1,5), n_boot=2000, seed=1415)
        if not bt.empty:
            bt.insert(0, "period", "2022–2026"); bootstrap_frames.append(bt)
    q=br[[c for c in ["issuer_cik","ticker","signal_date","cluster_value","cluster_value_2026","value_band_nominal_100_250","value_band_real_2026_100_250","excess_1","excess_5"] if c in br.columns]].copy()
    q.insert(0,"period","2022–2026"); value_audit_parts.append(q)

if summary_frames:
    vsummary = pd.concat(summary_frames, ignore_index=True)
    st.subheader("B vs C-NOMINAL vs C-REAL")
    vs = vsummary.copy()
    for c in ["mean_excess","trimmed_mean_excess_1pct","median_excess","win_rate_excess"]:
        if c in vs: vs[c]=(vs[c]*100).round(2)
    vs=vs.rename(columns={
        "mean_excess":"mean_excess_%",
        "trimmed_mean_excess_1pct":"trimmed_mean_1pct_%",
        "median_excess":"median_excess_%",
        "win_rate_excess":"win_rate_%",
    })
    st.dataframe(vs, use_container_width=True, hide_index=True)

    if overlap_frames:
        st.subheader("Quanto cambia realmente il campione?")
        ov=pd.concat(overlap_frames,ignore_index=True)
        st.dataframe(ov,use_container_width=True,hide_index=True)
        st.caption(
            "real_only = eventi esclusi dalla vecchia fascia nominale ma inclusi dopo la correzione CPI; nominal_only = eventi che la soglia nominale includeva ma che non equivalgono a $100k–250k del 2026."
        )

    if bootstrap_frames:
        st.subheader("Valore incrementale della fascia rispetto al resto di B")
        vb=pd.concat(bootstrap_frames,ignore_index=True)
        vbp=vb.copy()
        for c in ["observed_diff_vs_B_rest","ci95_low","ci95_high"]:
            if c in vbp: vbp[c]=(vbp[c]*100).round(2)
        vbp=vbp.rename(columns={
            "observed_diff_vs_B_rest":"diff_media_vs_B_rest_%",
            "ci95_low":"CI95_low_%","ci95_high":"CI95_high_%",
            "robustly_above_zero":"CI_interamente_>0",
        })
        st.dataframe(vbp,use_container_width=True,hide_index=True)

    # Frozen interpretation aid: no auto-selection of a winner, just consistency flags.
    direction=[]
    for _,r in vsummary.iterrows():
        direction.append({
            "period":r["period"], "rule":r["rule"], "horizon_sessions":int(r["horizon_sessions"]),
            "mean_>0":bool(r["mean_excess"]>0),
            "trimmed_>0":bool(r["trimmed_mean_excess_1pct"]>0),
            "median_>0":bool(r["median_excess"]>0),
            "win_rate_>50%":bool(r["win_rate_excess"]>0.5),
        })
    st.subheader("Coerenza direzionale (nessun ranking)")
    st.dataframe(pd.DataFrame(direction),use_container_width=True,hide_index=True)

    export_parts=[]
    x=vsummary.copy(); x.insert(0,"table","summary"); export_parts.append(x)
    if overlap_frames:
        x=pd.concat(overlap_frames,ignore_index=True); x.insert(0,"table","overlap"); export_parts.append(x)
    if bootstrap_frames:
        x=pd.concat(bootstrap_frames,ignore_index=True); x.insert(0,"table","bootstrap_vs_B_rest"); export_parts.append(x)
    x=thresholds_v014.copy(); x.insert(0,"table","cpi_thresholds"); export_parts.append(x)
    audit_export=pd.concat(export_parts,ignore_index=True,sort=False)
    st.download_button(
        "Scarica Value Normalization Audit v0.14 CSV",
        audit_export.to_csv(index=False).encode("utf-8"),
        file_name="insider_value_normalization_audit_v0_14.csv",
        mime="text/csv",
    )
else:
    st.info(
        "Per il Value Normalization Audit carica almeno uno dei due Event Study. Per il confronto completo usa sia 2006–2021 sia 2022–2026."
    )
