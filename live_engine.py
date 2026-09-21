from __future__ import annotations

import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from engine import build_issuer_day_signals, label_cluster_episodes, yahoo_symbol, _download_yahoo_prices

SEC_ARCHIVES = "https://www.sec.gov/Archives"
SEC_DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{quarter}/master.{yyyymmdd}.idx"

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
MISSING_TICKERS = {"", "NONE", "N/A", "NA", "NAN", "NULL", "NO TICKER"}
BAD_SECURITY_WORDS = re.compile(
    r"preferred|warrant|option|unit|note|bond|debenture|right|convertible|debt|phantom|restricted stock unit|rsu",
    re.I,
)
GOOD_SECURITY_WORDS = re.compile(
    r"common stock|common shares?|ordinary shares?|class [a-z] common|common class [a-z]",
    re.I,
)

COMPONENT_COLUMNS = [
    "ACCESSION_NUMBER", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL",
    "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
    "FILING_DATE", "TRANS_DATE", "DIRECT_INDIRECT_OWNERSHIP", "SECURITY_TITLE",
    "shares", "trade_value", "n_tranches", "vwap", "filing_lag_days",
    "sec_submission_url",
]

PROCESSED_COLUMNS = [
    "ACCESSION_NUMBER", "FILING_DATE", "filename", "status", "n_components",
    "error", "processed_at",
]


def _sec_headers(contact_email: str) -> dict[str, str]:
    email = str(contact_email or "").strip()
    if not email or "@" not in email:
        raise ValueError("Inserisci una email valida per il User-Agent SEC.")
    return {
        "User-Agent": f"IndependentInsiderRadarLive/1.2 {email}",
        "Accept-Encoding": "gzip, deflate",
    }


def _quarter_for_day(d: date) -> int:
    return (d.month - 1) // 3 + 1


def daily_master_url(d: date) -> str:
    return SEC_DAILY_INDEX.format(
        year=d.year,
        quarter=_quarter_for_day(d),
        yyyymmdd=d.strftime("%Y%m%d"),
    )


def _accession_from_filename(filename: str) -> str:
    base = Path(str(filename)).name
    if base.lower().endswith(".txt"):
        base = base[:-4]
    return base


def parse_master_index(text: str, source_date: date | None = None) -> pd.DataFrame:
    """Parse an SEC master daily index and keep original Form 4 filings only."""
    rows: list[dict] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form_type, filing_date, filename = [p.strip() for p in parts]
        if form_type != "4":
            continue
        if not filename.startswith("edgar/data/") or not filename.lower().endswith(".txt"):
            continue
        dt = pd.to_datetime(filing_date, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({
            "filer_cik": cik,
            "company_name_index": company,
            "form_type": form_type,
            "filing_date": pd.Timestamp(dt).normalize(),
            "filename": filename,
            "accession": _accession_from_filename(filename),
            "source_date": pd.Timestamp(source_date or dt.date()).normalize(),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "filer_cik", "company_name_index", "form_type", "filing_date",
            "filename", "accession", "source_date",
        ])
    return pd.DataFrame(rows).drop_duplicates("accession").reset_index(drop=True)


def fetch_daily_form4_index(
    d: date,
    contact_email: str,
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return (rows, status). 404/403 due to non-filing day is represented explicitly."""
    sess = session or requests.Session()
    url = daily_master_url(d)
    r = sess.get(url, headers=_sec_headers(contact_email), timeout=timeout)
    if r.status_code == 404:
        return parse_master_index("", d), "not_available"
    if r.status_code == 403:
        return parse_master_index("", d), "forbidden"
    r.raise_for_status()
    # SEC indexes are ASCII/Latin-1 friendly; requests often identifies them correctly.
    r.encoding = r.encoding or "latin-1"
    return parse_master_index(r.text, d), "ok"


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return root


def _text(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    el = node.find(path)
    if el is None or el.text is None:
        return default
    return str(el.text).strip()


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def _common_security(title: str) -> bool:
    t = str(title or "").strip()
    return bool(GOOD_SECURITY_WORDS.search(t)) and not bool(BAD_SECURITY_WORDS.search(t))


def _clean_ticker_value(value: str) -> str:
    t = str(value or "").strip().upper()
    if t in MISSING_TICKERS:
        return ""
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", t):
        return ""
    return t


def extract_ownership_xml(submission_text: str) -> str:
    """Extract the ownershipDocument XML from the complete SEC submission text."""
    text = str(submission_text or "")
    m = re.search(r"<ownershipDocument\b.*?</ownershipDocument>", text, flags=re.I | re.S)
    if not m:
        raise ValueError("ownershipDocument XML non trovato nel filing SEC")
    return m.group(0)


def parse_form4_xml(
    xml_text: str,
    *,
    filing_date,
    accession: str,
    submission_url: str = "",
    managers_only: bool = True,
    exclude_10b5_1: bool = True,
    drop_joint_filings: bool = True,
    min_trade_value: float = 10_000.0,
) -> tuple[pd.DataFrame, str]:
    """Parse one Form 4 ownership XML into conservative purchase components.

    Mirrors the frozen research rules: original Form 4, code P, acquired A,
    non-derivative common/ordinary shares, positive shares/price, one attributable
    owner, Officer/Director, and 10b5-1 excluded when explicitly marked.
    """
    try:
        root = _strip_namespaces(ET.fromstring(xml_text))
    except Exception as exc:
        return pd.DataFrame(columns=COMPONENT_COLUMNS), f"xml_error:{type(exc).__name__}"

    if exclude_10b5_1:
        aff = _text(root, ".//aff10b5One")
        if _truthy(aff):
            return pd.DataFrame(columns=COMPONENT_COLUMNS), "excluded_10b5_1"

    issuer = root.find("issuer")
    issuer_cik = _text(issuer, "issuerCik")
    issuer_name = _text(issuer, "issuerName")
    ticker = _clean_ticker_value(_text(issuer, "issuerTradingSymbol"))

    owners = root.findall("reportingOwner")
    if drop_joint_filings and len(owners) != 1:
        return pd.DataFrame(columns=COMPONENT_COLUMNS), "joint_or_missing_owner"
    if not owners:
        return pd.DataFrame(columns=COMPONENT_COLUMNS), "missing_owner"

    owner = owners[0]
    owner_id = owner.find("reportingOwnerId")
    owner_rel = owner.find("reportingOwnerRelationship")
    owner_cik = _text(owner_id, "rptOwnerCik")
    owner_name = _text(owner_id, "rptOwnerName")
    is_director = _truthy(_text(owner_rel, "isDirector"))
    is_officer = _truthy(_text(owner_rel, "isOfficer"))
    officer_title = _text(owner_rel, "officerTitle")
    relation_parts: list[str] = []
    if is_director:
        relation_parts.append("Director")
    if is_officer:
        relation_parts.append("Officer")
    if _truthy(_text(owner_rel, "isTenPercentOwner")):
        relation_parts.append("10% Owner")
    if _truthy(_text(owner_rel, "isOther")):
        other = _text(owner_rel, "otherText")
        relation_parts.append(other or "Other")
    relationship = " | ".join(dict.fromkeys(relation_parts))
    if managers_only and not (is_director or is_officer):
        return pd.DataFrame(columns=COMPONENT_COLUMNS), "not_officer_director"

    filing_ts = pd.to_datetime(filing_date, errors="coerce")
    if pd.isna(filing_ts):
        return pd.DataFrame(columns=COMPONENT_COLUMNS), "bad_filing_date"
    filing_ts = pd.Timestamp(filing_ts).normalize()

    rows: list[dict] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = _text(tx, "transactionCoding/transactionCode").upper()
        acquired = _text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value").upper()
        swap = _text(tx, "transactionCoding/equitySwapInvolved")
        if code != "P" or acquired != "A" or _truthy(swap):
            continue
        security_title = _text(tx, "securityTitle/value")
        if not _common_security(security_title):
            continue
        trans_date = pd.to_datetime(_text(tx, "transactionDate/value"), errors="coerce")
        shares = pd.to_numeric(_text(tx, "transactionAmounts/transactionShares/value"), errors="coerce")
        price = pd.to_numeric(_text(tx, "transactionAmounts/transactionPricePerShare/value"), errors="coerce")
        if pd.isna(trans_date) or not np.isfinite(shares) or not np.isfinite(price):
            continue
        shares = float(shares); price = float(price)
        if shares <= 0 or price <= 0:
            continue
        trans_date = pd.Timestamp(trans_date).normalize()
        lag = int((filing_ts - trans_date).days)
        if lag < 0 or lag > 366:
            continue
        trade_value = shares * price
        if trade_value < float(min_trade_value):
            continue
        rows.append({
            "ACCESSION_NUMBER": str(accession),
            "ISSUERCIK": str(issuer_cik),
            "ISSUERNAME": issuer_name,
            "ISSUERTRADINGSYMBOL": ticker,
            "RPTOWNERCIK": str(owner_cik),
            "RPTOWNERNAME": owner_name,
            "RPTOWNER_RELATIONSHIP": relationship,
            "RPTOWNER_TITLE": officer_title,
            "FILING_DATE": filing_ts,
            "TRANS_DATE": trans_date,
            "DIRECT_INDIRECT_OWNERSHIP": _text(tx, "ownershipNature/directOrIndirectOwnership/value"),
            "SECURITY_TITLE": security_title,
            "shares": shares,
            "trade_value": trade_value,
            "n_tranches": 1,
            "vwap": price,
            "filing_lag_days": lag,
            "sec_submission_url": submission_url,
        })

    if not rows:
        return pd.DataFrame(columns=COMPONENT_COLUMNS), "no_qualifying_purchase"

    raw = pd.DataFrame(rows)
    group_cols = [
        "ACCESSION_NUMBER", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL",
        "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
        "FILING_DATE", "TRANS_DATE", "DIRECT_INDIRECT_OWNERSHIP", "SECURITY_TITLE",
        "filing_lag_days", "sec_submission_url",
    ]
    comp = raw.groupby(group_cols, dropna=False, as_index=False).agg(
        shares=("shares", "sum"),
        trade_value=("trade_value", "sum"),
        n_tranches=("n_tranches", "sum"),
    )
    comp["vwap"] = comp["trade_value"] / comp["shares"]
    comp = comp[COMPONENT_COLUMNS]
    return comp.sort_values(["FILING_DATE", "TRANS_DATE"]).reset_index(drop=True), "qualified_purchase"


def fetch_and_parse_form4(
    filename: str,
    filing_date,
    accession: str,
    contact_email: str,
    *,
    timeout: int = 30,
    min_trade_value: float = 10_000.0,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, str]:
    sess = session or requests.Session()
    url = f"{SEC_ARCHIVES}/{str(filename).lstrip('/')}"
    r = sess.get(url, headers=_sec_headers(contact_email), timeout=timeout)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    xml = extract_ownership_xml(r.text)
    return parse_form4_xml(
        xml,
        filing_date=filing_date,
        accession=accession,
        submission_url=url,
        min_trade_value=min_trade_value,
    )


def _read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=columns or [])


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def live_state_paths(state_dir: str | Path) -> dict[str, Path]:
    root = Path(state_dir)
    return {
        "root": root,
        "index": root / "live_index.csv",
        "processed": root / "live_processed.csv",
        "components": root / "live_components.csv",
        "radar": root / "live_radar.csv",
        "prices": root / "live_radar_prices.csv",
        "forward_registry": root / "forward_registry_v1_2.csv",
        "forward_meta": root / "forward_registry_meta_v1_2.json",
        "index_cache": root / "index_cache",
    }


def discover_live_form4s(
    start_date: date,
    end_date: date,
    contact_email: str,
    state_dir: str | Path,
    *,
    progress_callback: Callable | None = None,
) -> tuple[pd.DataFrame, date | None]:
    paths = live_state_paths(state_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["index_cache"].mkdir(parents=True, exist_ok=True)
    existing = _read_csv_or_empty(paths["index"])
    session = requests.Session()

    days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    # No SEC filing indexes are expected on weekends. Skipping them saves requests.
    days = [d for d in days if d.weekday() < 5]
    frames: list[pd.DataFrame] = []
    latest_available: date | None = None
    for i, d in enumerate(days, start=1):
        cache = paths["index_cache"] / f"{d.strftime('%Y%m%d')}.csv"
        if cache.exists():
            day_df = _read_csv_or_empty(cache)
            status = "cached"
        else:
            try:
                day_df, status = fetch_daily_form4_index(d, contact_email, session=session)
            except Exception as exc:
                # Do not poison the cache on transient errors.
                day_df = pd.DataFrame()
                status = f"error:{type(exc).__name__}"
            if status == "ok":
                _atomic_write_csv(day_df, cache)
            time.sleep(0.13)
        if not day_df.empty:
            frames.append(day_df)
            latest_available = max(latest_available or d, d)
        if progress_callback:
            progress_callback("index", i, len(days), str(d), status)

    if frames:
        discovered = pd.concat(frames, ignore_index=True)
    else:
        discovered = pd.DataFrame(columns=[
            "filer_cik", "company_name_index", "form_type", "filing_date", "filename", "accession", "source_date"
        ])
    if not existing.empty:
        discovered = pd.concat([existing, discovered], ignore_index=True)
    if not discovered.empty:
        discovered = discovered.drop_duplicates("accession", keep="last")
        discovered["filing_date"] = pd.to_datetime(discovered["filing_date"], errors="coerce")
        discovered = discovered.sort_values(["filing_date", "accession"]).reset_index(drop=True)
        _atomic_write_csv(discovered, paths["index"])
        valid = discovered["filing_date"].dropna()
        if not valid.empty:
            latest_available = max(latest_available or valid.max().date(), valid.max().date())
    return discovered, latest_available


def sync_live_sec(
    *,
    start_date: date,
    end_date: date,
    contact_email: str,
    state_dir: str | Path,
    min_trade_value: float = 10_000.0,
    max_filings_per_run: int = 1200,
    progress_callback: Callable | None = None,
) -> dict:
    """Discover and checkpoint-process recent Form 4 filings.

    Every filing is marked processed, even when it contains no qualifying purchase, so
    repeated runs do not refetch the same document. The function intentionally processes
    oldest pending filings first because cluster context must be causal.
    """
    paths = live_state_paths(state_dir)
    discovered, latest_available = discover_live_form4s(
        start_date, end_date, contact_email, state_dir, progress_callback=progress_callback
    )
    processed = _read_csv_or_empty(paths["processed"], PROCESSED_COLUMNS)
    components = _read_csv_or_empty(paths["components"], COMPONENT_COLUMNS)

    done = set(processed.get("ACCESSION_NUMBER", pd.Series(dtype=str)).fillna("").astype(str))
    pending = discovered[~discovered["accession"].astype(str).isin(done)].copy()
    pending["filing_date"] = pd.to_datetime(pending["filing_date"], errors="coerce")
    pending = pending.sort_values(["filing_date", "accession"]).head(int(max_filings_per_run))
    total_pending_before = int(len(discovered) - discovered["accession"].astype(str).isin(done).sum())

    session = requests.Session()
    new_processed: list[dict] = []
    new_components: list[pd.DataFrame] = []
    for i, row in enumerate(pending.itertuples(index=False), start=1):
        accession = str(row.accession)
        filing_date = pd.Timestamp(row.filing_date).normalize()
        filename = str(row.filename)
        try:
            comp, status = fetch_and_parse_form4(
                filename, filing_date, accession, contact_email,
                min_trade_value=min_trade_value, session=session,
            )
            err = ""
            if not comp.empty:
                new_components.append(comp)
        except Exception as exc:
            status = f"fetch_error:{type(exc).__name__}"
            err = str(exc)[:300]
            comp = pd.DataFrame()
        new_processed.append({
            "ACCESSION_NUMBER": accession,
            "FILING_DATE": filing_date,
            "filename": filename,
            "status": status,
            "n_components": int(len(comp)),
            "error": err,
            "processed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

        # Flush often so a Streamlit restart loses at most a small batch.
        if i % 25 == 0 or i == len(pending):
            if new_processed:
                processed = pd.concat([processed, pd.DataFrame(new_processed)], ignore_index=True)
                processed = processed.drop_duplicates("ACCESSION_NUMBER", keep="last")
                _atomic_write_csv(processed, paths["processed"])
                new_processed = []
            if new_components:
                components = pd.concat([components] + new_components, ignore_index=True)
                dedupe = ["ISSUERCIK", "RPTOWNERCIK", "TRANS_DATE", "shares", "trade_value", "DIRECT_INDIRECT_OWNERSHIP"]
                for c in dedupe:
                    if c not in components.columns:
                        components[c] = pd.NA
                components["FILING_DATE"] = pd.to_datetime(components["FILING_DATE"], errors="coerce")
                components = components.sort_values("FILING_DATE").drop_duplicates(dedupe, keep="first")
                _atomic_write_csv(components, paths["components"])
                new_components = []
        if progress_callback:
            progress_callback("filings", i, len(pending), accession, status)
        time.sleep(0.13)

    processed = _read_csv_or_empty(paths["processed"], PROCESSED_COLUMNS)
    components = _read_csv_or_empty(paths["components"], COMPONENT_COLUMNS)
    processed_ids = set(processed.get("ACCESSION_NUMBER", pd.Series(dtype=str)).fillna("").astype(str))
    pending_after = int((~discovered["accession"].astype(str).isin(processed_ids)).sum()) if not discovered.empty else 0
    qualifying_filings = int(processed.get("status", pd.Series(dtype=str)).astype(str).eq("qualified_purchase").sum())
    return {
        "discovered_form4": int(len(discovered)),
        "processed_form4": int(len(processed)),
        "processed_this_run": int(len(pending)),
        "pending_form4": pending_after,
        "pending_before": total_pending_before,
        "qualifying_filings": qualifying_filings,
        "components": int(len(components)),
        "latest_available_index_date": latest_available,
        "complete": pending_after == 0,
    }


def load_live_components(state_dir: str | Path) -> pd.DataFrame:
    path = live_state_paths(state_dir)["components"]
    df = _read_csv_or_empty(path, COMPONENT_COLUMNS)
    if df.empty:
        return df
    for c in ["FILING_DATE", "TRANS_DATE"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["shares", "trade_value", "vwap", "filing_lag_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_live_radar(
    components: pd.DataFrame,
    *,
    window_days: int = 10,
    episode_gap_days: int = 10,
    today: date | None = None,
) -> pd.DataFrame:
    """Create operational WATCH/CORE labels using the frozen research rules."""
    if components is None or components.empty:
        return pd.DataFrame()
    today = today or date.today()
    sig = build_issuer_day_signals(components, window_days=window_days, min_insiders=2)
    if sig.empty:
        return sig

    lab2 = label_cluster_episodes(sig, min_insiders=2, episode_gap_days=episode_gap_days)
    lab3 = label_cluster_episodes(sig, min_insiders=3, episode_gap_days=episode_gap_days)
    sig = sig.copy()
    sig["group_ge2"] = lab2["analysis_group"].values
    sig["group_ge3"] = lab3["analysis_group"].values

    def priority(row) -> str:
        if row["group_ge3"] == "FIRST_CLUSTER":
            return "CORE"
        if row["group_ge2"] == "FIRST_CLUSTER":
            return "WATCH"
        if int(row.get("n_insiders", 0)) >= 3:
            return "REPEAT_CORE"
        if int(row.get("n_insiders", 0)) >= 2:
            return "REPEAT"
        return "SOLO"

    sig["priority"] = sig.apply(priority, axis=1)
    sig["value_tag"] = np.where(
        pd.to_numeric(sig["cluster_value"], errors="coerce").between(100_000, 250_000, inclusive="both"),
        "VALUE 100–250k (2026$)",
        "",
    )
    sig["signal_date"] = pd.to_datetime(sig["signal_date"], errors="coerce")
    sig["calendar_age_days"] = (pd.Timestamp(today) - sig["signal_date"]).dt.days
    context_start = pd.to_datetime(components["FILING_DATE"], errors="coerce").min()
    sig["context_complete"] = True
    if pd.notna(context_start):
        sig["context_complete"] = sig["signal_date"].ge(context_start + pd.Timedelta(days=window_days))

    def first_accession(v) -> str:
        parts = [p.strip() for p in str(v or "").split(";") if p.strip()]
        return parts[0] if parts else ""

    sig["primary_accession"] = sig["accessions"].map(first_accession)
    sig["sec_url"] = sig.apply(
        lambda r: (
            f"https://www.sec.gov/Archives/edgar/data/{str(r['issuer_cik']).lstrip('0')}/{str(r['primary_accession']).replace('-', '')}/"
            if str(r.get("primary_accession", "")) else ""
        ), axis=1,
    )
    sig["tradingview_url"] = sig["ticker"].fillna("").astype(str).map(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t}" if t and t.upper() not in MISSING_TICKERS else ""
    )
    rank = {"CORE": 0, "WATCH": 1, "REPEAT_CORE": 2, "REPEAT": 3, "SOLO": 4}
    sig["_rank"] = sig["priority"].map(rank).fillna(9)
    sig = sig.sort_values(["_rank", "signal_date", "cluster_value"], ascending=[True, False, False])
    return sig.drop(columns=["_rank"]).reset_index(drop=True)



def add_operational_columns(radar: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-only operational fields without changing the frozen signal rules.

    Buckets are deliberately mechanical:
    - NUOVO: no next trading session exists yet (e.g. weekend / same-day filing).
    - ATTIVO: entry exists and 1-4 sessions have been observed.
    - COMPLETATO: at least 5 sessions have been observed.
    - DA PREZZARE: Yahoo enrichment has not been run for the signal.
    - DA VERIFICARE: ticker/price/history could not be resolved reliably.
    """
    if radar is None:
        return pd.DataFrame()
    out = radar.copy()
    if out.empty:
        return out

    if "price_status" not in out.columns:
        out["price_status"] = "not_requested"
    if "sessions_observed" not in out.columns:
        out["sessions_observed"] = np.nan

    sessions = pd.to_numeric(out["sessions_observed"], errors="coerce")
    status = out["price_status"].fillna("not_requested").astype(str)

    bucket = pd.Series("DA VERIFICARE", index=out.index, dtype="object")
    bucket.loc[status.eq("not_requested")] = "DA PREZZARE"
    bucket.loc[status.eq("no_future_session")] = "NUOVO"
    bucket.loc[status.eq("ok") & sessions.between(1, 4, inclusive="both")] = "ATTIVO"
    bucket.loc[status.eq("ok") & sessions.ge(5)] = "COMPLETATO"
    out["operational_bucket"] = bucket

    def day_label(row) -> str:
        b = str(row.get("operational_bucket", ""))
        n = pd.to_numeric(pd.Series([row.get("sessions_observed")]), errors="coerce").iloc[0]
        if b == "NUOVO":
            return "0/5"
        if b in {"ATTIVO", "COMPLETATO"} and pd.notna(n):
            return f"{min(5, max(0, int(n)))}/5"
        return "—"

    out["day_5"] = out.apply(day_label, axis=1)

    n_insiders = pd.to_numeric(out.get("n_insiders", pd.Series(index=out.index, dtype=float)), errors="coerce")
    out["insider_band"] = np.select(
        [n_insiders.ge(5), n_insiders.eq(4), n_insiders.eq(3), n_insiders.eq(2)],
        ["5+", "4", "3", "2"],
        default="—",
    )

    roles = out.get("roles", pd.Series("", index=out.index)).fillna("").astype(str)
    def role_tag(text: str) -> str:
        t = str(text).lower()
        ceo = bool(re.search(r"\bceo\b|chief executive officer|president\s*[/,&-]?\s*ceo", t))
        cfo = bool(re.search(r"\bcfo\b|chief financial officer|co\s*cfo", t))
        if ceo and cfo:
            return "CEO+CFO"
        if ceo:
            return "CEO"
        if cfo:
            return "CFO"
        return "—"
    out["role_tag"] = roles.map(role_tag)

    # 2026 is the live reference year, so the nominal live value equals the 2026-dollar tag.
    out["value_flag"] = np.where(out.get("value_tag", pd.Series("", index=out.index)).fillna("").astype(str).ne(""), "VALUE", "")

    rank = {"CORE": 0, "WATCH": 1, "REPEAT_CORE": 2, "REPEAT": 3, "SOLO": 4}
    out["_priority_rank_v11"] = out.get("priority", pd.Series("", index=out.index)).map(rank).fillna(9)
    out["signal_date"] = pd.to_datetime(out.get("signal_date"), errors="coerce")
    out = out.sort_values(["_priority_rank_v11", "signal_date", "cluster_value"], ascending=[True, False, False])
    return out.drop(columns=["_priority_rank_v11"]).reset_index(drop=True)

def enrich_live_prices(
    radar: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    max_signal_age_days: int = 45,
    batch_size: int = 60,
) -> pd.DataFrame:
    """Add current/5-session price context to recent radar signals using Yahoo."""
    if radar is None or radar.empty:
        return pd.DataFrame() if radar is None else radar.copy()
    out = radar.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=max_signal_age_days)
    target = out[out["signal_date"].ge(cutoff) & out["priority"].isin(["CORE", "WATCH", "REPEAT_CORE"])].copy()
    if target.empty:
        return out

    target["yahoo_symbol"] = target["ticker"].map(yahoo_symbol)
    symbols = sorted({s for s in target["yahoo_symbol"].astype(str) if s})
    start = target["signal_date"].min() - pd.Timedelta(days=5)
    end = pd.Timestamp(date.today()) + pd.Timedelta(days=3)
    prices = _download_yahoo_prices(symbols, start, end, batch_size=batch_size)
    bench_map = _download_yahoo_prices([benchmark], start, end, batch_size=1)
    bench = bench_map.get(benchmark, pd.DataFrame())

    cols_defaults = {
        "entry_date": pd.NaT,
        "entry_open": np.nan,
        "price_date": pd.NaT,
        "current_close": np.nan,
        "sessions_observed": np.nan,
        "return_since_entry": np.nan,
        "excess_since_entry": np.nan,
        "return_5": np.nan,
        "excess_5": np.nan,
        "exit_5_date": pd.NaT,
        "price_status": "not_requested",
    }
    for c, v in cols_defaults.items():
        if c not in out.columns:
            out[c] = v

    if not bench.empty:
        bench = bench.copy()
        bench.index = pd.to_datetime(bench.index).tz_localize(None) if getattr(pd.to_datetime(bench.index), "tz", None) is not None else pd.to_datetime(bench.index)

    for idx, row in target.iterrows():
        sym = str(row.get("yahoo_symbol", ""))
        px = prices.get(sym, pd.DataFrame())
        if not sym:
            out.at[idx, "price_status"] = "missing_ticker"
            continue
        if px.empty or bench.empty:
            out.at[idx, "price_status"] = "missing_price"
            continue
        px = px.copy()
        px.index = pd.to_datetime(px.index)
        try:
            px.index = px.index.tz_localize(None)
        except TypeError:
            try:
                px.index = px.index.tz_convert(None)
            except Exception:
                pass
        signal = pd.Timestamp(row["signal_date"]).tz_localize(None) if pd.Timestamp(row["signal_date"]).tzinfo else pd.Timestamp(row["signal_date"])
        candidates = px.index[px.index > signal]
        if len(candidates) == 0:
            out.at[idx, "price_status"] = "no_future_session"
            continue
        entry = pd.Timestamp(candidates[0])
        if (entry.normalize() - signal.normalize()).days > 7:
            out.at[idx, "price_status"] = "stale_symbol_or_gap"
            continue
        loc = int(np.flatnonzero(px.index == entry)[0])
        entry_open = float(pd.to_numeric(pd.Series([px.iloc[loc]["Open"]]), errors="coerce").iloc[0])
        if not np.isfinite(entry_open) or entry_open <= 0:
            out.at[idx, "price_status"] = "bad_entry_price"
            continue
        tail = px.iloc[loc:].dropna(subset=["Close"])
        if tail.empty:
            out.at[idx, "price_status"] = "missing_close"
            continue
        price_date = pd.Timestamp(tail.index[-1])
        current_close = float(tail.iloc[-1]["Close"])
        sessions = int(len(tail))
        stock_ret = current_close / entry_open - 1.0

        bench_entry_candidates = bench.index[bench.index >= entry]
        if len(bench_entry_candidates) == 0:
            out.at[idx, "price_status"] = "benchmark_missing_entry"
            continue
        b_entry = pd.Timestamp(bench_entry_candidates[0])
        b_open = float(bench.loc[b_entry, "Open"])
        b_close_candidates = bench.index[bench.index <= price_date]
        b_current_date = pd.Timestamp(b_close_candidates[-1]) if len(b_close_candidates) else b_entry
        b_close = float(bench.loc[b_current_date, "Close"])
        excess_since = stock_ret - (b_close / b_open - 1.0)

        out.at[idx, "entry_date"] = entry
        out.at[idx, "entry_open"] = entry_open
        out.at[idx, "price_date"] = price_date
        out.at[idx, "current_close"] = current_close
        out.at[idx, "sessions_observed"] = sessions
        out.at[idx, "return_since_entry"] = stock_ret
        out.at[idx, "excess_since_entry"] = excess_since
        out.at[idx, "price_status"] = "ok"

        if sessions >= 5:
            exit_date = pd.Timestamp(tail.index[4])
            close5 = float(tail.iloc[4]["Close"])
            b5_candidates = bench.index[bench.index <= exit_date]
            if len(b5_candidates):
                b5_date = pd.Timestamp(b5_candidates[-1])
                b5_close = float(bench.loc[b5_date, "Close"])
                r5 = close5 / entry_open - 1.0
                out.at[idx, "return_5"] = r5
                out.at[idx, "excess_5"] = r5 - (b5_close / b_open - 1.0)
                out.at[idx, "exit_5_date"] = exit_date

    out["operational_status"] = np.where(
        (out["price_status"].eq("ok")) & (pd.to_numeric(out["sessions_observed"], errors="coerce") <= 5),
        "ACTIVE ≤5 sessions",
        np.where(out["calendar_age_days"].le(10), "RECENT", "AGED"),
    )
    return out



FORWARD_REGISTRY_COLUMNS = [
    "signal_key", "tracking_origin", "registered_at", "last_seen_at", "registration_version",
    "priority", "ticker", "issuer_cik", "issuer_name", "signal_date", "n_insiders", "cluster_value",
    "value_flag", "role_tag", "owners", "roles", "sec_url", "tradingview_url",
    "entry_date", "entry_open", "latest_price_date", "latest_close", "sessions_observed",
    "current_return_since_entry", "current_excess_since_entry", "price_status", "operational_bucket",
    "frozen", "completed_at", "exit_5_date", "return_5", "excess_5",
]


def _utc_now_text() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _registry_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().lower() in TRUE_VALUES


def _safe_date_text(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(dt) else pd.Timestamp(dt).strftime("%Y-%m-%d")


def _safe_num(value):
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and np.isfinite(x) else np.nan


def _forward_signal_key(row) -> str:
    issuer = str(row.get("issuer_cik", "") or "").strip()
    if not issuer or issuer.lower() == "nan":
        issuer = str(row.get("ticker", "") or "").strip().upper()
    priority = str(row.get("priority", "") or "").strip().upper()
    signal_date = _safe_date_text(row.get("signal_date"))
    return f"{issuer}|{signal_date}|{priority}"


def load_forward_registry(state_dir: str | Path) -> pd.DataFrame:
    path = live_state_paths(state_dir)["forward_registry"]
    df = _read_csv_or_empty(path, FORWARD_REGISTRY_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=FORWARD_REGISTRY_COLUMNS)
    for c in FORWARD_REGISTRY_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ["signal_date", "entry_date", "latest_price_date", "completed_at", "exit_5_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["n_insiders", "cluster_value", "entry_open", "latest_close", "sessions_observed",
              "current_return_since_entry", "current_excess_since_entry", "return_5", "excess_5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["frozen"] = df["frozen"].map(_registry_bool)
    return df[FORWARD_REGISTRY_COLUMNS].copy()


def load_forward_registry_meta(state_dir: str | Path) -> dict:
    path = live_state_paths(state_dir)["forward_meta"]
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_forward_meta(state_dir: str | Path, meta: dict) -> None:
    path = live_state_paths(state_dir)["forward_meta"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def update_forward_registry(snapshot: pd.DataFrame, state_dir: str | Path, *, now_utc: str | None = None) -> pd.DataFrame:
    """Idempotently register CORE/WATCH signals and freeze their 5-session result.

    The first call creates a BASELINE from signals already present, so pre-v1.2 observations
    are not silently counted as prospective forward evidence. Signals first appearing after
    that initialization are labelled FORWARD. Only context-complete CORE/WATCH signals enter.
    Once return_5/excess_5 are frozen, later price refreshes never rewrite them.
    """
    paths = live_state_paths(state_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    now = now_utc or _utc_now_text()
    meta = load_forward_registry_meta(state_dir)
    initialized = bool(meta.get("initialized_at"))
    if not initialized:
        meta = {"initialized_at": now, "version": "1.2", "note": "Existing eligible signals at initialization are BASELINE."}
        _write_forward_meta(state_dir, meta)

    reg = load_forward_registry(state_dir)
    if snapshot is None or snapshot.empty:
        if not paths["forward_registry"].exists():
            _atomic_write_csv(reg, paths["forward_registry"])
        return reg

    snap = add_operational_columns(snapshot)
    if "context_complete" not in snap.columns:
        snap["context_complete"] = False
    eligible = snap[
        snap.get("priority", pd.Series("", index=snap.index)).isin(["CORE", "WATCH"])
        & snap["context_complete"].fillna(False).astype(bool)
    ].copy()
    if eligible.empty:
        if not paths["forward_registry"].exists():
            _atomic_write_csv(reg, paths["forward_registry"])
        return reg

    eligible["signal_key"] = eligible.apply(_forward_signal_key, axis=1)
    eligible = eligible[eligible["signal_key"].str.contains(r"\|\d{4}-\d{2}-\d{2}\|", regex=True)].copy()
    existing = set(reg.get("signal_key", pd.Series(dtype=str)).fillna("").astype(str))
    origin_for_new = "FORWARD" if initialized else "BASELINE"

    new_rows = []
    for _, row in eligible.iterrows():
        key = str(row["signal_key"])
        if key in existing:
            continue
        new_rows.append({
            "signal_key": key,
            "tracking_origin": origin_for_new,
            "registered_at": now,
            "last_seen_at": now,
            "registration_version": "1.2",
            "priority": str(row.get("priority", "")),
            "ticker": str(row.get("ticker", "") or ""),
            "issuer_cik": str(row.get("issuer_cik", "") or ""),
            "issuer_name": str(row.get("issuer_name", "") or ""),
            "signal_date": _safe_date_text(row.get("signal_date")),
            "n_insiders": _safe_num(row.get("n_insiders")),
            "cluster_value": _safe_num(row.get("cluster_value")),
            "value_flag": str(row.get("value_flag", "") or ""),
            "role_tag": str(row.get("role_tag", "") or ""),
            "owners": str(row.get("owners", "") or ""),
            "roles": str(row.get("roles", "") or ""),
            "sec_url": str(row.get("sec_url", "") or ""),
            "tradingview_url": str(row.get("tradingview_url", "") or ""),
            "entry_date": _safe_date_text(row.get("entry_date")),
            "entry_open": _safe_num(row.get("entry_open")),
            "latest_price_date": _safe_date_text(row.get("price_date")),
            "latest_close": _safe_num(row.get("current_close")),
            "sessions_observed": _safe_num(row.get("sessions_observed")),
            "current_return_since_entry": _safe_num(row.get("return_since_entry")),
            "current_excess_since_entry": _safe_num(row.get("excess_since_entry")),
            "price_status": str(row.get("price_status", "not_requested") or "not_requested"),
            "operational_bucket": str(row.get("operational_bucket", "") or ""),
            "frozen": False,
            "completed_at": "",
            "exit_5_date": _safe_date_text(row.get("exit_5_date")),
            "return_5": np.nan,
            "excess_5": np.nan,
        })
        existing.add(key)
    if new_rows:
        reg = pd.concat([reg, pd.DataFrame(new_rows)], ignore_index=True)

    # Update dynamic fields. Frozen 5-session outcomes are immutable.
    by_key = {str(r["signal_key"]): r for _, r in eligible.iterrows()}
    for i in reg.index:
        key = str(reg.at[i, "signal_key"])
        row = by_key.get(key)
        if row is None:
            continue
        reg.at[i, "last_seen_at"] = now
        for dst, src in [
            ("entry_date", "entry_date"), ("entry_open", "entry_open"),
            ("latest_price_date", "price_date"), ("latest_close", "current_close"),
            ("sessions_observed", "sessions_observed"),
            ("current_return_since_entry", "return_since_entry"),
            ("current_excess_since_entry", "excess_since_entry"),
        ]:
            val = row.get(src)
            if dst.endswith("date"):
                txt = _safe_date_text(val)
                if txt:
                    reg.at[i, dst] = txt
            else:
                num = _safe_num(val)
                if np.isfinite(num):
                    reg.at[i, dst] = num
        reg.at[i, "price_status"] = str(row.get("price_status", reg.at[i, "price_status"]) or "")
        reg.at[i, "operational_bucket"] = str(row.get("operational_bucket", reg.at[i, "operational_bucket"]) or "")

        already_frozen = _registry_bool(reg.at[i, "frozen"])
        r5 = _safe_num(row.get("return_5"))
        x5 = _safe_num(row.get("excess_5"))
        sessions = _safe_num(row.get("sessions_observed"))
        if (not already_frozen) and np.isfinite(r5) and np.isfinite(x5) and np.isfinite(sessions) and sessions >= 5:
            reg.at[i, "frozen"] = True
            reg.at[i, "completed_at"] = now
            exit_txt = _safe_date_text(row.get("exit_5_date"))
            if exit_txt:
                reg.at[i, "exit_5_date"] = exit_txt
            reg.at[i, "return_5"] = r5
            reg.at[i, "excess_5"] = x5
            reg.at[i, "operational_bucket"] = "COMPLETATO"

    for c in FORWARD_REGISTRY_COLUMNS:
        if c not in reg.columns:
            reg[c] = pd.NA
    reg = reg.drop_duplicates("signal_key", keep="first")
    reg["signal_date"] = pd.to_datetime(reg["signal_date"], errors="coerce")
    reg = reg.sort_values(["signal_date", "priority"], ascending=[False, True]).reset_index(drop=True)
    _atomic_write_csv(reg[FORWARD_REGISTRY_COLUMNS], paths["forward_registry"])
    return load_forward_registry(state_dir)


def forward_registry_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Descriptive 5-session summary. BASELINE and FORWARD are kept separate."""
    if registry is None or registry.empty:
        return pd.DataFrame(columns=["tracking_origin", "priority", "n_registered", "n_completed", "mean_excess_5_pct", "median_excess_5_pct", "win_rate_excess_5_pct"])
    d = registry.copy()
    d["frozen"] = d["frozen"].map(_registry_bool)
    d["excess_5"] = pd.to_numeric(d["excess_5"], errors="coerce")
    rows = []
    for (origin, priority), g in d.groupby(["tracking_origin", "priority"], dropna=False):
        done = g[g["frozen"] & g["excess_5"].notna()].copy()
        rows.append({
            "tracking_origin": origin,
            "priority": priority,
            "n_registered": int(len(g)),
            "n_completed": int(len(done)),
            "mean_excess_5_pct": float(done["excess_5"].mean() * 100) if len(done) else np.nan,
            "median_excess_5_pct": float(done["excess_5"].median() * 100) if len(done) else np.nan,
            "win_rate_excess_5_pct": float((done["excess_5"] > 0).mean() * 100) if len(done) else np.nan,
        })
    return pd.DataFrame(rows)

def state_summary(state_dir: str | Path) -> dict:
    paths = live_state_paths(state_dir)
    idx = _read_csv_or_empty(paths["index"])
    proc = _read_csv_or_empty(paths["processed"], PROCESSED_COLUMNS)
    comp = _read_csv_or_empty(paths["components"], COMPONENT_COLUMNS)
    latest = None
    if not idx.empty and "filing_date" in idx.columns:
        d = pd.to_datetime(idx["filing_date"], errors="coerce").dropna()
        latest = d.max().date() if not d.empty else None
    done = set(proc.get("ACCESSION_NUMBER", pd.Series(dtype=str)).fillna("").astype(str))
    pending = int((~idx.get("accession", pd.Series(dtype=str)).fillna("").astype(str).isin(done)).sum()) if not idx.empty else 0
    return {
        "discovered": int(len(idx)),
        "processed": int(len(proc)),
        "pending": pending,
        "components": int(len(comp)),
        "latest_index_date": latest,
    }


def export_state_zip(state_dir: str | Path) -> bytes:
    paths = live_state_paths(state_dir)
    allowed = [paths["index"], paths["processed"], paths["components"], paths["radar"], paths["prices"], paths["forward_registry"], paths["forward_meta"]]
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in allowed:
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)
    return bio.getvalue()


def import_state_zip(data: bytes, state_dir: str | Path) -> list[str]:
    paths = live_state_paths(state_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    allowed = {p.name for p in [paths["index"], paths["processed"], paths["components"], paths["radar"], paths["prices"], paths["forward_registry"], paths["forward_meta"]]}
    restored: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base not in allowed:
                continue
            target = paths["root"] / base
            target.write_bytes(zf.read(name))
            restored.append(base)
    return restored
