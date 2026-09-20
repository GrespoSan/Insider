from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SEC_BULK_URL = (
    "https://www.sec.gov/files/datastandardsinnovation/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip"
)

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
BAD_SECURITY_WORDS = re.compile(
    r"preferred|warrant|option|unit|note|bond|debenture|right|convertible|debt|phantom|restricted stock unit|rsu",
    re.I,
)
GOOD_SECURITY_WORDS = re.compile(
    r"common stock|common shares?|ordinary shares?|class [a-z] common|common class [a-z]",
    re.I,
)


@dataclass(frozen=True)
class Quarter:
    year: int
    quarter: int

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.quarter}"

    @property
    def filename(self) -> str:
        return f"{self.year}q{self.quarter}_form345.zip"

    @property
    def url(self) -> str:
        return SEC_BULK_URL.format(year=self.year, quarter=self.quarter)


def quarter_range(start_year: int, start_quarter: int, end_year: int, end_quarter: int) -> list[Quarter]:
    out: list[Quarter] = []
    y, q = start_year, start_quarter
    while (y, q) <= (end_year, end_quarter):
        out.append(Quarter(y, q))
        q += 1
        if q == 5:
            y += 1
            q = 1
    return out


def latest_completed_quarter(today: date | None = None) -> Quarter:
    today = today or date.today()
    current_q = (today.month - 1) // 3 + 1
    if current_q == 1:
        return Quarter(today.year - 1, 4)
    return Quarter(today.year, current_q - 1)


def _headers(contact_email: str) -> dict[str, str]:
    email = contact_email.strip()
    if not email or "@" not in email:
        raise ValueError("Inserisci una email di contatto valida per il User-Agent SEC.")
    return {
        "User-Agent": f"IndependentInsiderRadar/0.1 {email}",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def download_quarter(q: Quarter, data_dir: str | Path, contact_email: str, timeout: int = 60) -> Path | None:
    """Download one SEC quarterly ZIP. Returns None if the quarter is not published yet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / q.filename
    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    r = requests.get(q.url, headers=_headers(contact_email), timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if len(r.content) < 1000:
        raise RuntimeError(f"Download SEC anomalo per {q.label}: file troppo piccolo.")
    dest.write_bytes(r.content)
    time.sleep(0.15)
    return dest


def _member_name(zf: zipfile.ZipFile, wanted: str) -> str:
    mapping = {Path(n).name.upper(): n for n in zf.namelist()}
    key = wanted.upper()
    if key not in mapping:
        raise KeyError(f"{wanted} non trovato nello ZIP. Presenti: {sorted(mapping)[:20]}")
    return mapping[key]


def _read_tsv(zf: zipfile.ZipFile, member: str, wanted_cols: Iterable[str]) -> pd.DataFrame:
    name = _member_name(zf, member)
    with zf.open(name) as fh:
        header = pd.read_csv(fh, sep="\t", dtype=str, nrows=0).columns.tolist()
    wanted = [c for c in wanted_cols if c in header]
    with zf.open(name) as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, usecols=wanted, low_memory=False)
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(wanted_cols)]


def load_quarter(zip_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the SEC columns needed by v0.1."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        submission = _read_tsv(
            zf,
            "SUBMISSION.tsv",
            [
                "ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "FORM_TYPE",
                "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE",
            ],
        )
        owners = _read_tsv(
            zf,
            "REPORTINGOWNER.tsv",
            [
                "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "RPTOWNER_TXT",
            ],
        )
        trans = _read_tsv(
            zf,
            "NONDERIV_TRANS.tsv",
            [
                "ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE",
                "TRANS_DATE", "TRANS_CODE", "EQUITY_SWAP_INVOLVED", "TRANS_SHARES",
                "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
                "DIRECT_INDIRECT_OWNERSHIP",
            ],
        )
    return submission, owners, trans


def _affirmative(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def _common_stock_mask(title: pd.Series) -> pd.Series:
    t = title.fillna("").astype(str).str.strip()
    return t.str.contains(GOOD_SECURITY_WORDS, na=False) & ~t.str.contains(BAD_SECURITY_WORDS, na=False)


def _clean_ticker(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str).str.strip().str.upper()
    out = out.where(out.str.match(r"^[A-Z0-9.\-]{1,12}$"), "")
    return out


def build_components(
    submissions: pd.DataFrame,
    owners: pd.DataFrame,
    trans: pd.DataFrame,
    *,
    managers_only: bool = True,
    drop_joint_filings: bool = True,
    exclude_10b5_1: bool = True,
    min_trade_value: float = 0.0,
) -> pd.DataFrame:
    """Create conservative, attributable code-P purchase components."""
    s = submissions.copy()
    o = owners.copy()
    t = trans.copy()

    form = s["DOCUMENT_TYPE"].fillna("").astype(str).str.strip()
    if form.eq("").all() and "FORM_TYPE" in s:
        form = s["FORM_TYPE"].fillna("").astype(str).str.strip()
    s["_FORM"] = form
    s = s[s["_FORM"].eq("4")].copy()  # exclude 4/A in v0.1

    if exclude_10b5_1 and "AFF10B5ONE" in s:
        s = s[~_affirmative(s["AFF10B5ONE"])].copy()

    s["FILING_DATE"] = pd.to_datetime(s["FILING_DATE"], errors="coerce")
    s["ISSUERTRADINGSYMBOL"] = _clean_ticker(s["ISSUERTRADINGSYMBOL"])
    s = s[s["ISSUERTRADINGSYMBOL"].ne("") & s["FILING_DATE"].notna()].copy()

    # A transaction row has no reporting-owner foreign key. Joint filings are ambiguous.
    owner_counts = o.groupby("ACCESSION_NUMBER")["RPTOWNERCIK"].nunique(dropna=True)
    if drop_joint_filings:
        valid_acc = owner_counts[owner_counts.eq(1)].index
        o = o[o["ACCESSION_NUMBER"].isin(valid_acc)].copy()
        s = s[s["ACCESSION_NUMBER"].isin(valid_acc)].copy()

    # Collapse duplicate owner relationship rows conservatively.
    o["RPTOWNER_RELATIONSHIP"] = o["RPTOWNER_RELATIONSHIP"].fillna("")
    o["RPTOWNER_TITLE"] = o["RPTOWNER_TITLE"].fillna("")
    o["RPTOWNER_TXT"] = o["RPTOWNER_TXT"].fillna("")
    o = (
        o.groupby(["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME"], dropna=False, as_index=False)
        .agg({
            "RPTOWNER_RELATIONSHIP": lambda x: " | ".join(sorted({v for v in x if v})),
            "RPTOWNER_TITLE": lambda x: " | ".join(sorted({v for v in x if v})),
            "RPTOWNER_TXT": lambda x: " | ".join(sorted({v for v in x if v})),
        })
    )
    if managers_only:
        rel = o["RPTOWNER_RELATIONSHIP"].str.upper()
        o = o[rel.str.contains("OFFICER|DIRECTOR", regex=True, na=False)].copy()

    t["TRANS_CODE"] = t["TRANS_CODE"].fillna("").astype(str).str.strip().str.upper()
    t["TRANS_ACQUIRED_DISP_CD"] = t["TRANS_ACQUIRED_DISP_CD"].fillna("").astype(str).str.strip().str.upper()
    t = t[t["TRANS_CODE"].eq("P") & t["TRANS_ACQUIRED_DISP_CD"].eq("A")].copy()
    if "EQUITY_SWAP_INVOLVED" in t:
        t = t[~_affirmative(t["EQUITY_SWAP_INVOLVED"])].copy()
    t = t[_common_stock_mask(t["SECURITY_TITLE"])].copy()
    t["TRANS_DATE"] = pd.to_datetime(t["TRANS_DATE"], errors="coerce")
    t["TRANS_SHARES"] = pd.to_numeric(t["TRANS_SHARES"], errors="coerce")
    t["TRANS_PRICEPERSHARE"] = pd.to_numeric(t["TRANS_PRICEPERSHARE"], errors="coerce")
    t = t[
        t["TRANS_DATE"].notna()
        & t["TRANS_SHARES"].gt(0)
        & t["TRANS_PRICEPERSHARE"].gt(0)
    ].copy()
    t["TRADE_VALUE"] = t["TRANS_SHARES"] * t["TRANS_PRICEPERSHARE"]

    x = t.merge(s, on="ACCESSION_NUMBER", how="inner", validate="many_to_one")
    x = x.merge(o, on="ACCESSION_NUMBER", how="inner", validate="many_to_one")
    x = x[x["FILING_DATE"].ge(x["TRANS_DATE"])].copy()
    x = x[(x["FILING_DATE"] - x["TRANS_DATE"]).dt.days.le(366)].copy()

    group_cols = [
        "ACCESSION_NUMBER", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL",
        "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
        "FILING_DATE", "TRANS_DATE", "DIRECT_INDIRECT_OWNERSHIP", "SECURITY_TITLE",
    ]
    comp = (
        x.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            shares=("TRANS_SHARES", "sum"),
            trade_value=("TRADE_VALUE", "sum"),
            n_tranches=("TRANS_SHARES", "size"),
        )
    )
    comp["vwap"] = comp["trade_value"] / comp["shares"]
    comp["filing_lag_days"] = (comp["FILING_DATE"] - comp["TRANS_DATE"]).dt.days
    comp = comp[comp["trade_value"].ge(float(min_trade_value))].copy()
    return comp.sort_values(["FILING_DATE", "ISSUERCIK", "TRANS_DATE"]).reset_index(drop=True)


def combine_quarters(zip_paths: Iterable[str | Path], **component_kwargs) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in zip_paths:
        s, o, t = load_quarter(path)
        frames.append(build_components(s, o, t, **component_kwargs))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Exact economic repeats occasionally appear across filings. Keep earliest public record.
    dedupe = ["ISSUERCIK", "RPTOWNERCIK", "TRANS_DATE", "shares", "trade_value", "DIRECT_INDIRECT_OWNERSHIP"]
    out = out.sort_values("FILING_DATE").drop_duplicates(dedupe, keep="first")
    return out.reset_index(drop=True)


def _owner_summary(w: pd.DataFrame) -> tuple[str, str]:
    owners = (
        w[["RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_TITLE", "RPTOWNER_RELATIONSHIP"]]
        .drop_duplicates("RPTOWNERCIK")
        .sort_values("RPTOWNERNAME")
    )
    names = "; ".join(owners["RPTOWNERNAME"].fillna("").astype(str))
    roles = "; ".join(
        (owners["RPTOWNER_TITLE"].fillna("").astype(str).str.strip()
         .where(owners["RPTOWNER_TITLE"].fillna("").astype(str).str.strip().ne(""), owners["RPTOWNER_RELATIONSHIP"].fillna("")))
    )
    return names, roles


def build_issuer_day_signals(
    components: pd.DataFrame,
    *,
    window_days: int = 10,
    min_insiders: int = 2,
) -> pd.DataFrame:
    """
    One public signal per issuer x filing date.

    A row is labelled cluster=True only using filings already public by that filing date.
    The cluster window is centered on newly disclosed transaction dates and spans +/- window_days
    in transaction-date space. This avoids using a future filing to label an earlier public signal.
    """
    if components.empty:
        return pd.DataFrame()

    results: list[dict] = []
    for issuer, g in components.groupby("ISSUERCIK", sort=False):
        g = g.sort_values(["FILING_DATE", "TRANS_DATE"]).reset_index(drop=True)
        for filing_date, newly_public in g.groupby("FILING_DATE", sort=True):
            public = g[g["FILING_DATE"].le(filing_date)]
            windows: list[pd.DataFrame] = []
            for t0 in newly_public["TRANS_DATE"].dropna().unique():
                t0 = pd.Timestamp(t0)
                lo = t0 - pd.Timedelta(days=window_days)
                hi = t0 + pd.Timedelta(days=window_days)
                w = public[public["TRANS_DATE"].between(lo, hi)].copy()
                if not w.empty:
                    windows.append(w)
            if windows:
                best = max(
                    windows,
                    key=lambda w: (w["RPTOWNERCIK"].nunique(), float(w["trade_value"].sum())),
                )
            else:
                best = newly_public.copy()

            n_insiders = int(best["RPTOWNERCIK"].nunique())
            names, roles = _owner_summary(best)
            new_value = float(newly_public["trade_value"].sum())
            total_value = float(best["trade_value"].sum())
            accessions = sorted(set(best["ACCESSION_NUMBER"].astype(str)))
            ticker = str(newly_public["ISSUERTRADINGSYMBOL"].dropna().iloc[0])
            issuer_name = str(newly_public["ISSUERNAME"].dropna().iloc[0])

            results.append({
                "issuer_cik": str(issuer),
                "ticker": ticker,
                "issuer_name": issuer_name,
                "signal_date": pd.Timestamp(filing_date).normalize(),
                "cluster": n_insiders >= int(min_insiders),
                "n_insiders": n_insiders,
                "new_filing_value": new_value,
                "cluster_value": total_value,
                "window_start": best["TRANS_DATE"].min(),
                "window_end": best["TRANS_DATE"].max(),
                "owners": names,
                "roles": roles,
                "mean_filing_lag_days": float(best["filing_lag_days"].mean()),
                "accessions": ";".join(accessions),
            })
    return pd.DataFrame(results).sort_values(["signal_date", "cluster_value"], ascending=[False, False]).reset_index(drop=True)


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def backtest_signals(
    signals: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    benchmark: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Free daily-price event study using yfinance.
    Entry: next available trading session OPEN after SEC filing date.
    Exit: CLOSE of horizon-th trading session starting at entry session.
    Excess return: stock return - benchmark return over the same dates.
    """
    import yfinance as yf

    if signals.empty:
        return signals.copy(), pd.DataFrame()

    sig = signals.copy()
    sig["signal_date"] = pd.to_datetime(sig["signal_date"])
    symbols = sorted({yahoo_symbol(x) for x in sig["ticker"].dropna().astype(str) if x})
    all_symbols = sorted(set(symbols + [benchmark]))
    start = sig["signal_date"].min() - pd.Timedelta(days=10)
    end = sig["signal_date"].max() + pd.Timedelta(days=max(horizons) * 2 + 30)

    raw = yf.download(
        all_symbols,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    def frame(sym: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            f = raw[sym].copy()
        else:
            f = raw.copy()
        if "Open" not in f.columns or "Close" not in f.columns:
            return pd.DataFrame()
        f = f[["Open", "Close"]].dropna(how="all")
        f.index = pd.to_datetime(f.index).tz_localize(None)
        return f

    bench = frame(benchmark)
    rows: list[dict] = []
    for _, row in sig.iterrows():
        sym = yahoo_symbol(str(row["ticker"]))
        px = frame(sym)
        base = row.to_dict()
        if px.empty or bench.empty:
            base["price_status"] = "missing"
            rows.append(base)
            continue

        entry_candidates = px.index[px.index > pd.Timestamp(row["signal_date"])]
        if len(entry_candidates) == 0:
            base["price_status"] = "no_future_session"
            rows.append(base)
            continue
        entry_date = entry_candidates[0]
        if entry_date not in bench.index:
            base["price_status"] = "benchmark_missing_entry"
            rows.append(base)
            continue
        entry_open = float(px.loc[entry_date, "Open"])
        bench_open = float(bench.loc[entry_date, "Open"])
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(bench_open) or bench_open <= 0:
            base["price_status"] = "bad_entry_price"
            rows.append(base)
            continue

        loc = px.index.get_loc(entry_date)
        base["entry_date"] = entry_date
        base["entry_open"] = entry_open
        base["price_status"] = "ok"
        for h in horizons:
            exit_pos = loc + h - 1
            if exit_pos >= len(px.index):
                base[f"ret_{h}"] = np.nan
                base[f"spy_{h}"] = np.nan
                base[f"excess_{h}"] = np.nan
                continue
            exit_date = px.index[exit_pos]
            if exit_date not in bench.index:
                base[f"ret_{h}"] = np.nan
                base[f"spy_{h}"] = np.nan
                base[f"excess_{h}"] = np.nan
                continue
            stock_close = float(px.loc[exit_date, "Close"])
            spy_close = float(bench.loc[exit_date, "Close"])
            ret = stock_close / entry_open - 1.0
            spy_ret = spy_close / bench_open - 1.0
            base[f"ret_{h}"] = ret
            base[f"spy_{h}"] = spy_ret
            base[f"excess_{h}"] = ret - spy_ret
        rows.append(base)

    event = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for cluster_value, subgroup in event.groupby("cluster", dropna=False):
        for h in horizons:
            x = pd.to_numeric(subgroup[f"excess_{h}"], errors="coerce").dropna()
            if x.empty:
                continue
            summary_rows.append({
                "group": "CLUSTER" if bool(cluster_value) else "SOLO",
                "horizon_sessions": h,
                "n": int(x.size),
                "mean_excess": float(x.mean()),
                "median_excess": float(x.median()),
                "win_rate_excess": float((x > 0).mean()),
            })
    summary = pd.DataFrame(summary_rows)
    return event, summary


def sec_filing_url(accession: str, issuer_cik: str) -> str:
    accession_clean = accession.replace("-", "")
    cik = str(issuer_cik).lstrip("0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
