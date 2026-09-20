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

SEC_BULK_URLS = (
    # Percorso storico/attuale usato dalla maggior parte dei trimestri SEC.
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip",
    # Nuovo percorso comparso per alcuni dataset recenti (es. 2026 Q2).
    "https://www.sec.gov/files/datastandardsinnovation/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip",
)

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
MISSING_TICKER_VALUES = {"", "NONE", "N/A", "NA", "NAN", "NULL", "NO TICKER"}
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
    def urls(self) -> list[str]:
        return [u.format(year=self.year, quarter=self.quarter) for u in SEC_BULK_URLS]

    @property
    def url(self) -> str:
        # Compatibilità: restituisce l'URL primario, ma il downloader prova tutti i fallback.
        return self.urls[0]


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

    last_error: Exception | None = None
    for url in q.urls:
        try:
            r = requests.get(url, headers=_headers(contact_email), timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            continue

        if r.status_code == 404:
            continue
        try:
            r.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            continue

        if len(r.content) < 1000:
            last_error = RuntimeError(
                f"Download SEC anomalo per {q.label} da {url}: file troppo piccolo."
            )
            continue

        # Controllo minimo: deve essere davvero uno ZIP valido prima di salvarlo.
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                if not zf.namelist():
                    raise RuntimeError("ZIP vuoto")
        except Exception as exc:
            last_error = RuntimeError(f"Risposta non valida per {q.label} da {url}: {exc}")
            continue

        dest.write_bytes(r.content)
        time.sleep(0.15)
        return dest

    if last_error is not None:
        raise RuntimeError(f"Impossibile scaricare {q.label}: {last_error}") from last_error
    return None


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
    out = out.where(~out.isin(MISSING_TICKER_VALUES), "")
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
    # Keep otherwise valid SEC purchases even when the issuer has no public ticker.
    # They remain useful for SEC statistics but will be marked unpriceable for Yahoo.
    s = s[s["FILING_DATE"].notna()].copy()

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
            # SEC occasionally leaves ISSUERNAME blank even when the ticker is valid.
            # Never let a missing display label invalidate an otherwise usable purchase.
            ticker_values = (
                newly_public["ISSUERTRADINGSYMBOL"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
            )
            ticker_values = ticker_values[~ticker_values.isin(MISSING_TICKER_VALUES)]
            ticker = ticker_values.iloc[0] if not ticker_values.empty else ""
            ticker_status = "ok" if ticker else "missing"

            issuer_values = (
                newly_public["ISSUERNAME"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            issuer_values = issuer_values[issuer_values.ne("")]
            issuer_name = issuer_values.iloc[0] if not issuer_values.empty else (ticker or str(issuer))

            # SEC filings can contain economically implausible values because of issuer input errors,
            # unit quirks or unusual securities. Preserve the observation but flag it for review;
            # do not silently delete it or use the flag as a trading score.
            value_review = bool(total_value >= 1_000_000_000)

            results.append({
                "issuer_cik": str(issuer),
                "ticker": ticker,
                "ticker_status": ticker_status,
                "issuer_name": issuer_name,
                "signal_date": pd.Timestamp(filing_date).normalize(),
                "cluster": n_insiders >= int(min_insiders),
                "n_insiders": n_insiders,
                "new_filing_value": new_value,
                "cluster_value": total_value,
                "value_review": value_review,
                "window_start": best["TRANS_DATE"].min(),
                "window_end": best["TRANS_DATE"].max(),
                "owners": names,
                "roles": roles,
                "mean_filing_lag_days": float(best["filing_lag_days"].mean()),
                "accessions": ";".join(accessions),
            })
    return pd.DataFrame(results).sort_values(["signal_date", "cluster_value"], ascending=[False, False]).reset_index(drop=True)


def _safe_text(value) -> str:
    """Return a clean string without ever evaluating pd.NA in boolean context."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _naive_timestamp(value) -> pd.Timestamp:
    """Normalize any datetime-like value to a tz-naive pandas Timestamp."""
    t = pd.Timestamp(value)
    if pd.isna(t):
        return pd.NaT
    try:
        if t.tzinfo is not None:
            # Strip timezone while preserving the displayed calendar time/date.
            t = t.tz_localize(None)
    except Exception:
        try:
            t = t.tz_convert(None)
        except Exception:
            pass
    return t


def _scalar_float(value) -> float:
    """Coerce Yahoo cells to one numeric scalar even if duplicate labels return a Series."""
    if isinstance(value, pd.Series):
        vals = pd.to_numeric(value, errors="coerce").dropna()
        return float(vals.iloc[0]) if not vals.empty else float("nan")
    if isinstance(value, (pd.DataFrame, np.ndarray, list, tuple)):
        arr = pd.to_numeric(pd.Series(np.asarray(value).ravel()), errors="coerce").dropna()
        return float(arr.iloc[0]) if not arr.empty else float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def yahoo_symbol(ticker: str) -> str:
    t = _safe_text(ticker).upper()
    if t in MISSING_TICKER_VALUES:
        return ""
    return t.replace(".", "-")


def _yf_extract_frame(raw: pd.DataFrame, sym: str, single_symbol: bool = False) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if sym in level0:
            f = raw[sym].copy()
        elif sym in level1:
            f = raw.xs(sym, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if not single_symbol:
            return pd.DataFrame()
        f = raw.copy()
    if "Open" not in f.columns or "Close" not in f.columns:
        return pd.DataFrame()
    f = f[["Open", "Close"]].dropna(how="all")
    if f.empty:
        return f
    idx = pd.to_datetime(f.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        try:
            idx = idx.tz_convert(None)
        except Exception:
            pass
    f.index = idx
    return f


def _download_yahoo_prices(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, *, batch_size: int = 80) -> dict[str, pd.DataFrame]:
    """Download Yahoo daily prices in conservative batches so 2k+ tickers do not hit one giant request."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    clean = [s for s in dict.fromkeys(symbols) if s]
    for pos in range(0, len(clean), batch_size):
        batch = clean[pos:pos + batch_size]
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch if len(batch) > 1 else batch[0],
                    start=start.date().isoformat(),
                    end=end.date().isoformat(),
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    timeout=30,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = pd.DataFrame()
            time.sleep(1.5 * (attempt + 1))
        for sym in batch:
            out[sym] = _yf_extract_frame(raw, sym, single_symbol=(len(batch) == 1))
        # Small pause between batches reduces Yahoo throttling without making the study excessively slow.
        time.sleep(0.35)
    return out


def backtest_signals(
    signals: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    benchmark: str = "SPY",
    batch_size: int = 60,
    max_entry_lag_days: int = 7,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-safe Yahoo event study.

    Instead of retaining the full price history for 2k+ symbols in RAM, this version
    downloads one ticker batch, calculates all SEC events belonging to that batch,
    releases the price frames, then continues. Benchmark SPY is held once.

    Entry: next available trading-session OPEN after SEC filing date.
    Exit: CLOSE of horizon-th trading session starting at entry session.
    Excess return: stock return - benchmark return over the same dates.
    """
    if signals.empty:
        return signals.copy(), pd.DataFrame()

    sig = signals.copy()
    sig["signal_date"] = sig["signal_date"].map(_naive_timestamp)
    sig["yahoo_symbol"] = sig["ticker"].map(yahoo_symbol)

    valid_symbols = sorted({s for s in sig["yahoo_symbol"].astype(str) if s})
    start = sig["signal_date"].min() - pd.Timedelta(days=10)
    end = sig["signal_date"].max() + pd.Timedelta(days=max(horizons) * 2 + 30)

    # Benchmark is small and kept once.
    bench_map = _download_yahoo_prices([benchmark], start, end, batch_size=1)
    bench = bench_map.get(benchmark, pd.DataFrame())
    if not bench.empty:
        bench = bench.copy()
        bench.index = pd.DatetimeIndex([_naive_timestamp(x) for x in bench.index])

    rows: list[dict] = []

    def blank_base(row):
        base = row.to_dict()
        for h in horizons:
            base[f"ret_{h}"] = np.nan
            base[f"spy_{h}"] = np.nan
            base[f"excess_{h}"] = np.nan
        return base

    # Preserve rows without usable ticker before Yahoo batches.
    missing_mask = sig["yahoo_symbol"].map(lambda x: not bool(_safe_text(x)))
    for _, row in sig[missing_mask].iterrows():
        base = blank_base(row)
        base["price_status"] = "missing_ticker"
        rows.append(base)

    n_batches = max(1, (len(valid_symbols) + batch_size - 1) // batch_size)
    for batch_no, pos in enumerate(range(0, len(valid_symbols), batch_size), start=1):
        batch = valid_symbols[pos:pos + batch_size]
        prices = _download_yahoo_prices(batch, start, end, batch_size=len(batch))
        batch_sig = sig[sig["yahoo_symbol"].isin(batch)]

        for _, row in batch_sig.iterrows():
            try:
                sym = _safe_text(row.get("yahoo_symbol", ""))
                base = blank_base(row)
                px = prices.get(sym, pd.DataFrame())
                if px.empty or bench.empty:
                    base["price_status"] = "missing_price"
                    rows.append(base)
                    continue

                signal_ts = _naive_timestamp(row.get("signal_date"))
                if pd.isna(signal_ts):
                    base["price_status"] = "invalid_signal_date"
                    rows.append(base)
                    continue

                px = px.copy()
                px.index = pd.DatetimeIndex([_naive_timestamp(x) for x in px.index])
                entry_candidates = px.index[px.index > signal_ts]
                if len(entry_candidates) == 0:
                    base["price_status"] = "no_future_session"
                    rows.append(base)
                    continue

                entry_date = _naive_timestamp(entry_candidates[0])
                entry_lag_days = int((entry_date.normalize() - signal_ts.normalize()).days)
                base["entry_lag_days"] = entry_lag_days
                if entry_lag_days > int(max_entry_lag_days):
                    base["entry_date"] = entry_date
                    base["price_status"] = "stale_symbol_or_gap"
                    rows.append(base)
                    continue
                if entry_date not in bench.index:
                    base["price_status"] = "benchmark_missing_entry"
                    rows.append(base)
                    continue

                entry_open = _scalar_float(px.loc[entry_date, "Open"])
                bench_open = _scalar_float(bench.loc[entry_date, "Open"])
                if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(bench_open) or bench_open <= 0:
                    base["price_status"] = "bad_entry_price"
                    rows.append(base)
                    continue

                loc = px.index.get_loc(entry_date)
                if not isinstance(loc, (int, np.integer)):
                    # Duplicate index rows are unusual; take the first exact location.
                    loc = int(np.flatnonzero(px.index == entry_date)[0])
                base["entry_date"] = entry_date
                base["entry_open"] = entry_open
                base["price_status"] = "ok"

                for h in horizons:
                    exit_pos = int(loc) + h - 1
                    if exit_pos >= len(px.index):
                        continue
                    exit_date = px.index[exit_pos]
                    if exit_date not in bench.index:
                        continue
                    stock_close = _scalar_float(px.iloc[exit_pos]["Close"])
                    spy_close = _scalar_float(bench.loc[exit_date, "Close"])
                    if not np.isfinite(stock_close) or stock_close <= 0 or not np.isfinite(spy_close) or spy_close <= 0:
                        continue
                    ret = stock_close / entry_open - 1.0
                    spy_ret = spy_close / bench_open - 1.0
                    base[f"ret_{h}"] = ret
                    base[f"spy_{h}"] = spy_ret
                    base[f"excess_{h}"] = ret - spy_ret
                rows.append(base)
            except Exception as exc:
                err = blank_base(row)
                err["price_status"] = f"row_error_{type(exc).__name__}"
                err["price_error"] = str(exc)[:240]
                rows.append(err)

        # Release the potentially large Yahoo frames before the next batch.
        del prices
        if progress_callback is not None:
            try:
                progress_callback(batch_no, n_batches, len(rows), len(sig))
            except Exception:
                pass

    event = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for cluster_value, subgroup in event.groupby("cluster", dropna=False):
        for h in horizons:
            x = pd.to_numeric(subgroup[f"excess_{h}"], errors="coerce").dropna()
            if x.empty:
                continue
            lo = x.quantile(0.01)
            hi = x.quantile(0.99)
            trimmed = x[(x >= lo) & (x <= hi)]
            issuer_count = int(subgroup.loc[x.index, "issuer_cik"].astype(str).nunique()) if "issuer_cik" in subgroup.columns else 0
            summary_rows.append({
                "group": "CLUSTER" if (not pd.isna(cluster_value) and bool(cluster_value)) else "SOLO",
                "horizon_sessions": h,
                "n": int(x.size),
                "n_issuers": issuer_count,
                "mean_excess": float(x.mean()),
                "trimmed_mean_excess_1pct": float(trimmed.mean()) if not trimmed.empty else np.nan,
                "median_excess": float(x.median()),
                "win_rate_excess": float((x > 0).mean()),
                "p01_excess": float(x.quantile(0.01)),
                "p99_excess": float(x.quantile(0.99)),
            })
    summary = pd.DataFrame(summary_rows)
    return event, summary

def normalize_loaded_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a previously exported insider_signals CSV.

    This allows the Streamlit app to resume directly from step 3 without rebuilding
    SEC Form 4 signals. The function is intentionally strict on the columns required
    by the event study, while preserving any extra audit columns present in the CSV.
    """
    if df is None or df.empty:
        raise ValueError("Il CSV segnali è vuoto.")

    out = df.copy()
    required = {"signal_date", "ticker", "cluster", "issuer_cik"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError("CSV segnali non compatibile. Colonne mancanti: " + ", ".join(missing))

    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    if out["signal_date"].isna().all():
        raise ValueError("La colonna signal_date non contiene date valide.")

    # CSV round-trips may turn booleans into strings or 0/1. Normalize explicitly.
    def _to_bool(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if pd.isna(v):
            return False
        t = str(v).strip().lower()
        if t in {"true", "1", "yes", "y", "si", "sì"}:
            return True
        if t in {"false", "0", "no", "n", ""}:
            return False
        raise ValueError(f"Valore booleano non riconosciuto nella colonna cluster: {v!r}")

    out["cluster"] = out["cluster"].map(_to_bool)
    if "value_review" in out.columns:
        try:
            out["value_review"] = out["value_review"].map(_to_bool)
        except ValueError:
            out["value_review"] = False
    else:
        out["value_review"] = False

    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()
    if "ticker_status" not in out.columns:
        out["ticker_status"] = np.where(out["ticker"].isin(MISSING_TICKER_VALUES) | out["ticker"].eq(""), "missing", "ok")
    else:
        out["ticker_status"] = out["ticker_status"].fillna("").astype(str)

    # Keep issuer_cik as text to avoid scientific notation / loss of leading zeroes.
    out["issuer_cik"] = out["issuer_cik"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

    # Optional columns used only by the display layer; create safe defaults if absent.
    defaults = {
        "issuer_name": "",
        "n_insiders": 1,
        "cluster_value": np.nan,
        "new_filing_value": np.nan,
        "window_start": pd.NaT,
        "window_end": pd.NaT,
        "roles": "",
        "owners": "",
        "mean_filing_lag_days": np.nan,
        "accessions": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    for col in ["window_start", "window_end"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    return out.sort_values(["signal_date", "cluster_value"], ascending=[False, False], na_position="last").reset_index(drop=True)


def sec_filing_url(accession: str, issuer_cik: str) -> str:
    accession_clean = accession.replace("-", "")
    cik = str(issuer_cik).lstrip("0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
