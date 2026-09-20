import pandas as pd
from engine import build_issuer_day_signals


def test_no_future_filing_lookahead():
    rows = [
        dict(ACCESSION_NUMBER="A1", ISSUERCIK="1", ISSUERNAME="Test", ISSUERTRADINGSYMBOL="TST",
             RPTOWNERCIK="101", RPTOWNERNAME="Alice", RPTOWNER_RELATIONSHIP="OFFICER", RPTOWNER_TITLE="CEO",
             FILING_DATE=pd.Timestamp("2026-01-05"), TRANS_DATE=pd.Timestamp("2026-01-03"),
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock", shares=100, trade_value=10000,
             n_tranches=1, vwap=100, filing_lag_days=2),
        dict(ACCESSION_NUMBER="A2", ISSUERCIK="1", ISSUERNAME="Test", ISSUERTRADINGSYMBOL="TST",
             RPTOWNERCIK="102", RPTOWNERNAME="Bob", RPTOWNER_RELATIONSHIP="OFFICER", RPTOWNER_TITLE="CFO",
             FILING_DATE=pd.Timestamp("2026-01-08"), TRANS_DATE=pd.Timestamp("2026-01-04"),
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock", shares=50, trade_value=6000,
             n_tranches=1, vwap=120, filing_lag_days=4),
    ]
    sig = build_issuer_day_signals(pd.DataFrame(rows), window_days=10, min_insiders=2)
    day1 = sig.loc[sig.signal_date.eq(pd.Timestamp("2026-01-05"))].iloc[0]
    day2 = sig.loc[sig.signal_date.eq(pd.Timestamp("2026-01-08"))].iloc[0]
    assert bool(day1.cluster) is False
    assert bool(day2.cluster) is True
    assert int(day2.n_insiders) == 2


if __name__ == "__main__":
    test_no_future_filing_lookahead()
    print("OK")


def test_missing_issuer_name_uses_ticker_fallback():
    rows = [
        dict(ACCESSION_NUMBER="A1", ISSUERCIK="77", ISSUERNAME=None, ISSUERTRADINGSYMBOL="MISS",
             RPTOWNERCIK="701", RPTOWNERNAME="Alice", RPTOWNER_RELATIONSHIP="OFFICER", RPTOWNER_TITLE="CEO",
             FILING_DATE=pd.Timestamp("2026-02-05"), TRANS_DATE=pd.Timestamp("2026-02-03"),
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock", shares=100, trade_value=10000,
             n_tranches=1, vwap=100, filing_lag_days=2),
    ]
    sig = build_issuer_day_signals(pd.DataFrame(rows), window_days=10, min_insiders=2)
    assert len(sig) == 1
    assert sig.iloc[0].ticker == "MISS"
    assert sig.iloc[0].issuer_name == "MISS"


def test_none_ticker_is_preserved_but_marked_missing():
    rows = [
        dict(ACCESSION_NUMBER="A1", ISSUERCIK="88", ISSUERNAME="Private-ish", ISSUERTRADINGSYMBOL="",
             RPTOWNERCIK="801", RPTOWNERNAME="Alice", RPTOWNER_RELATIONSHIP="OFFICER", RPTOWNER_TITLE="CEO",
             FILING_DATE=pd.Timestamp("2026-02-05"), TRANS_DATE=pd.Timestamp("2026-02-03"),
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock", shares=100, trade_value=10000,
             n_tranches=1, vwap=100, filing_lag_days=2),
    ]
    sig = build_issuer_day_signals(pd.DataFrame(rows), window_days=10, min_insiders=2)
    assert len(sig) == 1
    assert sig.iloc[0].ticker == ""
    assert sig.iloc[0].ticker_status == "missing"
    assert sig.iloc[0].issuer_name == "Private-ish"


def test_value_review_flag_only_marks_extreme_values():
    rows = [
        dict(ACCESSION_NUMBER="A1", ISSUERCIK="99", ISSUERNAME="Big", ISSUERTRADINGSYMBOL="BIG",
             RPTOWNERCIK="901", RPTOWNERNAME="Alice", RPTOWNER_RELATIONSHIP="OFFICER", RPTOWNER_TITLE="CEO",
             FILING_DATE=pd.Timestamp("2026-02-05"), TRANS_DATE=pd.Timestamp("2026-02-03"),
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock", shares=1, trade_value=1_500_000_000,
             n_tranches=1, vwap=1_500_000_000, filing_lag_days=2),
    ]
    sig = build_issuer_day_signals(pd.DataFrame(rows), window_days=10, min_insiders=2)
    assert bool(sig.iloc[0].value_review) is True


def test_backtest_rejects_stale_symbol_history(monkeypatch=None):
    import engine
    sig = pd.DataFrame([{
        'issuer_cik':'1','ticker':'TST','ticker_status':'ok','issuer_name':'Test',
        'signal_date':pd.Timestamp('2023-03-14'),'cluster':False,'n_insiders':1,
        'new_filing_value':10000,'cluster_value':10000,'value_review':False,
        'window_start':pd.Timestamp('2023-03-10'),'window_end':pd.Timestamp('2023-03-10'),
        'owners':'Alice','roles':'CEO','mean_filing_lag_days':2.0,'accessions':'A1'
    }])
    idx = pd.to_datetime(['2026-07-20','2026-07-21'])
    px = pd.DataFrame({'Open':[1.0,2.0],'Close':[2.0,2.5]}, index=idx)
    spy = pd.DataFrame({'Open':[100.0,101.0],'Close':[101.0,102.0]}, index=idx)
    original = engine._download_yahoo_prices
    try:
        engine._download_yahoo_prices = lambda symbols, start, end, batch_size=80: {'TST':px, 'SPY':spy}
        event, summary = engine.backtest_signals(sig, max_entry_lag_days=7)
    finally:
        engine._download_yahoo_prices = original
    assert event.iloc[0].price_status == 'stale_symbol_or_gap'
    assert pd.isna(event.iloc[0].get('excess_1', float('nan')))


def test_backtest_accepts_next_session_within_guard(monkeypatch=None):
    import engine
    sig = pd.DataFrame([{
        'issuer_cik':'1','ticker':'TST','ticker_status':'ok','issuer_name':'Test',
        'signal_date':pd.Timestamp('2026-01-02'),'cluster':False,'n_insiders':1,
        'new_filing_value':10000,'cluster_value':10000,'value_review':False,
        'window_start':pd.Timestamp('2026-01-02'),'window_end':pd.Timestamp('2026-01-02'),
        'owners':'Alice','roles':'CEO','mean_filing_lag_days':0.0,'accessions':'A1'
    }])
    idx = pd.to_datetime(['2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09'])
    px = pd.DataFrame({'Open':[10,10,10,10,10],'Close':[11,12,13,14,15]}, index=idx)
    spy = pd.DataFrame({'Open':[100,100,100,100,100],'Close':[101,102,103,104,105]}, index=idx)
    original = engine._download_yahoo_prices
    try:
        engine._download_yahoo_prices = lambda symbols, start, end, batch_size=80: {'TST':px, 'SPY':spy}
        event, summary = engine.backtest_signals(sig, horizons=(1,5), max_entry_lag_days=7)
    finally:
        engine._download_yahoo_prices = original
    assert event.iloc[0].price_status == 'ok'
    assert int(event.iloc[0].entry_lag_days) == 3
    assert abs(event.iloc[0].excess_1 - 0.09) < 1e-12
