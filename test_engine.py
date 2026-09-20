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


def test_backtest_handles_timezone_mismatch():
    import engine
    sig = pd.DataFrame([{
        'issuer_cik':'1','ticker':'TST','ticker_status':'ok','issuer_name':'Test',
        'signal_date':pd.Timestamp('2026-01-02', tz='UTC'),'cluster':False,'n_insiders':1,
        'new_filing_value':10000,'cluster_value':10000,'value_review':False,
        'window_start':pd.Timestamp('2026-01-02'),'window_end':pd.Timestamp('2026-01-02'),
        'owners':'Alice','roles':'CEO','mean_filing_lag_days':0.0,'accessions':'A1'
    }])
    idx = pd.DatetimeIndex(pd.to_datetime(['2026-01-05','2026-01-06'])).tz_localize('America/New_York')
    px = pd.DataFrame({'Open':[10,10],'Close':[11,12]}, index=idx)
    spy = pd.DataFrame({'Open':[100,100],'Close':[101,102]}, index=idx)
    original = engine._download_yahoo_prices
    try:
        engine._download_yahoo_prices = lambda symbols, start, end, batch_size=80: {'TST':px, 'SPY':spy}
        event, summary = engine.backtest_signals(sig, horizons=(1,), max_entry_lag_days=7)
    finally:
        engine._download_yahoo_prices = original
    assert event.iloc[0].price_status == 'ok'
    assert int(event.iloc[0].entry_lag_days) == 3


def test_yahoo_symbol_handles_pd_na():
    import engine
    assert engine.yahoo_symbol(pd.NA) == ''


def test_normalize_loaded_signals_csv_roundtrip():
    import pandas as pd
    from engine import normalize_loaded_signals

    raw = pd.DataFrame({
        "signal_date": ["2026-06-30", "2026-06-29"],
        "ticker": ["aapl", ""],
        "cluster": ["True", "False"],
        "issuer_cik": [320193, 123456],
    })
    out = normalize_loaded_signals(raw)
    assert len(out) == 2
    assert set(out["cluster"].tolist()) == {True, False}
    assert "AAPL" in set(out["ticker"])
    assert "ticker_status" in out.columns
    assert "value_review" in out.columns


def test_normalize_loaded_signals_rejects_view_missing_core_column():
    import pandas as pd
    import pytest
    from engine import normalize_loaded_signals

    raw = pd.DataFrame({"signal_date": ["2026-01-01"], "ticker": ["AAPL"], "cluster": [True]})
    with pytest.raises(ValueError):
        normalize_loaded_signals(raw)


def test_label_cluster_episodes_first_and_repeat():
    import pandas as pd
    from engine import label_cluster_episodes
    df = pd.DataFrame([
        {"issuer_cik":"1","signal_date":"2026-01-01","n_insiders":1,"cluster":False},
        {"issuer_cik":"1","signal_date":"2026-01-05","n_insiders":2,"cluster":True},
        {"issuer_cik":"1","signal_date":"2026-01-08","n_insiders":3,"cluster":True},
        {"issuer_cik":"1","signal_date":"2026-01-25","n_insiders":2,"cluster":True},
    ])
    out = label_cluster_episodes(df, min_insiders=2, episode_gap_days=10)
    assert out.analysis_group.tolist() == ["SOLO","FIRST_CLUSTER","REPEAT_CLUSTER","FIRST_CLUSTER"]


def test_threshold_reclassifies_two_insider_as_solo():
    import pandas as pd
    from engine import label_cluster_episodes
    df = pd.DataFrame([
        {"issuer_cik":"1","signal_date":"2026-01-05","n_insiders":2,"cluster":True},
        {"issuer_cik":"1","signal_date":"2026-01-08","n_insiders":3,"cluster":True},
    ])
    out = label_cluster_episodes(df, min_insiders=3, episode_gap_days=10)
    assert out.analysis_group.tolist() == ["SOLO","FIRST_CLUSTER"]


def test_bootstrap_difference_runs_and_returns_ci():
    import pandas as pd
    from engine import issuer_cluster_bootstrap_difference
    rows=[]
    for issuer in range(1,20):
        rows.append({"issuer_cik":str(issuer),"signal_date":pd.Timestamp("2026-01-01"),"n_insiders":1,"cluster":False,"excess_1":0.00})
        rows.append({"issuer_cik":str(issuer),"signal_date":pd.Timestamp("2026-01-20"),"n_insiders":2,"cluster":True,"excess_1":0.02})
    df=pd.DataFrame(rows)
    out=issuer_cluster_bootstrap_difference(df,horizons=(1,),n_boot=200,seed=1)
    assert len(out)==1
    assert out.iloc[0].observed_diff > 0
    assert "ci95_low" in out.columns


def test_normalize_loaded_event_study():
    import pandas as pd
    from engine import normalize_loaded_event_study
    raw=pd.DataFrame({"issuer_cik":[1],"signal_date":["2026-01-01"],"cluster":["True"],"n_insiders":[2],"excess_1":[0.01]})
    out=normalize_loaded_event_study(raw)
    assert bool(out.iloc[0].cluster) is True
    assert int(out.iloc[0].n_insiders)==2
