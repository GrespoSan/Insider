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
