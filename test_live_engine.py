from datetime import date
import pandas as pd
import numpy as np

from live_engine import parse_master_index, parse_form4_xml, build_live_radar, add_operational_columns, update_forward_registry, load_forward_registry, forward_registry_summary


def sample_xml(code="P", acquired="A", is_director="1", is_officer="0", aff="0", owners=1, title="Common Stock", price="10", shares="2000"):
    owner_xml = "".join([
        f"""
        <reportingOwner>
          <reportingOwnerId><rptOwnerCik>{100+i}</rptOwnerCik><rptOwnerName>Owner {i}</rptOwnerName></reportingOwnerId>
          <reportingOwnerRelationship><isDirector>{is_director}</isDirector><isOfficer>{is_officer}</isOfficer><isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
        </reportingOwner>""" for i in range(owners)
    ])
    return f"""<?xml version="1.0"?>
    <ownershipDocument>
      <aff10b5One>{aff}</aff10b5One>
      <issuer><issuerCik>123456</issuerCik><issuerName>Test Corp</issuerName><issuerTradingSymbol>TST</issuerTradingSymbol></issuer>
      {owner_xml}
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <securityTitle><value>{title}</value></securityTitle>
          <transactionDate><value>2026-09-10</value></transactionDate>
          <transactionCoding><transactionCode>{code}</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>{shares}</value></transactionShares>
            <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>{acquired}</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
          <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""


def test_master_index_keeps_only_form4():
    text = """Header
CIK|Company Name|Form Type|Date Filed|Filename
1|A|4|2026-09-10|edgar/data/1/0000000001-26-000001.txt
2|B|4/A|2026-09-10|edgar/data/2/0000000002-26-000002.txt
3|C|10-K|2026-09-10|edgar/data/3/0000000003-26-000003.txt
"""
    df = parse_master_index(text, date(2026, 9, 10))
    assert len(df) == 1
    assert df.iloc[0]["accession"] == "0000000001-26-000001"


def test_qualifying_purchase_parses():
    df, status = parse_form4_xml(sample_xml(), filing_date="2026-09-11", accession="x", min_trade_value=10_000)
    assert status == "qualified_purchase"
    assert len(df) == 1
    assert float(df.iloc[0]["trade_value"]) == 20_000
    assert df.iloc[0]["ISSUERTRADINGSYMBOL"] == "TST"


def test_filters_non_purchase_and_10b5_and_joint():
    df, status = parse_form4_xml(sample_xml(code="A"), filing_date="2026-09-11", accession="x")
    assert df.empty and status == "no_qualifying_purchase"
    df, status = parse_form4_xml(sample_xml(aff="1"), filing_date="2026-09-11", accession="x")
    assert df.empty and status == "excluded_10b5_1"
    df, status = parse_form4_xml(sample_xml(owners=2), filing_date="2026-09-11", accession="x")
    assert df.empty and status == "joint_or_missing_owner"


def test_build_live_radar_core_first_three():
    rows = []
    for i, d in enumerate(["2026-09-01", "2026-09-03", "2026-09-05"]):
        rows.append({
            "ACCESSION_NUMBER": f"a{i}", "ISSUERCIK": "999", "ISSUERNAME": "X Corp", "ISSUERTRADINGSYMBOL": "XYZ",
            "RPTOWNERCIK": f"o{i}", "RPTOWNERNAME": f"Owner {i}", "RPTOWNER_RELATIONSHIP": "Director", "RPTOWNER_TITLE": "",
            "FILING_DATE": pd.Timestamp(d), "TRANS_DATE": pd.Timestamp(d), "DIRECT_INDIRECT_OWNERSHIP": "D", "SECURITY_TITLE": "Common Stock",
            "shares": 2000, "trade_value": 50_000, "n_tranches": 1, "vwap": 25, "filing_lag_days": 0,
            "sec_submission_url": "",
        })
    radar = build_live_radar(pd.DataFrame(rows), today=date(2026, 9, 20))
    core = radar[radar["priority"].eq("CORE")]
    assert len(core) == 1
    assert int(core.iloc[0]["n_insiders"]) == 3
    assert core.iloc[0]["value_tag"] == "VALUE 100–250k (2026$)"


def test_operational_buckets_and_day_progress():
    df = pd.DataFrame([
        {"priority":"CORE","ticker":"A","signal_date":"2026-09-18","n_insiders":3,"cluster_value":120000,"value_tag":"VALUE 100–250k (2026$)","roles":"Chief Executive Officer; Director","price_status":"no_future_session","sessions_observed":np.nan},
        {"priority":"CORE","ticker":"B","signal_date":"2026-09-17","n_insiders":4,"cluster_value":400000,"value_tag":"","roles":"Chief Financial Officer; Director","price_status":"ok","sessions_observed":2},
        {"priority":"WATCH","ticker":"C","signal_date":"2026-09-10","n_insiders":2,"cluster_value":60000,"value_tag":"","roles":"Director","price_status":"ok","sessions_observed":6},
        {"priority":"CORE","ticker":"D","signal_date":"2026-09-10","n_insiders":5,"cluster_value":60000,"value_tag":"","roles":"CEO; CFO","price_status":"missing_price","sessions_observed":np.nan},
    ])
    out = add_operational_columns(df)
    got = dict(zip(out["ticker"], out["operational_bucket"]))
    assert got == {"A":"NUOVO","B":"ATTIVO","D":"DA VERIFICARE","C":"COMPLETATO"}
    row_a = out[out["ticker"].eq("A")].iloc[0]
    row_b = out[out["ticker"].eq("B")].iloc[0]
    row_c = out[out["ticker"].eq("C")].iloc[0]
    row_d = out[out["ticker"].eq("D")].iloc[0]
    assert row_a["day_5"] == "0/5" and row_a["value_flag"] == "VALUE" and row_a["role_tag"] == "CEO"
    assert row_b["day_5"] == "2/5" and row_b["insider_band"] == "4" and row_b["role_tag"] == "CFO"
    assert row_c["day_5"] == "5/5"
    assert row_d["insider_band"] == "5+" and row_d["role_tag"] == "CEO+CFO"



def _registry_snapshot(ticker="AAA", priority="CORE", signal_date="2026-09-18", context=True, sessions=1, r5=np.nan, x5=np.nan, status="ok"):
    return pd.DataFrame([{
        "priority": priority, "ticker": ticker, "issuer_cik": "123", "issuer_name": "Alpha Inc",
        "signal_date": pd.Timestamp(signal_date), "n_insiders": 3 if priority == "CORE" else 2,
        "cluster_value": 150000, "value_tag": "VALUE 100–250k (2026$)", "roles": "CEO; Director",
        "owners": "One; Two; Three", "context_complete": context, "sec_url": "https://sec.example",
        "tradingview_url": "https://tv.example", "entry_date": pd.Timestamp("2026-09-19"), "entry_open": 10.0,
        "price_date": pd.Timestamp("2026-09-20"), "current_close": 10.5, "sessions_observed": sessions,
        "return_since_entry": 0.05, "excess_since_entry": 0.03, "return_5": r5, "excess_5": x5,
        "exit_5_date": pd.Timestamp("2026-09-25") if np.isfinite(r5) else pd.NaT, "price_status": status,
    }])


def test_forward_registry_baseline_then_forward(tmp_path):
    first = _registry_snapshot("AAA")
    reg = update_forward_registry(first, tmp_path, now_utc="2026-09-21T08:00:00Z")
    assert len(reg) == 1
    assert reg.iloc[0]["tracking_origin"] == "BASELINE"

    second = pd.concat([first, _registry_snapshot("BBB", signal_date="2026-09-21")], ignore_index=True)
    second.loc[1, "issuer_cik"] = "456"
    reg2 = update_forward_registry(second, tmp_path, now_utc="2026-09-22T08:00:00Z")
    assert len(reg2) == 2
    origins = dict(zip(reg2["ticker"], reg2["tracking_origin"]))
    assert origins == {"BBB": "FORWARD", "AAA": "BASELINE"}


def test_forward_registry_freezes_5_session_result(tmp_path):
    update_forward_registry(_registry_snapshot("AAA"), tmp_path, now_utc="2026-09-21T08:00:00Z")
    snap_done = _registry_snapshot("BBB", signal_date="2026-09-21", sessions=5, r5=0.12, x5=0.08)
    snap_done.loc[0, "issuer_cik"] = "456"
    reg = update_forward_registry(snap_done, tmp_path, now_utc="2026-09-28T08:00:00Z")
    b = reg[reg["ticker"].eq("BBB")].iloc[0]
    assert bool(b["frozen"]) is True
    assert abs(float(b["excess_5"]) - 0.08) < 1e-12

    # A later refresh with different values must not rewrite the frozen outcome.
    changed = _registry_snapshot("BBB", signal_date="2026-09-21", sessions=8, r5=-0.30, x5=-0.40)
    changed.loc[0, "issuer_cik"] = "456"
    reg2 = update_forward_registry(changed, tmp_path, now_utc="2026-10-01T08:00:00Z")
    b2 = reg2[reg2["ticker"].eq("BBB")].iloc[0]
    assert abs(float(b2["excess_5"]) - 0.08) < 1e-12
    assert abs(float(b2["return_5"]) - 0.12) < 1e-12


def test_forward_registry_excludes_incomplete_context_and_is_idempotent(tmp_path):
    incomplete = _registry_snapshot("AAA", context=False)
    reg = update_forward_registry(incomplete, tmp_path, now_utc="2026-09-21T08:00:00Z")
    assert reg.empty
    complete = _registry_snapshot("AAA", context=True)
    reg2 = update_forward_registry(complete, tmp_path, now_utc="2026-09-22T08:00:00Z")
    reg3 = update_forward_registry(complete, tmp_path, now_utc="2026-09-23T08:00:00Z")
    assert len(reg2) == 1 and len(reg3) == 1
    assert reg2.iloc[0]["tracking_origin"] == "FORWARD"


def test_forward_registry_summary_separates_origins(tmp_path):
    base = _registry_snapshot("AAA", sessions=5, r5=0.10, x5=0.05)
    update_forward_registry(base, tmp_path, now_utc="2026-09-21T08:00:00Z")
    fwd = _registry_snapshot("BBB", signal_date="2026-09-21", sessions=5, r5=0.20, x5=0.10)
    fwd.loc[0, "issuer_cik"] = "456"
    reg = update_forward_registry(fwd, tmp_path, now_utc="2026-09-22T08:00:00Z")
    summ = forward_registry_summary(reg)
    assert set(summ["tracking_origin"]) == {"BASELINE", "FORWARD"}
    f = summ[summ["tracking_origin"].eq("FORWARD")].iloc[0]
    assert int(f["n_completed"]) == 1
    assert abs(float(f["mean_excess_5_pct"]) - 10.0) < 1e-12
