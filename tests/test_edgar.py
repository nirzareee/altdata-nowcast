import pandas as pd
import pytest

from nowcast.edgar import quarterly_revenue


def fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


@pytest.fixture
def facts():
    # FY2023 calendar year, then Q1 2024. Q2 2023 is restated in a later filing.
    main = [
        fact("2023-01-01", "2023-03-31", 100, "2023-04-25"),
        fact("2023-04-01", "2023-06-30", 110, "2023-07-25"),
        fact("2023-04-01", "2023-06-30", 112, "2024-07-25"),  # restated comparative
        fact("2023-01-01", "2023-06-30", 210, "2023-07-25"),  # 6-month YTD, ignored
        fact("2023-07-01", "2023-09-30", 120, "2023-10-25"),
        fact("2023-01-01", "2023-12-31", 460, "2024-02-10", "10-K"),
        fact("2024-01-01", "2024-03-31", 125, "2024-04-25"),
    ]
    # A narrower tag for one quarter must not override the total.
    narrow = [fact("2023-01-01", "2023-03-31", 90, "2023-04-25")]
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": main}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": narrow}},
    }}}


def test_quarters_and_derived_q4(facts):
    q = quarterly_revenue(facts).set_index("end")
    assert list(q["value"]) == [100, 110, 120, 130, 125]
    q4 = q.loc[pd.Timestamp("2023-12-31")]
    assert q4["source"] == "derived_q4"
    assert q4["start"] == pd.Timestamp("2023-10-01")
    assert q4["filed"] == pd.Timestamp("2024-02-10")


def test_point_in_time_keeps_original_value(facts):
    q2 = quarterly_revenue(facts).set_index("end").loc[pd.Timestamp("2023-06-30")]
    assert q2["value"] == 110
    assert q2["value_latest"] == 112
    assert q2["filed"] == pd.Timestamp("2023-07-25")


def test_yoy_growth(facts):
    q1_2024 = quarterly_revenue(facts).set_index("end").loc[pd.Timestamp("2024-03-31")]
    assert q1_2024["yoy_growth"] == pytest.approx(0.25)


def test_no_facts():
    assert quarterly_revenue({"facts": {}}).empty
