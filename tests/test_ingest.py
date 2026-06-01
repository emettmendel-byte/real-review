"""Unit tests for ingest SQL-building helpers and metro config.

These are pure-function tests — they don't touch the multi-GB JSON or build a DB.
"""

import pytest

from rrs.config import METROS, get_metro
from rrs.ingest import BUSINESS_COLS, _business_filter, _read_json, _sql_struct


def test_sql_struct_renders_duckdb_literal():
    assert _sql_struct({"a": "VARCHAR", "b": "BIGINT"}) == "{'a': 'VARCHAR', 'b': 'BIGINT'}"


def test_business_filter_states_only():
    clause = _business_filter(get_metro("philadelphia"))
    assert clause == "state IN ('PA', 'NJ', 'DE')"


def test_business_filter_includes_cities_when_set():
    clause = _business_filter(get_metro("santa_barbara"))
    assert clause.startswith("state IN ('CA') AND city IN (")
    assert "'Santa Barbara'" in clause and "'Goleta'" in clause


def test_read_json_call_quotes_path_and_sets_format():
    call = _read_json_for_business()
    assert call.startswith("read_json(")
    assert "format = 'newline_delimited'" in call
    assert "'business_id': 'VARCHAR'" in call


def _read_json_for_business():
    from rrs.config import RAW_FILES

    return _read_json(RAW_FILES["business"], BUSINESS_COLS, with_timestamp=False)


def test_read_json_adds_timestamp_format_only_when_requested():
    from rrs.config import RAW_FILES

    with_ts = _read_json(RAW_FILES["review"], {"date": "TIMESTAMP"}, with_timestamp=True)
    without_ts = _read_json(RAW_FILES["business"], BUSINESS_COLS, with_timestamp=False)
    assert "timestampformat" in with_ts
    assert "timestampformat" not in without_ts


def test_get_metro_rejects_unknown():
    with pytest.raises(SystemExit):
        get_metro("los_angeles")  # not in the dataset


def test_default_metro_present():
    assert "philadelphia" in METROS
