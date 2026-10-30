"""Offline fixtures for WS3 tests: WS2's synthetic fixture DB (self-written text)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("OTTER_LLM_MOCK", "1")
os.environ.setdefault("OTTER_NO_DENSE", "1")

from otter.storage import SqliteOntologyStore  # noqa: E402
from otter.storage.fixtures import FIXTURE_IDS, build_fixture_db  # noqa: E402


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("ws3") / "fx.sqlite3"
    build_fixture_db(p)
    return str(p)


@pytest.fixture()
def store(fixture_db):
    return SqliteOntologyStore(fixture_db, use_dense=False)


@pytest.fixture()
def ids():
    return FIXTURE_IDS
