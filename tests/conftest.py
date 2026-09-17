import pandas as pd
import pytest

from app.data_store import load_data_store


@pytest.fixture(scope="session")
def store():
    """The real support_tickets.csv, loaded once for the whole session."""
    return load_data_store()


@pytest.fixture(scope="session")
def anchor(store):
    return store.anchor


@pytest.fixture(scope="session")
def fixed_anchor():
    """A stable anchor for date-preset arithmetic tests.

    2024-03-30 is a Saturday, which makes week-boundary behaviour explicit.
    """
    return pd.Timestamp("2024-03-30 18:06")
