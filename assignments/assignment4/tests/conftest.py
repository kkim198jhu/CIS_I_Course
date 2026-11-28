import importlib
import os
import sys

import pytest

# Make `src` importable when running `pytest` from the assignment4 folder (or anywhere).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full simulated runs (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def student():
    """The package under test: src.student by default (set A4_STUDENT to test another one)."""
    return importlib.import_module(os.environ.get("A4_STUDENT", "src.student"))
