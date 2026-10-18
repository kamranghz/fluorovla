import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="session")
def cohort_lock():
    return json.load(open(ARTIFACTS / "provenance/cohort_lock.json"))


@pytest.fixture(scope="session")
def descriptor_rows():
    return json.load(open(ARTIFACTS / "references/candidate_pool_geometry/descriptor_rows.json"))
