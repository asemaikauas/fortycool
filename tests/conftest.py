"""Test-wide setup.

Point the run repository at a throwaway database before anything imports the
API module. Without this the suite writes into the real
``.fortycool-data/runs.sqlite3`` - the same file `/demo/verified-run` scans - so
running the tests polluted the store that holds the demo's verified run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DB = Path(tempfile.gettempdir()) / "fortycool-tests" / "runs.sqlite3"
_TEST_DB.parent.mkdir(parents=True, exist_ok=True)
# A fresh file per session keeps runs from one suite out of the next.
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ.setdefault("FORTYCOOL_DB_PATH", str(_TEST_DB))
os.environ["FORTYCOOL_DB_PATH"] = str(_TEST_DB)
