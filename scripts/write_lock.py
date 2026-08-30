"""Turn a pip install report into a hash-pinned requirements lock.

Usage:
    python -m pip install --dry-run --ignore-installed \
        --report report.json \
        earthengine-api fastapi httpx numpy pandas psycopg[binary] pydantic \
        reportlab uvicorn pytest pip-audit
    python scripts/write_lock.py report.json > requirements.lock

Install the result with ``pip install --require-hashes -r requirements.lock`` so
a build cannot silently pick up a different release than the one reviewed.
"""

from __future__ import annotations

import json
import platform
import sys

HEADER = """# Hash-pinned dependency lock for reproducible builds.
#
# Every dependency in pyproject.toml is an open range, so without this file
# each build resolved to whatever was newest on PyPI at the time and a
# compromised release inside the range would install silently.
#
# RESOLVED FOR: {target}
#
# The hashes below are for the wheels that fit that interpreter and platform.
# numpy, pandas and reportlab ship platform-specific wheels, so running
# --require-hashes on a different OS, architecture, or Python version fails with
# a hash mismatch. That is the lock working, not a corrupt file: regenerate it
# on the target platform, or install from pyproject.toml there instead.
#
# Regenerate with:
#   python -m pip install --dry-run --ignore-installed --report report.json <requirements>
#   python scripts/write_lock.py report.json > requirements.lock
#
# Install with:  pip install --require-hashes -r requirements.lock
"""


def target_description() -> str:
    return (
        f"{sys.implementation.name} "
        f"{sys.version_info.major}.{sys.version_info.minor} on "
        f"{platform.system().lower()} {platform.machine()}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    print(HEADER.format(target=target_description()))
    for item in sorted(report["install"], key=lambda i: i["metadata"]["name"].lower()):
        name = item["metadata"]["name"]
        version = item["metadata"]["version"]
        digest = (
            item.get("download_info", {})
            .get("archive_info", {})
            .get("hashes", {})
            .get("sha256")
        )
        if not digest:
            print(f"# {name}=={version} has no sha256 in the report", file=sys.stderr)
            continue
        print(f"{name}=={version} \\")
        print(f"    --hash=sha256:{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
