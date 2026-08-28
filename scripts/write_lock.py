"""Turn a pip install report into a hash-pinned requirements lock.

Usage:
    python -m pip install --dry-run --ignore-installed \
        --report report.json \
        earthengine-api fastapi httpx numpy pandas pydantic reportlab uvicorn pytest
    python scripts/write_lock.py report.json > requirements.lock

Install the result with ``pip install --require-hashes -r requirements.lock`` so
a build cannot silently pick up a different release than the one reviewed.
"""

from __future__ import annotations

import json
import sys

HEADER = """# Hash-pinned dependency lock for reproducible builds.
#
# Every dependency in pyproject.toml is an open range, so without this file
# each build resolved to whatever was newest on PyPI at the time and a
# compromised release inside the range would install silently.
#
# Regenerate with:
#   python -m pip install --dry-run --ignore-installed --report report.json <requirements>
#   python scripts/write_lock.py report.json > requirements.lock
#
# Install with:  pip install --require-hashes -r requirements.lock
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    print(HEADER)
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
