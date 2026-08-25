from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .models import AnalysisRequest
from .orchestrator import FortyCoolOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FortyCool agent analysis")
    parser.add_argument("--request", required=True, help="Path to an AnalysisRequest JSON file")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    return parser.parse_args()


async def execute(path: str) -> dict:
    request = AnalysisRequest.model_validate_json(Path(path).read_text())
    response = await FortyCoolOrchestrator().run(request)
    return response.model_dump(mode="json")


def main() -> None:
    args = parse_args()
    result = asyncio.run(execute(args.request))
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=not args.compact))


if __name__ == "__main__":
    main()
