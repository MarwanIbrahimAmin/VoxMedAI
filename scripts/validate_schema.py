from __future__ import annotations

import argparse
import json
import sys

from voxmed.schema_validation import dump_normalized_payload, validate_extraction_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and normalize a VoxMed extraction payload.")
    parser.add_argument("path", nargs="?", help="Path to a JSON file. Reads stdin if omitted.")
    args = parser.parse_args()

    raw_text = sys.stdin.read() if not args.path else open(args.path, "r", encoding="utf-8").read()
    payload = json.loads(raw_text)
    errors = validate_extraction_payload(payload)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(dump_normalized_payload(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())