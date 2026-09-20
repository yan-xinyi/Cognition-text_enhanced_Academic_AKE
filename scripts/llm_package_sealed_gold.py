from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _bootstrap import ROOT
from llm_ake.util import atomic_write_json, sha256_file, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Byte-copy raw test files into the sealed gold directory without parsing gold content")
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.sources) != 3:
        raise SystemExit("Exactly three raw corpus test JSON files are required")
    destination = ROOT / "data" / "_sealed_test_gold" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for source in args.sources:
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / f"{source.parent.name}_{source.name}"
        shutil.copy2(source, target)
        files.append({
            "path": target.relative_to(ROOT).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "source_path_at_packaging": str(source.resolve()),
            "copy_mode": "opaque_byte_copy_no_semantic_gold_parse",
        })
    manifest = {
        "schema_version": 1,
        "status": "SEALED",
        "packaged_at_utc": utc_now(),
        "formal_inference_access": "forbidden",
        "unseal_condition": "TEST_PREDICTIONS_FREEZE.json verified and explicit --unlock-test-gold",
        "files": files,
    }
    atomic_write_json(ROOT / "data" / "_sealed_test_gold" / "SEALED_GOLD_MANIFEST.json", manifest)
    print("Sealed raw test files:", len(files))


if __name__ == "__main__":
    main()

