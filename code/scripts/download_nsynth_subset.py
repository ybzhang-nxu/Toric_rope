from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/extract a small NSynth jsonwav subset.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--output-root", type=str, default="data/nsynth")
    parser.add_argument("--split-dir", type=str, default="nsynth-test")
    parser.add_argument("--max-audio", type=int, default=256, help="0 extracts all wav files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(path: str) -> str:
    return Path(path).name


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_root) / args.split_dir
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] | None = None
    selected: list[str] = []
    skipped_existing = 0
    max_audio = int(args.max_audio)

    with urllib.request.urlopen(args.url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                name = member.name
                if member.isfile() and name.endswith("examples.json"):
                    handle = tar.extractfile(member)
                    if handle is not None:
                        metadata = json.loads(handle.read().decode("utf-8"))
                    if max_audio > 0 and len(selected) >= max_audio:
                        break
                    continue
                if not (member.isfile() and name.endswith(".wav") and "/audio/" in name):
                    continue
                key = Path(name).stem
                if max_audio > 0 and len(selected) >= max_audio:
                    if metadata is not None:
                        break
                    continue
                target = audio_dir / safe_name(name)
                if target.exists() and not args.overwrite:
                    skipped_existing += 1
                    selected.append(key)
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                target.write_bytes(handle.read())
                selected.append(key)

    if metadata is not None:
        filtered = {key: metadata[key] for key in selected if key in metadata}
        (out_dir / "examples.json").write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    summary = {
        "url": args.url,
        "output_dir": str(out_dir),
        "audio_dir": str(audio_dir),
        "selected_audio": len(selected),
        "skipped_existing": skipped_existing,
        "metadata_written": metadata is not None,
        "examples_json": str(out_dir / "examples.json"),
        "next_command": (
            "PYTHONPATH=MetricToric/code python "
            "MetricToric/code/scripts/run_v14_nsynth_cqt_masked.py "
            f"--data-root {out_dir} --device cuda --max-samples {len(selected)} --steps 500"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "download_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
