"""Export completed Pilot-0 scientific records; never read live follow-up runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


RUNS = (
    "pilot0-pair1-seed11-20260825T125100Z",
    "pilot0-pair2-seed29-20260830T204211Z",
)
INPUTS = {
    f"{name}_manifest.json"
    for name in (
        "a_rl_train",
        "a_monitor",
        "a_cadence",
        "a_validation",
        "b_train",
        "b_validation",
    )
} | {"stage_a_prompt_pools.json"}
RECORDS = {"generations.jsonl", "rewards.jsonl", "metrics.jsonl"}
SUMMARIES = {"segment.json", "result.json", "matching.json", "pre-b-profiles.json"}
PRIVATE = {
    "project_id",
    "ledger",
    "ledger_snapshot",
    "token_and_cost_summary",
    "preflight_sha256",
    "evidence_index_sha256",
    "evidence_sha256",
    "evidence_file_count",
    "artifact_sha256",
    "source_bundle_sha256",
}
UNSAFE = re.compile(
    r'tinker://|/(?:Users|home)/|file://|(?i:api_key|access_token|authorization)"'
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()


def export(repo, output, run_id, pair):
    source = repo / "runs/pilot0" / run_id
    assert (
        json.loads((source / "result.json").read_text())["status"]
        == "evidence_collected"
    )
    root = output / f"pilot0-pair{pair}-data"
    root.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(p)
        for p in subprocess.check_output(
            ["rg", "--files", "--no-ignore", str(source)], text=True
        ).splitlines()
    ]
    paths = sorted(
        p
        for p in paths
        if (
            p.parent == source / "pilot-inputs"
            and p.name in INPUTS
            or p == source / "result.json"
            or p.relative_to(source).parts[0] == f"seed-{11 if pair == 1 else 29}"
            and p.name in RECORDS | SUMMARIES
        )
    )
    aliases, hashes, index = {}, {}, []
    totals = Counter()
    populations = {}

    def public(value):
        if isinstance(value, dict):
            return {
                k: public(v)
                for k, v in value.items()
                if k not in PRIVATE and "clock_cycle" not in k
            }
        if isinstance(value, list):
            if not value or isinstance(value[0], (int, float)):
                return value
            return [public(v) for v in value]
        if isinstance(value, str):
            if value.startswith("tinker://"):
                if value not in aliases:
                    aliases[value] = f"checkpoint:pair{pair}:{len(aliases) + 1:06d}"
                return aliases[value]
            return hashes.get(value, value)
        return value

    def destination(path):
        target = root / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def record(path, original, exported, rows=None):
        entry = {
            "path": path.relative_to(repo).as_posix(),
            "source_sha256": original,
            "public_sha256": exported,
        }
        if rows is not None:
            entry["rows"] = rows
        index.append(entry)
        hashes[original] = exported

    # Raw records first, so every summary can reference the exported byte hashes.
    for path in [p for p in paths if p.name in RECORDS]:
        old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
        counts, samples = defaultdict(lambda: ["", 0, 0]), {}
        rows = 0
        with path.open("rb") as inp, destination(path).open("xb") as out:
            for raw in inp:
                value = json.loads(raw)
                projected = public(value)
                for key in (
                    "prompt_text",
                    "completion_text",
                    "completion_token_ids",
                    "completion_logprobs",
                    "reward",
                    "exact_verification",
                ):
                    if key in value:
                        assert projected[key] == value[key], (path.name, key)
                payload = raw if projected == value else encode(projected)
                assert not UNSAFE.search(payload.decode()), path.relative_to(source)
                out.write(payload)
                old_hash.update(raw)
                new_hash.update(payload)
                rows += 1
                if path.name == "generations.jsonl":
                    sample = value["sample_id"]
                    assert sample not in samples
                    samples[sample] = (value["task_id"], value["reward"])
                    item = counts[value["task_id"]]
                    item[0] = value["panel_role"] or ""
                    item[1] += int(value["reward"] == 1)
                    item[2] += 1
                elif path.name == "rewards.jsonl":
                    expected = populations[path.parent][1].pop(value["sample_id"])
                    assert expected == (value["task_id"], value["reward"])
            if path.name == "generations.jsonl":
                populations[path.parent] = (counts, samples)
            elif path.name == "rewards.jsonl":
                assert not populations[path.parent][1], path
        record(
            path,
            "sha256:" + old_hash.hexdigest(),
            "sha256:" + new_hash.hexdigest(),
            rows,
        )
        totals[path.name] += rows

    for path in [p for p in paths if p.name not in RECORDS]:
        raw = path.read_bytes()
        value = json.loads(raw)
        if path.name == "result.json" and "item_counts" in value:
            counts, _ = populations[path.parent]
            expected = [
                dict(task_id=k, panel_role=v[0], successes=v[1], trials=v[2])
                for k, v in sorted(counts.items())
            ]
            assert value["item_counts"] == expected, path
            assert value["row_count"] == sum(v[2] for v in counts.values()), path
            totals["verified_evaluations"] += 1
        projected = public(value)
        payload = raw if projected == value else encode(projected)
        assert not UNSAFE.search(payload.decode()), path.relative_to(source)
        destination(path).write_bytes(payload)
        record(path, digest(raw), digest(payload))

    summary = {
        "pair": pair,
        "run_id": run_id,
        "totals": dict(totals),
        "files": index,
        "checkpoint_aliases": len(aliases),
    }
    (root / "data-index.json").write_bytes(encode(summary))
    shutil.copyfile(repo / "docs/pilot0-data.md", root / "README.md")
    shutil.copyfile(repo / "LICENSE", root / "LICENSE")
    shutil.copyfile(Path(__file__), root / "export_pilot0_data.py")
    print(json.dumps({"pair": pair, **dict(totals), "files": len(index)}), flush=True)
    archive = output / f"pilot0-pair{pair}-data.tar.gz"
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        names = [e["path"] for e in index] + [
            "data-index.json",
            "README.md",
            "LICENSE",
            "export_pilot0_data.py",
        ]
        for name in sorted(names):
            path = root / name
            info = tar.gettarinfo(str(path), arcname=f"{root.name}/{name}")
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    assert archive.stat().st_size < 2 * 1024**3, "Split archive before upload"
    print(f"Ready: {archive.name}, {archive.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for pair, run_id in enumerate(RUNS, 1):
        export(repo, args.output.resolve(), run_id, pair)


if __name__ == "__main__":
    main()
