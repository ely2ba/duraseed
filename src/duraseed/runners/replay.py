"""The fixed four-arm supervised replay package, composed from existing runtime."""

from __future__ import annotations

from duraseed.pilot0_data import stage_b_sources
from duraseed.replay_data import batch_indices
from duraseed.replay_evaluation import evaluate, raw_counts, write_profile
from duraseed.replay_inputs import block_inputs, candidate_manifest, read_json
from duraseed.replay_matching import ARMS, STAGE_A_GRID, STAGE_B_GRID, match, nominate
from duraseed.replay_remote import ReplayRemote, write_json
from duraseed.runtime import sft_datum


async def train_segment(remote, source, arm, stage, records, previous, start, stop):
    directory = remote.root / f"seed-{source.seed}" / arm / stage / f"u{stop}"
    coordinate = {
        "seed": source.seed,
        "replay_arm": arm,
        "stage": stage,
        "update": stop,
        "start": start,
        "parent_state_path": previous["state_path"],
    }
    checkpoint_file = directory / "checkpoint.json"
    if checkpoint_file.exists():
        checkpoint = read_json(checkpoint_file)
        if any(checkpoint.get(k) != v for k, v in coordinate.items()):
            raise ValueError("retained replay checkpoint changed its lineage")
        return checkpoint
    if any(directory.glob("update-*.json")):
        raise ValueError(
            "incomplete training segment requires reconciliation; no automatic replay"
        )
    await remote.restore(
        previous["state_path"],
        full_state=start > 0,
        coordinate=coordinate,
        sources=records if start == 0 else (),
    )
    datums = [
        sft_datum(remote.runtime, row, max_length=remote.config["max_length"])
        for row in records
    ]
    for update in range(start + 1, stop + 1):
        indices = (
            batch_indices(len(records), update)
            if stage == "stage_a"
            else tuple(((update - 1) * 32 + j) % len(records) for j in range(32))
        )
        await remote.update(
            [datums[i] for i in indices],
            remote.config[f"{stage}_lr"],
            directory / f"update-{update}.json",
            {**coordinate, "update": update},
        )
    checkpoint = await remote.save(
        f"{remote.run_id}-s{source.seed}-{arm}-{stage}-u{stop}",
        checkpoint_file,
        coordinate,
    )
    return checkpoint


async def stage_a(remote, source, arm, records):
    block = remote.config["blocks"][str(source.seed)]
    previous = {
        "state_path": block["m0_state_path"],
        "sampler_path": block["m0_sampler_path"],
    }
    start, cadence = 0, []
    for stop in STAGE_A_GRID:
        previous = await train_segment(
            remote, source, arm, "stage_a", records, previous, start, stop
        )
        directory = remote.root / f"seed-{source.seed}" / arm / "stage_a" / f"u{stop}"
        # Bind a renderer on clean resume if all preceding training was retained.
        if remote.runtime.renderer is None:
            await remote.restore(
                previous["state_path"],
                full_state=True,
                coordinate={
                    "seed": source.seed,
                    "replay_arm": arm,
                    "stage": "stage_a",
                    "update": stop,
                },
            )
        result = await evaluate(
            remote,
            source,
            previous,
            source.a_cadence,
            stage="stage_a",
            update=stop,
            arm=arm,
            purpose="cadence",
            draws=1,
            cap=4096,
            output=directory / "cadence",
        )
        cadence.append({"update": stop, **raw_counts(result, "targeted")})
        start = stop
    return cadence


async def stage_b(remote, source, arm, selected_update):
    arm_root = remote.root / f"seed-{source.seed}" / arm
    previous = read_json(
        arm_root / "stage_a" / f"u{selected_update}" / "checkpoint.json"
    )
    origin = previous["sampler_path"]
    previous["origin_sampler_path"] = origin
    records = stage_b_sources(source)
    pre_b = arm_root / "pre-b"
    for name, manifest, draws, cap in (
        ("a_validation", source.a_validation, 16, 4096),
        ("a_monitor", source.prompt_pools.a_monitor_manifest, 4, 4096),
        ("b_validation", source.b_validation, 16, 128),
    ):
        await evaluate(
            remote,
            source,
            previous,
            manifest,
            stage="stage_b",
            update=0,
            arm=arm,
            purpose=name,
            draws=draws,
            cap=cap,
            output=pre_b / name,
        )
    write_profile(source, arm, previous, pre_b / "a_validation", pre_b / "profile.json")
    start = 0
    for stop in STAGE_B_GRID[1:]:
        previous = await train_segment(
            remote, source, arm, "stage_b", records, previous, start, stop
        )
        previous["origin_sampler_path"] = origin
        directory = arm_root / "stage_b" / f"u{stop}"
        for name, manifest, draws, cap in (
            ("a_monitor", source.prompt_pools.a_monitor_manifest, 4, 4096),
            ("b_validation", source.b_validation, 16, 128),
            *(
                ([("a_validation", source.a_validation, 16, 4096)])
                if stop == 480
                else []
            ),
        ):
            await evaluate(
                remote,
                source,
                previous,
                manifest,
                stage="stage_b",
                update=stop,
                arm=arm,
                purpose=name,
                draws=draws,
                cap=cap,
                output=directory / name,
            )
        start = stop


async def run_package(remote: ReplayRemote) -> dict:
    sources, records, nominations = {}, {}, {}
    for seed in (11, 29):
        sources[seed], records[seed] = block_inputs(remote.repo, remote.config, seed)
    # No candidate or Stage-B outcomes are opened until all four Stage-A runs end.
    cadence_by_seed = {}
    for seed in (11, 29):
        cadence_by_seed[seed] = {
            arm: await stage_a(remote, sources[seed], arm, records[seed][arm])
            for arm in ARMS
        }
    for seed in (11, 29):
        nominations[str(seed)] = nominate(seed, cadence_by_seed[seed])
    nomination_path = remote.root / "nominations.json"
    if nomination_path.exists() and read_json(nomination_path) != nominations:
        raise ValueError("frozen candidate nominees changed; no replacements")
    write_json(nomination_path, nominations)
    matches = {}
    for seed in (11, 29):
        source, assessment = sources[seed], {}
        for arm in ARMS:
            assessment[arm] = []
            for candidate in nominations[str(seed)]["nominees"][arm]:
                step = candidate["update"]
                directory = remote.root / f"seed-{seed}" / arm / "stage_a" / f"u{step}"
                checkpoint = read_json(directory / "checkpoint.json")
                result = await evaluate(
                    remote,
                    source,
                    checkpoint,
                    candidate_manifest(source),
                    stage="stage_a",
                    update=step,
                    arm=arm,
                    purpose="candidate",
                    draws=16,
                    cap=4096,
                    output=directory / "candidate",
                )
                assessment[arm].append(
                    {"update": step, **raw_counts(result, "targeted")}
                )
        matches[str(seed)] = match(nominations[str(seed)], assessment)
    matching_path = remote.root / "matching.json"
    if matching_path.exists() and read_json(matching_path) != matches:
        raise ValueError("frozen matching decision changed")
    write_json(matching_path, matches)
    # Both block decisions exist before any new Stage-B outcomes are collected.
    for seed in (11, 29):
        selected = matches[str(seed)]["selected"]
        if selected is None:
            continue
        for arm in ARMS:
            await stage_b(remote, sources[seed], arm, selected[arm])
    result = {
        "status": "COMPLETED"
        if any(row["stage_b_allowed"] for row in matches.values())
        else "NO_MATCH",
        "matching": matches,
        "billing": remote.snapshot(),
    }
    write_json(remote.root / "result.json", result)
    return result
