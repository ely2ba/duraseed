#!/usr/bin/env python3
"""Run the authorized fixed-teacher cloning and common-continuation experiment."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from duraseed.endpoint_continuation import run_continuation
from duraseed.endpoint_inputs import corpus_coordinates, load_inputs, prepare
from duraseed.endpoint_matching import confirm, nominate, select
from duraseed.endpoint_remote import EndpointRemote
from duraseed.endpoint_training import train_student, train_teacher
from duraseed.pilot0_profiles import pre_b_capability_profile
from duraseed.replay_evaluation import evaluate
from duraseed.replay_inputs import read_json
from duraseed.replay_remote import write_json
from duraseed.replay_rolling import sample_manifest_groups
from duraseed.runners import pilot0_concurrent_sampling, pilot0_sampling


def progress(root, phase, **details):
    value = {"at": datetime.now(UTC).isoformat(), "phase": phase, **details}
    write_json(root / "package-progress.json", value)
    print(f"{value['at']} {phase}: {details}", flush=True)


def worker_remote(repo, root, config, worker):
    directory = root / (
        "acquisition" if worker == "acquisition" else f"continuations/{worker}"
    )
    budget = read_json(root / "preflight.json")["workers"][worker]
    preflight_path = directory / "preflight.json"
    write_json(preflight_path, budget)
    worker_config = {
        **config,
        "run_id": f"{root.name}-{worker}",
        "study": "endpoint-clone",
        "preflight": {"path": str(preflight_path.relative_to(repo))},
    }
    write_json(directory / "config.json", worker_config)
    return EndpointRemote(repo, worker_config, directory, config["project_id"])


async def profile(remote, source, checkpoint, manifest, name):
    output = remote.root / "profiles" / name
    await evaluate(
        remote,
        source,
        checkpoint,
        manifest,
        stage="stage_a",
        update=checkpoint["update"],
        arm=name,
        purpose="clone_profile",
        draws=16,
        cap=4096,
        output=output,
        seed_namespace=f"endpoint-clone.profile.{name}",
    )
    value = pre_b_capability_profile(
        origin_kind="endpoint-clone-profile",
        seed=source.seed,
        method=None,
        manifest=manifest,
        evaluation_directories=(output,),
        cover_thresholds=(0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75),
        expected_sampler_path=checkpoint["sampler_path"],
    )
    value.update(
        update=checkpoint["update"],
        stage_a_update=checkpoint["update"],
        checkpoint=checkpoint,
        profile_label=name,
    )
    write_json(output / "profile.json", value)
    return value


async def execute(repo, root):
    if list(root.rglob("remote-call-state.json")) or (root / "result.json").exists():
        raise RuntimeError(
            "already dispatched; inspect existing requests, never blindly relaunch"
        )
    config, source, clone_manifest, confirmation_manifest = load_inputs(repo, root)
    pilot0_concurrent_sampling.EVALUATION_CONCURRENCY = 8
    pilot0_sampling.sample_manifest_groups = sample_manifest_groups
    remote = worker_remote(repo, root, config, "acquisition")
    progress(root, "teacher acquisition", target_updates=30)
    m0 = {
        "state_path": config["m0_state_path"],
        "sampler_path": config["m0_sampler_path"],
    }
    teacher = await train_teacher(remote, source, m0)
    write_json(root / "teacher.json", teacher)

    progress(root, "frozen-teacher corpus", target_completions=6400)
    corpus = await remote.sample_corpus(
        source, clone_manifest, teacher["checkpoint"], remote.root / "corpus"
    )
    progress(root, "student acquisition", target_updates=294)
    indices = {
        (row["generation"]["task_id"], row["generation"]["sample_index"]): index
        for index, row in enumerate(corpus)
    }
    order = [indices[pair] for pair in corpus_coordinates(clone_manifest)]
    student = await train_student(remote, source, m0, corpus, order)
    write_json(root / "student.json", student)
    student_checkpoints = {row["update"]: row for row in student["checkpoints"]}

    progress(root, "clone selection")
    nomination = nominate(student["cadence"], teacher["cadence"][-1])
    write_json(root / "nominations.json", nomination)
    teacher_profile = await profile(
        remote, source, teacher["checkpoint"], source.a_validation, "teacher-selection"
    )
    candidates = {}
    for row in nomination["nominees"]:
        update = row["update"]
        checkpoint = student_checkpoints[update]
        candidates[update] = await profile(
            remote,
            source,
            checkpoint,
            source.a_validation,
            f"student-selection-u{update}",
        )
    selection = select(nomination, teacher_profile, candidates)
    write_json(root / "selection.json", selection)
    if selection["status"] != "SELECTED":
        write_json(root / "matching.json", selection)
        result = {
            "status": "CLONE_UNAVAILABLE",
            "selection": selection,
            "stage_b_started": False,
        }
    else:
        update = selection["selected_update"]
        checkpoint = student_checkpoints[update]
        progress(root, "item-disjoint clone confirmation", selected_update=update)
        first = await profile(
            remote,
            source,
            teacher["checkpoint"],
            confirmation_manifest,
            "teacher-confirmation-1",
        )
        repeat = await profile(
            remote,
            source,
            teacher["checkpoint"],
            confirmation_manifest,
            "teacher-confirmation-2",
        )
        clone = await profile(
            remote, source, checkpoint, confirmation_manifest, "student-confirmation"
        )
        matching = confirm(selection, first, repeat, clone)
        write_json(root / "matching.json", matching)
        if matching["status"] != "MATCHED":
            result = {
                "status": "CLONE_UNAVAILABLE",
                "matching": matching,
                "stage_b_started": False,
            }
        else:
            progress(root, "common continuation", runs=["T1", "T2", "S1", "S2"])

            async def continue_one(label):
                child = worker_remote(repo, root, config, label)
                origin = teacher["checkpoint"] if label.startswith("T") else checkpoint
                try:
                    return await run_continuation(
                        child, source, origin, label, 480 if label.endswith("1") else 20
                    )
                except Exception as error:
                    write_json(
                        child.root / "error.json",
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "type": type(error).__name__,
                            "message": str(error),
                            "action": "inspect durable calls; no automatic retry",
                        },
                    )
                    raise

            # A failed branch does not cancel another branch's in-flight calls.
            outcomes = await asyncio.gather(
                *(continue_one(label) for label in ("T1", "T2", "S1", "S2")),
                return_exceptions=True,
            )
            failures = [
                str(value) for value in outcomes if isinstance(value, BaseException)
            ]
            result = {
                "status": "EXECUTION_FAILED" if failures else "COMPLETED",
                "matching": matching,
                "stage_b_started": True,
                "errors": failures,
                "continuations": [
                    value for value in outcomes if not isinstance(value, BaseException)
                ],
            }
    result["completed_at"] = datetime.now(UTC).isoformat()
    result["acquisition_billing"] = remote.snapshot()
    write_json(root / "result.json", result)
    progress(root, "terminal", status=result["status"])
    if result["status"] == "EXECUTION_FAILED":
        raise RuntimeError("continuation failure; inspect preserved branch evidence")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a simple directory name")
    repo = args.repo.resolve()
    root = repo / "runs/endpoint-clone" / args.run_id
    if not args.execute:
        preflight = prepare(repo, root)
        print(f"Prepared {root}: {preflight['main_usd']} USD maximum", flush=True)
        return
    try:
        asyncio.run(execute(repo, root))
    except Exception as error:
        write_json(
            root / "error.json",
            {
                "at": datetime.now(UTC).isoformat(),
                "type": type(error).__name__,
                "message": str(error),
                "action": "inspect durable calls; never automatically retry",
            },
        )
        raise


if __name__ == "__main__":
    main()
