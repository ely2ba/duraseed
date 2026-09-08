"""Synthetic replay corpus tests; fixtures are not scientific evidence."""

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from duraseed.data.leakage import LeakageAuditError
from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.data.splits import derive_tces_split_seed
from duraseed.pilot0_data import _tces_completion
from duraseed.replay_data import batch_indices, build_corpus, order_key, selection_key
from duraseed.replay_data_tokens import measure_source
from duraseed import replay_data_archive
from duraseed.runtime import RuntimeBundle, SDKBundle, sft_datum
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig, render_prompt
from duraseed.training.reward import verify_task_completion
from duraseed.training.sft import SourceKind, VerifiedSourceRecord


@pytest.fixture(scope="module")
def synthetic_source():
    generator = TCESGenerator(
        derive_tces_split_seed(501, "a_rl_train"),
        TCESGeneratorConfig(
            n_operands=3,
            operand_min=2,
            operand_max=30,
            max_tree_depth=3,
            max_ast_nodes=5,
            max_attempts=256,
            split="a_rl_train",
        ),
    )
    rows, numeric = [], set()
    for index in range(100):
        row = build_tces_record(generator.generate(index))
        key = (tuple(sorted(row.operands)), row.target)
        if key not in numeric:
            rows.append(row)
            numeric.add(key)
        if len(rows) == 32:
            break
    train = build_manifest(
        name="synthetic-replay",
        split="a_rl_train",
        generator_version="1.0.0",
        root_seed=501,
        records=rows,
    )
    empty = build_manifest(
        name="synthetic-empty",
        split="a_monitor",
        generator_version="1.0.0",
        root_seed=501,
        records=[],
        task_family="tces",
    )
    validation = build_manifest(
        name="synthetic-empty-validation",
        split="a_validation",
        generator_version="1.0.0",
        root_seed=501,
        records=[],
        task_family="tces",
    )
    artifact = SimpleNamespace(
        boundary_family_ids=tuple(sorted({row.intended_family for row in rows})),
        intermediate_family_ids=(),
        broad_random_family_ids=(),
        sentinel_family_ids=(),
    )
    return SimpleNamespace(
        seed=11,
        prompt_pools=SimpleNamespace(
            artifact=artifact, a_rl_train_manifest=train, a_monitor_manifest=empty
        ),
        a_validation=validation,
    )


def sample(source, index, *, label="one", step=1, capped=False, incorrect=False):
    task = source.prompt_pools.a_rl_train_manifest.records[index]
    text = "Synthetic rationale preserved verbatim.\n" + _tces_completion(task)
    if incorrect:
        text = "not an answer"
    verification = verify_task_completion(text, task.to_task())
    identifier = f"synthetic:{index}:{label}"
    generation = dict(
        sample_id=identifier,
        task_id=task.task_id,
        task_manifest_id=source.prompt_pools.a_rl_train_manifest.manifest_id,
        seed=11,
        method="B-G",
        purpose="training",
        source_split="a_rl_train",
        task_family="tces",
        assigned_family_id=task.intended_family,
        item_index=task.item_index,
        completion_text=text,
        prompt_text=render_prompt(task.to_task()),
        training_step=step,
        stop_reason="length" if capped else "stop",
        sampled_tokens=4096 if capped else 100,
        sampling_max_tokens=4096,
        sample_index=0,
        sampling_seed=42,
        sampler_checkpoint_path="synthetic:pre-update",
        reward=verification.reward,
    )
    return {
        "generation": generation,
        "reward": {
            "sample_id": identifier,
            "task_id": task.task_id,
            "reward": verification.reward,
            "exact_verification": verification.model_dump(mode="json"),
        },
        "source": {"synthetic": True},
    }


def test_selection_and_order_are_exact_sha_rules_with_equal_prompts(synthetic_source):
    records = [
        sample(synthetic_source, index, label=label)
        for index in range(32)
        for label in ("one", "two")
    ]
    corpus, decisions, audit = build_corpus(synthetic_source, records)
    reverse, _, _ = build_corpus(synthetic_source, reversed(records))
    assert corpus == reverse and audit["status"] == "READY"
    assert len(corpus) == 32 and sum(row["selected"] for row in decisions) == 32
    assert [row["task_id"] for row in corpus] == sorted(
        (row["task_id"] for row in corpus),
        key=lambda value: (
            sha256(f"duraseed-replay-order-v1|11|{value}".encode()).hexdigest(),
            value,
        ),
    )
    for row in corpus:
        candidates = [
            item["generation"]
            for item in records
            if item["generation"]["task_id"] == row["task_id"]
        ]
        chosen = min(
            candidates,
            key=lambda item: selection_key(11, item["task_id"], item["sample_id"]),
        )
        assert row["source"]["sample_id"] == chosen["sample_id"]
        assert row["R-P"]["verified_completion_text"] == chosen["completion_text"]
        assert row["R-S"]["prompt_text"] == row["R-P"]["prompt_text"]
    assert len(set(batch_indices(32, 1))) == 32
    assert sum(len(batch_indices(32, update)) for update in range(1, 295)) == 9408
    assert batch_indices(35, 2) == tuple((32 + j) % 35 for j in range(32))
    assert order_key(11, "task") != order_key(29, "task")


def test_cutoff_cap_verifier_and_capacity_are_not_rerolled(synthetic_source):
    values = [
        sample(synthetic_source, 0, label="cap", capped=True),
        sample(synthetic_source, 1, label="bad", incorrect=True),
        sample(synthetic_source, 2, label="late", step=51),
        sample(synthetic_source, 3, label="keep"),
    ]
    corpus, decisions, audit = build_corpus(synthetic_source, values)
    assert len(corpus) == 1 and audit["status"] == "DATA_BLOCKED"
    assert [row["reasons"] for row in decisions] == [
        ["capped"],
        ["not_verifier_correct"],
        ["outside_fixed_update_cutoff"],
        ["selected"],
    ]
    with pytest.raises(ValueError, match=">=32"):
        batch_indices(31, 1)


def test_verifier_disagreement_and_foreign_or_duplicate_identity_fail(synthetic_source):
    value = sample(synthetic_source, 0)
    bad = deepcopy(value)
    bad["reward"]["reward"] = 0.0
    with pytest.raises(ValueError, match="verifier disagreement"):
        build_corpus(synthetic_source, [bad])
    bad = deepcopy(value)
    bad["generation"]["task_manifest_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="authorized manifest"):
        build_corpus(synthetic_source, [bad])
    with pytest.raises(ValueError, match="duplicate"):
        build_corpus(synthetic_source, [value, value])


def test_sentinel_and_evaluation_leakage_fail(synthetic_source):
    source = deepcopy(synthetic_source)
    family = source.prompt_pools.a_rl_train_manifest.records[0].intended_family
    source.prompt_pools.artifact.sentinel_family_ids = (family,)
    with pytest.raises(ValueError, match="sentinel separation"):
        build_corpus(source, [sample(source, 0)])
    source = deepcopy(synthetic_source)
    source.a_validation = source.prompt_pools.a_rl_train_manifest
    with pytest.raises(LeakageAuditError):
        build_corpus(source, [sample(source, 0)])


class Input:
    def __init__(self, tokens):
        self.tokens, self.length = tokens, len(tokens)

    def to_ints(self):
        return self.tokens


class Renderer:
    def build_supervised_example(self, messages, *, train_on_what):
        tokens = [1, 2, 3] + list(messages[1]["content"].encode()) + [0]
        return Input(tokens), [0, 0, 0] + [1] * (len(tokens) - 3)

    def build_generation_prompt(self, messages, *, role):
        return Input([1, 2, 3])


def test_long_trace_full_conversion_and_completion_only_mean_mask():
    source = VerifiedSourceRecord(
        prompt_text="synthetic prompt",
        verified_completion_text="long text " * 200,
        task_id="sha256:" + "1" * 64,
        task_family="format",
        source_split="format_train",
        source_kind=SourceKind.TASK_AGNOSTIC_FORMAT,
        strategy_family_id=None,
        exact_verification=None,
        source_manifest_id="sha256:" + "2" * 64,
    )

    def convert(messages, renderer, *, max_length, train_on_what, reduction):
        full, mask = renderer.build_supervised_example(
            messages, train_on_what=train_on_what
        )
        assert max_length == full.length > 1024 and reduction == "mean"
        weights = [value / sum(mask[1:]) for value in mask[1:]]
        return SimpleNamespace(
            model_input=Input(full.tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": SimpleNamespace(data=full.tokens[1:]),
                "weights": SimpleNamespace(data=weights),
            },
        )

    sdk = SDKBundle(
        None,
        None,
        SimpleNamespace(LAST_ASSISTANT_MESSAGE="last"),
        convert,
        "synthetic",
        "synthetic",
    )
    runtime = RuntimeBundle(sdk, None, None, Renderer(), None)
    measured = measure_source(runtime, source)
    assert measured == {
        "full_rendered_tokens": 2004,
        "train_tokens": 2003,
        "prompt_tokens": 3,
        "target_tokens": 2001,
    }
    with pytest.raises(ValueError, match="truncated"):
        sft_datum(runtime, source)


def test_archive_requires_sampler_before_its_committed_optimizer_update(
    tmp_path, monkeypatch
):
    run = tmp_path / "synthetic-run"
    source = SimpleNamespace(
        prompt_pools=SimpleNamespace(
            artifact=SimpleNamespace(bg_group_order=()),
            a_rl_train_manifest=SimpleNamespace(manifest_id="manifest"),
        )
    )
    preflight = {"lineage": {"m0_state_path": "state", "m0_sampler_path": "sampler"}}
    segment = dict(
        parent_state_path="state",
        parent_sampler_path="sampler",
        origin_state_path="state",
        run_id="synthetic-run",
        source_manifest_ids={"a_rl_train_manifest_id": "manifest"},
    )
    monkeypatch.setattr(replay_data_archive, "ordered_stage_a_pools", lambda source: {})
    monkeypatch.setattr(replay_data_archive, "read_segment", lambda *args: segment)
    metrics = [{"training_step": step, "method": "B-G"} for step in range(1, 11)]
    calls = []
    for step in range(1, 11):
        key = replay_data_archive.canonical_json_hash(
            {"seed": 11, "method": "B-G", "step": step}
        ).removeprefix("sha256:")
        calls.extend(
            [
                {
                    "operation": "pilot0-ephemeral-sampler",
                    "path": f"ephemeral:synthetic-run:{key}",
                    "sequence": step * 100 + 99,
                    "status": "completed",
                },
                {
                    "operation": "pilot0-stage-a-rl-update",
                    "step": step,
                    "sequence": step * 100 + 18,
                    "status": "completed",
                },
            ]
        )
    monkeypatch.setattr(
        replay_data_archive,
        "jsonl",
        lambda path: iter(metrics if path.name == "metrics.jsonl" else calls),
    )
    with pytest.raises(ValueError, match="before its optimizer update"):
        list(
            replay_data_archive.authenticated_rollouts(run, 11, None, source, preflight)
        )
