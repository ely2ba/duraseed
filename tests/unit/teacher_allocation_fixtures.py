"""Deterministic allocation fixture shared by equivalence tests."""

from __future__ import annotations

from itertools import permutations

from duraseed.data.manifests import (
    TCESTaskManifestRecord,
    build_manifest,
    task_semantic_hash,
)
from duraseed.data.matching import (
    FamilyBlockRecord,
    StructuralCovariates,
    TeacherExampleRecord,
)
from duraseed.data.panel_matching import parse_tces_family_structure
from duraseed.data.panels import (
    FamilyPanelArtifact,
    PanelLabel,
    SeedBlockPanelAssignment,
)
from duraseed.provenance import canonical_json_hash, derive_namespaced_seed
from duraseed.schemas import ExactRational, TCESTask
from duraseed.tasks.tces import generate_teacher_trace, parse_expression
from duraseed.training.reward import verify_task_completion
from duraseed.training.teacher_allocation import (
    TeacherTokenCounts,
    TeacherTraceCandidate,
)
from duraseed.training.teacher_allocation_sources import TeacherAllocationSources

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def fixed_token_measurer(_record) -> TeacherTokenCounts:
    return TeacherTokenCounts(prompt=3, target=2)


def freezer_sources(config) -> TeacherAllocationSources:
    artifact, candidates = _teacher_fixture(
        protected_rows=24, root_seed=config.seed, operand_stride=1
    )
    panel_id = canonical_json_hash(artifact)
    metadata = {
        "panel_artifact_id": panel_id,
        "source_broad_manifest_id": "",
        "source_confirmation_manifest_id": "",
    }
    from duraseed.data.manifests import build_manifest

    broad = build_manifest(
        name="broad",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=5,
        records=[],
        task_family="tces",
    )
    confirmation = build_manifest(
        name="confirmation",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=5,
        records=[],
        task_family="tces",
        parent_manifest_id=broad.manifest_id,
    )
    metadata.update(
        source_broad_manifest_id=broad.manifest_id,
        source_confirmation_manifest_id=confirmation.manifest_id,
    )

    def empty(name: str, split: str):
        return build_manifest(
            name=name,
            split=split,
            generator_version="1.0.0",
            root_seed=config.seed,
            records=[],
            task_family="tces",
        )

    panel_families = artifact.panel_a_family_ids + artifact.panel_b_family_ids
    target_records = tuple(
        row.source_record
        for family in panel_families
        for row in [
            value
            for value in candidates
            if value.source_record.intended_family == family
        ][:16]
    )
    gate_records = tuple(
        row.source_record.model_copy(
            update={
                "split": "a_seed_gate",
                "generator_seed": derive_namespaced_seed(
                    config.seed, "dataset.tces.split", "a_seed_gate"
                ),
            }
        )
        for family in panel_families
        for row in [
            value
            for value in candidates
            if value.source_record.intended_family == family
        ][16 : 16 + config.teacher_dose.gate_items_per_family]
    )
    target = build_manifest(
        name="target",
        split="a_seed_train",
        generator_version="1.0.0",
        root_seed=config.seed,
        records=list(target_records),
        metadata=metadata,
    )
    gate = build_manifest(
        name="gate",
        split="a_seed_gate",
        generator_version="1.0.0",
        root_seed=config.seed,
        records=list(gate_records),
        metadata=metadata,
    )
    return TeacherAllocationSources(
        config,
        artifact,
        broad,
        confirmation,
        empty("rl", "a_rl_train"),
        empty("monitor", "a_monitor"),
        target,
        gate,
        1,
        16,
    )


def _teacher_fixture(
    protected_rows: int = 2,
    root_seed: int = 419,
    operand_stride: int = 5,
) -> tuple[
    FamilyPanelArtifact,
    tuple[TeacherTraceCandidate, ...],
]:
    split = "a_seed_train"
    generator_seed = derive_namespaced_seed(
        root_seed,
        "dataset.tces.split",
        split,
    )
    templates: dict[str, tuple[int, ...]] = {}
    for permutation in permutations(range(5)):
        operands = (10, 11, 12, 13, 14)
        ordered = tuple(operands[index] for index in permutation)
        expression = (
            f"(((({ordered[0]}+{ordered[1]})+{ordered[2]})+{ordered[3]})+{ordered[4]})"
        )
        provisional = TCESTask(
            operands=operands,
            target=ExactRational(numerator=sum(operands)),
            split=split,
        )
        task_id = task_semantic_hash(provisional)
        task = provisional.model_copy(update={"task_id": task_id})
        completion = f"<answer>{expression}</answer>"
        verification = verify_task_completion(completion, task)
        assert verification.reward == 1.0
        family = verification.strategy_family_id
        assert family is not None
        structure = parse_tces_family_structure(family)
        if structure.tree_depth == 5:
            templates.setdefault(family, permutation)
        if len(templates) == 36:
            break
    assert len(templates) == 36

    family_templates = tuple(sorted(templates.items()))
    family_ids = tuple(family for family, _ in family_templates)
    panel_a = family_ids[:12]
    panel_b = family_ids[12:24]
    protected = set(panel_a) | set(panel_b)
    rows: list[tuple[TCESTaskManifestRecord, str]] = []
    for family, permutation in family_templates:
        rows_per_family = protected_rows if family in protected else 16
        for _ in range(rows_per_family):
            item_index = len(rows)
            start = 10 + operand_stride * item_index
            operands = tuple(range(start, start + 5))
            ordered = tuple(operands[index] for index in permutation)
            expression = (
                f"(((({ordered[0]}+{ordered[1]})+{ordered[2]})+{ordered[3]})"
                f"+{ordered[4]})"
            )
            provisional = TCESTask(
                operands=operands,
                target=ExactRational(numerator=sum(operands)),
                split=split,
            )
            task_id = task_semantic_hash(provisional)
            task = provisional.model_copy(update={"task_id": task_id})
            completion = generate_teacher_trace(parse_expression(expression))
            verification = verify_task_completion(completion, task)
            assert verification.reward == 1.0
            assert verification.strategy_family_id == family
            structure = parse_tces_family_structure(family)
            rows.append(
                (
                    TCESTaskManifestRecord(
                        task_id=task_id,
                        split=split,
                        generator_version="1.0.0",
                        generator_seed=generator_seed,
                        item_index=item_index,
                        accepted_attempt=0,
                        prompt_template_id="tces_v1",
                        content_hash=task_id,
                        operands=operands,
                        target=task.target,
                        allowed_ops=task.allowed_ops,
                        constraints=task.constraints,
                        intended_family=family,
                        valid_family_ids=(family,),
                        valid_family_count=1,
                        valid_expression_count=1,
                        minimum_depth=structure.tree_depth,
                    ),
                    completion,
                )
            )
    assert len(rows) == 24 * protected_rows + 12 * 16
    manifest = build_manifest(
        name="teacher-allocation-fixture",
        split=split,
        generator_version="1.0.0",
        root_seed=root_seed,
        records=[record for record, _ in rows],
    )
    completions = {record.task_id: completion for record, completion in rows}
    artifact = FamilyPanelArtifact(
        m0_checkpoint_path="tinker://m0",
        candidate_family_table_manifest_id=SHA_A,
        panel_matching_report_id=SHA_B,
        allocation_seed=3,
        panel_a_family_ids=panel_a,
        panel_b_family_ids=panel_b,
        seed_block_assignments=(
            SeedBlockPanelAssignment(
                training_seed=17,
                targeted_panel=PanelLabel.A,
                sentinel_panel=PanelLabel.B,
            ),
            SeedBlockPanelAssignment(
                training_seed=37,
                targeted_panel=PanelLabel.B,
                sentinel_panel=PanelLabel.A,
            ),
        ),
    )
    candidates: list[TeacherTraceCandidate] = []
    for source in manifest.records:
        assert isinstance(source, TCESTaskManifestRecord)
        structure = parse_tces_family_structure(source.intended_family)
        matching = TeacherExampleRecord(
            record_id=source.task_id,
            family_id=source.intended_family,
            allocation_group=(
                "boundary" if source.intended_family in protected else "random"
            ),
            covariates=StructuralCovariates(
                tree_depth=structure.tree_depth,
                operand_count=len(source.operands),
                operator_multiset=("+", "+", "+", "+"),
                noncommutative_count=0,
                fractional_intermediate=False,
                target_magnitude_bin="fixture",
                valid_family_count_bin="fixture",
                teacher_trace_token_bin="fixture",
            ),
            teacher_prompt_tokens=100,
            teacher_target_tokens=20,
            teacher_trace_format="concise_derivation_v1",
        )
        candidates.append(
            TeacherTraceCandidate(
                source_manifest=manifest,
                source_record=source,
                completion=completions[source.task_id],
                matching_record=matching,
                family_block_record=FamilyBlockRecord(
                    record=matching,
                    fractional_profile=structure.fractional_intermediate_profile,
                    absolute_target=abs(
                        source.target.numerator / source.target.denominator
                    ),
                    valid_family_count=source.valid_family_count,
                ),
            )
        )
    return artifact, tuple(candidates)
