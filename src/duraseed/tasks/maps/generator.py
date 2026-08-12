"""Deterministic, solver-backed MAPS task generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random

from duraseed.schemas import (
    MAPSInstruction,
    MAPSInstructionKind,
    MAPSTask,
    MAPS_MAX_CONSTANT_DIGITS,
    MAPS_MAX_PROGRAM_LENGTH,
)

from .interpreter import (
    Program,
    canonical_program,
    execute_program,
)
from .solver import MAPSSolveResult, solve_bfs


# Versioned rather than inferred at runtime: generation must not depend on an
# ambient prime library or on iteration order.  Pilot calibration may replace
# this explicit set before manifests are frozen.
FROZEN_PRIMES_V1: tuple[int, ...] = (17, 19, 23, 29, 31)
DEFAULT_ADD_CONSTANTS: tuple[int, ...] = (2, 3, 5, 7, 9)
DEFAULT_MUL_CONSTANTS: tuple[int, ...] = (2, 3, 5)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


@dataclass(frozen=True, slots=True)
class MAPSGeneratorConfig:
    """Frozen difficulty controls for deterministic MAPS generation."""

    moduli: tuple[int, ...] = FROZEN_PRIMES_V1
    add_constants: tuple[int, ...] = DEFAULT_ADD_CONSTANTS
    mul_constants: tuple[int, ...] = DEFAULT_MUL_CONSTANTS
    include_neg: bool = True
    allowed_instruction_count: int = 6
    latent_program_min_length: int = 2
    latent_program_max_length: int = 5
    min_shortest_length: int = 2
    max_shortest_length: int = 5
    max_shortest_programs: int = 24
    max_shortest_families: int = 12
    min_latent_distractors: int = 1
    require_inverse_pair: bool = False
    blocked_shortest_family_ids: tuple[str, ...] = ()
    max_attempts: int = 4_096

    def __post_init__(self) -> None:
        integer_fields = (
            "allowed_instruction_count",
            "latent_program_min_length",
            "latent_program_max_length",
            "min_shortest_length",
            "max_shortest_length",
            "max_shortest_programs",
            "max_shortest_families",
            "min_latent_distractors",
            "max_attempts",
        )
        if any(
            isinstance(getattr(self, field), bool)
            or not isinstance(getattr(self, field), int)
            for field in integer_fields
        ):
            raise ValueError("generator count and length controls must be integers")
        if not isinstance(self.include_neg, bool) or not isinstance(
            self.require_inverse_pair, bool
        ):
            raise ValueError("generator feature switches must be booleans")
        if (
            not self.moduli
            or any(
                isinstance(modulus, bool) or not isinstance(modulus, int)
                for modulus in self.moduli
            )
            or any(not _is_prime(modulus) for modulus in self.moduli)
        ):
            raise ValueError("moduli must be a non-empty tuple of primes")
        if len(set(self.moduli)) != len(self.moduli):
            raise ValueError("moduli must not contain duplicates")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.add_constants
        ):
            raise ValueError("ADD constants must be integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.mul_constants
        ):
            raise ValueError("MUL constants must be integers")
        if any(
            len(str(abs(value))) > MAPS_MAX_CONSTANT_DIGITS
            for value in (*self.add_constants, *self.mul_constants)
        ):
            raise ValueError(
                f"instruction constants may have at most "
                f"{MAPS_MAX_CONSTANT_DIGITS} digits"
            )
        if len(set(self.add_constants)) != len(self.add_constants):
            raise ValueError("ADD constants must not contain duplicates")
        if len(set(self.mul_constants)) != len(self.mul_constants):
            raise ValueError("MUL constants must not contain duplicates")

        pool_size = (
            len(self.add_constants) + len(self.mul_constants) + int(self.include_neg)
        )
        if not (1 <= self.allowed_instruction_count <= pool_size):
            raise ValueError("allowed_instruction_count exceeds the instruction pool")
        if not 1 <= self.latent_program_min_length <= self.latent_program_max_length:
            raise ValueError("invalid latent-program length interval")
        if not 1 <= self.min_shortest_length <= self.max_shortest_length:
            raise ValueError("invalid shortest-program length interval")
        if self.max_shortest_length > self.latent_program_max_length:
            raise ValueError("shortest length cannot exceed the maximum latent length")
        if self.latent_program_max_length > MAPS_MAX_PROGRAM_LENGTH:
            raise ValueError(
                f"program lengths may not exceed {MAPS_MAX_PROGRAM_LENGTH}"
            )
        if self.max_shortest_programs < 1 or self.max_shortest_families < 1:
            raise ValueError("ambiguity limits must be positive")
        if not (0 <= self.min_latent_distractors < self.allowed_instruction_count):
            raise ValueError("min_latent_distractors leaves no latent instruction")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if len(set(self.blocked_shortest_family_ids)) != len(
            self.blocked_shortest_family_ids
        ):
            raise ValueError("blocked family IDs must not contain duplicates")
        if not all(
            isinstance(family_id, str) for family_id in self.blocked_shortest_family_ids
        ):
            raise ValueError("blocked family IDs must be strings")


@dataclass(frozen=True, slots=True)
class GeneratedMAPSInstance:
    """A task together with auditable latent and shortest-path provenance."""

    task: MAPSTask
    latent_program: Program
    solution: MAPSSolveResult
    root_seed: int
    item_index: int
    accepted_attempt: int

    @property
    def teacher_program(self) -> Program:
        """The first canonical shortest program, suitable as a teacher trace."""

        return self.solution.shortest_programs[0]


class MAPSGenerationError(RuntimeError):
    """Raised when deterministic rejection sampling exhausts its budget."""


def _instruction_pool(config: MAPSGeneratorConfig) -> tuple[MAPSInstruction, ...]:
    instructions = [
        MAPSInstruction(op=MAPSInstructionKind.ADD, argument=value)
        for value in config.add_constants
    ]
    instructions.extend(
        MAPSInstruction(op=MAPSInstructionKind.MUL, argument=value)
        for value in config.mul_constants
    )
    if config.include_neg:
        instructions.append(MAPSInstruction(op=MAPSInstructionKind.NEG))
    return tuple(sorted(instructions, key=lambda instruction: instruction.canonical()))


def _attempt_rng(root_seed: int, item_index: int, attempt: int) -> random.Random:
    material = f"duraseed:maps:v1:{root_seed}:{item_index}:{attempt}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(seed)


def _has_inverse_pair(
    instructions: tuple[MAPSInstruction, ...],
    modulus: int,
) -> bool:
    for left in instructions:
        for right in instructions:
            if (
                left.op is MAPSInstructionKind.NEG
                and right.op is MAPSInstructionKind.NEG
            ):
                return True
            if (
                left.op is MAPSInstructionKind.ADD
                and right.op is MAPSInstructionKind.ADD
                and left.argument is not None
                and right.argument is not None
                and (left.argument + right.argument) % modulus == 0
            ):
                return True
            if (
                left.op is MAPSInstructionKind.MUL
                and right.op is MAPSInstructionKind.MUL
                and left.argument is not None
                and right.argument is not None
                and (left.argument * right.argument) % modulus == 1
            ):
                return True
    return False


def _task_id(task: MAPSTask) -> str:
    payload = {
        "allowed_instructions": [
            instruction.canonical() for instruction in task.allowed_instructions
        ],
        "max_program_length": task.max_program_length,
        "modulus": task.modulus,
        "start": task.start,
        "target": task.target,
        "version": 1,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "maps-" + hashlib.sha256(encoded).hexdigest()


class MAPSGenerator:
    """Generate each item from an independent deterministic seed stream."""

    def __init__(
        self,
        root_seed: int,
        config: MAPSGeneratorConfig | None = None,
    ) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer")
        self.root_seed = root_seed
        self.config = config or MAPSGeneratorConfig()
        self._pool = _instruction_pool(self.config)

    def generate(self, item_index: int = 0) -> GeneratedMAPSInstance:
        """Generate one accepted instance or fail after ``max_attempts``."""

        if (
            isinstance(item_index, bool)
            or not isinstance(item_index, int)
            or item_index < 0
        ):
            raise ValueError("item_index must be a non-negative integer")
        config = self.config
        blocked_families = set(config.blocked_shortest_family_ids)

        for attempt in range(config.max_attempts):
            rng = _attempt_rng(self.root_seed, item_index, attempt)
            modulus = rng.choice(config.moduli)
            start = rng.randrange(modulus)
            allowed = tuple(
                sorted(
                    rng.sample(self._pool, config.allowed_instruction_count),
                    key=lambda instruction: instruction.canonical(),
                )
            )
            if config.require_inverse_pair and not _has_inverse_pair(allowed, modulus):
                continue

            latent_pool_size = (
                config.allowed_instruction_count - config.min_latent_distractors
            )
            latent_pool = tuple(rng.sample(allowed, latent_pool_size))
            latent_length = rng.randint(
                config.latent_program_min_length,
                config.latent_program_max_length,
            )
            latent_program: Program = tuple(
                rng.choice(latent_pool) for _ in range(latent_length)
            )
            target = execute_program(start, modulus, latent_program)
            if target == start:
                continue

            provisional_task = MAPSTask(
                start=start,
                modulus=modulus,
                target=target,
                allowed_instructions=allowed,
                max_program_length=config.max_shortest_length,
            )
            solution = solve_bfs(
                provisional_task,
                max_programs=config.max_shortest_programs + 1,
            )
            shortest_length = solution.shortest_length
            if shortest_length is None:
                continue
            if not (
                config.min_shortest_length
                <= shortest_length
                <= config.max_shortest_length
            ):
                continue
            if (
                solution.truncated
                or len(solution.shortest_programs) > config.max_shortest_programs
            ):
                continue
            if len(solution.shortest_family_ids) > config.max_shortest_families:
                continue
            if blocked_families.intersection(solution.shortest_family_ids):
                continue

            task = MAPSTask(
                start=provisional_task.start,
                modulus=provisional_task.modulus,
                target=provisional_task.target,
                allowed_instructions=provisional_task.allowed_instructions,
                max_program_length=provisional_task.max_program_length,
                task_id=_task_id(provisional_task),
            )
            return GeneratedMAPSInstance(
                task=task,
                latent_program=latent_program,
                solution=solution,
                root_seed=self.root_seed,
                item_index=item_index,
                accepted_attempt=attempt,
            )

        raise MAPSGenerationError(
            "could not generate an accepted MAPS instance within "
            f"{config.max_attempts} deterministic attempts for item {item_index}"
        )

    def generate_task(self, item_index: int = 0) -> MAPSTask:
        """Generate one task while discarding solver provenance."""

        return self.generate(item_index).task

    def generate_many(
        self,
        count: int,
        *,
        start_index: int = 0,
    ) -> tuple[GeneratedMAPSInstance, ...]:
        """Generate an indexed batch and fail closed on a content duplicate."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
        ):
            raise ValueError("start_index must be a non-negative integer")
        instances = tuple(
            self.generate(index) for index in range(start_index, start_index + count)
        )
        task_ids = [instance.task.task_id for instance in instances]
        if len(task_ids) != len(set(task_ids)):
            raise MAPSGenerationError("generated batch contains a duplicate task")
        return instances


def generate_instance(
    root_seed: int,
    item_index: int = 0,
    config: MAPSGeneratorConfig | None = None,
) -> GeneratedMAPSInstance:
    """Functional entry point for deterministic instance generation."""

    return MAPSGenerator(root_seed, config).generate(item_index)


def generate_task(
    root_seed: int,
    item_index: int = 0,
    config: MAPSGeneratorConfig | None = None,
) -> MAPSTask:
    """Functional entry point returning only the task schema."""

    return MAPSGenerator(root_seed, config).generate_task(item_index)


def render_prompt(task: MAPSTask) -> str:
    """Render the exact Stage-B prompt shape declared in the specification."""

    allowed = "\n".join(
        f"- {instruction.canonical()}" for instruction in task.allowed_instructions
    )
    return (
        f"Start value: {task.start}\n"
        f"Modulus: {task.modulus}\n"
        f"Target: {task.target}\n"
        "Allowed instructions:\n"
        f"{allowed}\n"
        f"Maximum program length: {task.max_program_length}\n\n"
        f"Instructions execute from top to bottom modulo {task.modulus}.\n"
        "Separate multiple instructions with semicolons; newlines alone are "
        "not valid separators.\n"
        "Return one valid program inside <answer>...</answer>."
    )


def render_teacher_answer(instance: GeneratedMAPSInstance) -> str:
    """Render a verified canonical shortest teacher completion."""

    return f"<answer>{canonical_program(instance.teacher_program)}</answer>"
