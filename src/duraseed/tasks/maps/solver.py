"""Breadth-first shortest-program solver for MAPS tasks."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from duraseed.schemas import MAPSInstruction, MAPSTask

from .interpreter import (
    Program,
    apply_instruction,
    canonical_program,
    program_family_id,
)


@dataclass(frozen=True, slots=True)
class MAPSSolveResult:
    """All discovered canonical shortest programs and their families."""

    shortest_length: int | None
    shortest_programs: tuple[Program, ...]
    shortest_family_ids: tuple[str, ...]
    explored_states: int
    truncated: bool = False

    @property
    def reachable(self) -> bool:
        return self.shortest_length is not None

    @property
    def programs(self) -> tuple[Program, ...]:
        """Readable alias for callers that already know these are shortest."""

        return self.shortest_programs

    @property
    def families(self) -> tuple[str, ...]:
        """Readable alias for the canonical shortest family IDs."""

        return self.shortest_family_ids


def _canonical_instruction_set(
    instructions: Sequence[MAPSInstruction],
) -> tuple[MAPSInstruction, ...]:
    by_text: dict[str, MAPSInstruction] = {}
    for instruction in instructions:
        by_text[instruction.canonical()] = instruction
    return tuple(by_text[text] for text in sorted(by_text))


def solve_bfs(
    task: MAPSTask,
    *,
    max_programs: int | None = None,
) -> MAPSSolveResult:
    """Solve ``task`` using BFS over the finite residue-state graph.

    BFS records every predecessor edge at the first-seen distance, then
    backtracks those edges to enumerate every canonical shortest program.  If
    ``max_programs`` is supplied, enumeration is bounded and ``truncated``
    reports that further shortest programs exist.
    """

    if max_programs is not None and max_programs < 1:
        raise ValueError("max_programs must be positive when supplied")

    start = task.start % task.modulus
    target = task.target % task.modulus
    if start == target:
        empty: Program = ()
        return MAPSSolveResult(
            shortest_length=0,
            shortest_programs=(empty,),
            shortest_family_ids=(program_family_id(empty),),
            explored_states=1,
        )

    instructions = _canonical_instruction_set(task.allowed_instructions)
    distances: dict[int, int] = {start: 0}
    parents: dict[int, list[tuple[int, MAPSInstruction]]] = {start: []}
    queue: deque[int] = deque([start])
    target_distance: int | None = None

    while queue:
        state = queue.popleft()
        distance = distances[state]
        if distance >= task.max_program_length:
            continue
        if target_distance is not None and distance + 1 > target_distance:
            continue

        for instruction in instructions:
            next_state = apply_instruction(state, instruction, task.modulus)
            next_distance = distance + 1
            known_distance = distances.get(next_state)
            edge = (state, instruction)
            if known_distance is None:
                distances[next_state] = next_distance
                parents[next_state] = [edge]
                queue.append(next_state)
                if next_state == target:
                    target_distance = next_distance
            elif known_distance == next_distance:
                # Different instructions may induce the same transition modulo
                # p.  They remain distinct canonical programs.
                parents[next_state].append(edge)

    if target_distance is None:
        return MAPSSolveResult(
            shortest_length=None,
            shortest_programs=(),
            shortest_family_ids=(),
            explored_states=len(distances),
        )

    def backtrack(state: int) -> Iterator[Program]:
        if state == start:
            yield ()
            return
        ordered_edges = sorted(
            parents[state],
            key=lambda edge: (edge[1].canonical(), edge[0]),
        )
        for previous_state, instruction in ordered_edges:
            for prefix in backtrack(previous_state):
                yield (*prefix, instruction)

    programs_by_text: dict[str, Program] = {}
    truncated = False
    for program in backtrack(target):
        text = canonical_program(program)
        programs_by_text.setdefault(text, program)
        if max_programs is not None and len(programs_by_text) > max_programs:
            truncated = True
            break

    ordered_text = sorted(programs_by_text)
    if max_programs is not None:
        ordered_text = ordered_text[:max_programs]
    programs = tuple(programs_by_text[text] for text in ordered_text)
    families = tuple(sorted({program_family_id(program) for program in programs}))
    return MAPSSolveResult(
        shortest_length=target_distance,
        shortest_programs=programs,
        shortest_family_ids=families,
        explored_states=len(distances),
        truncated=truncated,
    )


def shortest_program(task: MAPSTask) -> Program | None:
    """Return the lexicographically first canonical shortest program."""

    result = solve_bfs(task)
    return result.shortest_programs[0] if result.reachable else None


# The shorter name is useful at call sites while retaining the algorithm in
# the public ``solve_bfs`` name.
solve = solve_bfs
