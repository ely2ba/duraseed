from __future__ import annotations

from itertools import product
import unittest

from duraseed.tasks.maps import (
    MAPSGenerator,
    MAPSInstruction,
    MAPSInstructionKind,
    MAPSTask,
    canonical_program,
    execute_program,
    render_teacher_answer,
    solve_bfs,
    verify_maps,
)


def _brute_shortest_length(task: MAPSTask) -> int | None:
    for length in range(task.max_program_length + 1):
        for program in product(task.allowed_instructions, repeat=length):
            if execute_program(task.start, task.modulus, program) == task.target:
                return length
    return None


class MAPSProperties(unittest.TestCase):
    def test_bfs_matches_brute_force_on_small_state_spaces(self) -> None:
        instruction_sets = (
            (
                MAPSInstruction(op=MAPSInstructionKind.ADD, argument=1),
                MAPSInstruction(op=MAPSInstructionKind.MUL, argument=2),
            ),
            (
                MAPSInstruction(op=MAPSInstructionKind.ADD, argument=2),
                MAPSInstruction(op=MAPSInstructionKind.NEG),
            ),
        )
        for modulus in (3, 5, 7):
            for allowed in instruction_sets:
                for start in range(modulus):
                    for target in range(modulus):
                        if start == target:
                            continue
                        with self.subTest(
                            modulus=modulus,
                            start=start,
                            target=target,
                            allowed=allowed,
                        ):
                            task = MAPSTask(
                                start=start,
                                modulus=modulus,
                                target=target,
                                allowed_instructions=allowed,
                                max_program_length=4,
                            )
                            result = solve_bfs(task)
                            self.assertEqual(
                                result.shortest_length,
                                _brute_shortest_length(task),
                            )
                            for program in result.shortest_programs:
                                self.assertEqual(len(program), result.shortest_length)
                                self.assertEqual(
                                    execute_program(start, modulus, program),
                                    target,
                                )

    def test_generated_shortest_programs_always_verify(self) -> None:
        for seed in range(40):
            with self.subTest(seed=seed):
                instance = MAPSGenerator(seed).generate(seed % 5)
                self.assertFalse(instance.solution.truncated)
                self.assertGreaterEqual(instance.solution.shortest_length or 0, 2)
                self.assertLessEqual(
                    len(instance.solution.shortest_programs),
                    MAPSGenerator(seed).config.max_shortest_programs,
                )
                for program in instance.solution.shortest_programs:
                    completion = f"<answer>{canonical_program(program)}</answer>"
                    result = verify_maps(completion, instance.task)
                    self.assertTrue(result.valid, result.to_dict())
                    self.assertEqual(result.final_value, instance.task.target)

    def test_generation_is_deterministic_per_root_seed_and_index(self) -> None:
        for seed in range(20):
            generator = MAPSGenerator(seed)
            forward = tuple(generator.generate(index) for index in range(3))
            reverse = tuple(generator.generate(index) for index in reversed(range(3)))
            self.assertEqual(forward, tuple(reversed(reverse)))
            self.assertEqual(
                forward,
                tuple(MAPSGenerator(seed).generate(index) for index in range(3)),
            )

    def test_adversarial_or_invalid_programs_are_never_accepted(self) -> None:
        fixed_invalid = (
            "<answer>RUN 1</answer>",
            "<answer>ADD 1; python()</answer>",
            "<answer>ADD 1 || id</answer>",
            "<answer>ADD 1; </answer>",
            "<answer>ＭＵＬ 2</answer>",
            "<answer>MUL 2</answer> trailing",
            "prefix <answer>NEG</answer>",
            "<answer><script>alert(1)</script></answer>",
        )
        for seed in range(25):
            instance = MAPSGenerator(seed).generate()
            # Default generation requires shortest length >= 2, so no legal
            # one-instruction completion can already be a solution.
            legal_but_wrong = (
                f"<answer>{instance.task.allowed_instructions[0].canonical()}</answer>"
            )
            for completion in (*fixed_invalid, legal_but_wrong):
                with self.subTest(seed=seed, completion=completion):
                    self.assertFalse(verify_maps(completion, instance.task).valid)

    def test_teacher_render_is_canonical_and_verifiable(self) -> None:
        for seed in range(25):
            instance = MAPSGenerator(seed).generate(9)
            rendered = render_teacher_answer(instance)
            self.assertEqual(
                rendered,
                f"<answer>{canonical_program(instance.teacher_program)}</answer>",
            )
            self.assertTrue(verify_maps(rendered, instance.task).valid)


if __name__ == "__main__":
    unittest.main()
