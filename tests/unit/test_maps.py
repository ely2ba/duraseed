from __future__ import annotations

import unittest

from duraseed.tasks.maps import (
    MAPSFailureCode,
    MAPSGenerator,
    MAPSGeneratorConfig,
    MAPSInstruction,
    MAPSInstructionKind,
    MAPSTask,
    ProgramParseError,
    canonical_program,
    execute_program,
    parse_program,
    render_prompt,
    render_teacher_answer,
    solve_bfs,
    verify_maps,
)


def instruction(
    op: MAPSInstructionKind,
    argument: int | None = None,
) -> MAPSInstruction:
    return MAPSInstruction(op=op, argument=argument)


class MAPSInterpreterTests(unittest.TestCase):
    def test_strict_parse_canonicalize_and_execute(self) -> None:
        program = parse_program(" \nMUL 3;\tADD 5 ; NEG\n")
        self.assertEqual(canonical_program(program), "MUL 3; ADD 5; NEG")
        self.assertEqual(execute_program(7, 31, program), 5)

    def test_signed_legal_constants_are_exact(self) -> None:
        program = parse_program("ADD -2; MUL -3")
        self.assertEqual(canonical_program(program), "ADD -2; MUL -3")
        self.assertEqual(execute_program(1, 7, program), 3)

    def test_non_dsl_text_is_rejected(self) -> None:
        invalid_programs = (
            "",
            "add 2",
            "ADD 02",
            "ADD +2",
            "ADD -0",
            "ADD\t2",
            "ADD 2;",
            "ADD 2\nMUL 3",
            "x = 2",
            "__import__('os').system('id')",
            "ADD 2 # comment",
            "NEG()",
            "MUL ２",
            "ＮＥＧ",
        )
        for program in invalid_programs:
            with self.subTest(program=program), self.assertRaises(ProgramParseError):
                parse_program(program)

    def test_instruction_and_task_validation_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            instruction(MAPSInstructionKind.NEG, 1)
        with self.assertRaises(ValueError):
            instruction(MAPSInstructionKind.ADD)
        add = instruction(MAPSInstructionKind.ADD, 2)
        with self.assertRaises(ValueError):
            MAPSTask(
                start=0,
                modulus=7,
                target=1,
                allowed_instructions=(add, add),
            )

        task = MAPSTask(
            start=38,
            modulus=31,
            target=-12,
            allowed_instructions=(add,),
        )
        self.assertEqual((task.start, task.target), (7, 19))


class MAPSSolverTests(unittest.TestCase):
    def test_bfs_returns_all_canonical_shortest_programs(self) -> None:
        task = MAPSTask(
            start=0,
            modulus=5,
            target=2,
            allowed_instructions=(
                instruction(MAPSInstructionKind.MUL, 2),
                instruction(MAPSInstructionKind.ADD, 1),
            ),
            max_program_length=3,
        )
        result = solve_bfs(task)
        self.assertTrue(result.reachable)
        self.assertEqual(result.shortest_length, 2)
        self.assertEqual(
            tuple(canonical_program(program) for program in result.shortest_programs),
            ("ADD 1; ADD 1", "ADD 1; MUL 2"),
        )
        self.assertEqual(
            result.shortest_family_ids,
            ("maps:ADD>ADD", "maps:ADD>MUL"),
        )
        for program in result.shortest_programs:
            self.assertEqual(
                execute_program(task.start, task.modulus, program),
                task.target,
            )

    def test_bfs_honours_maximum_length_and_unreachable_tasks(self) -> None:
        multiply = instruction(MAPSInstructionKind.MUL, 2)
        task = MAPSTask(
            start=0,
            modulus=7,
            target=1,
            allowed_instructions=(multiply,),
            max_program_length=5,
        )
        result = solve_bfs(task)
        self.assertFalse(result.reachable)
        self.assertIsNone(result.shortest_length)
        self.assertEqual(result.shortest_programs, ())

    def test_task_schema_rejects_empty_program_problems(self) -> None:
        add = instruction(MAPSInstructionKind.ADD, 1)
        with self.assertRaises(ValueError):
            MAPSTask(
                start=3,
                modulus=7,
                target=3,
                allowed_instructions=(add,),
            )
        with self.assertRaises(ValueError):
            MAPSTask(
                start=3,
                modulus=7,
                target=4,
                allowed_instructions=(add,),
                max_program_length=0,
            )


class MAPSVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = MAPSTask(
            start=7,
            modulus=31,
            target=5,
            allowed_instructions=(
                instruction(MAPSInstructionKind.ADD, 5),
                instruction(MAPSInstructionKind.MUL, 3),
                instruction(MAPSInstructionKind.NEG),
            ),
            max_program_length=3,
        )

    def test_valid_program_produces_structured_exact_result(self) -> None:
        result = verify_maps("<answer>MUL 3; ADD 5; NEG</answer>", self.task)
        self.assertTrue(result.valid)
        self.assertEqual(result.reward, 1.0)
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.final_value, 5)
        self.assertEqual(result.canonical_program, "MUL 3; ADD 5; NEG")
        self.assertEqual(result.strategy_family_id, "maps:MUL>ADD>NEG")
        self.assertEqual(result.to_dict()["reward"], 1.0)
        self.assertEqual(result.to_shared_result().reward, 1.0)
        shared = result.to_shared_result()
        self.assertEqual(shared.reward, 1.0)
        self.assertEqual(shared.canonical_expression, result.canonical_program)

    def test_wrong_target_and_illegal_instruction_are_distinct(self) -> None:
        wrong = verify_maps("<answer>ADD 5</answer>", self.task)
        self.assertEqual(wrong.failure_code, MAPSFailureCode.WRONG_TARGET)
        self.assertEqual(wrong.final_value, 12)
        self.assertEqual(
            wrong.to_shared_result().failure_code,
            MAPSFailureCode.WRONG_TARGET,
        )

        illegal = verify_maps("<answer>ADD 6</answer>", self.task)
        self.assertEqual(illegal.failure_code, MAPSFailureCode.ILLEGAL_INSTRUCTION)
        self.assertIsNone(illegal.final_value)

    def test_program_length_is_enforced(self) -> None:
        result = verify_maps(
            "<answer>NEG; NEG; NEG; NEG</answer>",
            self.task,
        )
        self.assertEqual(result.failure_code, MAPSFailureCode.PROGRAM_TOO_LONG)

    def test_largest_schema_program_fits_the_verifier_character_budget(self) -> None:
        argument = 10**31
        repeats = 100
        task = MAPSTask(
            start=0,
            modulus=1009,
            target=(argument * repeats) % 1009,
            allowed_instructions=(instruction(MAPSInstructionKind.ADD, argument),),
            max_program_length=repeats,
        )
        program = "; ".join([f"ADD {argument}"] * repeats)

        result = verify_maps(f"<answer>{program}</answer>", task)

        self.assertTrue(result.valid, result.to_dict())

    def test_answer_wrapper_is_exact_and_rejects_code_or_extra_text(self) -> None:
        cases = {
            "MUL 3": MAPSFailureCode.MISSING_ANSWER_TAG,
            "<answer></answer>": MAPSFailureCode.EMPTY_ANSWER,
            "<answer>NEG</answer><answer>NEG</answer>": (
                MAPSFailureCode.MULTIPLE_ANSWER_TAGS
            ),
            "Here: <answer>NEG</answer>": MAPSFailureCode.INVALID_PROGRAM,
            "```<answer>NEG</answer>```": MAPSFailureCode.INVALID_PROGRAM,
            "<answer>__import__('os')</answer>": MAPSFailureCode.INVALID_PROGRAM,
        }
        for completion, expected_code in cases.items():
            with self.subTest(completion=completion):
                result = verify_maps(completion, self.task)
                self.assertFalse(result.valid)
                self.assertEqual(result.failure_code, expected_code)


class MAPSGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_solver_backed(self) -> None:
        first = MAPSGenerator(1234).generate(7)
        second = MAPSGenerator(1234).generate(7)
        self.assertEqual(first, second)
        self.assertTrue(first.task.task_id.startswith("maps-"))
        self.assertEqual(len(first.task.task_id), 69)
        self.assertGreaterEqual(first.solution.shortest_length or 0, 2)
        for program in first.solution.shortest_programs:
            self.assertEqual(
                execute_program(first.task.start, first.task.modulus, program),
                first.task.target,
            )
        self.assertTrue(verify_maps(render_teacher_answer(first), first.task).valid)

    def test_prompt_declares_every_execution_constraint(self) -> None:
        instance = MAPSGenerator(77).generate()
        prompt = render_prompt(instance.task)
        self.assertIn(f"Start value: {instance.task.start}", prompt)
        self.assertIn(f"Modulus: {instance.task.modulus}", prompt)
        self.assertIn(f"Target: {instance.task.target}", prompt)
        self.assertIn("Allowed instructions:", prompt)
        self.assertIn(
            f"Maximum program length: {instance.task.max_program_length}",
            prompt,
        )
        self.assertIn("Separate multiple instructions with semicolons", prompt)
        self.assertIn("newlines alone are not valid separators", prompt)
        self.assertIn("<answer>...</answer>", prompt)

    def test_generator_config_rejects_nonprime_modulus(self) -> None:
        with self.assertRaises(ValueError):
            MAPSGeneratorConfig(moduli=(15,))

    def test_generator_config_rejects_unrenderable_constants(self) -> None:
        with self.assertRaises(ValueError):
            MAPSGeneratorConfig(
                add_constants=(10**32,),
                mul_constants=(),
                include_neg=False,
                allowed_instruction_count=1,
                min_latent_distractors=0,
            )
        with self.assertRaises(ValueError):
            MAPSGeneratorConfig(
                latent_program_max_length=101,
                max_shortest_length=5,
            )


if __name__ == "__main__":
    unittest.main()
