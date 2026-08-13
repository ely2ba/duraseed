from duraseed.calibration_seeds import ephemeral_sampler_id


def test_ephemeral_sampler_fallback_binds_attempt_lr_and_step() -> None:
    first = ephemeral_sampler_id("run-one", "attempt-0001", 1e-5, 7)
    assert first == "ephemeral:run-one:attempt-0001:B-G:1.0000000000000001e-05:step-7"
    assert first != ephemeral_sampler_id("run-one", "attempt-0002", 3e-5, 7)
