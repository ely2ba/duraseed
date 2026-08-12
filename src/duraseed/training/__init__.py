"""Runtime-independent training records and calibration reducers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "GateStatus": "teacher_dose",
    "PairedControlChange": "teacher_dose",
    "TeacherDoseAssessment": "teacher_dose",
    "TeacherDoseCriterion": "teacher_dose",
    "TeacherDoseDecision": "teacher_dose",
    "TeacherDoseDecisionStatus": "teacher_dose",
    "TeacherDoseEvidenceError": "teacher_dose",
    "TeacherDoseGateSummary": "teacher_dose",
    "assess_teacher_dose": "teacher_dose",
    "decide_teacher_dose": "teacher_dose",
    "summarize_paired_control_change": "teacher_dose",
    "summarize_teacher_dose_gate": "teacher_dose",
    "STAGE_A_LEARNING_RATE_GRIDS": "stage_a_calibration",
    "StageACalibrationEvidenceError": "stage_a_calibration",
    "StageADurationDecision": "stage_a_calibration",
    "StageADurationDecisionStatus": "stage_a_calibration",
    "StageAFinalAssessment": "stage_a_calibration",
    "StageAFinalCriterion": "stage_a_calibration",
    "StageAFinalEvidence": "stage_a_calibration",
    "StageALearningRateDecision": "stage_a_calibration",
    "StageALearningRateDecisionStatus": "stage_a_calibration",
    "StageAPairedItemEvidence": "stage_a_calibration",
    "StageAScreenAssessment": "stage_a_calibration",
    "StageAScreenEvidence": "stage_a_calibration",
    "assess_stage_a_final": "stage_a_calibration",
    "assess_stage_a_screen": "stage_a_calibration",
    "decide_stage_a_duration": "stage_a_calibration",
    "select_stage_a_learning_rate": "stage_a_calibration",
    "PanelTeacherAllocation": "teacher_allocation",
    "RANDOM_TEACHER_ALLOCATION_SEED": "teacher_allocation",
    "TeacherAllocationError": "teacher_allocation",
    "TeacherTokenCounts": "teacher_allocation",
    "TeacherTraceCandidate": "teacher_allocation",
    "build_crossed_teacher_allocations": "teacher_allocation",
    "build_teacher_trace_candidate": "teacher_allocation",
    "verify_task_completion": "reward",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
