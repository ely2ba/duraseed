"""Typed completed-response failures for training metrics."""


class NonFiniteTrainingMetricError(RuntimeError):
    """A completed optimizer response contained a non-finite metric."""

    def __init__(self, metric_name: str) -> None:
        super().__init__(f"non-finite Tinker metric {metric_name}")
        self.metric_name = metric_name


__all__ = ["NonFiniteTrainingMetricError"]
