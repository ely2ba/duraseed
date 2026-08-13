"""Fail-closed local reservations for paid Tinker calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

from duraseed.runtime.billing import PRICE_SNAPSHOT, PriceSnapshot, UsageQuantities


class ReservationError(RuntimeError):
    """A paid call lacks an affordable, bounded reservation."""


@dataclass(frozen=True, slots=True)
class TokenBudget:
    prefill: int
    sample: int
    train: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.prefill, self.sample, self.train)
        ):
            raise ValueError("token budgets must be nonnegative integers")

    def plus(self, other: "TokenBudget") -> "TokenBudget":
        return TokenBudget(
            self.prefill + other.prefill,
            self.sample + other.sample,
            self.train + other.train,
        )


ZERO_TOKENS = TokenBudget(0, 0, 0)


def _add_usd(left: float, right: float) -> float:
    return float(Decimal(str(left)) + Decimal(str(right)))


class TokenLedger:
    """Reserve worst-case usage before a call, then record its actual usage."""

    def __init__(
        self,
        limits: TokenBudget,
        authorized_usd: float,
        *,
        prices: PriceSnapshot = PRICE_SNAPSHOT,
    ) -> None:
        if not math.isfinite(authorized_usd) or authorized_usd < 0:
            raise ValueError("authorized_usd must be finite and nonnegative")
        self.limits = limits
        self.authorized_usd = float(authorized_usd)
        self.prices = prices
        self.committed = ZERO_TOKENS
        self.observed = ZERO_TOKENS
        self.committed_fixed_usd = 0.0
        self.observed_fixed_usd = 0.0
        self._pending: tuple[TokenBudget, float] | None = None

    @staticmethod
    def _usage(tokens: TokenBudget) -> UsageQuantities:
        return UsageQuantities(
            prefill_tokens=tokens.prefill,
            sample_tokens=tokens.sample,
            train_tokens=tokens.train,
        )

    @property
    def committed_cost_usd(self) -> float:
        return self.prices.cost(self._usage(self.committed)) + self.committed_fixed_usd

    @property
    def observed_cost_usd(self) -> float:
        return self.prices.cost(self._usage(self.observed)) + self.observed_fixed_usd

    @property
    def has_pending_call(self) -> bool:
        return self._pending is not None

    def reserve_call(self, tokens: TokenBudget, *, fixed_usd: float = 0.0) -> None:
        """Commit a worst-case reservation before beginning one paid operation."""

        if self._pending is not None:
            raise ReservationError("the previous paid call has not been settled")
        if not math.isfinite(fixed_usd) or fixed_usd < 0:
            raise ValueError("fixed_usd must be finite and nonnegative")
        candidate = self.committed.plus(tokens)
        if (
            candidate.prefill > self.limits.prefill
            or candidate.sample > self.limits.sample
            or candidate.train > self.limits.train
        ):
            raise ReservationError("token reservation exceeds the authorized budget")
        candidate_cost = (
            self.prices.cost(self._usage(candidate))
            + self.committed_fixed_usd
            + fixed_usd
        )
        if candidate_cost > self.authorized_usd:
            raise ReservationError("reservation exceeds the authorized dollar cap")
        self.committed = candidate
        self.committed_fixed_usd = _add_usd(self.committed_fixed_usd, fixed_usd)
        self._pending = (tokens, fixed_usd)

    def settle_call(self, actual: TokenBudget) -> None:
        if self._pending is None:
            raise ReservationError("no paid call is awaiting settlement")
        reserved, fixed_usd = self._pending
        if (
            actual.prefill > reserved.prefill
            or actual.sample > reserved.sample
            or actual.train > reserved.train
        ):
            raise ReservationError("observed usage exceeds the call reservation")
        self.observed = self.observed.plus(actual)
        self.observed_fixed_usd = _add_usd(self.observed_fixed_usd, fixed_usd)
        self._pending = None

    def abort_call(self) -> None:
        """Close a failed call conservatively at its reserved token usage."""

        if self._pending is None:
            raise ReservationError("no paid call is awaiting settlement")
        reserved, fixed_usd = self._pending
        self.observed = self.observed.plus(reserved)
        self.observed_fixed_usd = _add_usd(self.observed_fixed_usd, fixed_usd)
        self._pending = None


__all__ = ["ReservationError", "TokenBudget", "TokenLedger", "ZERO_TOKENS"]
