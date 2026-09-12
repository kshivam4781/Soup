"""#899 — shared initial/final training-loss summary for trainer wrappers.

Every ``transformers.Trainer``-backed wrapper (SFT, DPO, GRPO, KTO, BCO, IPO,
SimPO, ORPO, online-DPO, reward-model, classifier, embedding, distill, ASR,
pretrain, MoLE-routing) built its own "Training Complete!" summary the same
way::

    logs = self.trainer.state.log_history
    train_losses = [entry["loss"] for entry in logs if "loss" in entry]
    ...
    "initial_loss": train_losses[0] if train_losses else 0,
    "final_loss": train_losses[-1] if train_losses else 0,

HF Trainer only appends an intermediate ``log_history`` entry keyed
``"loss"`` every ``training.logging_steps`` (default 10). A run with fewer
total steps than that never produces one; the only ``log_history`` entry is
the final summary, keyed ``"train_loss"`` instead. Because the filter is
``"loss" in entry`` (an exact key match), it never matches that summary
entry, ``train_losses`` ends up empty, and both fields silently became a
literal ``0`` — printed by the CLI panel as ``Loss: 0.0000 -> 0.0000``,
indistinguishable from a genuinely dead run even though training completed
fine and the real loss is sitting right there under a different key.

The obvious fix — fall back to that ``train_loss`` entry for both ends when
no per-step samples exist — creates a second, subtler wrong statement: the
summary value is a mean over the whole run, not a first observation, so
printing it as both ends of an arrow (``Loss: 2.3830 -> 2.3830``) reads as
"the loss did not move", which this run's data cannot support either way.
``summarize_train_loss`` therefore also reports whether the two ends are
real, independent observations (``has_delta``); callers must only render an
arrow between them when it is True (see ``format_loss_summary``).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence


class LossSummary:
    """Initial/final loss for a completed run, plus whether they're a real pair.

    ``initial_loss`` and ``final_loss`` are always floats fit to print. When
    ``has_delta`` is False they are equal (either a single per-step
    observation, or the run-level mean used as a fallback), and a caller
    must not present them as a before/after pair.
    """

    __slots__ = ("initial_loss", "final_loss", "has_delta")

    def __init__(self, initial_loss: float, final_loss: float, has_delta: bool) -> None:
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.has_delta = has_delta

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"LossSummary(initial_loss={self.initial_loss!r}, "
            f"final_loss={self.final_loss!r}, has_delta={self.has_delta!r})"
        )


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_train_loss(log_history: Sequence[Any]) -> LossSummary:
    """Build the initial/final loss summary from a Trainer's ``log_history``.

    ``log_history`` is ``self.trainer.state.log_history`` (a list of dicts
    HF Trainer appends to on every ``self.log(...)`` call, the last of which
    is the final-summary dict returned by ``Trainer.train()``).

    - Two or more per-step ``"loss"`` entries: a real pair, ``has_delta=True``.
    - Exactly one: that single observation is both ends, honestly — a
      one-step run truly did not move, so ``has_delta=False`` (nothing to
      draw an arrow between) but it is not a fallback.
    - None (the run had fewer total steps than ``logging_steps``): fall back
      to the final entry's ``"train_loss"`` (the run-level mean) for both
      ends, still with ``has_delta=False``. If even that is missing,
      fall back to ``0.0``.
    """
    train_losses: List[float] = [
        entry["loss"]
        for entry in log_history
        if isinstance(entry, Mapping) and "loss" in entry
    ]
    if len(train_losses) >= 2:
        return LossSummary(
            initial_loss=_coerce_float(train_losses[0]),
            final_loss=_coerce_float(train_losses[-1]),
            has_delta=True,
        )
    if len(train_losses) == 1:
        value = _coerce_float(train_losses[0])
        return LossSummary(initial_loss=value, final_loss=value, has_delta=False)

    fallback = 0.0
    for entry in reversed(log_history):
        if isinstance(entry, Mapping) and "train_loss" in entry:
            fallback = _coerce_float(entry["train_loss"])
            break
    return LossSummary(initial_loss=fallback, final_loss=fallback, has_delta=False)


def format_loss_summary(summary: LossSummary, *, total_steps: int | None = None) -> str:
    """Render ``summary`` for the "Training Complete!" panel.

    Only prints an arrow when ``summary.has_delta`` is True — i.e. when
    there are two real, independent observations to connect. Otherwise
    prints a single value so the panel never claims a delta it does not
    have the data for.
    """
    if summary.has_delta:
        return f"{summary.initial_loss:.4f} -> {summary.final_loss:.4f}"
    steps_note = f" over {total_steps} steps" if total_steps else ""
    return (
        f"{summary.initial_loss:.4f} (single value{steps_note}; not enough "
        "per-step history for a delta)"
    )
