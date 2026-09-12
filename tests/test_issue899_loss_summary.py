"""#899 — "Training Complete!" panel always showed "Loss: 0.0000 -> 0.0000"
when a run had fewer total steps than ``training.logging_steps`` (default
10, see ``config/schema.py``).

Root cause, verified across every trainer wrapper: each built its own
initial/final loss pair as ``train_losses[0 or -1] if train_losses else 0``,
where ``train_losses`` filtered ``self.trainer.state.log_history`` on a
literal ``"loss"`` key. HF Trainer only writes that key on the periodic
intermediate log entries it emits every ``logging_steps``; a run shorter
than that has none, so the only ``log_history`` entry is the final summary
dict — keyed ``"train_loss"``, not ``"loss"`` — and both fields silently
became the literal ``0``.

The naive fix (fall back to that ``train_loss`` entry for both ends) trades
one wrong statement for a subtler one: the summary value is a run-level
mean, not a first observation, so printing it as both ends of an arrow
(``2.3830 -> 2.3830``) reads as "the loss never moved" — not something a
single mean value can support either way. ``summarize_train_loss`` reports
``has_delta`` alongside the pair so a caller can tell the two cases apart;
``format_loss_summary`` (used by the CLI panel) only draws the arrow when
``has_delta`` is True.

This file pins both branches per the fix's own design discussion on the
issue thread: a short run (no per-step samples, fallback to the summary
mean, no arrow) and a normal run (a real pair, arrow drawn) — plus a
source-grep guard so a seventeenth trainer, or a revert on one of the
sixteen fixed here, can't quietly reintroduce a private copy of the old
``... if train_losses else 0`` pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soup_cli.utils.loss_summary import format_loss_summary, summarize_train_loss

# Every trainer wrapper this issue named as sharing the bug (sft.py is the
# file the issue quotes verbatim; the other fifteen were confirmed to share
# it byte-for-byte in the same PR). `ppo.py` has a second, unrelated
# `"loss" in entry` filter over a hand-built log (every manual PPO step logs
# a "loss" itself, so it can't hit this bug) and is deliberately not touched.
FIXED_TRAINER_MODULES = [
    "kto", "reward_model", "dpo", "asr", "mole_routing", "pretrain", "grpo",
    "online_dpo", "bco", "orpo", "ipo", "sft", "simpo", "embedding",
    "distill", "classifier",
]

TRAINER_DIR = Path(__file__).parent.parent / "src" / "soup_cli" / "trainer"


class TestSummarizeTrainLoss:
    def test_short_run_falls_back_to_the_final_summary_mean_not_zero(self):
        # Reproduces the issue's own repro: 5 steps against logging_steps=10,
        # so log_history holds only the final HF Trainer summary entry.
        log_history = [
            {
                "train_runtime": 4183,
                "train_loss": 2.383,
                "epoch": 1,
            }
        ]
        summary = summarize_train_loss(log_history)
        assert summary.initial_loss == pytest.approx(2.383)
        assert summary.final_loss == pytest.approx(2.383)
        assert summary.has_delta is False

    def test_short_run_with_no_summary_entry_either_falls_back_to_zero(self):
        summary = summarize_train_loss([])
        assert summary.initial_loss == 0.0
        assert summary.final_loss == 0.0
        assert summary.has_delta is False

    def test_normal_run_reports_a_real_delta(self):
        log_history = [
            {"loss": 2.9, "step": 10},
            {"loss": 1.7, "step": 20},
            {"loss": 0.9, "step": 30},
            {"train_runtime": 900, "train_loss": 1.83},
        ]
        summary = summarize_train_loss(log_history)
        assert summary.initial_loss == pytest.approx(2.9)
        assert summary.final_loss == pytest.approx(0.9)
        assert summary.has_delta is True

    def test_single_per_step_sample_is_an_honest_non_delta_not_a_fallback(self):
        log_history = [{"loss": 1.5, "step": 10}, {"train_runtime": 100, "train_loss": 1.5}]
        summary = summarize_train_loss(log_history)
        assert summary.initial_loss == pytest.approx(1.5)
        assert summary.final_loss == pytest.approx(1.5)
        assert summary.has_delta is False

    def test_non_mapping_entries_are_ignored(self):
        # log_history should always be dicts, but stay defensive.
        summary = summarize_train_loss([None, "not-a-dict", {"loss": 1.0}, {"loss": 0.5}])
        assert summary.has_delta is True
        assert summary.initial_loss == pytest.approx(1.0)
        assert summary.final_loss == pytest.approx(0.5)

    def test_non_numeric_train_loss_falls_back_to_zero_instead_of_raising(self):
        summary = summarize_train_loss([{"train_loss": "not-a-number"}])
        assert summary.initial_loss == 0.0
        assert summary.final_loss == 0.0
        assert summary.has_delta is False


class TestFormatLossSummary:
    def test_delta_renders_an_arrow(self):
        summary = summarize_train_loss([{"loss": 2.9}, {"loss": 0.9}, {"train_loss": 1.5}])
        assert format_loss_summary(summary) == "2.9000 -> 0.9000"

    def test_no_delta_renders_a_single_value_with_no_arrow(self):
        summary = summarize_train_loss([{"train_loss": 2.383}])
        rendered = format_loss_summary(summary)
        assert "->" not in rendered
        assert "2.3830" in rendered

    def test_no_delta_can_include_the_step_count(self):
        summary = summarize_train_loss([{"train_loss": 2.383}])
        rendered = format_loss_summary(summary, total_steps=5)
        assert "5 steps" in rendered
        assert "->" not in rendered


class TestNoTrainerKeepsAPrivateCopy:
    """Guard against a 17th trainer, or a revert on one of these 16,
    reintroducing `train_losses[0 or -1] if train_losses else 0` instead of
    calling the shared helper. A grep rather than an import check, so it
    also catches a copy that never imports `summarize_train_loss` at all.
    """

    @pytest.mark.parametrize("module_name", FIXED_TRAINER_MODULES)
    def test_module_uses_the_shared_helper(self, module_name):
        source = (TRAINER_DIR / f"{module_name}.py").read_text()
        assert "summarize_train_loss" in source, (
            f"{module_name}.py no longer calls the shared loss-summary helper"
        )
        assert "if train_losses else 0" not in source, (
            f"{module_name}.py reintroduced the old zero-fallback pattern"
        )

    @pytest.mark.parametrize("module_name", FIXED_TRAINER_MODULES)
    def test_module_is_still_valid_python(self, module_name):
        # Cheap sanity check that the source-level patch didn't corrupt
        # indentation or leave a dangling reference.
        ast.parse((TRAINER_DIR / f"{module_name}.py").read_text())
