"""An empty answer is only a real failure when the token budget caused it.

gpt-oss splits reasoning and answer across channels. When the provider ends the turn
without emitting the answer channel, the draw says nothing about the model -- but it was
being graded as a wrong answer. That hit 43.1% of gpt-oss-20b eval draws at the 4096 cap
against 0.0% for the locally served scout, and 16-19% survived the move to 32768 tokens,
so it is not truncation. finish_reason separates the two cases.
"""

from __future__ import annotations

import asyncio

from pipelinerl.swe.scripts.livecodebench.collect_lcb_expert import _is_complete

REV = "rev1"
COMMIT = None


def _row(**kw):
    from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
        LCB_EVALUATOR_COMMIT,
    )
    row = {
        "full_output": "print(1)",
        "resolved": False,
        "public_resolved": False,
        "result_codes": [-1],
        "public_result_codes": [-1],
        "eval_metadata": {},
        "public_eval_metadata": {},
        "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
        "_lcb_dataset_revision": REV,
    }
    row.update(kw)
    return row


def test_budget_exhausted_empty_is_a_real_failure():
    row = _row(full_output="", eval_metadata={"error_message": "EmptyGeneration"},
               finish_reason="length")
    assert _is_complete(row, REV)


def test_provider_returned_nothing_is_invalid():
    row = _row(full_output="", eval_metadata={"error_message": "EmptyGeneration"},
               finish_reason="stop")
    assert not _is_complete(row, REV)


def test_legacy_rows_without_finish_reason_keep_old_semantics():
    row = _row(full_output="", eval_metadata={"error_message": "EmptyGeneration"})
    assert _is_complete(row, REV)


def test_normal_answer_unaffected_by_finish_reason():
    assert _is_complete(_row(finish_reason="stop"), REV)
    assert _is_complete(_row(finish_reason="length"), REV)


def _fake_call_sequence(outs):
    """Drive openrouter_call's retry loop without touching the network."""
    from pipelinerl.swe.scripts.livecodebench import collect_lcb_trajectories as m

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, payload):
            self._p = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return self._p

    class FakeSession:
        def post(self, *a, **kw):
            i = min(calls["n"], len(outs) - 1)
            calls["n"] += 1
            content, finish = outs[i]
            return FakeResp({
                "choices": [{"message": {"content": content, "reasoning": "think"},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            })

    out = asyncio.run(m.openrouter_call(
        FakeSession(), "m", "sys", "user", "key", empty_retries=2, gen_timeout=1))
    return out, calls["n"]


def test_empty_stop_is_retried_until_content_arrives():
    out, n = _fake_call_sequence([("", "stop"), ("", "stop"), ("print(1)", "stop")])
    assert out["full_output"] == "print(1)"
    assert n == 3


def test_empty_length_is_not_retried():
    out, n = _fake_call_sequence([("", "length"), ("print(1)", "stop")])
    assert out["full_output"] == ""
    assert out["finish_reason"] == "length"
    assert n == 1


def test_retries_are_bounded():
    out, n = _fake_call_sequence([("", "stop")])
    assert out["full_output"] == ""
    assert n == 3
