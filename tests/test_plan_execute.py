import json

import numpy as np

from pipelinerl.swe.scripts.livecodebench import analyze_plan_execute as ape
from pipelinerl.swe.scripts.livecodebench.collect_lcb_plan_execute import (
    NULL_PLAN,
    _plan_is_complete,
    _read_plans,
    make_execute_prompt,
    make_plan_prompt,
)

ROW = {
    "question_content": "Given n, print n squared.",
    "starter_code": "",
    "metadata": {},
    "platform": "atcoder",
    "question_id": "x1",
}


def test_plan_prompt_forbids_code_and_execute_prompt_wraps_plan():
    plan_prompt = make_plan_prompt(ROW)
    assert "Do NOT write any code" in plan_prompt
    assert ROW["question_content"] in plan_prompt
    exec_prompt = make_execute_prompt(ROW, "  read n; print n*n  ")
    assert exec_prompt.index(ROW["question_content"]) < exec_prompt.index("<plan>")
    assert "<plan>\nread n; print n*n\n</plan>" in exec_prompt
    assert "```" not in NULL_PLAN


def test_plan_resume_only_reuses_nonempty_plans(tmp_path):
    path = tmp_path / "plan120_eval.jsonl"
    rows = [
        {"problem_id": "a", "plan_text": "do it", "model": "m"},
        {"problem_id": "b", "plan_text": "   ", "model": "m"},  # API failure row
        {"problem_id": "a", "plan_text": "do it better", "model": "m"},  # later row wins
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    plans = _read_plans(path)
    assert set(plans) == {"a"}
    assert plans["a"]["plan_text"] == "do it better"
    assert not _plan_is_complete(rows[1])


def test_analysis_statistics_on_synthetic_outcomes():
    # 4 problems, 3 draws. Problem 3 is solved only by the composite (off-ladder).
    comp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]], bool)
    ref = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]], bool)
    assert ape.pass_at_k(comp).tolist() == [0.25, 0.5, 0.75]
    h = ape.hazard(comp)
    assert h[0] == 0.25 and abs(h[1] - 1 / 3) < 1e-9 and h[2] == 0.5
    # union within k=3: all 4 solved; best single: ref 3/4 -> c_inf = 25pt
    assert abs(ape.c_inf(comp, ref, 3) - 25.0) < 1e-9
    # stop-on-pass: fixed cost 1 each, per-draw cost 1; problem 0 stops after draw 1, ...
    per_draw = np.ones((4, 3))
    fixed = np.ones(4)
    cost, solve = ape.stop_on_pass_cost(comp, per_draw, fixed, 3)
    # spends: p0: 1+1, p1: 1+2, p2: 1+3, p3: 1+3 -> mean 3.25; solves 3/4
    assert abs(cost - 3.25) < 1e-9 and solve == 0.75
