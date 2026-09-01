import json

import numpy as np

from pipelinerl.swe.scripts.livecodebench import audit_truncation as at


def _bundle(tmp_path):
    """3 problems x 2 routes x 2 draws. p1 is solved; p2 and p3 are pool-unsolved."""
    d = tmp_path / "tensors"
    d.mkdir()
    outcome = np.array([[[True, False], [False, False]],
                        [[False, False], [False, False]],
                        [[False, False], [False, False]]])
    valid = np.ones_like(outcome)
    completion = np.array([[[100, 4096], [4096, 4096]],
                           [[4090, 4096], [4096, 4096]],
                           [[200, 300], [400, 500]]], dtype=np.float32)
    np.savez(d / "tensors.npz", final_outcome=outcome, valid=valid,
             completion_tokens=completion, problem_ids=np.array(["p1", "p2", "p3"]),
             model_slots=np.array(["scout", "oss120"]))
    (d / "problems.jsonl").write_text("\n".join(
        json.dumps({"problem_id": p, "source_temporal_split": s})
        for p, s in [("p1", "train"), ("p2", "train"), ("p3", "eval")]) + "\n")
    return d


def test_dump_lists_only_pool_unsolved_and_measures_cap_rate(tmp_path, capsys):
    bundle = _bundle(tmp_path)
    ids = tmp_path / "ids.txt"
    at.dump_unsolved(bundle, ids, old_cap=4096)
    assert ids.read_text().split() == ["p2", "p3"]
    summary = json.loads((tmp_path / "unsolved_summary.json").read_text())
    assert summary["n_pool_unsolved"] == 2
    assert summary["by_split"] == {"eval": 1, "train": 1}
    # p2's oss120 draws are both capped, p3's are not -> 2 of 4 unsolved draws at cap
    assert summary["per_route"]["oss120"]["unsolved_draws_at_cap"] == 0.5


def test_compare_counts_rescues_and_corrects_the_impossible_fraction(tmp_path, capsys):
    bundle = _bundle(tmp_path)
    ids = tmp_path / "ids.txt"
    at.dump_unsolved(bundle, ids, old_cap=4096)
    rec = tmp_path / "recollect"
    rec.mkdir()
    # p2 falls at the larger cap on draw 1; p3 still fails. One row exceeds the old cap.
    rows = {
        "oss120_32k_train_d0.jsonl": [{"problem_id": "p2", "resolved": False, "completion_tokens": 9000}],
        "oss120_32k_train_d1.jsonl": [{"problem_id": "p2", "resolved": True, "completion_tokens": 12000}],
        "oss120_32k_eval_d0.jsonl": [{"problem_id": "p3", "resolved": False, "completion_tokens": 500}],
    }
    for name, rs in rows.items():
        (rec / name).write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    at.compare(bundle, rec, "oss120_32k", ids, new_cap=32768)
    report = json.loads((rec / "truncation_audit.json").read_text())
    assert report["n_now_solved"] == 1 and report["rescued_ids"] == ["p2"]
    assert report["n_recollected"] == 2 and report["n_draws"] == 3
    assert abs(report["impossible_fraction_before"] - 2 / 3) < 1e-9
    assert abs(report["impossible_fraction_after"] - 1 / 3) < 1e-9
    assert abs(report["draws_exceeding_old_4096_cap"] - 2 / 3) < 1e-9
    assert "labels are cap-limited" in capsys.readouterr().out
