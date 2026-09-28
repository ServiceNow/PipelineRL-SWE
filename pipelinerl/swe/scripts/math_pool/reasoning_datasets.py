"""More reasoning datasets for the cost-predictability screens (NEW_PATH 4.A.13): loaders + graders for
ZebraLogic (grid), Knights & Knaves, SuperGPQA and MMLU-Pro (multiple choice), BIG-Bench Extra Hard (exact match).

Each task: {problem_id, problem, prompt, answer, difficulty, subject, kind}. `prompt` is the full user message (the math
collector uses it instead of its boxed-math template); `kind` selects the grader. Samples are stratified and seeded
(seed 0) so the screen and any later full pool see the same problems. `difficulty` is the dataset's own graded label
mapped to a number where one exists (ZebraLogic: houses x features; K&K: people; SuperGPQA: easy/middle/hard), else 0.
"""
from __future__ import annotations
import ast, json, random, re
from collections import defaultdict

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _stratify(items, key, n, seed=0):
    rng = random.Random(seed); by = defaultdict(list)
    for x in items:
        by[key(x)].append(x)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by); out = []
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def _mc_prompt(q, options):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f"Answer the following multiple-choice question. Reason step by step, then give the letter of the correct "
            f"option within \\boxed{{}} (e.g. \\boxed{{C}}).\n\n{q}\n\n{opts}")


def load(name, n=1000):
    from datasets import load_dataset
    if name == "zebra":
        d = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        out = []
        for r in d:
            sol = ast.literal_eval(r["solution"]) if isinstance(r["solution"], str) else r["solution"]
            h, f = (int(x) for x in r["size"].split("*"))
            hdr = sol["header"][1:]
            fmt = "{" + ", ".join(f'"House {i+1}": {{' + ", ".join(f'"{c}": "..."' for c in hdr) + "}" for i in range(h)) + "}"
            prompt = (f"{r['puzzle']}\n\nSolve the puzzle. Reason step by step, then give the complete solution as a JSON "
                      f"object in a ```json code block, exactly in this format:\n{fmt}")
            out.append({"problem_id": f"zebra_{r['id']}", "problem": r["puzzle"], "prompt": prompt,
                        "answer": json.dumps(sol), "difficulty": float(h * f), "subject": r["size"], "kind": "zebra"})
        return out[:n]
    if name == "kk":
        out = []
        for ppl in range(2, 9):
            d = load_dataset("K-and-K/knights-and-knaves", "test", split=f"{ppl}ppl")
            for i, r in enumerate(d):
                names = ast.literal_eval(r["names"]) if isinstance(r["names"], str) else r["names"]
                sol = ast.literal_eval(r["solution"]) if isinstance(r["solution"], str) else r["solution"]
                prompt = (f"{r['quiz']}\n\nReason step by step. Then finish with one line per person, exactly in the form "
                          f"'NAME is a knight' or 'NAME is a knave', covering: {', '.join(names)}.")
                out.append({"problem_id": f"kk_{ppl}_{i}", "problem": r["quiz"], "prompt": prompt,
                            "answer": json.dumps(dict(zip(names, sol))), "difficulty": float(ppl), "subject": f"{ppl}ppl", "kind": "kk"})
        return _stratify(out, lambda x: x["difficulty"], n)
    if name == "supergpqa":
        d = load_dataset("m-a-p/SuperGPQA", split="train")
        tier = {"easy": 1.0, "middle": 2.0, "hard": 3.0}
        out = []
        for r in d:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            out.append({"problem_id": f"sgpqa_{r['uuid']}", "problem": r["question"], "prompt": _mc_prompt(r["question"], opts),
                        "answer": r["answer_letter"], "difficulty": tier.get(r["difficulty"], 0.0), "subject": r["discipline"], "kind": "mc"})
        return _stratify(out, lambda x: (x["difficulty"], x["subject"]), n)
    if name == "mmlupro":
        d = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        out = []
        for r in d:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            out.append({"problem_id": f"mmlup_{r['question_id']}", "problem": r["question"], "prompt": _mc_prompt(r["question"], opts),
                        "answer": r["answer"], "difficulty": 0.0, "subject": r["category"], "kind": "mc"})
        return _stratify(out, lambda x: x["subject"], n)
    if name == "bbeh":
        d = load_dataset("BBEH/bbeh", split="train")
        out = []
        for i, r in enumerate(d):
            prompt = f"{r['input']}\n\nReason step by step, then give your final answer within \\boxed{{}}."
            out.append({"problem_id": f"bbeh_{i}", "problem": r["input"], "prompt": prompt, "answer": str(r["target"]),
                        "difficulty": 0.0, "subject": r["task"], "kind": "exact"})
        return _stratify(out, lambda x: x["subject"], n)
    raise KeyError(name)


def _boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
    return text[i + 7:j - 1] if depth == 0 else None


def _norm(s):
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(s))
    return re.sub(r"[\s\.\,\;\:\!\"\'`\$]+", " ", s.lower()).strip()


def grade(kind, text, answer):
    text = text or ""
    if kind == "mc":
        b = _boxed(text)
        m = re.search(r"[A-Z]", (b or "").upper()) if b else None
        if not m:
            m2 = re.findall(r"(?:answer|option)\s*(?:is|:)?\s*\(?([A-J])\)?\b", text, flags=re.I)
            return bool(m2) and m2[-1].upper() == answer.upper()
        return m.group(0) == answer.upper()
    if kind == "exact":
        b = _boxed(text)
        if b is None:
            return False
        if _norm(b) == _norm(answer):
            return True
        try:
            from math_verify import parse, verify
            g, p = parse(f"${answer}$"), parse(f"${b}$")
            return bool(g) and bool(p) and bool(verify(g, p))
        except Exception:
            return False
    if kind == "kk":
        sol = json.loads(answer); tail = text[-3000:]
        for name, is_knight in sol.items():
            hits = re.findall(rf"\b{re.escape(name)}\s+is\s+an?\s+(knight|knave)", tail, flags=re.I)
            if not hits or (hits[-1].lower() == "knight") != bool(is_knight):
                return False
        return True
    if kind == "zebra":
        sol = json.loads(answer); hdr = sol["header"]
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        cand = blocks[-1] if blocks else (re.findall(r"(\{\s*\"House 1\".*\})", text, flags=re.S) or [None])[-1]
        if not cand:
            return False
        try:
            pred = json.loads(cand)
        except Exception:
            return False
        pred = pred.get("solution", pred)
        for row in sol["rows"]:
            house = pred.get(f"House {row[0]}") or pred.get(str(row[0]))
            if not isinstance(house, dict):
                return False
            got = {_norm(k): _norm(v) for k, v in house.items()}
            for col, val in zip(hdr[1:], row[1:]):
                if got.get(_norm(col)) != _norm(val):
                    return False
        return True
    raise KeyError(kind)
