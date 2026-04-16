#!/usr/bin/env python
import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def _decode_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token_id]) for token_id in token_ids]


def _summarize_pattern(
    tokenizer: Any,
    name: str,
    values: list[str],
    formatter: Callable[[str], str],
    examples: int,
) -> dict[str, Any]:
    one_token: list[str] = []
    split_examples: list[dict[str, Any]] = []
    token_len_counts: dict[int, int] = {}
    for value in values:
        text = formatter(value)
        ids = _token_ids(tokenizer, text)
        token_len_counts[len(ids)] = token_len_counts.get(len(ids), 0) + 1
        if len(ids) == 1:
            one_token.append(value)
        elif len(split_examples) < examples:
            split_examples.append(
                {
                    "value": value,
                    "text": text,
                    "token_ids": ids,
                    "pieces": _decode_pieces(tokenizer, ids),
                }
            )
    return {
        "pattern": name,
        "n_values": len(values),
        "one_token_count": len(one_token),
        "one_token_rate": len(one_token) / len(values) if values else None,
        "token_len_counts": {str(key): token_len_counts[key] for key in sorted(token_len_counts)},
        "first_one_token_values": one_token[:examples],
        "split_examples": split_examples,
    }


def _inspect_values(tokenizer: Any, values: list[str], examples: int) -> list[dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    for value in values[:examples]:
        variants = [
            ("plain", value),
            ("leading_space", f" {value}"),
            ("leading_newline", f"\n{value}"),
            ("answer_newline", f"Answer:\n{value}"),
            ("json_after_comma", f",{value}"),
            ("json_after_colon", f":{value}"),
        ]
        inspected.append(
            {
                "value": value,
                "variants": [
                    {
                        "name": variant_name,
                        "text": text,
                        "token_ids": _token_ids(tokenizer, text),
                        "pieces": _decode_pieces(tokenizer, _token_ids(tokenizer, text)),
                    }
                    for variant_name, text in variants
                ],
            }
        )
    return inspected


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe reward-value tokenization for offline-router text modes.")
    parser.add_argument(
        "--model-path",
        default="/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current",
        help="Tokenizer path or model name.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the full JSON report.",
    )
    parser.add_argument("--examples", type=int, default=8, help="Number of examples to print/store per section.")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )

    two_decimal_values = [f"{idx / 100.0:.2f}" for idx in range(101)]
    two_digit_suffixes = [f"{idx:02d}" for idx in range(100)]
    decimal_suffixes = [f".{idx:02d}" for idx in range(100)]
    five_point_grid_values = [f"{idx / 20.0:.2f}" for idx in range(21)]

    patterns = [
        ("reward_plain_0.xx", two_decimal_values, lambda value: value),
        ("reward_leading_space_0.xx", two_decimal_values, lambda value: f" {value}"),
        ("reward_leading_newline_0.xx", two_decimal_values, lambda value: f"\n{value}"),
        ("reward_json_after_comma_0.xx", two_decimal_values, lambda value: f",{value}"),
        ("reward_json_after_colon_0.xx", two_decimal_values, lambda value: f":{value}"),
        ("decimal_suffix_.xx", decimal_suffixes, lambda value: value),
        ("decimal_suffix_leading_space_.xx", decimal_suffixes, lambda value: f" {value}"),
        ("two_digit_suffix_xx", two_digit_suffixes, lambda value: value),
        ("two_digit_suffix_leading_space_xx", two_digit_suffixes, lambda value: f" {value}"),
        ("five_point_grid_plain", five_point_grid_values, lambda value: value),
        ("five_point_grid_leading_space", five_point_grid_values, lambda value: f" {value}"),
    ]

    selected_values = ["0.00", "0.05", "0.10", "0.37", "0.50", "0.95", "1.00"]
    report = {
        "model_path": args.model_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "patterns": [
            _summarize_pattern(tokenizer, name, values, formatter, examples=args.examples)
            for name, values, formatter in patterns
        ],
        "selected_values": _inspect_values(
            tokenizer,
            selected_values,
            examples=len(selected_values),
        ),
    }

    for pattern in report["patterns"]:
        print(
            f"{pattern['pattern']}: "
            f"{pattern['one_token_count']}/{pattern['n_values']} one-token, "
            f"token_len_counts={pattern['token_len_counts']}"
        )
        if pattern["split_examples"]:
            first = pattern["split_examples"][0]
            print(
                "  first split: "
                f"{first['text']!r} -> ids={first['token_ids']} pieces={first['pieces']!r}"
            )

    print("\nSelected values:")
    for item in report["selected_values"]:
        print(f"  {item['value']}")
        for variant in item["variants"]:
            print(
                f"    {variant['name']}: {variant['text']!r} -> "
                f"ids={variant['token_ids']} pieces={variant['pieces']!r}"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
