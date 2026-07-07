#!/usr/bin/env python3
"""Quick check that an Anthropic API key is valid."""
import os
import sys
from pathlib import Path


def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and len(sys.argv) > 1:
        key = Path(sys.argv[1]).read_text().strip()
    if not key:
        print("Usage: ANTHROPIC_API_KEY=sk-ant-... python check_anthropic_key.py")
        print("   or: python check_anthropic_key.py /path/to/key_file")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=key)
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: key_ok"}],
        )
        resp = msg.content[0].text.strip()
        print(f"OK  key is valid | model replied: {resp!r}")
        print(f"    input_tokens={msg.usage.input_tokens}  output_tokens={msg.usage.output_tokens}")
    except anthropic.AuthenticationError as exc:
        print(f"INVALID KEY: {exc}")
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
