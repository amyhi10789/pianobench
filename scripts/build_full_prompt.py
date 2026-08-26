#!/usr/bin/env python3
"""Combine shared instructions with a specific prompt file."""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
PROMPTS = DATA / "prompts"

PROMPT_FILES = {
    "1A": "level1/prompt_1a_middle_c.txt",
    "1B": "level1/prompt_1b_e4.txt",
    "1C": "level1/prompt_1c_g4.txt",
}


def build(prompt_id: str) -> str:
    shared = (PROMPTS / "shared_instructions.txt").read_text(encoding="utf-8").strip()
    specific_name = PROMPT_FILES[prompt_id]
    specific = (PROMPTS / specific_name).read_text(encoding="utf-8").strip()
    return f"{shared}\n\n---\n\n{specific}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full prompt from shared + specific text.")
    parser.add_argument("--prompt-id", choices=sorted(PROMPT_FILES), default="1A")
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output .txt path. If omitted, print to the terminal.",
    )
    args = parser.parse_args()

    text = build(args.prompt_id)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
