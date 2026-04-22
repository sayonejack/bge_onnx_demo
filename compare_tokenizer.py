from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def join_ids(values):
    return " ".join(str(v) for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HF tokenizer output")
    parser.add_argument("text", help="single sentence text")
    parser.add_argument(
        "--text-b",
        dest="text_b",
        default=None,
        help="optional second sentence for pair encoding",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "bge_onnx_demo"
            / "Xenova-bge-large-zh-v1.5"
        ),
        help="local tokenizer directory",
    )
    parser.add_argument("--max-length", dest="max_length", type=int, default=512)
    parser.add_argument(
        "--pad-to-max-length",
        dest="pad_to_max_length",
        action="store_true",
        help="pad to max_length",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    if args.text_b is None:
        encoding = tokenizer(
            args.text,
            truncation=True,
            padding="max_length" if args.pad_to_max_length else False,
            max_length=args.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
        )
    else:
        encoding = tokenizer(
            args.text,
            args.text_b,
            truncation=True,
            padding="max_length" if args.pad_to_max_length else False,
            max_length=args.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
        )

    print("input_ids:", join_ids(encoding["input_ids"]))
    print("attention_mask:", join_ids(encoding["attention_mask"]))
    print("token_type_ids:", join_ids(encoding["token_type_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
