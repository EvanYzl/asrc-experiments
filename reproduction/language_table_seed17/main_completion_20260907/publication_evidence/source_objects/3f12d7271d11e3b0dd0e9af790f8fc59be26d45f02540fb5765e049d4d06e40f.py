"""Rebuild the SS-AGA DBP-5L text feature table when the released file is unavailable.

Rows follow the exact sorted language/file order consumed by SS-AGA.  DBP-5L
ships entity URIs rather than abstracts, so this fallback embeds the decoded URI
labels.  The resulting provenance JSON makes that distinction explicit.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def entity_label(uri: str) -> str:
    value = unquote(uri.strip().rsplit("/", 1)[-1])
    return value.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    args = parser.parse_args()

    entity_files = sorted((args.dataset / "entity").glob("*.tsv"))
    rows = []
    counts = {}
    for path in entity_files:
        labels = [entity_label(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.extend(labels)
        counts[path.stem] = len(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    hidden_size = int(model.config.hidden_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=(len(rows), hidden_size)
    )
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            encoded = tokenizer(
                rows[start:end], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            if args.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            output[start:end] = pooled.float().cpu().numpy()
            if start == 0 or end == len(rows) or end % (args.batch_size * 20) == 0:
                print(f"encoded {end}/{len(rows)}", flush=True)
    output.flush()

    provenance = {
        "purpose": "SS-AGA fallback text feature table",
        "official_artifact_available": False,
        "official_google_drive_id": "1-R_2lqS5AQtWqLZXC45SrfkK5XETREe5",
        "source_text": "URL-decoded DBP-5L entity URI labels",
        "language_order": [path.stem for path in entity_files],
        "language_counts": counts,
        "rows": len(rows),
        "dimensions": hidden_size,
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "pooling": args.pooling,
        "max_length": args.max_length,
        "dtype": "float32",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
