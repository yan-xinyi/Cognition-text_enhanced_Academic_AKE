from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDocumentDataset(Dataset):
    """Load one precomputed SciBERT/Proxy-ET `.npz` file per document."""

    def __init__(self, index_path: str | Path, domain: str | None = None) -> None:
        self.index_path = Path(index_path)
        with self.index_path.open("r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        self.rows = [row for row in rows if domain is None or row["domain"] == domain]
        if not self.rows:
            raise ValueError(f"No documents found in {self.index_path} for domain={domain}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        meta = dict(self.rows[index])
        path = Path(meta["npz"])
        if not path.is_absolute():
            path = self.index_path.parent / path
        with np.load(path, allow_pickle=True) as payload:
            embedding = np.asarray(payload["embedding"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
            cognitive = np.asarray(payload["proxy_et"], dtype=np.float32)
            feature_mask = np.asarray(payload["proxy_mask"], dtype=np.float32)
            valid = np.asarray(payload["proxy_valid"], dtype=np.float32)
            tokens = [str(value) for value in payload["tokens"].tolist()]
        size = embedding.shape[0]
        expected = ((size,), (size, 5), (size, 5), (size,))
        actual = (labels.shape, cognitive.shape, feature_mask.shape, valid.shape)
        if actual != expected:
            raise ValueError(f"Shape mismatch in {path}: expected {expected}, got {actual}")
        meta["tokens"] = tokens
        return {
            "embedding": torch.from_numpy(embedding),
            "labels": torch.from_numpy(labels),
            "cognitive": torch.from_numpy(cognitive),
            "feature_mask": torch.from_numpy(feature_mask),
            "valid": torch.from_numpy(valid),
            "meta": meta,
        }


def collate_documents(items: Sequence[dict]) -> dict:
    lengths = torch.tensor([item["labels"].numel() for item in items], dtype=torch.long)
    maximum = int(lengths.max().item())
    embedding_dim = int(items[0]["embedding"].shape[1])
    batch = len(items)
    embedding = torch.zeros(batch, maximum, embedding_dim, dtype=torch.float32)
    labels = torch.zeros(batch, maximum, dtype=torch.long)
    cognitive = torch.zeros(batch, maximum, 5, dtype=torch.float32)
    feature_mask = torch.zeros(batch, maximum, 5, dtype=torch.float32)
    valid = torch.zeros(batch, maximum, dtype=torch.float32)
    token_mask = torch.zeros(batch, maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        embedding[index, :length] = item["embedding"]
        labels[index, :length] = item["labels"]
        cognitive[index, :length] = item["cognitive"]
        feature_mask[index, :length] = item["feature_mask"]
        valid[index, :length] = item["valid"]
        token_mask[index, :length] = True
    return {
        "embedding": embedding,
        "labels": labels,
        "cognitive": cognitive,
        "feature_mask": feature_mask,
        "valid": valid,
        "lengths": lengths,
        "token_mask": token_mask,
        "meta": [item["meta"] for item in items],
    }


def spans_from_bio(tags: Sequence[int], scores: Sequence[float], tokens: Sequence[str]) -> List[tuple[str, float]]:
    spans: List[tuple[str, float]] = []
    start = None
    for index, tag in enumerate(list(tags) + [0]):
        if tag == 1 or (tag == 2 and start is None):
            if start is not None:
                spans.append((" ".join(tokens[start:index]), float(max(scores[start:index]))))
            start = index
        elif tag == 0 and start is not None:
            spans.append((" ".join(tokens[start:index]), float(max(scores[start:index]))))
            start = None
    return sorted(spans, key=lambda item: (-item[1], item[0]))

