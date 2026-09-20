from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SOURCE_TARGETS = ("nFix", "FFD", "GPT", "TRT", "fixProp")
FEATURES = ("NFIX", "FFD", "GPT", "TRT", "FIXPROP")
LABEL_PAD = -100.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def eligible_word(value: object) -> bool:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return len(text) <= 64 and any(character.isalpha() for character in text)


@dataclass(frozen=True)
class ETScaler:
    mean: np.ndarray
    std: np.ndarray
    z_min: float = -5.0
    z_max: float = 5.0

    @classmethod
    def fit(cls, frame: pd.DataFrame, z_min: float = -5.0, z_max: float = 5.0) -> "ETScaler":
        values = frame[list(SOURCE_TARGETS)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("ET training targets contain non-finite values")
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(values.mean(axis=0), std, z_min, z_max)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.std, self.z_min, self.z_max)

    def to_json(self) -> dict:
        return {"features": list(FEATURES), "source_targets": list(SOURCE_TARGETS), "mean": self.mean.tolist(), "std": self.std.tolist(), "z_clip": [self.z_min, self.z_max]}


def document_windows(size: int, width: int = 256, stride: int = 192) -> List[Tuple[int, int]]:
    if size < 0 or width <= 0 or stride <= 0:
        raise ValueError("Invalid window arguments")
    output: List[Tuple[int, int]] = []
    start = 0
    while start < size:
        end = min(size, start + width)
        output.append((start, end))
        if end == size:
            break
        start += stride
    return output


def select_owner_windows(size: int, width: int = 256, stride: int = 192) -> np.ndarray:
    """Assign each word to the overlapping window with the largest edge margin."""
    owner = np.full(size, -1, dtype=np.int64)
    margin = np.full(size, -1, dtype=np.int64)
    for start, end in document_windows(size, width, stride):
        positions = np.arange(start, end)
        candidate = np.minimum(positions - start, end - 1 - positions)
        update = candidate > margin[positions]
        owner[positions[update]] = start
        margin[positions[update]] = candidate[update]
    return owner


def validate_et_frame(frame: pd.DataFrame) -> None:
    required = {"sentence_id", "word_id", "word", *SOURCE_TARGETS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing ET columns: {sorted(missing)}")
    if not np.isfinite(frame[list(SOURCE_TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("ET targets contain NaN or infinity")


def write_proxy_jsonl(documents: Sequence[Mapping], predictions: Sequence[np.ndarray], path: str | Path, scaler: ETScaler) -> None:
    if len(documents) != len(predictions):
        raise ValueError("Document and prediction counts differ")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for document, raw in zip(documents, predictions):
            tokens = list(document["visible_tokens"])
            if raw.shape != (len(tokens), 5):
                raise ValueError(f"Prediction shape mismatch for {document['doc_id']}")
            valid = np.asarray([eligible_word(token) for token in tokens], dtype=np.uint8)
            standardized = scaler.transform(np.asarray(raw, dtype=np.float64))
            standardized[valid == 0] = 0.0
            row = dict(document)
            row["proxy_et"] = standardized.tolist()
            row["proxy_mask"] = np.repeat(valid[:, None], 5, axis=1).tolist()
            row["proxy_valid"] = valid.tolist()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

