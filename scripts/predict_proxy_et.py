from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import RobertaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_ake.et_regressor import RobertaETRegressor
from cognition_ake.proxy_et import ETScaler, document_windows, eligible_word, select_owner_windows, write_proxy_jsonl


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def predict_document(model, tokenizer, tokens, device, width, stride):
    output = np.zeros((len(tokens), 5), dtype=np.float32)
    owner = select_owner_windows(len(tokens), width, stride)
    for start, end in document_windows(len(tokens), width, stride):
        encoded = tokenizer([tokens[start:end]], is_split_into_words=True, add_special_tokens=True, truncation=False, return_attention_mask=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 512:
            raise RuntimeError("Word window exceeds RoBERTa's 512-subword limit; reduce prediction.word_window")
        with torch.no_grad():
            prediction = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))[0].cpu().numpy()
        first = {}
        previous = None
        for position, word_id in enumerate(encoded.word_ids(batch_index=0)):
            if word_id is not None and word_id != previous:
                first[int(word_id)] = position
            previous = word_id
        for local in range(end - start):
            global_position = start + local
            if owner[global_position] == start and eligible_word(tokens[global_position]):
                output[global_position] = np.clip(prediction[first[local]], 0.0, 100.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate occurrence-specific Proxy ET for academic text")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "proxy_et.yaml")
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(resolve(cfg["output"]["checkpoint"]), map_location="cpu")
    scaler_json = json.loads(resolve(cfg["output"]["scaler"]).read_text(encoding="utf-8"))
    scaler = ETScaler(np.asarray(scaler_json["mean"]), np.asarray(scaler_json["std"]), *scaler_json["z_clip"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint["model_id"], add_prefix_space=True)
    model = RobertaETRegressor(checkpoint["model_id"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    documents_path = args.documents or resolve(cfg["data"]["academic_documents"])
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    width, stride = int(cfg["prediction"]["word_window"]), int(cfg["prediction"]["word_stride"])
    predictions = [predict_document(model, tokenizer, row["visible_tokens"], device, width, stride) for row in documents]
    destination = args.output or resolve(cfg["output"]["academic_proxy"])
    write_proxy_jsonl(documents, predictions, destination, scaler)
    print(json.dumps({"documents": len(documents), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()

