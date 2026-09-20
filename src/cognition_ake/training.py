from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import NPZDocumentDataset, collate_documents, spans_from_bio
from .evaluation import evaluate_micro
from .models import CognitionAKETagger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def branch_loss(crf, emissions, labels, mask, class_weights, auxiliary_weight: float) -> torch.Tensor:
    crf_loss = -crf(emissions, labels, mask=mask.bool(), reduction="mean")
    selected = mask.reshape(-1)
    ce = nn.functional.cross_entropy(emissions.reshape(-1, emissions.size(-1))[selected], labels.reshape(-1)[selected], weight=class_weights)
    return crf_loss + auxiliary_weight * ce


def model_loss(model: CognitionAKETagger, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], cfg: Mapping, class_weights: torch.Tensor) -> torch.Tensor:
    task = branch_loss(model.crf, outputs["emissions"], batch["labels"], batch["token_mask"], class_weights, float(cfg.get("auxiliary_ce_weight", 0.2)))
    if model.method in {"CogAlign", "FS-CA"}:
        cognitive = branch_loss(model.cogalign.cognitive_crf, outputs["cognitive_emissions"], batch["labels"], batch["token_mask"], class_weights, float(cfg.get("auxiliary_ce_weight", 0.2)))
        adversarial = nn.functional.cross_entropy(outputs["discriminator_logits"], outputs["modality_labels"])
        return task + float(cfg.get("lambda_cog", 0.5)) * cognitive + float(cfg.get("lambda_adv", 0.05)) * adversarial
    if model.method == "GAZESUP":
        available = (batch["feature_mask"] > 0) & batch["token_mask"].unsqueeze(-1)
        gaze = (outputs["gaze_prediction"] - batch["cognitive"]).pow(2)[available].mean()
        return task + float(cfg.get("lambda_gaze", 0.1)) * gaze
    if model.method == "CIB":
        available = (batch["valid"] > 0) & batch["token_mask"]
        kl = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()).sum(dim=-1)
        return task + float(cfg.get("cib_beta", 0.001)) * kl[available].mean()
    return task


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def infer(model: CognitionAKETagger, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    records = []
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            outputs = model.forward_all(batch["embedding"], batch["cognitive"], batch["feature_mask"], batch["valid"], batch["lengths"], batch["token_mask"], inference=True)
            probabilities = torch.softmax(outputs["emissions"], dim=-1)[:, :, 1:].max(dim=-1).values.cpu().numpy()
            decoded = model.decode(outputs["emissions"], batch["token_mask"])
            for index, tags in enumerate(decoded):
                meta = raw["meta"][index]
                predictions = spans_from_bio(tags, probabilities[index, : len(tags)], meta["tokens"])
                records.append({"doc_id": meta["doc_id"], "domain": meta["domain"], "predictions": predictions, "gold_present": meta.get("gold_present", [])})
    return evaluate_micro(records, 5), records


def train(config: Mapping, method: str, domain: str, seed: int, output_dir: str | Path) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = NPZDocumentDataset(config["data"]["train_index"], domain)
    validation_set = NPZDocumentDataset(config["data"]["validation_index"], domain)
    batch_size = int(config["training"].get("batch_size", 64))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_documents)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_documents)
    model = CognitionAKETagger(method, config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3)), weight_decay=float(config["training"].get("weight_decay", 1e-4)))
    class_weights = torch.tensor(config["training"].get("class_weights", [1.0, 3.0, 3.0]), dtype=torch.float32, device=device)
    best = -1.0
    stale = 0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{domain}_{method}_seed{seed}.pt"
    history = []
    for epoch in range(1, int(config["training"].get("epochs", 15)) + 1):
        model.train()
        losses = []
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad()
            outputs = model.forward_all(batch["embedding"], batch["cognitive"], batch["feature_mask"], batch["valid"], batch["lengths"], batch["token_mask"])
            loss = model_loss(model, outputs, batch, config["training"], class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip", 5.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = infer(model, validation_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_micro_f1_at_5": float(metrics["f1"])}
        history.append(row)
        if metrics["f1"] > best:
            best = float(metrics["f1"])
            stale = 0
            torch.save({"method": method, "domain": domain, "seed": seed, "model_state": model.state_dict(), "config": dict(config), "best_validation_micro_f1_at_5": best}, checkpoint)
        else:
            stale += 1
        if stale >= int(config["training"].get("early_stopping_patience", 3)):
            break
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    metrics, records = infer(model, validation_loader, device)
    prediction_path = output / f"{domain}_{method}_seed{seed}_validation.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"checkpoint": str(checkpoint), "predictions": str(prediction_path), "metrics": metrics, "history": history}
