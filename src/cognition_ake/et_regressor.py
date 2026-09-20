from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizerFast

from .proxy_et import LABEL_PAD, SOURCE_TARGETS, validate_et_frame


class EyeTrackingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: RobertaTokenizerFast) -> None:
        validate_et_frame(frame)
        self.sentence_ids = sorted(frame["sentence_id"].unique().tolist())
        self.rows = [frame[frame["sentence_id"] == sentence].sort_values("word_id") for sentence in self.sentence_ids]
        words = [rows["word"].astype(str).tolist() for rows in self.rows]
        self.encodings = tokenizer(words, is_split_into_words=True, add_special_tokens=True, padding=True, truncation=False, return_attention_mask=True, return_tensors="pt")
        labels = torch.full((len(words), self.encodings["input_ids"].shape[1], 5), LABEL_PAD, dtype=torch.float32)
        for index, rows in enumerate(self.rows):
            first = []
            previous = None
            for position, word_id in enumerate(self.encodings.word_ids(batch_index=index)):
                if word_id is not None and word_id != previous:
                    first.append(position)
                previous = word_id
            if len(first) != len(rows):
                raise RuntimeError(f"Subword alignment failure for sentence {self.sentence_ids[index]}")
            labels[index, first, :] = torch.tensor(rows[list(SOURCE_TARGETS)].to_numpy(np.float32))
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {"input_ids": self.encodings["input_ids"][index], "attention_mask": self.encodings["attention_mask"][index], "labels": self.labels[index]}


class RobertaETRegressor(nn.Module):
    def __init__(self, model_id: str = "roberta-base") -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_id)
        self.decoder = nn.Linear(self.roberta.config.hidden_size, 5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden)


@dataclass(frozen=True)
class TrainStage:
    name: str
    epochs: int


def train_stage(model: RobertaETRegressor, dataset: EyeTrackingDataset, device: torch.device, stage: TrainStage, *, learning_rate: float, micro_batch: int, gradient_accumulation: int) -> list[dict]:
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history = []
    for epoch in range(1, stage.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        count = 0
        for step, batch in enumerate(loader, 1):
            labels = batch["labels"].to(device)
            prediction = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            mask = labels[:, :, 0] != LABEL_PAD
            loss = (prediction[mask] - labels[mask]).pow(2).mean() / gradient_accumulation
            loss.backward()
            total += float(loss.detach().cpu()) * gradient_accumulation * int(mask.sum())
            count += int(mask.sum())
            if step % gradient_accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()
        history.append({"stage": stage.name, "epoch": epoch, "mean_token_mse": total / max(count, 1)})
    return history


def predict_dataset(model: RobertaETRegressor, dataset: EyeTrackingDataset, device: torch.device, batch_size: int = 4) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            prediction = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu()
            mask = batch["labels"][:, :, 0] != LABEL_PAD
            output.append(prediction[mask].numpy())
    return np.clip(np.vstack(output), 0.0, 100.0)

