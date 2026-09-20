from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import RobertaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_ake.et_regressor import EyeTrackingDataset, RobertaETRegressor, TrainStage, predict_dataset, train_stage
from cognition_ake.proxy_et import ETScaler, SOURCE_TARGETS, set_seed


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Provo -> CMCL/ZuCo-NR Proxy ET predictor")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "proxy_et.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model_id"], add_prefix_space=True)
    model = RobertaETRegressor(cfg["model_id"]).to(device)
    frames = [pd.read_csv(resolve(cfg["data"][name])) for name in ("provo", "cmcl_nr_train")]
    datasets = [EyeTrackingDataset(frame, tokenizer) for frame in frames]
    history = []
    for stage_cfg, dataset in zip(cfg["training"]["stages"], datasets):
        history.extend(train_stage(model, dataset, device, TrainStage(stage_cfg["name"], int(stage_cfg["epochs"])), learning_rate=float(cfg["training"]["learning_rate"]), micro_batch=int(cfg["training"]["micro_batch"]), gradient_accumulation=int(cfg["training"]["gradient_accumulation"])))
    train_frame = frames[1]
    scaler = ETScaler.fit(train_frame, *map(float, cfg["prediction"]["standardized_clip"]))
    checkpoint = resolve(cfg["output"]["checkpoint"])
    scaler_path = resolve(cfg["output"]["scaler"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": cfg["model_id"], "targets": list(SOURCE_TARGETS), "seed": cfg["seed"], "state_dict": model.state_dict(), "history": history}, checkpoint)
    scaler_path.write_text(json.dumps(scaler.to_json(), indent=2) + "\n", encoding="utf-8")
    validation = pd.read_csv(resolve(cfg["data"]["cmcl_nr_validation"]))
    prediction = predict_dataset(model, EyeTrackingDataset(validation, tokenizer), device, int(cfg["prediction"]["batch_size"]))
    mae = abs(validation[list(SOURCE_TARGETS)].to_numpy() - prediction).mean(axis=0)
    print(json.dumps({"checkpoint": str(checkpoint), "validation_mae": dict(zip(SOURCE_TARGETS, map(float, mae))), "overall_mae": float(mae.mean())}, indent=2))


if __name__ == "__main__":
    main()

