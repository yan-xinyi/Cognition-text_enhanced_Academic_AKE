"""Controlled cognition-text fusion models used by the release trainer."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


METHODS = ("TEXT", "DC", "GIB", "GAZESUP", "CIB", "CogAlign", "FS-CA", "ET-ATT")


def bilstm(input_dim: int, hidden: int, layers: int, dropout: float) -> nn.LSTM:
    return nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)


def run_lstm(module: nn.LSTM, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    packed = pack_padded_sequence(values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
    packed_output, _ = module(packed)
    output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=values.size(1))
    return output


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(values: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(values, coefficient)


def masked_cognition(values: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
    finite = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return finite * feature_mask


class CogAlignBlock(nn.Module):
    """Shared/private multimodal alignment with text-only test inference."""

    def __init__(self, text_dim: int, hidden: int, align_hidden: int, layers: int, dropout: float, num_tags: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.text_aware_logits = nn.Linear(text_dim, 5)
        self.cognitive_projection = nn.Sequential(nn.Linear(11, text_dim), nn.ReLU(), nn.Dropout(dropout))
        self.private_text = bilstm(text_dim, align_hidden, layers, dropout)
        self.private_cognitive = bilstm(text_dim, align_hidden, layers, dropout)
        self.shared = bilstm(text_dim, align_hidden, layers, dropout)
        align_dim = 2 * align_hidden
        self.text_head = nn.Sequential(nn.Linear(2 * align_dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.cognitive_head = nn.Sequential(nn.Linear(2 * align_dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.text_classifier = nn.Linear(hidden, num_tags)
        self.cognitive_classifier = nn.Linear(hidden, num_tags)
        self.cognitive_crf = CRF(num_tags, batch_first=True)
        self.pool_score = nn.Linear(align_dim, 1)
        self.discriminator = nn.Sequential(nn.Linear(align_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def cognitive_input(self, text: torch.Tensor, cognitive: torch.Tensor, feature_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = masked_cognition(cognitive, feature_mask)
        logits = self.text_aware_logits(text).masked_fill(feature_mask <= 0, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where((feature_mask > 0).any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        return torch.cat([values * weights * 5.0, feature_mask, valid.unsqueeze(-1)], dim=-1)

    def pooled(self, hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        scores = self.pool_score(hidden).squeeze(-1).masked_fill(~token_mask, -1e4)
        return torch.sum(hidden * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)

    def forward(self, text: torch.Tensor, cognitive: torch.Tensor, feature_mask: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor, token_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        cognitive_hidden = self.cognitive_projection(self.cognitive_input(text, cognitive, feature_mask, valid))
        private_text = run_lstm(self.private_text, self.dropout(text), lengths)
        shared_text = run_lstm(self.shared, self.dropout(text), lengths)
        private_cognitive = run_lstm(self.private_cognitive, self.dropout(cognitive_hidden), lengths)
        shared_cognitive = run_lstm(self.shared, self.dropout(cognitive_hidden), lengths)
        text_hidden = self.text_head(torch.cat([private_text, shared_text], dim=-1))
        cognitive_branch = self.cognitive_head(torch.cat([private_cognitive, shared_cognitive], dim=-1))
        text_pool = self.pooled(shared_text, token_mask)
        cognitive_pool = self.pooled(shared_cognitive, token_mask)
        modality = gradient_reverse(torch.cat([text_pool, cognitive_pool], dim=0))
        labels = torch.cat([torch.zeros(text_pool.size(0), dtype=torch.long, device=text_pool.device), torch.ones(cognitive_pool.size(0), dtype=torch.long, device=text_pool.device)])
        return {
            "emissions": self.text_classifier(self.dropout(text_hidden)),
            "cognitive_emissions": self.cognitive_classifier(self.dropout(cognitive_branch)),
            "discriminator_logits": self.discriminator(modality),
            "modality_labels": labels,
        }

    def text_only(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        private_text = run_lstm(self.private_text, self.dropout(text), lengths)
        shared_text = run_lstm(self.shared, self.dropout(text), lengths)
        return self.text_classifier(self.dropout(self.text_head(torch.cat([private_text, shared_text], dim=-1))))


class CognitionAKETagger(nn.Module):
    """Eight controlled systems sharing a BiLSTM-CRF sequence tagger."""

    def __init__(self, method: str, config: Mapping, num_tags: int = 3) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        embedding_dim = int(config.get("embedding_dim", 768))
        hidden = int(config.get("hidden_size", 128))
        align_hidden = int(config.get("alignment_hidden", 64))
        layers = int(config.get("lstm_layers", 1))
        dropout = float(config.get("dropout", 0.2))
        self.dropout = nn.Dropout(dropout)
        input_dim = embedding_dim + (5 if method == "ET-ATT" else 0)
        latent_dim = int(config.get("cib_latent_dim", 8))
        if method == "CIB":
            input_dim += latent_dim
        self.backbone = bilstm(input_dim, hidden, layers, dropout)
        self.word_dim = 2 * hidden
        self.crf = CRF(num_tags, batch_first=True)
        self.classifier = nn.Linear(self.word_dim, num_tags)
        self.feature_gate = nn.Parameter(torch.zeros(5)) if method == "FS-CA" else None

        if method in {"CogAlign", "FS-CA"}:
            self.cogalign = CogAlignBlock(self.word_dim, align_hidden, align_hidden, layers, dropout, num_tags)
        elif method == "DC":
            self.cognitive_projection = nn.Sequential(nn.Linear(11, align_hidden), nn.ReLU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(self.word_dim + align_hidden, num_tags)
        elif method == "GIB":
            self.feature_embeddings = nn.ModuleList([nn.Linear(1, self.word_dim) for _ in range(5)])
            self.gib_norm = nn.LayerNorm(self.word_dim)
        elif method == "GAZESUP":
            self.gaze_head = nn.Linear(self.word_dim, 5)
        elif method == "CIB":
            bottleneck = int(config.get("cib_hidden", 32))
            self.cib_encoder = nn.Sequential(nn.Linear(5, bottleneck), nn.ReLU(), nn.Dropout(dropout))
            self.cib_mu = nn.Linear(bottleneck, latent_dim)
            self.cib_logvar = nn.Linear(bottleneck, latent_dim)
        elif method == "ET-ATT":
            self.task_attention = nn.Linear(self.word_dim, 1, bias=False)
            self.classifier = nn.Linear(2 * self.word_dim, num_tags)

    def _text_hidden(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return run_lstm(self.backbone, self.dropout(values), lengths)

    def forward_all(self, embedding: torch.Tensor, cognitive: torch.Tensor, feature_mask: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor, token_mask: torch.Tensor, *, inference: bool = False) -> Dict[str, torch.Tensor]:
        values = masked_cognition(cognitive, feature_mask)
        if self.method in {"CogAlign", "FS-CA"}:
            text = self._text_hidden(embedding, lengths)
            if inference:
                return {"emissions": self.cogalign.text_only(text, lengths)}
            if self.feature_gate is not None:
                gate = torch.softmax(self.feature_gate, dim=0) * 5.0
                values = values * gate.view(1, 1, 5)
            outputs = self.cogalign(text, values, feature_mask, valid, lengths, token_mask)
            if self.feature_gate is not None:
                outputs["feature_gate"] = gate
            return outputs

        diagnostics: Dict[str, torch.Tensor] = {}
        if self.method == "CIB":
            encoded = self.cib_encoder(values)
            mu = self.cib_mu(encoded)
            logvar = self.cib_logvar(encoded).clamp(-10.0, 10.0)
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training and not inference else mu
            hidden = self._text_hidden(torch.cat([embedding, latent * valid.unsqueeze(-1)], dim=-1), lengths)
            diagnostics.update({"mu": mu, "logvar": logvar})
        elif self.method == "ET-ATT":
            hidden = self._text_hidden(torch.cat([embedding, values], dim=-1), lengths)
            scores = self.task_attention(hidden).squeeze(-1) / math.sqrt(float(self.word_dim))
            scores = scores.masked_fill(~token_mask, -1e4)
            attention = torch.softmax(scores, dim=1) * token_mask.float()
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
            context = torch.bmm(attention.unsqueeze(1), hidden).squeeze(1).unsqueeze(1).expand(-1, hidden.size(1), -1)
            diagnostics["attention"] = attention
            return {"emissions": self.classifier(self.dropout(torch.cat([hidden, context], dim=-1))), **diagnostics}
        else:
            hidden = self._text_hidden(embedding, lengths)

        if self.method == "DC":
            cognitive_input = torch.cat([values, feature_mask, valid.unsqueeze(-1)], dim=-1)
            task_hidden = torch.cat([hidden, self.cognitive_projection(cognitive_input)], dim=-1)
        elif self.method == "GIB":
            available = (feature_mask > 0) & token_mask.unsqueeze(-1)
            probability = torch.relu(values) * available.float()
            probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
            entropy = -(probability.clamp_min(1e-8).log() * probability).sum(dim=1)
            diversity = (1.0 - entropy / available.sum(dim=1).float().clamp_min(2).log()).clamp_min(0)
            weights = diversity / diversity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            embedded = torch.stack([torch.tanh(layer(values[:, :, i:i+1])) for i, layer in enumerate(self.feature_embeddings)], dim=2)
            task_hidden = self.gib_norm(hidden + 0.5 * (embedded * weights[:, None, :, None]).sum(dim=2) * valid.unsqueeze(-1))
            diagnostics["feature_weights"] = weights
        else:
            task_hidden = hidden
        if self.method == "GAZESUP":
            diagnostics["gaze_prediction"] = self.gaze_head(hidden)
        return {"emissions": self.classifier(self.dropout(task_hidden)), **diagnostics}

    def decode(self, emissions: torch.Tensor, token_mask: torch.Tensor) -> Sequence[Sequence[int]]:
        return self.crf.decode(emissions, mask=token_mask.bool())
