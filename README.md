# Cognition-text Enhanced Academic AKE

Research code for predicting word-level eye-tracking (ET) features and using
them in academic keyphrase extraction (AKE). This pre-publication repository
contains implementation, configuration, compact numerical outputs, and
synthetic examples. Manuscript prose, detailed interpretation, licensed
corpora, gold test labels, checkpoints, logs, and raw API responses are not
included.

## Components

- `src/`: Proxy-ET, AKE, evaluation, bootstrap, and LLM utilities
- `scripts/`: command-line training, inference, scoring, and audit tools
- `configs/`: portable experiment settings
- `data/`: schemas, manifests, and synthetic data only
- `results/`: machine-readable aggregate outputs
- `figures/`: plotting code and source tables
- `tests/`: data-free unit tests

The five ET variables are `NFIX`, `FFD`, `GPT`, `TRT`, and `FIXPROP`.
Predictions for academic tokens are called **Proxy ET**; they are model outputs,
not eye movements recorded from readers of the academic documents.

## Installation

```bash
conda env create -f environment.yml
conda activate cognition-ake-release
pip install -e .
```

Install a PyTorch/CUDA build compatible with the local GPU driver when GPU
training is required.

## Data-free checks

```bash
python -m pytest -q
python scripts/reproduce_tables.py
python scripts/plot_figure2.py
python scripts/audit_release.py
```

## Reproduction entry points

```bash
python scripts/train_proxy_et.py --config configs/proxy_et.yaml
python scripts/predict_proxy_et.py --config configs/proxy_et.yaml \
  --documents data/processed/academic_inference.jsonl
python scripts/train_ake.py --config configs/ake.yaml \
  --method ET-ATT --domain PMC --seed 42
python scripts/evaluate_predictions.py predictions.jsonl --k 3 5 10
```

Input schemas are stored under `data/schema/`; metric and bootstrap behaviour
is implemented in `src/cognition_ake/evaluation.py` and
`src/cognition_ake/bootstrap.py`.

## Data and release boundary

Third-party datasets must be obtained from their official distributors. Keep
raw documents, participant data, annotations, sealed test gold, embeddings,
checkpoints, and API responses outside Git. The synthetic JSONL file documents
the expected interface without reproducing source texts.

## Manuscript status and citation

This repository accompanies the following manuscript, currently under review
at iConference 2027 (the 22nd annual conference of the iSchools):

> Yan, X., Peng, J., Wang, J., Lu, C., & Zhang, C. (2027). From natural
> reading to academic keyphrase extraction: Cross-domain transfer of open
> eye-tracking signals. *Manuscript submitted to iConference 2027*. Under
> review.

Authors: Xinyi Yan, Jitong Peng, Jiafeng Wang, Chao Lu, and Chengzhi Zhang\*.
\* Corresponding author.

```bibtex
@unpublished{yan2027naturalreading,
  author    = {Xinyi Yan and Jitong Peng and Jiafeng Wang and Chao Lu and Chengzhi Zhang},
  title     = {From Natural Reading to Academic Keyphrase Extraction: Cross-Domain Transfer of Open Eye-Tracking Signals},
  year      = {2027},
  note      = {Manuscript submitted to iConference 2027; under review}
}
```

Please update the citation with the final proceedings metadata if the paper is
accepted and published.
