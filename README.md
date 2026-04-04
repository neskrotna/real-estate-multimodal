# real-estate-multimodal

This repo builds a clean, reproducible pipeline for a real-estate dataset:
- Freeze dataset snapshot
- Build manifest (one row per image)
- Create leak-free train/val/test splits (by listing_id)
- Build match/mismatch pairs (positive + negative)
- Run baselines and experiments:
  - Multimodal: Multilingual CLIP (German-capable)
  - Multimodal: Max-image similarity pooling
  - Multimodal: CLIP + metadata classifier
  - Multimodal: Lightweight projection-head fine-tuning

  - Text: TF-IDF (classic) + SBERT multilingual MPNet (strong)
  - Image: ConvNeXt Tiny (strong) + optional ResNet18 (classic)

## 0) Setup

### Create & activate venv (on Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Data Pipeline

### Build image manifest
```bash
python -m src.data.build_manifest
```

### Create splits
```bash
python -m src.data.create_splits
```
#### Output:
```bash
data/processed/split_v1.json
```

### Build pairs:
```bash
python -m src.data.build_pairs
```
#### Output:
```bash
data/processed/pairs_train.jsonl
data/processed/pairs_val.jsonl
data/processed/pairs_test.jsonl
```

## 2) Experiments

All experiments are located in:
```bash
src/experiments/
```
Outputs are written to:
```bash
runs/
```