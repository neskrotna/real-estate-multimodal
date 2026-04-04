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

### Experiment 1: CLIP with max-image pooling
```bash
python -m src.experiments.exp1_clip_maxpool
```
#### Output:
```bash
runs/exp1_clip_maxpool/
  scored_val.jsonl
  scored_test.jsonl
  val_metrics.json
  test_metrics.json
```

### Experiment 2: CLIP + metadata classifier
```bash
python -m src.experiments.exp2_clip_metadata_classifier
```
#### Output:
```bash
runs/exp2_clip_metadata_classifier/
  classifier.joblib
  feature_names.json
  feature_coefficients.json
  scored_val.jsonl
  scored_test.jsonl
  val_metrics.json
  test_metrics.json
```

### Experiment 3: Projection-head fine-tuning
```bash
python -m src.experiments.exp3_clip_projection_finetune
```
#### Output:
```bash
runs/exp3_clip_projection_finetune/
  best_model.pt
  history.json
  scored_val.jsonl
  scored_test.jsonl
  val_metrics.json
  test_metrics.json
```

## Visualizations

All visualization scripts are in:
```bash
src/visualizations/
```
Outputs are written to:
```bash
reports/figures/
reports/examples/
```

### Experiment comparsion
```bash
python -m src.visualizations.plot_experiment_comparison
```
Creates:
```bash
experiment_comparison.png
```

### Confusion matrices
```bash
python -m src.visualizations.plot_confusion_matrices
```
Creates:
```bash
exp1_confusion_matrix.png
exp2_confusion_matrix.png
exp3_confusion_matrix.png
```

### Similarity distributions
```bash
python -m src.visualizations.plot_similarity_distributions \
  --scored-file runs/exp1_clip_maxpool/scored_test.jsonl \
  --title "Exp1 similarity distribution" \
  --output reports/figures/exp1_similarity_distribution.png
```

### Fine-tuning training curve
```bash
python -m src.visualizations.plot_finetune_history
```
Creates:
```bash
exp3_training_curve.png
```

### Qualitative examples
```bash
python -m src.visualizations.export_qualitative_examples \
  --scored-file runs/exp1_clip_maxpool/scored_test.jsonl \
  --output-dir reports/examples/exp1
```

## 4) Project structure
```bash
data/
  raw/
  processed/

reports/
runs/

src/
  baselines/
  data/
  experiments/
  utils/
  visualizations/

tools/
  annotation_tool/
  image_conversion_tool/
```