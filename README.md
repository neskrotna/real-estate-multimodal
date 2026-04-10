# real-estate-multimodal

This repository implements a reproducible multimodal pipeline for real estate listings, combining text and images for similarity and retrieval tasks.

## Overview

The pipeline includes:

  - Dataset preparation and standardization
  - Train/val/test splits (by listing_id)
  - Positive and negative pair construction
  - Baseline and multimodal experiments
  - Retrieval-based evaluation (Recall@K, MRR)
  - Visualization and qualitative analysis
  - Interactive demo for search and recommendation

## Models & Baselines

### Text Models

  - TF-IDF (classical baseline)
  - Multilingual SBert (semantic baseline)

### Image Model

  - ResNet-50 (CNN-based)

### Multimodal Models

 - CLIP Vision Encoder (ViT-B/32)
    - CLIP (image + text shared embedding space)
    - Max-image similarity pooling
    - Projection-head fine-tuning
    - CLIP + metadata classifier

## 0) Setup

### Create & activate venv (Windows)
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

### Build positive and negative pairs:
```bash
python -m src.data.build_pairs
```
#### Output:
```bash
data/processed/pairs_train.jsonl
data/processed/pairs_val.jsonl
data/processed/pairs_test.jsonl
```

## 2) Baselines

Baselines provide controlled comparison across model types.

### Text baseline - TF-IDF

```bash
python -m src.baselines.text_tfidf_baseline
```

### Text baseline - SBert

```bash
python -m src.baselines.text_sbert_baseline
```

### Image baseline - ResNet

```bash
python -m src.baselines.image_resnet_baseline
```

### Output example

```bash
metrics.json
```

## 3) Experiments

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

### Experiment 2: CLIP + metadata classifier
```bash
python -m src.experiments.exp2_clip_metadata_classifier
```

### Experiment 3: Projection-head fine-tuning
```bash
python -m src.experiments.exp3_clip_projection_finetune
```

### Experiment 4: CLIP-only classifier
```bash
python -m src.experiments.exp4_clip_only_classifier
```

### Experiment 5: Projection + max pooling
```bash
python -m src.experiments.exp5_clip_projection_maxpool
```

### Experiment 7: Hard negative fine-tuning
```bash
python -m src.experiments.exp7_clip_hard_negative_finetune
```

### Experiment 8: Pair classifier
```bash
python -m src.experiments.exp8_clip_pair_classifier
```

## 4) Visualizations

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

## 5) Demo (Search & Recommendation)

The project includes a lightweight web demo for interactive search and recommendation using all valuable trained models.

What it does:

  - Search listings using text queries
  - Retrieve top-k similar listings
  - Click a listing to get recommendations
  - Switch between models (TF-IDF, SBERT, ResNet, CLIP, CLIP-Exp3-Projection-Head Fine-Tuning)

## 6) Project structure
```bash
data/
  raw/
  processed/

demo/
  artifacts/
  backend/
  frontend/
  scripts

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