# real-estate-multimodal

This repo builds a clean, reproducible pipeline for a real-estate dataset:
- Freeze dataset snapshot
- Build manifest (one row per image)
- Create leak-free train/val/test splits (by listing_id)
- Build match/mismatch pairs (positive + negative)
- Run baselines (pretrained-first):
  - Text: TF-IDF (classic) + SBERT multilingual MPNet (strong)
  - Image: ConvNeXt Tiny (strong) + optional ResNet18 (classic)
  - Multimodal: Multilingual CLIP (German-capable)

## 0) Setup

### Create & activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt