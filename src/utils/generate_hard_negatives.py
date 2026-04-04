from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LISTINGS_FILE = PROJECT_ROOT / "data" / "raw" / "descriptions" / "listings.xlsx"
IMAGES_ROOT = PROJECT_ROOT / "data" / "raw" / "images"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "descriptions" / "hard_negatives.xlsx"

# optional room labels file
# expected columns:
# listing_id | room_type
# e.g. VIE_A_001 | kitchen

ROOM_LABELS_FILE = PROJECT_ROOT / "data" / "raw" / "descriptions" / "room_labels.xlsx"

# SETTINGS
TOP_K_PER_LISTING = 3
MIN_TOTAL_SCORE = 8.0

# metadata weights
W_SAME_CITY = 2.0
W_SAME_TRANSACTION = 2.0
W_SAME_PROPERTY_TYPE = 2.0
W_SAME_CONDITION = 1.0
W_SAME_ROOMS = 2.0
W_AREA_CLOSE_10 = 2.0
W_AREA_CLOSE_20 = 1.0
W_PRICE_CLOSE_15 = 2.0
W_PRICE_CLOSE_30 = 1.0
W_SAME_OUTDOOR = 1.0

# similarity weights
W_TEXT_SIM = 3.0
W_IMAGE_SIM = 3.0
W_ROOM_COMP = 2.0

# representative image count per listing for embeddings
MAX_IMAGES_PER_LISTING = 8
IMAGE_SIZE = 224

# OPTIONAL IMAGE EMBEDDINGS
def load_image_model():
    """
    Loads a pretrained ResNet18 feature extractor from torchvision.
    First run may download weights.
    If weights cannot be downloaded, the script falls back to no image embeddings.
    """
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()

        transform = weights.transforms()

        return torch, model, transform
    except Exception as e:
        warnings.warn(
            f"Image embedding model could not be loaded. "
            f"Image similarity will be skipped.\nReason: {e}"
        )
        return None, None, None

# HELPERS
def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def cosine_sim(v1, v2):
    v1 = np.asarray(v1).reshape(1, -1)
    v2 = np.asarray(v2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0, 0])


def list_images(folder_path: Path):
    if not folder_path.exists():
        return []

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in folder_path.iterdir() if p.suffix.lower() in exts]
    files = sorted(files)

    if len(files) <= MAX_IMAGES_PER_LISTING:
        return files

    # evenly sample representative images
    idxs = np.linspace(0, len(files) - 1, MAX_IMAGES_PER_LISTING, dtype=int)
    return [files[i] for i in idxs]


def compute_listing_embedding(image_paths, torch, model, transform):
    if torch is None or model is None or transform is None:
        return None

    features = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = model(x).squeeze(0).cpu().numpy()

            # normalize each image embedding
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            features.append(feat)
        except Exception:
            continue

    if not features:
        return None

    emb = np.mean(features, axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def load_room_composition(room_labels_file: Path):
    """
    Expected columns:
    listing_id | room_type
    Each row = one room label for one listing
    Example:
    VIE_A_001 | kitchen
    VIE_A_001 | bathroom
    VIE_A_001 | bedroom
    """
    if not room_labels_file.exists():
        return {}

    room_df = pd.read_excel(room_labels_file)

    required = {"listing_id", "room_type"}
    if not required.issubset(room_df.columns):
        warnings.warn(
            f"{room_labels_file.name} exists but does not contain the required columns: {required}"
        )
        return {}

    room_df["listing_id"] = room_df["listing_id"].astype(str).str.strip()
    room_df["room_type"] = room_df["room_type"].astype(str).str.strip().str.lower()

    room_map = (
        room_df.groupby("listing_id")["room_type"]
        .apply(lambda x: sorted(set(x.tolist())))
        .to_dict()
    )
    return room_map


def jaccard_similarity(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)

    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b
    inter = set_a & set_b

    if not union:
        return 0.0

    return len(inter) / len(union)

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading listings from: {LISTINGS_FILE}")

    if not LISTINGS_FILE.exists():
        raise FileNotFoundError(f"Listings file not found: {LISTINGS_FILE}")

    df = pd.read_excel(LISTINGS_FILE).copy()

    # required columns from your real file
    required_cols = {
        "anzeigen_id",
        "titel",
        "beschreibung",
        "stadt",
        "preis_brutto",
        "zimmer",
        "wohnfläche_m2",
        "immobilientyp",
        "zustand",
        "image_folder",
        "miet_kauf",
        "freifläche",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in listings.xlsx: {missing}")

    # normalize text columns
    text_cols = [
        "anzeigen_id",
        "titel",
        "beschreibung",
        "strasse",
        "stadt",
        "immobilientyp",
        "zustand",
        "image_folder",
        "miet_kauf",
        "freifläche",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)

    # numeric columns
    for col in ["preis_brutto", "zimmer", "wohnfläche_m2", "hausnummer", "image_count"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    # drop rows with critical missing values
    df = df.dropna(subset=["anzeigen_id", "stadt", "preis_brutto", "zimmer", "wohnfläche_m2"])

    # text similarity
    print("Computing text similarity...")
    df["text_for_similarity"] = (
        df["titel"].fillna("") + " " + df["beschreibung"].fillna("")
    ).str.strip()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_for_similarity"])
    text_sim_matrix = cosine_similarity(tfidf_matrix)

    id_to_pos = {listing_id: idx for idx, listing_id in enumerate(df["anzeigen_id"].tolist())}

    # image embeddings
    print("Loading image embedding model...")
    torch, model, transform = load_image_model()

    print("Computing listing-level image embeddings...")
    image_embeddings = {}

    for _, row in df.iterrows():
        listing_id = row["anzeigen_id"]
        image_folder = row["image_folder"]

        folder_path = IMAGES_ROOT / image_folder
        image_paths = list_images(folder_path)

        emb = compute_listing_embedding(image_paths, torch, model, transform)
        image_embeddings[listing_id] = emb

    # room composition
    room_map = load_room_composition(ROOM_LABELS_FILE)
    if room_map:
        print(f"Loaded room composition labels from: {ROOM_LABELS_FILE}")
    else:
        print("No room_labels.xlsx found. Room composition similarity will be skipped.")

    # pair scoring
    print("Scoring candidate pairs...")
    results = []

    for i, j in combinations(df.index, 2):
        row1 = df.loc[i]
        row2 = df.loc[j]

        id1 = row1["anzeigen_id"]
        id2 = row2["anzeigen_id"]

        if id1 == id2:
            continue

        # skip likely same-address duplicates
        same_street = normalize_text(row1.get("strasse", "")) == normalize_text(row2.get("strasse", ""))
        same_house = normalize_text(row1.get("hausnummer", "")) == normalize_text(row2.get("hausnummer", ""))
        if same_street and same_house and same_street != "":
            continue

        score = 0.0
        reasons = []

        # metadata similarity
        if row1["stadt"] == row2["stadt"]:
            score += W_SAME_CITY
            reasons.append("same city")

        if row1["miet_kauf"] == row2["miet_kauf"]:
            score += W_SAME_TRANSACTION
            reasons.append("same miet_kauf")

        if row1["immobilientyp"] == row2["immobilientyp"]:
            score += W_SAME_PROPERTY_TYPE
            reasons.append("same immobilientyp")

        if row1["zustand"] == row2["zustand"]:
            score += W_SAME_CONDITION
            reasons.append("same zustand")

        if row1["zimmer"] == row2["zimmer"]:
            score += W_SAME_ROOMS
            reasons.append("same zimmer")

        area_diff = abs(row1["wohnfläche_m2"] - row2["wohnfläche_m2"])
        if area_diff <= 10:
            score += W_AREA_CLOSE_10
            reasons.append(f"wohnfläche diff <= 10 ({area_diff:.1f})")
        elif area_diff <= 20:
            score += W_AREA_CLOSE_20
            reasons.append(f"wohnfläche diff <= 20 ({area_diff:.1f})")

        p1, p2 = row1["preis_brutto"], row2["preis_brutto"]
        if max(p1, p2) > 0:
            price_diff_ratio = abs(p1 - p2) / max(p1, p2)
            if price_diff_ratio <= 0.15:
                score += W_PRICE_CLOSE_15
                reasons.append(f"price diff <= 15% ({price_diff_ratio:.1%})")
            elif price_diff_ratio <= 0.30:
                score += W_PRICE_CLOSE_30
                reasons.append(f"price diff <= 30% ({price_diff_ratio:.1%})")

        if row1["freifläche"] == row2["freifläche"]:
            score += W_SAME_OUTDOOR
            reasons.append("same freifläche")

        # text similarity
        pos1 = id_to_pos[id1]
        pos2 = id_to_pos[id2]
        text_sim = float(text_sim_matrix[pos1, pos2])
        score += text_sim * W_TEXT_SIM
        reasons.append(f"text_sim={text_sim:.3f}")

        # image similarity
        img_sim = None
        emb1 = image_embeddings.get(id1)
        emb2 = image_embeddings.get(id2)
        if emb1 is not None and emb2 is not None:
            img_sim = cosine_sim(emb1, emb2)
            score += img_sim * W_IMAGE_SIM
            reasons.append(f"image_sim={img_sim:.3f}")

        # room composition
        room_sim = None
        rooms1 = room_map.get(id1, [])
        rooms2 = room_map.get(id2, [])
        if rooms1 or rooms2:
            room_sim = jaccard_similarity(rooms1, rooms2)
            score += room_sim * W_ROOM_COMP
            reasons.append(f"room_comp_sim={room_sim:.3f}")

        if score >= MIN_TOTAL_SCORE:
            results.append({
                "query_listing_id": id1,
                "candidate_listing_id": id2,
                "hard_negative_score": round(score, 4),
                "text_similarity": round(text_sim, 4),
                "image_similarity": round(img_sim, 4) if img_sim is not None else None,
                "room_composition_similarity": round(room_sim, 4) if room_sim is not None else None,
                "reason": ", ".join(reasons)
            })
            results.append({
                "query_listing_id": id2,
                "candidate_listing_id": id1,
                "hard_negative_score": round(score, 4),
                "text_similarity": round(text_sim, 4),
                "image_similarity": round(img_sim, 4) if img_sim is not None else None,
                "room_composition_similarity": round(room_sim, 4) if room_sim is not None else None,
                "reason": ", ".join(reasons)
            })

    hard_df = pd.DataFrame(results)

    if hard_df.empty:
        print("No hard negatives found with the current threshold.")
        return

    # keep top k per listing
    hard_df = hard_df.sort_values(
        by=["query_listing_id", "hard_negative_score"],
        ascending=[True, False]
    )
    hard_df = hard_df.groupby("query_listing_id").head(TOP_K_PER_LISTING).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    hard_df.to_excel(OUTPUT_FILE, index=False)

    print(f"Saved {len(hard_df)} hard negative pairs to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()