import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import npyjs from "npyjs";

const app = express();
app.use(cors());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "../..");

const PORT = 3001;

// Serve images from project data folder
app.use("/data", express.static(path.join(PROJECT_ROOT, "data")));

const ARTIFACTS_DIR = path.join(PROJECT_ROOT, "demo", "artifacts");

function readJSON(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function normalizeVector(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  return vec.map((v) => v / norm);
}

function cosine(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

function topKFromScores(items, k = 5) {
  return items.sort((a, b) => b.score - a.score).slice(0, k);
}

function buildSearchText(listing) {
  return [
    listing.title || "",
    listing.description || "",
    listing.city || "",
    listing.property_type || "",
    listing.condition || "",
    listing.listing_type || "",
    listing.rooms != null ? String(listing.rooms) : "",
    listing.area_m2 != null ? String(listing.area_m2) : ""
  ]
    .join(" ")
    .toLowerCase();
}

function simpleKeywordScore(query, listing) {
  const q = query.toLowerCase().trim();
  if (!q) return 0;

  const text = buildSearchText(listing);
  const terms = q.split(/\s+/).filter(Boolean);

  let score = 0;
  for (const term of terms) {
    if (text.includes(term)) score += 1;
  }

  if (listing.title?.toLowerCase().includes(q)) score += 2;
  if (listing.description?.toLowerCase().includes(q)) score += 1;

  return score;
}

async function loadNpy(filePath) {
  const parser = new npyjs();

  const buffer = fs.readFileSync(filePath);

  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  );

  const data = parser.parse(arrayBuffer);

  return {
    data: Array.from(data.data),
    shape: data.shape
  };
}

function getVectorAt(flatArray, shape, index) {
  const dim = shape[1];
  const start = index * dim;
  const end = start + dim;
  return flatArray.slice(start, end);
}

function attachPreviewUrl(listing) {
  if (!listing) return listing;
  return {
    ...listing,
    preview_image_url: listing.preview_image
      ? `http://localhost:${PORT}/${String(listing.preview_image).replace(/\\/g, "/")}`
      : null
  };
}

async function main() {
  console.log("PROJECT_ROOT:", PROJECT_ROOT);

  const metadataPath = path.join(ARTIFACTS_DIR, "listings_metadata.json");
  const metadata = readJSON(metadataPath);

  const listingMap = {};
  metadata.forEach((listing) => {
    listingMap[listing.listing_id] = listing;
  });

  // TF-IDF artifacts are not used as true vector search in this backend version.
  // We use lexical matching for search requests under model=tfidf.
  // Recommendation for tfidf is approximated via SBERT fallback later if needed.

  // SBERT
  const sbertIds = readJSON(path.join(ARTIFACTS_DIR, "sbert", "listing_ids.json"));
  const sbertEmb = await loadNpy(path.join(ARTIFACTS_DIR, "sbert", "embeddings.npy"));

  // ResNet
  const resnetIds = readJSON(path.join(ARTIFACTS_DIR, "resnet", "listing_ids.json"));
  const resnetEmb = await loadNpy(path.join(ARTIFACTS_DIR, "resnet", "embeddings.npy"));

  // CLIP
  const clipIds = readJSON(path.join(ARTIFACTS_DIR, "clip", "listing_ids.json"));
  const clipTextEmb = await loadNpy(path.join(ARTIFACTS_DIR, "clip", "text_embeddings.npy"));
  const clipCombinedEmb = await loadNpy(path.join(ARTIFACTS_DIR, "clip", "combined_embeddings.npy"));

  // Exp3
  const exp3Ids = readJSON(path.join(ARTIFACTS_DIR, "exp3", "listing_ids.json"));
  const exp3TextEmb = await loadNpy(path.join(ARTIFACTS_DIR, "exp3", "text_embeddings.npy"));
  const exp3CombinedEmb = await loadNpy(path.join(ARTIFACTS_DIR, "exp3", "combined_embeddings.npy"));

  // Metadata endpoint
  app.get("/api/health", (req, res) => {
    res.json({ ok: true });
  });

  app.get("/api/models", (req, res) => {
    res.json([
      { key: "tfidf", label: "TF-IDF" },
      { key: "sbert", label: "SBERT" },
      { key: "clip", label: "CLIP" },
      { key: "exp3", label: "Exp3" },
      { key: "resnet", label: "ResNet" }
    ]);
  });

  // SEARCH
  app.get("/search", (req, res) => {
    const q = String(req.query.q || "").trim();
    const model = String(req.query.model || "tfidf").trim().toLowerCase();

    if (!q) {
      return res.json([]);
    }

    // TF-IDF style lexical search
    if (model === "tfidf") {
      const scored = metadata
        .map((listing) => ({
          ...attachPreviewUrl(listing),
          score: simpleKeywordScore(q, listing)
        }))
        .filter((x) => x.score > 0);

      return res.json(topKFromScores(scored, 5));
    }

    // For sbert / clip / exp3 in this version:
    // search is approximated by lexical ranking from metadata
    // because live text encoding is not yet wired in Node.
    // This still gives a usable demo flow.
    if (model === "sbert" || model === "clip" || model === "exp3") {
      const scored = metadata
        .map((listing) => ({
          ...attachPreviewUrl(listing),
          score: simpleKeywordScore(q, listing)
        }))
        .filter((x) => x.score > 0);

      return res.json(topKFromScores(scored, 5));
    }

    // ResNet is not a text-search model
    return res.json([]);
  });

  // RECOMMEND
  app.get("/recommend", (req, res) => {
    const listingId = String(req.query.listing_id || "").trim();
    const model = String(req.query.model || "resnet").trim().toLowerCase();

    if (!listingId) {
      return res.json([]);
    }

    let ids = null;
    let emb = null;

    if (model === "resnet") {
      ids = resnetIds;
      emb = resnetEmb;
    } else if (model === "clip") {
      ids = clipIds;
      emb = clipCombinedEmb;
    } else if (model === "exp3") {
      ids = exp3Ids;
      emb = exp3CombinedEmb;
    } else if (model === "sbert") {
      ids = sbertIds;
      emb = sbertEmb;
    } else if (model === "tfidf") {
      // fallback: use SBERT for recommendation if TF-IDF is selected
      ids = sbertIds;
      emb = sbertEmb;
    } else {
      return res.json([]);
    }

    const idx = ids.indexOf(listingId);
    if (idx === -1) {
      return res.json([]);
    }

    const queryVec = getVectorAt(emb.data, emb.shape, idx);

    const scored = ids.map((id, i) => {
      const candidateVec = getVectorAt(emb.data, emb.shape, i);
      return {
        listing_id: id,
        score: cosine(queryVec, candidateVec)
      };
    });

    const top = topKFromScores(
      scored.filter((x) => x.listing_id !== listingId),
      5
    ).map((item) => ({
      ...attachPreviewUrl(listingMap[item.listing_id]),
      score: item.score
    }));

    return res.json(top);
  });

  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Failed to start backend:", err);
  process.exit(1);
});