import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import multer from "multer";
import npyjs from "npyjs";

const app = express();

app.use(cors());
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const PORT = 3001;

const ARTIFACTS_DIR = path.join(
  PROJECT_ROOT,
  "demo",
  "artifacts"
);

const UPLOADS_DIR = path.join(
  __dirname,
  "temporary_uploads"
);

const RESNET_INFERENCE_SCRIPT = path.join(
  __dirname,
  "resnet_inference.py"
);

fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// Makes files under PROJECT_ROOT/data available through /data
app.use(
  "/data",
  express.static(path.join(PROJECT_ROOT, "data"))
);

const uploadStorage = multer.diskStorage({
  destination: (_req, _file, callback) => {
    callback(null, UPLOADS_DIR);
  },

  filename: (_req, file, callback) => {
    const safeExtension = path
      .extname(file.originalname)
      .toLowerCase()
      .replace(/[^a-z0-9.]/g, "");

    const uniqueName = [
      Date.now(),
      Math.round(Math.random() * 1_000_000),
    ].join("-");

    callback(
      null,
      `${uniqueName}${safeExtension || ".jpg"}`
    );
  },
});

const upload = multer({
  storage: uploadStorage,

  limits: {
    fileSize: 10 * 1024 * 1024,
  },

  fileFilter: (_req, file, callback) => {
    if (!file.mimetype.startsWith("image/")) {
      callback(
        new Error("Only image files are accepted.")
      );
      return;
    }

    callback(null, true);
  },
});

function readJSON(filePath) {
  return JSON.parse(
    fs.readFileSync(filePath, "utf-8")
  );
}

function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error(
      `Embedding dimension mismatch: ${a.length} and ${b.length}.`
    );
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (normA * normB);
}

function topKFromScores(items, k = 5) {
  return [...items]
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

function buildSearchText(listing) {
  return [
    listing.title || "",
    listing.description || "",
    listing.city || "",
    listing.country || "",
    listing.address || "",
    listing.postal_code || "",
    listing.property_type || "",
    listing.condition || "",
    listing.listing_type || "",
    listing.rooms != null
      ? String(listing.rooms)
      : "",
    listing.area_m2 != null
      ? String(listing.area_m2)
      : "",
  ]
    .join(" ")
    .toLowerCase();
}

function simpleKeywordScore(query, listing) {
  const normalizedQuery = query.toLowerCase().trim();

  if (!normalizedQuery) {
    return 0;
  }

  const text = buildSearchText(listing);

  const terms = normalizedQuery
    .split(/\s+/)
    .filter(Boolean);

  let rawScore = 0;

  for (const term of terms) {
    if (text.includes(term)) {
      rawScore += 1;
    }
  }

  if (
    listing.title
      ?.toLowerCase()
      .includes(normalizedQuery)
  ) {
    rawScore += 2;
  }

  if (
    listing.description
      ?.toLowerCase()
      .includes(normalizedQuery)
  ) {
    rawScore += 1;
  }

  /*
   * Maximum possible lexical score:
   * one point for each query term,
   * two points for an exact title match,
   * one point for an exact description match.
   */
  const maximumScore = terms.length + 3;

  if (maximumScore === 0) {
    return 0;
  }

  return Math.min(rawScore / maximumScore, 1);
}

async function loadNpy(filePath) {
  const parser = new npyjs();
  const buffer = fs.readFileSync(filePath);

  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  );

  const parsed = parser.parse(arrayBuffer);

  return {
    data: Array.from(parsed.data),
    shape: parsed.shape,
  };
}

function getVectorAt(flatArray, shape, index) {
  if (
    !Array.isArray(shape) ||
    shape.length !== 2
  ) {
    throw new Error(
      `Expected a two-dimensional embedding matrix. Received shape: ${shape}`
    );
  }

  const dimension = shape[1];
  const start = index * dimension;
  const end = start + dimension;

  return flatArray.slice(start, end);
}

function toPublicImageUrl(relativePath) {
  if (!relativePath) {
    return null;
  }

  const normalizedPath = String(relativePath)
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");

  return `http://localhost:${PORT}/${normalizedPath}`;
}

function attachListingUrls(listing) {
  if (!listing) {
    return null;
  }

  const imagePaths = Array.isArray(
    listing.image_paths
  )
    ? listing.image_paths
    : [];

  return {
    ...listing,

    preview_image_url: toPublicImageUrl(
      listing.preview_image
    ),

    image_urls: imagePaths
      .map(toPublicImageUrl)
      .filter(Boolean),
  };
}

function deleteTemporaryFile(filePath) {
  if (!filePath) {
    return;
  }

  fs.unlink(filePath, (error) => {
    if (
      error &&
      error.code !== "ENOENT"
    ) {
      console.error(
        "Could not delete temporary upload:",
        error
      );
    }
  });
}

function runResNetInference(imagePath) {
  return new Promise((resolve, reject) => {
    /*
      When the backend is started from an activated virtual
      environment, "python" resolves to that environment.

      A custom executable may also be supplied:
      $env:PYTHON_BIN = "C:\\path\\to\\python.exe"
     */
    const pythonExecutable =
      process.env.PYTHON_BIN || "python";

    const child = spawn(
      pythonExecutable,
      [
        RESNET_INFERENCE_SCRIPT,
        imagePath,
      ],
      {
        cwd: __dirname,
        windowsHide: true,
      }
    );

    let stdout = "";
    let stderr = "";

    const timeout = setTimeout(() => {
      child.kill();

      reject(
        new Error(
          "ResNet inference exceeded the 120-second time limit."
        )
      );
    }, 120_000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (exitCode) => {
      clearTimeout(timeout);

      if (exitCode !== 0) {
        reject(
          new Error(
            stderr.trim() ||
              `Python inference exited with code ${exitCode}.`
          )
        );
        return;
      }

      try {
        const lines = stdout
          .trim()
          .split(/\r?\n/)
          .filter(Boolean);

        const finalLine =
          lines[lines.length - 1];

        const embedding =
          JSON.parse(finalLine);

        if (!Array.isArray(embedding)) {
          throw new Error(
            "Python did not return an embedding array."
          );
        }

        resolve(embedding);
      } catch (error) {
        reject(
          new Error(
            `Could not parse the ResNet embedding: ${error.message}`
          )
        );
      }
    });
  });
}

async function main() {
  console.log("PROJECT_ROOT:", PROJECT_ROOT);

  const metadataPath = path.join(
    ARTIFACTS_DIR,
    "listings_metadata.json"
  );

  const metadata = readJSON(metadataPath);

  const listingMap = {};

  metadata.forEach((listing) => {
    listingMap[listing.listing_id] =
      listing;
  });

  // SBERT artifacts
  const sbertIds = readJSON(
    path.join(
      ARTIFACTS_DIR,
      "sbert",
      "listing_ids.json"
    )
  );

  const sbertEmb = await loadNpy(
    path.join(
      ARTIFACTS_DIR,
      "sbert",
      "embeddings.npy"
    )
  );

  // ResNet artifacts
  const resnetIds = readJSON(
    path.join(
      ARTIFACTS_DIR,
      "resnet",
      "listing_ids.json"
    )
  );

  const resnetEmb = await loadNpy(
    path.join(
      ARTIFACTS_DIR,
      "resnet",
      "embeddings.npy"
    )
  );

  // CLIP artifacts
  const clipIds = readJSON(
    path.join(
      ARTIFACTS_DIR,
      "clip",
      "listing_ids.json"
    )
  );

  const clipCombinedEmb = await loadNpy(
    path.join(
      ARTIFACTS_DIR,
      "clip",
      "combined_embeddings.npy"
    )
  );

  // Experiment 3 artifacts
  const exp3Ids = readJSON(
    path.join(
      ARTIFACTS_DIR,
      "exp3",
      "listing_ids.json"
    )
  );

  const exp3CombinedEmb = await loadNpy(
    path.join(
      ARTIFACTS_DIR,
      "exp3",
      "combined_embeddings.npy"
    )
  );

  app.get("/api/health", (_req, res) => {
    res.json({
      ok: true,
      listing_count: metadata.length,
      resnet_embedding_shape:
        resnetEmb.shape,
    });
  });

  app.get("/api/models", (_req, res) => {
    res.json([
      {
        key: "tfidf",
        label: "TF-IDF",
        input_type: "text",
      },
      {
        key: "sbert",
        label: "SBERT",
        input_type: "text",
      },
      {
        key: "clip",
        label: "CLIP",
        input_type: "text",
      },
      {
        key: "exp3",
        label:
          "Projection Fine-Tuning",
        input_type: "text",
      },
      {
        key: "resnet",
        label: "ResNet-50",
        input_type: "image",
      },
    ]);
  });

  /*
    Returns the complete listing used by the
    double-click details modal.
   */
  app.get(
    "/listing/:listingId",
    (req, res) => {
      const listingId = String(
        req.params.listingId || ""
      ).trim();

      const listing =
        listingMap[listingId];

      if (!listing) {
        return res.status(404).json({
          error: "Listing not found.",
        });
      }

      return res.json(
        attachListingUrls(listing)
      );
    }
  );

  /*
    Text search

    TF-IDF, SBERT, CLIP and Experiment 3 use the existing lexical demo logic
   */
  app.get("/search", (req, res) => {
    const query = String(
      req.query.q || ""
    ).trim();

    const model = String(
      req.query.model || "tfidf"
    )
      .trim()
      .toLowerCase();

    if (!query) {
      return res.json([]);
    }

    if (model === "resnet") {
      return res.status(400).json({
        error:
          "ResNet-50 requires an image upload.",
      });
    }

    const supportedTextModels = [
      "tfidf",
      "sbert",
      "clip",
      "exp3",
    ];

    if (
      !supportedTextModels.includes(model)
    ) {
      return res.status(400).json({
        error: "Unknown search model.",
      });
    }

    const scored = metadata
      .map((listing) => ({
        ...attachListingUrls(listing),
        score: simpleKeywordScore(
          query,
          listing
        ),
      }))
      .filter((listing) => {
        return listing.score > 0;
      });

    return res.json(
      topKFromScores(scored, 5)
    );
  });

  /*
  ResNet-50 image search
   */
  app.post(
    "/search/image",
    upload.single("image"),
    async (req, res) => {
      if (!req.file) {
        return res.status(400).json({
          error:
            "No image file was uploaded.",
        });
      }

      try {
        const queryEmbedding =
          await runResNetInference(
            req.file.path
          );

        const expectedDimension =
          resnetEmb.shape[1];

        if (
          queryEmbedding.length !==
          expectedDimension
        ) {
          throw new Error(
            `Expected a ${expectedDimension}-dimensional embedding, but received ${queryEmbedding.length}.`
          );
        }

        const scored = resnetIds.map(
          (listingId, index) => {
            const listingEmbedding =
              getVectorAt(
                resnetEmb.data,
                resnetEmb.shape,
                index
              );

            return {
              listing_id: listingId,
              score: cosine(
                queryEmbedding,
                listingEmbedding
              ),
            };
          }
        );

        const topResults =
          topKFromScores(scored, 5)
            .map((item) => {
              const listing =
                listingMap[
                  item.listing_id
                ];

              if (!listing) {
                return null;
              }

              return {
                ...attachListingUrls(
                  listing
                ),
                score: item.score,
              };
            })
            .filter(Boolean);

        return res.json(topResults);
      } catch (error) {
        console.error(
          "ResNet image search failed:",
          error
        );

        return res.status(500).json({
          error:
            error.message ||
            "ResNet image search failed.",
        });
      } finally {
        deleteTemporaryFile(
          req.file.path
        );
      }
    }
  );

  app.get("/recommend", (req, res) => {
    const listingId = String(
      req.query.listing_id || ""
    ).trim();

    const model = String(
      req.query.model || "resnet"
    )
      .trim()
      .toLowerCase();

    if (!listingId) {
      return res.json([]);
    }

    let ids;
    let embeddings;

    if (model === "resnet") {
      ids = resnetIds;
      embeddings = resnetEmb;
    } else if (model === "clip") {
      ids = clipIds;
      embeddings = clipCombinedEmb;
    } else if (model === "exp3") {
      ids = exp3Ids;
      embeddings = exp3CombinedEmb;
    } else if (model === "sbert") {
      ids = sbertIds;
      embeddings = sbertEmb;
    } else if (model === "tfidf") {
      /*
      The current demo has no stored TF-IDF listing vectors. 
      SBERT remains the recommendation fallback.
       */
      ids = sbertIds;
      embeddings = sbertEmb;
    } else {
      return res.status(400).json({
        error: "Unknown model.",
      });
    }

    const queryIndex =
      ids.indexOf(listingId);

    if (queryIndex === -1) {
      return res.json([]);
    }

    const queryVector = getVectorAt(
      embeddings.data,
      embeddings.shape,
      queryIndex
    );

    const scored = ids.map(
      (candidateId, index) => {
        const candidateVector =
          getVectorAt(
            embeddings.data,
            embeddings.shape,
            index
          );

        return {
          listing_id: candidateId,
          score: cosine(
            queryVector,
            candidateVector
          ),
        };
      }
    );

    const recommendations =
      topKFromScores(
        scored.filter(
          (item) =>
            item.listing_id !==
            listingId
        ),
        5
      )
        .map((item) => {
          const listing =
            listingMap[
              item.listing_id
            ];

          if (!listing) {
            return null;
          }

          return {
            ...attachListingUrls(
              listing
            ),
            score: item.score,
          };
        })
        .filter(Boolean);

    return res.json(recommendations);
  });

  app.use(
    (
      error,
      _req,
      res,
      _next
    ) => {
      console.error(
        "Backend request failed:",
        error
      );

      if (
        error instanceof
        multer.MulterError
      ) {
        return res.status(400).json({
          error:
            error.code ===
            "LIMIT_FILE_SIZE"
              ? "The image is larger than 10 MB."
              : error.message,
        });
      }

      return res.status(500).json({
        error:
          error.message ||
          "Unexpected backend error.",
      });
    }
  );

  app.listen(PORT, () => {
    console.log(
      `Server running on http://localhost:${PORT}`
    );
  });
}

main().catch((error) => {
  console.error(
    "Failed to start backend:",
    error
  );

  process.exit(1);
});