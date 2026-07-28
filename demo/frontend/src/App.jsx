import {
  useEffect,
  useRef,
  useState,
} from "react";

const API_BASE = "http://localhost:3001";

const MODEL_LABELS = {
  tfidf: "TF-IDF",
  sbert: "SBERT",
  clip: "CLIP",
  exp3: "Projection Fine-Tuning",
  resnet: "ResNet-50",
};

function ListingCard({
  listing,
  onSingleClick,
  onDoubleClick,
}) {
  const clickTimerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (clickTimerRef.current) {
        clearTimeout(
          clickTimerRef.current
        );
      }
    };
  }, []);

  const handleClick = () => {
    if (!onSingleClick) {
      return;
    }

    if (clickTimerRef.current) {
      clearTimeout(
        clickTimerRef.current
      );
    }

    /*
    Wait briefly before treating the interaction as
    a single click. A double-click cancels this timer.
     */
    clickTimerRef.current = setTimeout(
      () => {
        onSingleClick(
          listing.listing_id,
          listing.title
        );

        clickTimerRef.current = null;
      },
      240
    );
  };

  const handleDoubleClick = (
    event
  ) => {
    event.preventDefault();

    if (clickTimerRef.current) {
      clearTimeout(
        clickTimerRef.current
      );

      clickTimerRef.current = null;
    }

    onDoubleClick?.(
      listing.listing_id
    );
  };

  return (
    <article
      className={[
        "listing-card",
        onSingleClick ||
        onDoubleClick
          ? "listing-card-clickable"
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={handleClick}
      onDoubleClick={
        handleDoubleClick
      }
      title={
        onDoubleClick
          ? "Double-click to open the complete listing"
          : undefined
      }
    >
      {listing.preview_image_url ? (
        <img
          src={
            listing.preview_image_url
          }
          alt={listing.title}
          className="listing-image"
        />
      ) : (
        <div className="listing-image listing-image-placeholder">
          No image
        </div>
      )}

      <div className="listing-content">
        <div className="listing-title">
          {listing.title}
        </div>

        <div className="listing-meta">
          {[
            listing.city,
            listing.address,
          ]
            .filter(Boolean)
            .join(" · ")}
        </div>

        <div className="listing-description">
          {listing.description
            ? listing.description.slice(
                0,
                140
              ) +
              (listing.description
                .length > 140
                ? "..."
                : "")
            : ""}
        </div>

        <div className="listing-footer">
          {listing.price_eur !=
          null ? (
            <span>
              {listing.price_eur} €
            </span>
          ) : null}

          {listing.rooms != null ? (
            <span>
              {listing.rooms} rooms
            </span>
          ) : null}

          {listing.area_m2 !=
          null ? (
            <span>
              {listing.area_m2} m²
            </span>
          ) : null}
        </div>

        {listing.score !==
          undefined && (
          <div className="listing-score">
            Similarity:{" "}
            {formatScore(
              listing.score
            )}
          </div>
        )}

        {onDoubleClick ? (
          <div className="listing-hint">
            Double-click for details
          </div>
        ) : null}
      </div>
    </article>
  );
}

function formatScore(score) {
  const value = Number(score);

  if (!Number.isFinite(value)) {
    return "n/a";
  }

  /*
   * Cosine similarity normally lies between -1 and 1.
   * Lexical scores in the existing text demo may exceed 1.
   */
  if (value >= -1 && value <= 1) {
    return `${(
      value * 100
    ).toFixed(1)}%`;
  }

  return value.toFixed(3);
}

function ListingModal({
  listing,
  isLoading,
  error,
  onClose,
}) {
  const [
    selectedImageIndex,
    setSelectedImageIndex,
  ] = useState(0);

  useEffect(() => {
    setSelectedImageIndex(0);
  }, [listing?.listing_id]);

  useEffect(() => {
    const handleKeyDown = (
      event
    ) => {
      if (event.key === "Escape") {
        onClose();
      }

      if (
        event.key ===
          "ArrowRight" &&
        listing?.image_urls?.length
      ) {
        setSelectedImageIndex(
          (current) =>
            (current + 1) %
            listing.image_urls
              .length
        );
      }

      if (
        event.key ===
          "ArrowLeft" &&
        listing?.image_urls?.length
      ) {
        setSelectedImageIndex(
          (current) =>
            (current -
              1 +
              listing.image_urls
                .length) %
            listing.image_urls
              .length
        );
      }
    };

    document.addEventListener(
      "keydown",
      handleKeyDown
    );

    const previousOverflow =
      document.body.style
        .overflow;

    document.body.style.overflow =
      "hidden";

    return () => {
      document.removeEventListener(
        "keydown",
        handleKeyDown
      );

      document.body.style.overflow =
        previousOverflow;
    };
  }, [listing, onClose]);

  const imageUrls =
    listing?.image_urls?.length
      ? listing.image_urls
      : listing?.preview_image_url
        ? [
            listing.preview_image_url,
          ]
        : [];

  const showPreviousImage = () => {
    if (imageUrls.length === 0) {
      return;
    }

    setSelectedImageIndex(
      (current) =>
        (current -
          1 +
          imageUrls.length) %
        imageUrls.length
    );
  };

  const showNextImage = () => {
    if (imageUrls.length === 0) {
      return;
    }

    setSelectedImageIndex(
      (current) =>
        (current + 1) %
        imageUrls.length
    );
  };

  const handleOverlayClick = (
    event
  ) => {
    if (
      event.target ===
      event.currentTarget
    ) {
      onClose();
    }
  };

  return (
    <div
      className="modal-overlay"
      onMouseDown={
        handleOverlayClick
      }
      role="presentation"
    >
      <section
        className="listing-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="listing-modal-title"
      >
        <button
          type="button"
          className="modal-close"
          onClick={onClose}
          aria-label="Close listing"
        >
          ×
        </button>

        {isLoading ? (
          <div className="modal-status">
            Loading complete
            listing...
          </div>
        ) : error ? (
          <div className="modal-status modal-error">
            {error}
          </div>
        ) : listing ? (
          <>
            <div className="modal-gallery">
              <div className="modal-main-image-shell">
                {imageUrls.length >
                0 ? (
                  <img
                    src={
                      imageUrls[
                        selectedImageIndex
                      ]
                    }
                    alt={`${listing.title}, image ${
                      selectedImageIndex +
                      1
                    }`}
                    className="modal-main-image"
                  />
                ) : (
                  <div className="modal-main-image modal-image-placeholder">
                    No images available
                  </div>
                )}

                {imageUrls.length >
                1 ? (
                  <>
                    <button
                      type="button"
                      className="gallery-arrow gallery-arrow-left"
                      onClick={
                        showPreviousImage
                      }
                      aria-label="Previous image"
                    >
                      ‹
                    </button>

                    <button
                      type="button"
                      className="gallery-arrow gallery-arrow-right"
                      onClick={
                        showNextImage
                      }
                      aria-label="Next image"
                    >
                      ›
                    </button>

                    <div className="gallery-counter">
                      {selectedImageIndex +
                        1}{" "}
                      / {imageUrls.length}
                    </div>
                  </>
                ) : null}
              </div>

              {imageUrls.length >
              1 ? (
                <div className="modal-thumbnails">
                  {imageUrls.map(
                    (
                      imageUrl,
                      index
                    ) => (
                      <button
                        type="button"
                        key={`${imageUrl}-${index}`}
                        className={[
                          "thumbnail-button",
                          index ===
                          selectedImageIndex
                            ? "thumbnail-button-active"
                            : "",
                        ]
                          .filter(
                            Boolean
                          )
                          .join(" ")}
                        onClick={() =>
                          setSelectedImageIndex(
                            index
                          )
                        }
                        aria-label={`Show image ${
                          index + 1
                        }`}
                      >
                        <img
                          src={
                            imageUrl
                          }
                          alt=""
                          className="thumbnail-image"
                        />
                      </button>
                    )
                  )}
                </div>
              ) : null}
            </div>

            <div className="modal-details">
              <div className="modal-location">
                {[
                  listing.postal_code,
                  listing.city,
                  listing.address,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </div>

              <h2 id="listing-modal-title">
                {listing.title}
              </h2>

              <div className="modal-price">
                {listing.price_eur !=
                null
                  ? `${listing.price_eur} €`
                  : "Price not available"}
              </div>

              <div className="modal-facts">
                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Rooms
                  </span>
                  <span>
                    {listing.rooms ??
                      "Not specified"}
                  </span>
                </div>

                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Area
                  </span>
                  <span>
                    {listing.area_m2 !=
                    null
                      ? `${listing.area_m2} m²`
                      : "Not specified"}
                  </span>
                </div>

                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Property type
                  </span>
                  <span>
                    {listing.property_type ||
                      "Not specified"}
                  </span>
                </div>

                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Condition
                  </span>
                  <span>
                    {listing.condition ||
                      "Not specified"}
                  </span>
                </div>

                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Listing type
                  </span>
                  <span>
                    {listing.listing_type ||
                      "Not specified"}
                  </span>
                </div>

                <div className="modal-fact">
                  <span className="modal-fact-label">
                    Outdoor space
                  </span>
                  <span>
                    {listing.has_outdoor_space
                      ? "Yes"
                      : "No"}
                  </span>
                </div>
              </div>

              <div className="modal-description-section">
                <h3>Description</h3>

                <p>
                  {listing.description ||
                    "No description available."}
                </p>
              </div>

              <div className="modal-listing-id">
                Listing ID:{" "}
                {listing.listing_id}
              </div>
            </div>
          </>
        ) : null}
      </section>
    </div>
  );
}

export default function App() {
  const [query, setQuery] =
    useState("");

  const [model, setModel] =
    useState("tfidf");

  const [results, setResults] =
    useState([]);

  const [similar, setSimilar] =
    useState([]);

  const [
    selectedTitle,
    setSelectedTitle,
  ] = useState("");

  const [
    isLoadingSearch,
    setIsLoadingSearch,
  ] = useState(false);

  const [
    isLoadingSimilar,
    setIsLoadingSimilar,
  ] = useState(false);

  const [
    searchError,
    setSearchError,
  ] = useState("");

  const [
    selectedImage,
    setSelectedImage,
  ] = useState(null);

  const [
    selectedImagePreview,
    setSelectedImagePreview,
  ] = useState("");

  const [
    isModalOpen,
    setIsModalOpen,
  ] = useState(false);

  const [
    modalListing,
    setModalListing,
  ] = useState(null);

  const [
    isLoadingListing,
    setIsLoadingListing,
  ] = useState(false);

  const [
    listingError,
    setListingError,
  ] = useState("");

  const fileInputRef =
    useRef(null);

  const isImageSearch =
    model === "resnet";

  useEffect(() => {
    return () => {
      if (
        selectedImagePreview
      ) {
        URL.revokeObjectURL(
          selectedImagePreview
        );
      }
    };
  }, [selectedImagePreview]);

  const resetSearchResults = () => {
    setResults([]);
    setSimilar([]);
    setSelectedTitle("");
    setSearchError("");
  };

  const handleModelChange = (
    event
  ) => {
    const nextModel =
      event.target.value;

    setModel(nextModel);
    resetSearchResults();

    if (nextModel === "resnet") {
      setQuery("");
    } else {
      setSelectedImage(null);

      if (selectedImagePreview) {
        URL.revokeObjectURL(
          selectedImagePreview
        );
      }

      setSelectedImagePreview("");
    }
  };

  const handleImageSelection = (
    event
  ) => {
    const file =
      event.target.files?.[0];

    setSearchError("");

    if (!file) {
      return;
    }

    if (
      !file.type.startsWith(
        "image/"
      )
    ) {
      setSearchError(
        "Please select an image file."
      );
      return;
    }

    if (
      file.size >
      10 * 1024 * 1024
    ) {
      setSearchError(
        "The selected image is larger than 10 MB."
      );
      return;
    }

    if (selectedImagePreview) {
      URL.revokeObjectURL(
        selectedImagePreview
      );
    }

    setSelectedImage(file);
    setSelectedImagePreview(
      URL.createObjectURL(file)
    );
    resetSearchResults();
  };

  const removeSelectedImage = () => {
    setSelectedImage(null);

    if (selectedImagePreview) {
      URL.revokeObjectURL(
        selectedImagePreview
      );
    }

    setSelectedImagePreview("");

    if (fileInputRef.current) {
      fileInputRef.current.value =
        "";
    }

    resetSearchResults();
  };

  const handleSearch = async () => {
    setSearchError("");
    setSimilar([]);
    setSelectedTitle("");

    if (isImageSearch) {
      if (!selectedImage) {
        setSearchError(
          "Select an image before starting the ResNet-50 search."
        );
        return;
      }
    } else if (!query.trim()) {
      setSearchError(
        "Enter a search query."
      );
      return;
    }

    setIsLoadingSearch(true);

    try {
      let response;

      if (isImageSearch) {
        const formData =
          new FormData();

        formData.append(
          "image",
          selectedImage
        );

        response = await fetch(
          `${API_BASE}/search/image`,
          {
            method: "POST",
            body: formData,
          }
        );
      } else {
        response = await fetch(
          `${API_BASE}/search?q=${encodeURIComponent(
            query.trim()
          )}&model=${encodeURIComponent(
            model
          )}`
        );
      }

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            "Search failed."
        );
      }

      setResults(
        Array.isArray(data)
          ? data
          : []
      );
    } catch (error) {
      console.error(
        "Search failed:",
        error
      );

      setResults([]);

      setSearchError(
        error.message ||
          "Search failed."
      );
    } finally {
      setIsLoadingSearch(false);
    }
  };

  const handleResultClick = async (
    listingId,
    title
  ) => {
    setIsLoadingSimilar(true);
    setSelectedTitle(title);
    setSimilar([]);

    try {
      const response = await fetch(
        `${API_BASE}/recommend?listing_id=${encodeURIComponent(
          listingId
        )}&model=${encodeURIComponent(
          model
        )}`
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            "Recommendation request failed."
        );
      }

      setSimilar(
        Array.isArray(data)
          ? data
          : []
      );
    } catch (error) {
      console.error(
        "Recommendation failed:",
        error
      );

      setSimilar([]);
    } finally {
      setIsLoadingSimilar(false);
    }
  };

  const handleOpenListing = async (
    listingId
  ) => {
    setIsModalOpen(true);
    setIsLoadingListing(true);
    setListingError("");
    setModalListing(null);

    try {
      const response = await fetch(
        `${API_BASE}/listing/${encodeURIComponent(
          listingId
        )}`
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            "Could not load the complete listing."
        );
      }

      setModalListing(data);
    } catch (error) {
      console.error(
        "Listing request failed:",
        error
      );

      setListingError(
        error.message ||
          "Could not load the complete listing."
      );
    } finally {
      setIsLoadingListing(false);
    }
  };

  const handleCloseListing = () => {
    setIsModalOpen(false);
    setModalListing(null);
    setListingError("");
  };

  const handleKeyDown = (
    event
  ) => {
    if (
      event.key === "Enter" &&
      !isImageSearch
    ) {
      handleSearch();
    }
  };

  return (
    <div className="page">
      <div className="search-shell">
        <div className="brand">
          Real Estate Search
        </div>

        <div
          className={[
            "search-row",
            isImageSearch
              ? "search-row-image"
              : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          {isImageSearch ? (
            <div className="image-search-control">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="visually-hidden"
                onChange={
                  handleImageSelection
                }
              />

              {selectedImagePreview ? (
                <div className="image-upload-preview">
                  <img
                    src={
                      selectedImagePreview
                    }
                    alt="Selected ResNet search"
                  />

                  <div className="image-upload-preview-info">
                    <span className="image-upload-name">
                      {selectedImage.name}
                    </span>

                    <span className="image-upload-help">
                      This image will be
                      compared with the
                      representative image
                      of each listing.
                    </span>
                  </div>

                  <button
                    type="button"
                    className="image-remove-button"
                    onClick={
                      removeSelectedImage
                    }
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="image-upload-button"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  <span className="image-upload-icon">
                    ↑
                  </span>

                  <span>
                    Select an image for
                    ResNet-50 search
                  </span>

                  <small>
                    JPG, PNG or WEBP,
                    maximum 10 MB
                  </small>
                </button>
              )}
            </div>
          ) : (
            <input
              type="text"
              className="search-input"
              placeholder="Search listings..."
              value={query}
              onChange={(event) =>
                setQuery(
                  event.target.value
                )
              }
              onKeyDown={
                handleKeyDown
              }
            />
          )}

          <select
            className="search-select"
            value={model}
            onChange={
              handleModelChange
            }
          >
            <option value="tfidf">
              TF-IDF
            </option>

            <option value="sbert">
              SBERT
            </option>

            <option value="clip">
              CLIP
            </option>

            <option value="exp3">
              Projection Fine-Tuning
            </option>

            <option value="resnet">
              ResNet-50
            </option>
          </select>

          <button
            type="button"
            className="search-button"
            onClick={handleSearch}
            disabled={
              isLoadingSearch ||
              (isImageSearch &&
                !selectedImage)
            }
          >
            {isLoadingSearch
              ? isImageSearch
                ? "Encoding..."
                : "Searching..."
              : "Search"}
          </button>
        </div>

        {searchError ? (
          <div
            className="search-error"
            role="alert"
          >
            {searchError}
          </div>
        ) : null}

        {isImageSearch ? (
          <div className="search-mode-note">
            ResNet-50 performs image-to-image
            retrieval. Upload a property image
            instead of entering a text query.
          </div>
        ) : null}
      </div>

      <div className="section">
        <div className="section-header">
          <h2>Results</h2>

          <span className="model-badge">
            {MODEL_LABELS[model]}
          </span>
        </div>

        {isLoadingSearch ? (
          <div className="empty">
            {isImageSearch
              ? "Generating the image embedding and finding visually similar listings..."
              : "Searching..."}
          </div>
        ) : results.length === 0 ? (
          <div className="empty">
            {isImageSearch
              ? "Upload an image to retrieve visually similar listings."
              : "No search results yet."}
          </div>
        ) : (
          <>
            <div className="interaction-note">
              Single-click a result to load
              recommendations. Double-click it
              to open the complete listing.
            </div>

            <div className="results-grid">
              {results.map(
                (listing) => (
                  <ListingCard
                    key={
                      listing.listing_id
                    }
                    listing={listing}
                    onSingleClick={
                      handleResultClick
                    }
                    onDoubleClick={
                      handleOpenListing
                    }
                  />
                )
              )}
            </div>
          </>
        )}
      </div>

      <div className="section">
        <div className="section-header">
          <h2>
            Similar Listings

            {selectedTitle ? (
              <span className="inline-note">
                {" "}
                for “
                {selectedTitle}”
              </span>
            ) : null}
          </h2>
        </div>

        {isLoadingSimilar ? (
          <div className="empty">
            Loading similar listings...
          </div>
        ) : similar.length === 0 ? (
          <div className="empty">
            Single-click a result to load
            recommendations.
          </div>
        ) : (
          <div className="results-grid">
            {similar.map(
              (listing) => (
                <ListingCard
                  key={
                    listing.listing_id
                  }
                  listing={listing}
                  onSingleClick={
                    handleResultClick
                  }
                  onDoubleClick={
                    handleOpenListing
                  }
                />
              )
            )}
          </div>
        )}
      </div>

      <div className="page-footer">
        Made by @Gabriela
      </div>

      {isModalOpen ? (
        <ListingModal
          listing={modalListing}
          isLoading={
            isLoadingListing
          }
          error={listingError}
          onClose={
            handleCloseListing
          }
        />
      ) : null}
    </div>
  );
}