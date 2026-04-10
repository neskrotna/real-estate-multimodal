import { useState } from "react";

const API_BASE = "http://localhost:3001";

function ListingCard({ listing, onClick }) {
  return (
    <div
      className={`listing-card ${onClick ? "listing-card-clickable" : ""}`}
      onClick={() => onClick?.(listing.listing_id, listing.title)}
    >
      {listing.preview_image_url ? (
        <img src={listing.preview_image_url} alt={listing.title} className="listing-image" />
      ) : (
        <div className="listing-image listing-image-placeholder" />
      )}

      <div className="listing-content">
        <div className="listing-title">{listing.title}</div>

        <div className="listing-meta">
          {[listing.city, listing.address].filter(Boolean).join(" · ")}
        </div>

        <div className="listing-description">
          {listing.description
            ? listing.description.slice(0, 140) + (listing.description.length > 140 ? "..." : "")
            : ""}
        </div>

        <div className="listing-footer">
          {listing.price_eur ? <span>{listing.price_eur} €</span> : null}
          {listing.rooms ? <span>{listing.rooms} rooms</span> : null}
          {listing.area_m2 ? <span>{listing.area_m2} m²</span> : null}
        </div>

        {listing.score !== undefined && (
          <div className="listing-score">Score: {Number(listing.score).toFixed(3)}</div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [model, setModel] = useState("tfidf");
  const [results, setResults] = useState([]);
  const [similar, setSimilar] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState("");
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(false);

  const handleSearch = async () => {
    const q = query.trim();
    if (!q) return;

    setIsLoadingSearch(true);
    setSimilar([]);
    setSelectedTitle("");

    try {
      const res = await fetch(
        `${API_BASE}/search?q=${encodeURIComponent(q)}&model=${encodeURIComponent(model)}`
      );
      const data = await res.json();
      setResults(data);
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    } finally {
      setIsLoadingSearch(false);
    }
  };

  const handleResultClick = async (listingId, title) => {
    setIsLoadingSimilar(true);
    setSelectedTitle(title);

    try {
      const res = await fetch(
        `${API_BASE}/recommend?listing_id=${encodeURIComponent(listingId)}&model=${encodeURIComponent(model)}`
      );
      const data = await res.json();
      setSimilar(data);
    } catch (error) {
      console.error("Recommendation failed:", error);
      setSimilar([]);
    } finally {
      setIsLoadingSimilar(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="page">
      <div className="search-shell">
        <div className="brand">Real Estate Search</div>

        <div className="search-row">
          <input
            type="text"
            className="search-input"
            placeholder="Search listings..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <select
            className="search-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="tfidf">TF-IDF</option>
            <option value="sbert">SBERT</option>
            <option value="clip">CLIP</option>
            <option value="exp3">Exp3</option>
            <option value="resnet">ResNet</option>
          </select>

          <button className="search-button" onClick={handleSearch}>
            Search
          </button>
        </div>
      </div>

      <div className="section">
        <div className="section-header">
          <h2>Results</h2>
          <span className="model-badge">{model.toUpperCase()}</span>
        </div>

        {isLoadingSearch ? (
          <div className="empty">Searching...</div>
        ) : results.length === 0 ? (
          <div className="empty">No search results yet.</div>
        ) : (
          <div className="results-grid">
            {results.map((listing) => (
              <ListingCard
                key={listing.listing_id}
                listing={listing}
                onClick={handleResultClick}
              />
            ))}
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-header">
          <h2>
            Similar Listings
            {selectedTitle ? <span className="inline-note"> for “{selectedTitle}”</span> : null}
          </h2>
        </div>

        {isLoadingSimilar ? (
          <div className="empty">Loading similar listings...</div>
        ) : similar.length === 0 ? (
          <div className="empty">Click a result to load recommendations.</div>
        ) : (
          <div className="results-grid">
            {similar.map((listing) => (
              <ListingCard key={listing.listing_id} listing={listing} />
            ))}
          </div>
        )}
      </div>
      <div className="page-footer">Made by @Gabriela</div>
    </div>
  );
}