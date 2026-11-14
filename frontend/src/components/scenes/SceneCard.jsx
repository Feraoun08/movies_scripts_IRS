import { useState } from 'react';
import { formatRating, formatScore, formatYear } from '../../utils/format.js';
import GenreChips from '../common/GenreChips.jsx';
import GlassPanel from '../common/GlassPanel.jsx';

export default function SceneCard({ scene }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <GlassPanel as="article" className="card scene-card">
      <header className="card-head">
        <div>
          <h3>
            {scene.title} <span className="muted">({formatYear(scene.year)})</span>
          </h3>
          <p className="muted">
            Scene {scene.scene_idx} - {scene.heading}
          </p>
        </div>
        <span className="score" title="Relevance score (not comparable between BM25 and TF-IDF)">
          {Number(scene.score).toFixed(3)}
        </span>
      </header>

      <GenreChips genres={scene.genres} />

      <p className="meta">
        IMDb {formatRating(scene.imdb_rating)} - Metascore {formatScore(scene.metascore)} -{' '}
        {scene.n_words} words - {scene.matched_chunk_count} matching passage
        {scene.matched_chunk_count === 1 ? '' : 's'}
      </p>

      <pre className={`scene-text ${expanded ? 'expanded' : ''}`}>{scene.text}</pre>

      <button type="button" className="link-button" onClick={() => setExpanded((value) => !value)}>
        {expanded ? 'Collapse scene' : 'Show full scene'}
      </button>
    </GlassPanel>
  );
}
