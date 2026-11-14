import { formatRating, formatScore, formatVotes, formatYear } from '../../utils/format.js';
import GenreChips from '../common/GenreChips.jsx';
import GlassPanel from '../common/GlassPanel.jsx';

export default function MovieCard({ movie }) {
  return (
    <GlassPanel as="article" className="card">
      <header className="card-head">
        <h3>{movie.title}</h3>
        <span className="muted">{formatYear(movie.year)}</span>
      </header>

      <GenreChips genres={movie.genres} />

      <dl className="stats">
        <div>
          <dt>IMDb</dt>
          <dd>{formatRating(movie.imdb_rating)}</dd>
        </div>
        <div>
          <dt>Metascore</dt>
          <dd>{formatScore(movie.metascore)}</dd>
        </div>
        <div>
          <dt>Votes</dt>
          <dd>{formatVotes(movie.imdb_votes)}</dd>
        </div>
      </dl>
    </GlassPanel>
  );
}
