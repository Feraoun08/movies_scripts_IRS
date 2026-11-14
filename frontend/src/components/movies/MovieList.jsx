import MovieCard from './MovieCard.jsx';

export default function MovieList({ movies }) {
  return (
    <>
      <p className="result-summary">{movies.length} movies</p>
      <div className="grid">
        {movies.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} />
        ))}
      </div>
    </>
  );
}
