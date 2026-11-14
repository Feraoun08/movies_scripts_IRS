export default function GenreChips({ genres }) {
  if (!genres || genres.length === 0) return null;

  return (
    <ul className="chips">
      {genres.map((genre) => (
        <li key={genre}>{genre}</li>
      ))}
    </ul>
  );
}
