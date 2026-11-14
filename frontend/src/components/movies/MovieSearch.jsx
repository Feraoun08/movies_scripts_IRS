import { searchMovies } from '../../api/client.js';
import { useSearch } from '../../hooks/useSearch.js';
import StatusMessage from '../common/StatusMessage.jsx';
import MovieList from './MovieList.jsx';
import MovieSearchForm from './MovieSearchForm.jsx';

export default function MovieSearch() {
  const { status, data, error, run } = useSearch(searchMovies);

  return (
    <>
      <MovieSearchForm loading={status === 'loading'} onSubmit={run} />

      <StatusMessage
        status={status}
        error={error}
        isEmpty={status === 'success' && data.length === 0}
        emptyText="No movies match these filters."
        loadingText="Searching movies..."
      />

      {status === 'success' && data.length > 0 && <MovieList movies={data} />}
    </>
  );
}
