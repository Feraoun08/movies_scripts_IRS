import { searchScenes } from '../../api/client.js';
import { useSearch } from '../../hooks/useSearch.js';
import StatusMessage from '../common/StatusMessage.jsx';
import SceneResultList from './SceneResultList.jsx';
import SceneSearchForm from './SceneSearchForm.jsx';

export default function SceneSearch() {
  const { status, data, error, run } = useSearch(searchScenes);

  return (
    <>
      <SceneSearchForm loading={status === 'loading'} onSubmit={run} />

      <StatusMessage
        status={status}
        error={error}
        isEmpty={status === 'success' && data.count === 0}
        emptyText="No scenes found. Try different words or fewer filters."
        loadingText="Searching scenes..."
      />

      {status === 'success' && data.count > 0 && <SceneResultList data={data} />}
    </>
  );
}
