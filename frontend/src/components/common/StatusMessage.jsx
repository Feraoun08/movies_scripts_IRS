import GlassPanel from './GlassPanel.jsx';

export default function StatusMessage({ status, error, isEmpty, emptyText, loadingText }) {
  if (status === 'loading') {
    return (
      <GlassPanel className="status" role="status">
        <span className="spinner" aria-hidden="true" />
        {loadingText}
      </GlassPanel>
    );
  }

  if (status === 'error') {
    return (
      <GlassPanel className="status status-error" role="alert">
        {error}
      </GlassPanel>
    );
  }

  if (status === 'success' && isEmpty) {
    return <GlassPanel className="status">{emptyText}</GlassPanel>;
  }

  return null;
}
