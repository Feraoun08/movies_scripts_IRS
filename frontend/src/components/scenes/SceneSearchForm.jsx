import { useState } from 'react';
import { validateFilters } from '../../utils/validation.js';
import Field from '../common/Field.jsx';
import GlassPanel from '../common/GlassPanel.jsx';

const EMPTY = {
  query: '',
  method: 'bm25',
  topK: '10',
  title: '',
  year: '',
  minRating: '',
  maxRating: '',
};

export default function SceneSearchForm({ loading, onSubmit }) {
  const [values, setValues] = useState(EMPTY);
  const [error, setError] = useState('');

  const update = (name) => (event) =>
    setValues((current) => ({ ...current, [name]: event.target.value }));

  const handleSubmit = (event) => {
    event.preventDefault();

    const problem = validateFilters(values);
    setError(problem);
    if (problem) return;

    onSubmit({
      q: values.query.trim(),
      method: values.method,
      top_k: values.topK,
      title: values.title.trim(),
      year: values.year,
      min_rating: values.minRating,
      max_rating: values.maxRating,
    });
  };

  return (
    <GlassPanel as="form" className="form form-column" onSubmit={handleSubmit}>
      <div className="form-row">
        <Field label="Describe the scene" className="grow">
          <input
            value={values.query}
            onChange={update('query')}
            placeholder="e.g. love friendship and relationships"
            autoFocus
          />
        </Field>

        <button type="submit" className="primary" disabled={loading || !values.query.trim()}>
          {loading ? 'Searching...' : 'Search scenes'}
        </button>
      </div>

      <details className="filters">
        <summary>Options and filters</summary>

        <div className="form-row">
          <Field label="Ranking">
            <select value={values.method} onChange={update('method')}>
              <option value="bm25">BM25</option>
              <option value="tfidf">TF-IDF</option>
            </select>
          </Field>
          <Field label="Results">
            <select value={values.topK} onChange={update('topK')}>
              <option value="5">5</option>
              <option value="10">10</option>
              <option value="20">20</option>
            </select>
          </Field>
          <Field label="Movie title" className="grow">
            <input value={values.title} onChange={update('title')} />
          </Field>
          <Field label="Year">
            <input type="number" value={values.year} onChange={update('year')} />
          </Field>
          <Field label="Min rating">
            <input type="number" step="0.1" value={values.minRating} onChange={update('minRating')} />
          </Field>
          <Field label="Max rating">
            <input type="number" step="0.1" value={values.maxRating} onChange={update('maxRating')} />
          </Field>
        </div>
      </details>

      {error && <p className="form-error">{error}</p>}
    </GlassPanel>
  );
}
