import { useState } from 'react';
import { validateFilters } from '../../utils/validation.js';
import Field from '../common/Field.jsx';
import GlassPanel from '../common/GlassPanel.jsx';

const EMPTY = { title: '', year: '', minRating: '', maxRating: '' };

export default function MovieSearchForm({ loading, onSubmit }) {
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
      title: values.title.trim(),
      year: values.year,
      min_rating: values.minRating,
      max_rating: values.maxRating,
      limit: 20,
    });
  };

  return (
    <GlassPanel as="form" className="form" onSubmit={handleSubmit}>
      <Field label="Title" className="grow">
        <input value={values.title} onChange={update('title')} placeholder="e.g. Inside Out" />
      </Field>
      <Field label="Year">
        <input type="number" value={values.year} onChange={update('year')} placeholder="2015" />
      </Field>
      <Field label="Min rating">
        <input type="number" step="0.1" value={values.minRating} onChange={update('minRating')} />
      </Field>
      <Field label="Max rating">
        <input type="number" step="0.1" value={values.maxRating} onChange={update('maxRating')} />
      </Field>

      <button type="submit" className="primary" disabled={loading}>
        {loading ? 'Searching...' : 'Search movies'}
      </button>

      {error && <p className="form-error">{error}</p>}
    </GlassPanel>
  );
}
