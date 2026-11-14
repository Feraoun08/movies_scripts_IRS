const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function toQuery(params) {
  const query = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== '' && value !== null && value !== undefined) {
      query.append(key, value);
    }
  });

  return query.toString();
}

function errorMessage(body, status) {
  if (typeof body?.detail === 'string') return body.detail;
  if (Array.isArray(body?.detail)) return body.detail.map((d) => d.msg).join(', ');
  return `Request failed (${status}).`;
}

async function get(path, params = {}) {
  let response;

  try {
    response = await fetch(`${BASE_URL}${path}?${toQuery(params)}`);
  } catch {
    throw new Error('Cannot reach the API. Is the backend running on ' + BASE_URL + '?');
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(errorMessage(body, response.status));
  }

  return response.json();
}

export const searchMovies = (params) => get('/movies', params);
export const searchScenes = (params) => get('/scenes/search', params);
