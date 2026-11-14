import { useCallback, useRef, useState } from 'react';

export function useSearch(fetcher) {
  const [state, setState] = useState({ status: 'idle', data: null, error: '' });
  const latest = useRef(0);

  const run = useCallback(
    async (params) => {
      const id = ++latest.current;
      setState((s) => ({ ...s, status: 'loading', error: '' }));

      try {
        const data = await fetcher(params);
        if (id === latest.current) setState({ status: 'success', data, error: '' });
      } catch (error) {
        if (id === latest.current) setState({ status: 'error', data: null, error: error.message });
      }
    },
    [fetcher]
  );

  return { ...state, run };
}
