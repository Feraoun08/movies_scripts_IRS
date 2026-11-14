import { useState } from 'react';
import GlassTabs from './components/common/GlassTabs.jsx';
import MovieSearch from './components/movies/MovieSearch.jsx';
import SceneSearch from './components/scenes/SceneSearch.jsx';

const TABS = [
  { id: 'scenes', label: 'Scene Search' },
  { id: 'movies', label: 'Movie Search' },
];

export default function App() {
  const [tab, setTab] = useState('scenes');

  return (
    <div className="app">
      <header className="app-header">
        <h1>CinemaSearch</h1>
        <p>Find the exact movie scene, or browse movies by metadata.</p>
        <GlassTabs tabs={TABS} active={tab} onChange={setTab} />
      </header>

      <main>
        <section hidden={tab !== 'scenes'}>
          <SceneSearch />
        </section>
        <section hidden={tab !== 'movies'}>
          <MovieSearch />
        </section>
      </main>
    </div>
  );
}
