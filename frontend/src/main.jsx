import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles/index.css';

// StrictMode is off on purpose: liquid-glass has no destroy(),
// so double-mounting in dev would leak WebGL contexts.
createRoot(document.getElementById('root')).render(<App />);
