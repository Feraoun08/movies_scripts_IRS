/* global Container, Button */
import { useEffect, useRef } from 'react';

const libraryReady = () =>
  typeof Container !== 'undefined' && typeof Button !== 'undefined';

export default function GlassTabs({ tabs, active, onChange }) {
  const hostRef = useRef(null);
  const buttonsRef = useRef({});
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const useLibrary = libraryReady();

  // Phase 1: create the liquid-glass tabs once.
  useEffect(() => {
    if (!useLibrary) return undefined;

    const container = new Container({
      type: 'rounded',
      borderRadius: 28,
      tintOpacity: 0.15,
    });

    tabs.forEach((tab) => {
      const button = new Button({
        text: tab.label,
        size: 18,
        type: 'pill',
        tintOpacity: 0.25,
        onClick: () => onChangeRef.current(tab.id),
      });

      container.addChild(button);
      buttonsRef.current[tab.id] = button;
    });

    hostRef.current.appendChild(container.element);

    return () => container.element.remove();
  }, []);

  // Phase 2: reflect the active tab.
  useEffect(() => {
    Object.entries(buttonsRef.current).forEach(([id, button]) => {
      button.element.classList.toggle('is-active', id === active);
    });
  }, [active]);

  if (!useLibrary) {
    return (
      <div className="tabs-fallback">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={tab.id === active ? 'is-active' : ''}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
    );
  }

  return <div className="glass-tabs" ref={hostRef} />;
}
