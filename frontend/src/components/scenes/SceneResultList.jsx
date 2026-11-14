import SceneCard from './SceneCard.jsx';

export default function SceneResultList({ data }) {
  return (
    <>
      <p className="result-summary">
        {data.count} scenes - {data.method.toUpperCase()}
      </p>

      <div className="list">
        {data.results.map((scene) => (
          <SceneCard key={`${scene.movie_id}:${scene.scene_idx}`} scene={scene} />
        ))}
      </div>
    </>
  );
}
