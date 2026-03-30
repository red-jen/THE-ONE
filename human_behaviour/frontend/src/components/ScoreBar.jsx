export default function ScoreBar({ score, label }) {
  const pct = Math.min(100, Math.max(0, score));
  const color =
    pct >= 70 ? 'bg-red-500' :
    pct >= 40 ? 'bg-amber-500' :
    'bg-emerald-500';

  return (
    <div className="space-y-1">
      {label && (
        <div className="flex justify-between text-xs text-gray-400">
          <span>{label}</span>
          <span className="font-mono">{pct.toFixed(1)}</span>
        </div>
      )}
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
