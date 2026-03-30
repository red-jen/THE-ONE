import { User, Megaphone, Flag, MessageSquare } from 'lucide-react';
import ScoreBar from './ScoreBar';
import PersonCropImage from './PersonCropImage';

export default function PersonCard({ person, rank }) {
  const {
    person_id, crop_url, leader_score, reasoning,
    observations, cameras_seen, duration_sec,
    front_count, megaphone_count, banner_count, gesture_total,
  } = person;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-gray-700 transition">
      <div className="mb-4">
        <p className="text-xs text-gray-500 mb-2">Detected appearance (saved crop)</p>
        <PersonCropImage cropUrl={crop_url} personId={person_id} />
      </div>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-indigo-600/20 flex items-center justify-center">
            <User className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Person #{person_id}</h3>
            <p className="text-xs text-gray-500">Rank #{rank}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">{(leader_score ?? 0).toFixed(1)}</div>
          <div className="text-xs text-gray-500">/ 100</div>
        </div>
      </div>

      <ScoreBar score={leader_score ?? 0} label="Leadership Score" />

      <div className="mt-3">
        <p className="text-xs font-medium text-gray-500 mb-1">Why this leadership score?</p>
        {reasoning ? (
          <p className="text-sm text-gray-300 leading-relaxed border-l-2 border-indigo-500/50 pl-3">
            {reasoning}
          </p>
        ) : (
          <p className="text-sm text-gray-500 italic">
            No explanation stored yet. Re-run &quot;Re-score with LLM&quot; after Ollama is running, or check that the scorer returned a &quot;reasoning&quot; field.
          </p>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-gray-400">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Seen:</span>
          <span className="text-gray-300">{observations ?? '—'} times</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Duration:</span>
          <span className="text-gray-300">{(duration_sec ?? 0).toFixed(1)}s</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Front:</span>
          <span className="text-gray-300">{front_count ?? 0}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Cameras:</span>
          <span className="text-gray-300">{cameras_seen ?? 0}</span>
        </div>
      </div>

      <div className="mt-3 flex gap-2 flex-wrap">
        {megaphone_count > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 text-xs">
            <Megaphone className="w-3 h-3" /> Megaphone ({megaphone_count})
          </span>
        )}
        {banner_count > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-xs">
            <Flag className="w-3 h-3" /> Banner ({banner_count})
          </span>
        )}
        {gesture_total > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 text-xs">
            <MessageSquare className="w-3 h-3" /> Gestures ({gesture_total})
          </span>
        )}
      </div>
    </div>
  );
}
