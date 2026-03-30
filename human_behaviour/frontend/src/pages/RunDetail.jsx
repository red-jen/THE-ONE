import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getRun, analyzeRun } from '../lib/api';
import PersonCard from '../components/PersonCard';
import StatusBadge from '../components/StatusBadge';
import { MessageCircle, RefreshCw, Loader2, Brain } from 'lucide-react';

export default function RunDetail() {
  const { runId } = useParams();
  const role = localStorage.getItem('role') || 'viewer';
  const canAnalyze = role === 'admin' || role === 'analyst';
  const [run, setRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState('');

  const fetchRun = () => {
    setLoading(true);
    getRun(runId)
      .then(r => setRun(r.data))
      .catch(err => setError(err.response?.data?.detail || 'Failed to load run'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchRun(); }, [runId]);

  const handleReScore = async () => {
    if (!canAnalyze) {
      setError('Your role cannot re-score runs');
      return;
    }
    setScoring(true);
    try {
      await analyzeRun(runId);
      fetchRun();
    } catch (err) {
      setError(err.response?.data?.detail || 'Scoring failed');
    } finally {
      setScoring(false);
    }
  };

  if (loading) return <div className="text-center py-16 text-gray-500">Loading...</div>;
  if (error) return <div className="text-center py-16 text-red-400">{error}</div>;
  if (!run) return null;

  const rawPersons = run.persons || [];
  const leaderScores = run.leader_scores || [];
  const queries = run.queries || [];

  const scoreByPid = Object.fromEntries(
    leaderScores.map((s) => [s.person_id, s]),
  );
  const persons = rawPersons.map((p) => {
    const s = scoreByPid[p.person_id];
    return {
      ...p,
      leader_score: p.leader_score ?? s?.leader_score ?? null,
      reasoning: p.reasoning || s?.reasoning || null,
    };
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">{run.run_name}</h1>
          <div className="flex items-center gap-3 mt-2">
            <StatusBadge status={run.status} />
            <span className="text-sm text-gray-500">
              {run.videos_count} video(s) &middot; {persons.length} person(s) detected
            </span>
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={handleReScore} disabled={scoring || !canAnalyze}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm
              font-medium text-gray-300 transition disabled:opacity-50">
            {scoring ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
            Re-score with LLM
          </button>
          <Link to={`/runs/${runId}/chat`}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm
              font-medium transition">
            <MessageCircle className="w-4 h-4" /> Ask Questions
          </Link>
          <button onClick={fetchRun}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 transition">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error message */}
      {run.error_message && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-400">
          {run.error_message}
        </div>
      )}

      {/* Leader scores from RAG */}
      {leaderScores.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">
            Leadership Ranking
            <span className="ml-2 text-sm font-normal text-gray-500">
              (scored by {leaderScores[0]?.backend_used || 'LLM'})
            </span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {persons
              .filter(p => p.leader_score != null)
              .sort((a, b) => (b.leader_score ?? 0) - (a.leader_score ?? 0))
              .map((person, idx) => (
                <PersonCard key={person.person_id} person={person} rank={idx + 1} />
              ))}
          </div>
        </div>
      )}

      {/* All detected persons (if no scores yet) */}
      {leaderScores.length === 0 && persons.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Detected Persons</h2>
          <p className="text-sm text-gray-400 mb-4">
            No leadership scoring yet.
            {canAnalyze && (
              <button onClick={handleReScore} className="text-indigo-400 hover:underline ml-1">
                Run RAG scoring
              </button>
            )}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {persons.map((person, idx) => (
              <PersonCard key={person.person_id} person={person} rank={idx + 1} />
            ))}
          </div>
        </div>
      )}

      {/* Recent Q&A */}
      {queries.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Recent Questions</h2>
          <div className="space-y-3">
            {queries.slice(0, 5).map(q => (
              <div key={q.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <p className="text-sm font-medium text-indigo-300 mb-2">{q.question}</p>
                <p className="text-sm text-gray-400 whitespace-pre-wrap">{q.answer}</p>
                <p className="text-xs text-gray-600 mt-2">
                  {q.asked_at ? new Date(q.asked_at).toLocaleString() : ''}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
