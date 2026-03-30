import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listRuns, healthCheck } from '../lib/api';
import StatusBadge from '../components/StatusBadge';
import { ArrowRight, Server, Database, Shield } from 'lucide-react';

export default function Dashboard() {
  const [runs, setRuns] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const role = localStorage.getItem('role') || 'viewer';
  const canRunPipeline = role === 'admin' || role === 'analyst';

  useEffect(() => {
    Promise.all([
      listRuns().then(r => setRuns(r.data)).catch(() => {}),
      healthCheck().then(r => setHealth(r.data)).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-8">
      {/* Health cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <Server className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-medium text-gray-300">API Status</h3>
          </div>
          <p className={`text-lg font-bold ${health?.status === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
            {health ? health.status.toUpperCase() : 'Checking...'}
          </p>
          <p className="text-xs text-gray-500 mt-1">v{health?.version || '—'}</p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <Database className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-medium text-gray-300">Database</h3>
          </div>
          <p className={`text-lg font-bold ${health?.database === 'connected' ? 'text-emerald-400' : 'text-red-400'}`}>
            {health?.database?.toUpperCase() || 'UNKNOWN'}
          </p>
          <p className="text-xs text-gray-500 mt-1">PostgreSQL</p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-2">
            <Shield className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-medium text-gray-300">Scoring</h3>
          </div>
          <p className="text-lg font-bold text-indigo-400">
            {health?.scoring_backend?.toUpperCase() || 'RAG_LLM'}
          </p>
          <p className="text-xs text-gray-500 mt-1">Ollama: {health?.ollama_url || '—'}</p>
        </div>
      </div>

      {/* Runs table */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Pipeline Runs</h2>
          {canRunPipeline && (
            <Link to="/upload"
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition">
              + New Analysis
            </Link>
          )}
        </div>

        {loading ? (
          <div className="text-center py-12 text-gray-500">Loading...</div>
        ) : runs.length === 0 ? (
          <div className="text-center py-16 bg-gray-900 border border-gray-800 rounded-xl">
            <p className="text-gray-400 mb-4">No pipeline runs yet.</p>
            {canRunPipeline ? (
              <Link to="/upload"
                className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition">
                Start your first analysis
                <ArrowRight className="w-4 h-4" />
              </Link>
            ) : (
              <p className="text-sm text-gray-500">Your role can view runs and ask questions.</p>
            )}
          </div>
        ) : (
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 text-left">
                  <th className="px-5 py-3 font-medium">ID</th>
                  <th className="px-5 py-3 font-medium">Run Name</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                  <th className="px-5 py-3 font-medium">Videos</th>
                  <th className="px-5 py-3 font-medium">Scoring</th>
                  <th className="px-5 py-3 font-medium">Created</th>
                  <th className="px-5 py-3"></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition">
                    <td className="px-5 py-3 text-gray-400 font-mono text-xs">#{run.id}</td>
                    <td className="px-5 py-3 text-white font-medium">{run.run_name}</td>
                    <td className="px-5 py-3"><StatusBadge status={run.status} /></td>
                    <td className="px-5 py-3 text-gray-400">{run.videos_count}</td>
                    <td className="px-5 py-3 text-gray-400">{run.scoring_backend || '—'}</td>
                    <td className="px-5 py-3 text-gray-500 text-xs">
                      {run.created_at ? new Date(run.created_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3">
                      <Link to={`/runs/${run.id}`}
                        className="text-indigo-400 hover:text-indigo-300 text-xs font-medium transition">
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
