import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadPipeline, runPipeline } from '../lib/api';
import { Upload as UploadIcon, Video, Loader2, FolderOpen } from 'lucide-react';

export default function Upload() {
  const navigate = useNavigate();
  const role = localStorage.getItem('role') || 'viewer';
  const canRunPipeline = role === 'admin' || role === 'analyst';
  const [mode, setMode] = useState('upload');
  const [runName, setRunName] = useState('');
  const [question, setQuestion] = useState('Who is the most likely protest leader and why?');
  const [files, setFiles] = useState([]);
  const [videosDir, setVideosDir] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canRunPipeline) {
      setError('Your role cannot start new pipeline runs');
      return;
    }
    if (!runName.trim()) { setError('Run name is required'); return; }
    setLoading(true);
    setError('');

    try {
      let res;
      if (mode === 'upload') {
        if (files.length === 0) { setError('Select at least one video'); setLoading(false); return; }
        const formData = new FormData();
        formData.append('run_name', runName);
        formData.append('question', question);
        files.forEach(f => formData.append('files', f));
        res = await uploadPipeline(formData);
      } else {
        if (!videosDir.trim()) { setError('Videos directory is required'); setLoading(false); return; }
        res = await runPipeline({ run_name: runName, videos_dir: videosDir, question });
      }
      navigate(`/runs/${res.data.run_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Pipeline failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">New Analysis</h1>
      {!canRunPipeline && (
        <div className="mb-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3 text-sm text-yellow-300">
          Viewer role is read-only. Ask an analyst/admin account to run new analyses.
        </div>
      )}

      {/* Mode toggle */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'upload', label: 'Upload Videos', icon: UploadIcon },
          { id: 'path', label: 'Local Path', icon: FolderOpen },
        ].map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setMode(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition
              ${mode === id ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'}`}>
            <Icon className="w-4 h-4" /> {label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Run name */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1.5">Run Name</label>
          <input type="text" value={runName} onChange={e => setRunName(e.target.value)}
            placeholder="e.g. protest_day1_analysis"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white
              placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent" />
        </div>

        {/* Upload mode */}
        {mode === 'upload' ? (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">Video Files</label>
            <div className="border-2 border-dashed border-gray-700 rounded-xl p-8 text-center
              hover:border-indigo-500/50 transition cursor-pointer"
              onClick={() => document.getElementById('file-input').click()}>
              <Video className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-sm text-gray-400">
                {files.length > 0
                  ? `${files.length} file(s) selected`
                  : 'Click to select video files (.mp4, .avi, .mov)'}
              </p>
              <input id="file-input" type="file" multiple accept="video/*" className="hidden"
                onChange={e => setFiles(Array.from(e.target.files))} />
            </div>
            {files.length > 0 && (
              <ul className="mt-2 space-y-1">
                {files.map((f, i) => (
                  <li key={i} className="text-xs text-gray-500 flex items-center gap-1">
                    <Video className="w-3 h-3" /> {f.name} ({(f.size / 1e6).toFixed(1)} MB)
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">Videos Directory</label>
            <input type="text" value={videosDir} onChange={e => setVideosDir(e.target.value)}
              placeholder="e.g. /path/to/camera/videos"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white
                placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500" />
          </div>
        )}

        {/* Question */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1.5">Analysis Question</label>
          <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={2}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white
              placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none" />
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        <button type="submit" disabled={loading || !canRunPipeline}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500
            disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UploadIcon className="w-4 h-4" />}
          {loading ? 'Running Pipeline...' : 'Start Analysis'}
        </button>
      </form>
    </div>
  );
}
