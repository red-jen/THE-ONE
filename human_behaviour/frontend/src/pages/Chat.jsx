import { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { askQuestion, getRun } from '../lib/api';
import { Send, Loader2, ArrowLeft, Bot, User } from 'lucide-react';

export default function Chat() {
  const { runId } = useParams();
  const [run, setRun] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    getRun(runId).then(r => {
      setRun(r.data);
      if (r.data.queries?.length) {
        const history = r.data.queries.reverse().flatMap(q => [
          { role: 'user', text: q.question },
          { role: 'assistant', text: q.answer },
        ]);
        setMessages(history);
      }
    }).catch(() => {});
  }, [runId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: question }]);
    setLoading(true);

    try {
      const res = await askQuestion(runId, {
        question,
        generator_backend: 'ollama',
        top_leaders: 5,
        top_evidence: 8,
      });
      setMessages(prev => [...prev, { role: 'assistant', text: res.data.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `Error: ${err.response?.data?.detail || err.message}`,
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center gap-4 pb-4 border-b border-gray-800">
        <Link to={`/runs/${runId}`}
          className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400 transition">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-lg font-bold text-white">Analysis Chat</h1>
          <p className="text-xs text-gray-500">{run?.run_name || `Run #${runId}`}</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <Bot className="w-12 h-12 text-gray-700 mx-auto mb-4" />
            <p className="text-gray-500 text-sm">Ask questions about the analysis results.</p>
            <p className="text-gray-600 text-xs mt-1">e.g. "Who is the most likely leader and why?"</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-indigo-600/20 flex items-center justify-center shrink-0 mt-1">
                <Bot className="w-4 h-4 text-indigo-400" />
              </div>
            )}
            <div className={`max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
              ${msg.role === 'user'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-800 text-gray-300 border border-gray-700'}`}>
              {msg.text}
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1">
                <User className="w-4 h-4 text-gray-300" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-indigo-600/20 flex items-center justify-center shrink-0">
              <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-500">
              Analyzing evidence...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 pt-4">
        <form onSubmit={(e) => { e.preventDefault(); handleSend(); }}
          className="flex gap-3">
          <input type="text" value={input} onChange={e => setInput(e.target.value)}
            placeholder="Ask about the protest analysis..."
            disabled={loading}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white
              placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500
              disabled:opacity-50" />
          <button type="submit" disabled={loading || !input.trim()}
            className="px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
              disabled:cursor-not-allowed rounded-xl transition">
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
