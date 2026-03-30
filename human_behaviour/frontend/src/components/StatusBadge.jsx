const STYLES = {
  completed: 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/30',
  running:   'bg-blue-500/15 text-blue-400 ring-blue-500/30',
  pending:   'bg-gray-500/15 text-gray-400 ring-gray-500/30',
  failed:    'bg-red-500/15 text-red-400 ring-red-500/30',
};

export default function StatusBadge({ status }) {
  const style = STYLES[status] || STYLES.pending;
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ring-1 ring-inset ${style}`}>
      {status}
    </span>
  );
}
