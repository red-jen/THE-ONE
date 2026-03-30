import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { Activity, Upload, LayoutDashboard, LogOut } from 'lucide-react';

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/upload', label: 'New Analysis', icon: Upload },
];

export default function Layout() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const role = localStorage.getItem('role') || 'viewer';
  const canRunPipeline = role === 'admin' || role === 'analyst';

  const onLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    navigate('/login', { replace: true });
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Top bar */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <Activity className="w-6 h-6 text-indigo-400 group-hover:text-indigo-300 transition" />
            <span className="text-lg font-bold tracking-tight">
              Protest Leader <span className="text-indigo-400">Detection</span>
            </span>
          </Link>

          <nav className="flex items-center gap-1">
            {NAV.filter((item) => canRunPipeline || item.to !== '/upload').map(({ to, label, icon: Icon }) => {
              const active = pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition
                    ${active
                      ? 'bg-indigo-600/20 text-indigo-300'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                    }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </Link>
              );
            })}
            <span className="px-2 py-1 rounded bg-gray-800 text-xs text-gray-400 uppercase tracking-wide">{role}</span>
            <button
              onClick={onLogout}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}
