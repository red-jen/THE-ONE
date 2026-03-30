import axios from 'axios';

/**
 * API base URL (no trailing slash, no /api suffix — paths are like /pipeline/upload).
 *
 * - Set VITE_API_ORIGIN in any mode (dev, preview, production build) to skip /api and avoid 413
 *   from proxies (e.g. http://127.0.0.1:8000 or http://127.0.0.1:8010).
 * - VITE_API_DIRECT_URL still works if set (alias).
 * - Dev default: direct to :8000. Built app default: same-origin /api (needs nginx 2g or rebuild).
 */
export function resolveApiBaseURL() {
  const explicit =
    (import.meta.env.VITE_API_ORIGIN && String(import.meta.env.VITE_API_ORIGIN).trim()) ||
    (import.meta.env.VITE_API_DIRECT_URL && String(import.meta.env.VITE_API_DIRECT_URL).trim());
  if (explicit) {
    return explicit.replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return 'http://127.0.0.1:8000';
  }
  return '/api';
}

/** Full URL for an API path like `/runs/1/persons/2/crop` (for authenticated `fetch` / images). */
export function getAbsoluteApiUrl(apiPath) {
  const base = resolveApiBaseURL();
  const path = apiPath.startsWith('/') ? apiPath : `/${apiPath}`;
  if (base.startsWith('http')) {
    return `${base}${path}`;
  }
  return `${base}${path}`;
}

const api = axios.create({
  baseURL: resolveApiBaseURL(),
  timeout: 120000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('role');
      // Redirect to login for expired/invalid tokens.
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  },
);

export const healthCheck = () => api.get('/health');
export const login = (username, password) =>
  api.post('/auth/token', { username, password });
export const me = () => api.get('/auth/me');

export const listRuns = (limit = 50, offset = 0) =>
  api.get('/runs', { params: { limit, offset } });
export const getRun = (runId) => api.get(`/runs/${runId}`);
export const getPersons = (runId) => api.get(`/runs/${runId}/persons`);
export const getQueries = (runId) => api.get(`/runs/${runId}/queries`);

export const runPipeline = (data) => api.post('/pipeline/run', data);
export const uploadPipeline = (formData) =>
  api.post('/pipeline/upload', formData, {
    // Let the browser set multipart boundary (manual Content-Type breaks uploads).
    timeout: 600000,
  });

export const analyzeRun = (runId, data = {}) =>
  api.post(`/analyze/${runId}`, data);
export const askQuestion = (runId, data) =>
  api.post(`/ask/${runId}`, data);

export default api;
