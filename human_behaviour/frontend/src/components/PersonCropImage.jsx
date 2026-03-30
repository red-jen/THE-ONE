import { useEffect, useState, useRef } from 'react';
import { getAbsoluteApiUrl } from '../lib/api';
import { ImageOff } from 'lucide-react';

/**
 * Person crop requires Authorization — plain <img src> never sends the Bearer token.
 */
export default function PersonCropImage({ cropUrl, personId }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(!!cropUrl);
  const objectUrlRef = useRef(null);

  useEffect(() => {
    if (!cropUrl) {
      setLoading(false);
      setFailed(true);
      return;
    }

    const url = getAbsoluteApiUrl(cropUrl);
    const token = localStorage.getItem('token');
    let cancelled = false;

    setLoading(true);
    setFailed(false);
    setBlobUrl(null);

    fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        if (objectUrlRef.current) {
          URL.revokeObjectURL(objectUrlRef.current);
          objectUrlRef.current = null;
        }
        const u = URL.createObjectURL(blob);
        objectUrlRef.current = u;
        setBlobUrl(u);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [cropUrl]);

  if (!cropUrl || failed) {
    return (
      <div className="flex flex-col items-center justify-center h-40 rounded-lg border border-dashed border-gray-700 bg-gray-800/50 text-gray-500 text-sm">
        <ImageOff className="w-8 h-8 mb-2 opacity-50" />
        <span>{cropUrl ? 'No crop image available' : 'No crop URL'}</span>
        <span className="text-xs mt-1 text-gray-600">Person #{personId}</span>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-40 rounded-lg border border-gray-800 bg-gray-800/30 animate-pulse flex items-center justify-center text-xs text-gray-500">
        Loading image…
      </div>
    );
  }

  return (
    <img
      src={blobUrl || undefined}
      alt={`Person ${personId} crop`}
      className="w-full h-40 object-cover rounded-lg border border-gray-800"
    />
  );
}
