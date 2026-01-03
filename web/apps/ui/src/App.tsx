import React, { useMemo, useState } from 'react';
import { MemoryCache, openPack, openPackSet, Pack, PackSet } from '@annpack/client';

const DEFAULT_MANIFEST = '/pack/pack.manifest.json';

const parseVector = (raw: string): number[] => {
  const parts = raw.split(/[\s,]+/).filter(Boolean);
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => Number.isNaN(n))) {
    throw new Error('Vector contains non-numeric values.');
  }
  return nums;
};

export default function App() {
  const [manifestUrl, setManifestUrl] = useState(DEFAULT_MANIFEST);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState<string | null>(null);
  const [pack, setPack] = useState<Pack | PackSet | null>(null);
  const [header, setHeader] = useState<Record<string, number> | null>(null);
  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [vectorInput, setVectorInput] = useState('');
  const [results, setResults] = useState<Array<Record<string, unknown>>>([]);
  const [isPackSet, setIsPackSet] = useState(false);
  const [verifyIntegrity, setVerifyIntegrity] = useState(false);

  const cache = useMemo(() => new MemoryCache(), []);

  const handleLoad = async () => {
    setError(null);
    setStatus('Loading...');
    try {
      const loaded = isPackSet
        ? await openPackSet(manifestUrl, [], {
            cache,
            telemetry: (event) => console.log('[annpack]', event.name, event.detail ?? {}),
            verify: verifyIntegrity,
          })
        : await openPack(manifestUrl, {
            cache,
            telemetry: (event) => console.log('[annpack]', event.name, event.detail ?? {}),
            verify: verifyIntegrity,
          });
      const hdr = await loaded.readHeader();
      setPack(loaded);
      setHeader(hdr);
      setManifest(loaded.manifest as Record<string, unknown>);
      setStatus('Ready');
    } catch (err) {
      setStatus('Error');
      setError(String(err));
    }
  };

  const handleSearch = async () => {
    setError(null);
    if (!pack) {
      setError('Load a pack first.');
      return;
    }
    try {
      const vec = parseVector(vectorInput);
      const hits = await pack.search({ queryVector: vec, topK: 5 });
      setResults(hits);
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <div className="app">
      <h1>ANNPack UI</h1>
      <p className="badge">Demo mode expects a pack mounted at /pack/</p>

      <div className="card">
        <h2>Load Pack</h2>
        <label>Manifest URL</label>
        <input value={manifestUrl} onChange={(e) => setManifestUrl(e.target.value)} />
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            type="checkbox"
            checked={isPackSet}
            onChange={(e) => setIsPackSet(e.target.checked)}
            id="packsetToggle"
          />
          <label htmlFor="packsetToggle">PackSet manifest</label>
        </div>
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            type="checkbox"
            checked={verifyIntegrity}
            onChange={(e) => setVerifyIntegrity(e.target.checked)}
            id="verifyToggle"
          />
          <label htmlFor="verifyToggle">Verify checksums</label>
        </div>
        <div style={{ marginTop: 12, display: 'flex', gap: 12 }}>
          <button onClick={handleLoad}>Load</button>
          <span className="badge">Status: {status}</span>
        </div>
        {error && (
          <div className="error" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Search</h2>
        <div className="grid">
          <div>
            <label>Vector query</label>
            <textarea
              rows={3}
              placeholder="0.1, 0.2, 0.3..."
              value={vectorInput}
              onChange={(e) => setVectorInput(e.target.value)}
            />
          </div>
          <div>
            <label>Text query (embedding plugin required)</label>
            <input disabled placeholder="Configure embedding plugin" />
            <p className="badge" style={{ marginTop: 8 }}>
              Embeddings disabled by default.
            </p>
          </div>
        </div>
        <button style={{ marginTop: 12 }} onClick={handleSearch}>
          Search
        </button>
      </div>

      <div className="card">
        <h2>Results</h2>
        <ul className="results">
          {results.length === 0 && <li>No results yet.</li>}
          {results.map((row, idx) => (
            <li key={idx}>
              <strong>ID:</strong> {String(row.id)} &nbsp; <strong>Score:</strong>{' '}
              {String(row.score)}
              <div>{row.meta ? JSON.stringify(row.meta) : 'No metadata'}</div>
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <h2>Inspector</h2>
        <div className="grid">
          <div>
            <label>Header</label>
            <pre>{header ? JSON.stringify(header, null, 2) : 'Header not loaded'}</pre>
          </div>
          <div>
            <label>Cache</label>
            <pre>{JSON.stringify(cache.stats(), null, 2)}</pre>
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <label>Manifest</label>
          <pre>{manifest ? JSON.stringify(manifest, null, 2) : 'Manifest not loaded'}</pre>
        </div>
      </div>
    </div>
  );
}
