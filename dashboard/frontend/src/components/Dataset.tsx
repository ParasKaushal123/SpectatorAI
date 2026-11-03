import { useEffect, useState } from "react";
import { request } from "../workspaceApi";

type Entry = { id: string; videos: { key: string; folder: string; name: string; url: string }[] };

export default function Dataset({ id }: { id: string }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [variant, setVariant] = useState(0);
  const [query, setQuery] = useState("");
  useEffect(() => {
    let active = true;
    void request<Entry[]>("/api/dataset").then(rows => { if (active) { setEntries(rows); setLoaded(true); } })
      .catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, []);
  useEffect(() => { setVariant(0); }, [id]);
  const selected = entries.find(row => row.id === id);
  const video = selected?.videos[variant] ?? selected?.videos[0];
  const visibleEntries = entries.filter(row => row.id.toLowerCase().includes(query.trim().toLowerCase()));
  return <div className="library dataset-page">
    {error && <p className="error" role="alert">{error}</p>}
    {id ? <>
      <div className="library-actions"><a className="btn-small" href="#/dataset">← All IDs</a><p>{id}</p></div>
      {selected && video && <section className="dataset-viewer">
        <header className="dataset-viewer-head"><div><span className="eyebrow">Mapped render</span><h2>{selected.id}</h2></div>
          {selected.videos.length > 1 && <label>Version <select aria-label="Dataset video version" value={variant} onChange={e => setVariant(Number(e.target.value))}>
            {selected.videos.map((v, i) => <option key={v.key} value={i}>{v.folder}</option>)}
          </select></label>}
        </header>
        <video key={video.key} src={video.url} controls autoPlay muted playsInline preload="metadata" aria-label={`Dataset video ${id}`} />
        <p>Source: {video.folder} · {video.name}</p>
      </section>}
      {loaded && !selected && <div className="card-soft"><h2>No video found for this ID.</h2><p>Return to the dataset list and choose another ID.</p></div>}
    </> : <>
      <div className="library-actions dataset-actions"><p>{entries.length} IDs · {entries.reduce((n, row) => n + row.videos.length, 0)} videos</p>
        <label className="dataset-search"><span className="sr-only">Search dataset IDs</span><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search an ID" /></label>
      </div>
      {!loaded && !error && <p>Loading dataset…</p>}
      {loaded && !entries.length && <div className="card-soft"><h2>No dataset videos found.</h2><p>Videos in the two outputs folders appear here automatically.</p></div>}
      {loaded && !!entries.length && !visibleEntries.length && <div className="card-soft"><h2>No matching IDs.</h2><p>Try a shorter part of the ID.</p></div>}
      {visibleEntries.map(row => <article className="library-row dataset-row" key={row.id}>
        <a href={`#/dataset/${encodeURIComponent(row.id)}`}><b>{row.id}</b><small>{row.videos.map(v => v.folder).join(" · ")}</small></a>
        <span className="pill-sm">{row.videos.length} {row.videos.length === 1 ? "video" : "videos"}</span>
        <a className="btn-small" href={`#/dataset/${encodeURIComponent(row.id)}`}>Play ↗</a>
      </article>)}
    </>}
  </div>;
}
