import { useEffect, useState } from "react";
import { request } from "../workspaceApi";

type Entry = { id: string; videos: { key: string; folder: string; name: string; url: string }[] };

export default function Dataset({ id }: { id: string }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [variant, setVariant] = useState(0);
  useEffect(() => {
    let active = true;
    void request<Entry[]>("/api/dataset").then(rows => { if (active) { setEntries(rows); setLoaded(true); } })
      .catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, []);
  useEffect(() => { setVariant(0); }, [id]);
  const selected = entries.find(row => row.id === id);
  const video = selected?.videos[variant] ?? selected?.videos[0];
  return <div className="library dataset-page">
    {error && <p className="error" role="alert">{error}</p>}
    {id ? <>
      <div className="library-actions"><a className="btn-small" href="#/dataset">← All IDs</a><p>{id}</p></div>
      {selected && video && <section className="dataset-viewer">
        {selected.videos.length > 1 && <label>Video version <select aria-label="Dataset video version" value={variant} onChange={e => setVariant(Number(e.target.value))}>
          {selected.videos.map((v, i) => <option key={v.key} value={i}>{v.folder}</option>)}
        </select></label>}
        <video key={video.key} src={video.url} controls autoPlay muted playsInline preload="metadata" aria-label={`Dataset video ${id}`} />
        <p>{video.folder} · {video.name}</p>
      </section>}
      {loaded && !selected && <p>No video found for this ID.</p>}
    </> : <>
      <div className="library-actions"><p>{entries.length} IDs · {entries.reduce((n, row) => n + row.videos.length, 0)} videos</p></div>
      {!loaded && !error && <p>Loading dataset…</p>}
      {loaded && !entries.length && <div className="card-soft"><h2>No dataset videos found.</h2><p>Videos in the two outputs folders appear here automatically.</p></div>}
      {entries.map(row => <article className="library-row" key={row.id}>
        <a href={`#/dataset/${encodeURIComponent(row.id)}`}><b>{row.id}</b><small>{row.videos.map(v => v.folder).join(" · ")}</small></a>
        <span className="pill-sm">{row.videos.length} {row.videos.length === 1 ? "video" : "videos"}</span>
        <a className="btn-small" href={`#/dataset/${encodeURIComponent(row.id)}`}>Play ↗</a>
      </article>)}
    </>}
  </div>;
}
