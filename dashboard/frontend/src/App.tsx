import { useCallback, useEffect, useRef, useState } from "react";
import Landing from "./Landing";
import Dataset from "./components/Dataset";
import Icon, { BrandMark } from "./components/Icon";
import Showcase from "./Showcase";
import Safe from "./components/Safe";
import WorkspacePlayer from "./components/WorkspacePlayer";
import type { ClipData, ClipSummary } from "./api";
import { api } from "./api";
import { base, request, type Connection, type WorkspaceJob } from "./workspaceApi";
import "./workspace.css";

function useRoute() {
  const [route, setRoute] = useState(location.hash || "#/landing");
  useEffect(() => { const fn = () => setRoute(location.hash || "#/landing"); window.addEventListener("hashchange", fn); return () => window.removeEventListener("hashchange", fn); }, []);
  return route;
}

export default function App() {
  const route = useRoute();
  const [path, query] = route.replace(/^#\/?/, "").split("?");
  const parts = path.split("/");
  const page = ["landing", "workspace", "library", "dataset", "room", "showcase", "settings"].includes(parts[0]) ? parts[0] : "landing";
  const wid = parts[1] ? decodeURIComponent(parts[1]) : "";
  const [connection, setConnection] = useState<Connection | null>(null);
  const [model, setModel] = useState("Pretrained baseline");
  const [jobs, setJobs] = useState<WorkspaceJob[]>([]);
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [job, setJob] = useState<WorkspaceJob | null>(null);
  const [clip, setClip] = useState<ClipData | null>(null);
  const [panel, setPanel] = useState(false);
  const [navOpen, setNavOpen] = useState(true);
  const [exportOpen, setExportOpen] = useState(false);
  useEffect(() => { setPanel(Boolean(wid)); setNavOpen(!wid); setExportOpen(false); }, [wid]);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [manual, setManual] = useState("");
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [cpu, setCpu] = useState<string | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const refresh = useCallback(async () => {
    const [js, cs] = await Promise.all([request<WorkspaceJob[]>("/api/workspaces"), api.clips()]); setJobs(js); setClips(cs);
  }, []);
  const check = useCallback(async (force = false) => {
    const c = await request<Connection>(`/api/colab/status?force=${force}`); setConnection(c);
    if (!settingsDirty) setManual(c.settings.url);
  }, [settingsDirty]);
  useEffect(() => {
    void check().catch(e => setError(String(e)));
    const timer = setInterval(() => { void check().catch(() => {}); }, 10000);
    return () => clearInterval(timer);
  }, [check]);
  useEffect(() => { void api.model().then(m => setModel(m.label)).catch(() => {}); void refresh().catch(e => setError(String(e))); }, [refresh]);
  useEffect(() => {
    setClip(null); setJob(null); setError("");
    if (!wid || page !== "workspace") return;
    let active = true, timer: ReturnType<typeof setTimeout>;
    if (wid.startsWith("demo:")) {
      void api.clip(wid.slice(5)).then(c => { if (active) setClip(c); }).catch(e => { if (active) setError(String(e)); });
    } else {
      let loaded = false;
      const tick = async () => {
        try {
          const [j, allJobs] = await Promise.all([request<WorkspaceJob>(base(wid)), request<WorkspaceJob[]>("/api/workspaces")]);
          if (!active) return;
          setJob(j); setJobs(allJobs);
          if (j.status === "ready" && !loaded) {
            const c = await request<ClipData>(`${base(wid)}/clip`);
            if (active) { setClip(c); loaded = true; }
          }
        } catch (e) { if (active) setError(String(e)); }
        if (active) timer = setTimeout(tick, 1000);
      };
      void tick();
    }
    return () => { active = false; clearTimeout(timer); };
  }, [page, wid]);
  useEffect(() => {
    if (!cpu) return;
    const timer = setInterval(() => { void api.job(cpu).then(j => {
      if (j.status === "done" && j.clip_key) { location.hash = `/workspace/demo:${encodeURIComponent(j.clip_key)}`; setCpu(null); }
      if (j.status === "error") { setError(j.error ?? "CPU preview failed"); setCpu(null); }
    }).catch(e => setError(String(e))); }, 1500);
    return () => clearInterval(timer);
  }, [cpu]);
  useEffect(() => { if (page === "library") void refresh(); }, [page, refresh]);
  async function upload(video: File | undefined) {
    if (!video || uploading) return;
    setError(""); setUploading(true);
    try {
      const body = new FormData(); body.append("file", video);
      const j = await request<WorkspaceJob>("/api/workspaces", { method: "POST", body });
      location.hash = `/workspace/${j.id}`; await refresh();
    } catch (e) { setError(String(e)); }
    finally { setUploading(false); if (file.current) file.current.value = ""; }
  }
  async function remove(id?: string) {
    try {
      await request(id ? base(id) : "/api/data", { method: "DELETE" });
      if (!id || id === wid) location.hash = "/library";
      await refresh();
    } catch (e) { setError(String(e)); }
  }
  async function save(values: Record<string, unknown>) {
    try {
      setConnection(await request<Connection>("/api/colab/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) }));
      setSettingsDirty(false);
    } catch (e) { setError(String(e)); }
  }
  const ahead = jobs.find(j => j.id !== wid && ["preparing", "uploading", "processing", "fetching"].includes(j.status));
  const earlierReady = job && jobs.find(j => j.id !== wid && j.name === job.name && j.status === "ready");
  const pill = connection?.connected ? `${connection.gpu ?? "Colab"} · ${connection.warm ? "ready" : "warming up"}` : "Offline · Settings";
  if (page === "landing") return <Landing onUpload={() => { location.hash = "/workspace"; }} />;
  const ready = !!clip;
  const active = page === "workspace" && !!wid && !ready;
  const title = page === "dataset" ? wid || "Dataset" : job?.name ?? clip?.name ?? "No clip loaded";
  return <div className={`app-shell ${navOpen ? "nav-expanded" : "nav-collapsed"} ${page === "workspace" ? "studio-workspace" : ""}`}>
    <header className="app-topbar"><a className="app-brand" href="#/landing"><BrandMark />SpectatorAI</a>
      <nav className="breadcrumbs" aria-label="Breadcrumb"><a href="#/library">Local workspace</a><span>›</span><span>{title}</span></nav>
      {ready && <span className="status-pill ready"><i />Ready</span>}
      {active && <span className="status-pill">{job?.status === "failed" ? "Failed" : job?.stage ?? "Opening…"}</span>}
      <a className={`connection-pill ${connection?.connected ? "online" : ""}`} href="#/settings" title={pill}><i />{connection?.connected ? "Colab connected" : "Colab offline"}</a>
      {ready && <span className="clip-spec">{clip.width}×{clip.height} · {clip.fps.toFixed(0)} fps · {clip.n_frames} frames</span>}
      <div className="export-control"><button className="studio-export" disabled={!ready} aria-expanded={exportOpen} onClick={() => setExportOpen(!exportOpen)}>Export<Icon name="down" size={16} /></button>
        {exportOpen && <div className="export-menu" role="menu">
          {job && (job.render.status === "done" ? <a role="menuitem" href={`${base(job.id)}/export/render`} download>Rendered video <span>MP4</span></a> : <button role="menuitem" disabled>Rendered video · {job.render.status}<span>MP4</span></button>)}
          {job && <button role="menuitem" onClick={() => { setPanel(true); window.dispatchEvent(new CustomEvent("workspace-export-view")); setExportOpen(false); }}>My 3D view <span>MP4</span></button>}
          {["hands.jsonl", "robot_wrist_gripper.csv", "stats.json", ...(!wid.startsWith("demo:") ? ["quality.json", "scene.json"] : [])].map(n => <a key={n} role="menuitem" href={wid.startsWith("demo:") ? `/api/clips/${encodeURIComponent(wid.slice(5))}/download/${n}` : `${base(wid)}/download/${n}`} download>{n}<span>{n.endsWith("csv") ? "CSV" : "JSON"}</span></a>)}
        </div>}
      </div>
      <button className="panel-toggle" aria-label="Toggle right panel" aria-pressed={panel} onClick={() => setPanel(!panel)}><Icon name="panel" /></button>
      <input ref={file} type="file" accept="video/*" hidden onChange={e => void upload(e.target.files?.[0])} />
    </header>
    <nav className="app-sidebar" aria-label="Main navigation">
      <button className="add-video" aria-label="Add video" disabled={uploading} onClick={() => { location.hash = "/workspace"; file.current?.click(); }}><Icon name="plus" /><span>{uploading ? "Uploading…" : "Add video"}</span></button>
      <div className="nav-spacer" />
      {[["workspace", "Workspace", "workspace"], ["room", "Room", "room"], ["library", "Library", "library"], ["dataset", "Dataset", "library"], ["showcase", "Compare", "compare"]].map(([p, label, icon]) => <a key={p} aria-label={label} title={label} className={page === p ? "active" : ""} href={p === "showcase" ? "#compare" : `#/${p}`}><Icon name={icon} /><span>{label}</span></a>)}
      <button aria-label="Exports" title="Exports" onClick={() => { setPanel(true); window.dispatchEvent(new CustomEvent("workspace-details", { detail: "exports" })); }}><Icon name="export" /><span>Exports</span></button>
      <a href="#/settings" aria-label="Settings" title="Settings" className={page === "settings" ? "active" : ""}><Icon name="settings" /><span>Settings</span></a>
      <button className="collapse-nav" aria-label={navOpen ? "Collapse navigation" : "Expand navigation"} onClick={() => setNavOpen(!navOpen)}><Icon name={navOpen ? "left" : "right"} /><span>Collapse</span></button>
    </nav>
    <main className={`app-content page-${page}`}>
      {error && <p className="error" role="alert">{error}</p>}
      {page === "showcase" ? <Showcase /> : <>
        <div className={`page-heading ${page === "workspace" ? "workspace-heading" : ""}`}><div><span className="eyebrow">SpectatorAI / {page}</span><h1>{page === "workspace" ? job?.name ?? clip?.name ?? "Your motion, in 3D." : page[0].toUpperCase() + page.slice(1)}</h1></div>
          {page === "workspace" && job && <button className="btn-small" onClick={() => void remove(job.id)}>Delete workspace</button>}
        </div>
        {page === "workspace" && <>
          {!wid && <div className="empty-workspace"><div className="empty-content">
            <div className="empty-heading"><span className="outline-pill">Workspace</span><h1>Drop in a clip.<br />Watch it become 3D.</h1></div>
            <section className="upload-card"><h2>Add a head-camera clip</h2><p>Fisheye or a normal phone video, MP4 or MOV. The video plays as soon as it uploads; tracking runs alongside.</p>
              <div className="drop-zone" role="button" tabIndex={0} onClick={() => file.current?.click()} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") file.current?.click(); }} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); void upload(e.dataTransfer.files[0]); }}>{uploading ? "Uploading…" : "Drop a video here or choose a file"}</div>
              <small>{connection?.connected ? "Colab connected · processing starts automatically" : "Colab offline · uploads wait locally until connected"}</small>
            </section>
            <section className="recent-card"><div className="recent-heading"><h2>Recent clips</h2><a href="#/library">Open library</a></div>
              {jobs.slice(0, 5).map(j => <a className="recent-row" href={`#/workspace/${j.id}`} key={j.id}><span>{j.name}</span><span>Local workspace</span><span>{j.video ? `${j.video.duration_s.toFixed(0)} s` : "—"}</span><span className={`recent-status ${j.status}`}>{j.status}</span><span>{new Date(j.created * 1000).toLocaleDateString()}</span></a>)}
              {!jobs.length && <p className="empty-recents">Your recent clips will appear here.</p>}
            </section>
          </div></div>}
          {job && job.status !== "ready" && <div className="processing-card"><span className="eyebrow">{job.status}</span><h2>{job.status === "uploaded" && ahead ? "Waiting for an earlier upload" : job.stage}</h2>
            {job.status === "uploaded" && ahead && <p><b>{ahead.name}</b> is currently {ahead.stage?.toLowerCase() ?? ahead.status}. Your clip starts automatically when it finishes.</p>}
            {earlierReady && <p>An earlier upload with this filename is already ready. <a href={`#/workspace/${earlierReady.id}`} style={{ textDecoration: "underline" }}>Open its results ↗</a></p>}
            <progress max={1} value={job.progress} /><p>{Math.round(job.progress * 100)}% · {Math.round(Date.now() / 1000 - (job.started ?? job.created))} s elapsed{job.queue_position ? ` · queue position ${job.queue_position}` : ""}</p>
            <div className="steps">{["preparing", "uploading", "processing", "fetching"].map(s => <span className={job.timings[s] ? "done" : ""} key={s}>{job.timings[s] ? "✓ " : "○ "}{s}{job.timings[s] ? ` · ${job.timings[s].toFixed(1)} s` : ""}</span>)}</div>
            {Object.entries(job.stage_seconds).map(([k, v]) => <small key={k}>{k}: {v.toFixed(1)} s · </small>)}
            {job.video && <video src={`${base(job.id)}/video`} controls muted className="processing-video" />}
            {job.error && <p className="error">{job.error}</p>}
            {job.status === "failed" && <><button className="btn-small" onClick={() => void request(`${base(job.id)}/retry`, { method: "POST" }).catch(e => setError(String(e)))}>Retry</button><details><summary>Server details</summary><pre>{job.traceback ?? job.error}</pre></details></>}
            {job.video && <button className="btn-small" disabled={!!cpu} onClick={() => void request<{ id: string }>(`${base(job.id)}/cpu`, { method: "POST" }).then(j => setCpu(j.id)).catch(e => setError(String(e)))}>{cpu ? "CPU preview running…" : "Run locally · 10 s CPU preview"}</button>}
          </div>}
          {clip && <Safe name="3D workspace"><WorkspacePlayer key={wid} clip={clip} job={job ?? undefined} demo={wid.startsWith("demo:") ? wid.slice(5) : undefined} panel={panel} initialFrame={Number(new URLSearchParams(query).get("frame") ?? 0)} /></Safe>}
          {wid && !clip && !job && !error && <p>Opening workspace…</p>}
          {!clip && <section className="empty-transport"><div><button disabled><Icon name="play" size={14} />Play</button><button disabled aria-label="Previous frame"><Icon name="prev" size={16} /></button><button disabled aria-label="Next frame"><Icon name="next" size={16} /></button><span>00:00.00</span><small>frame — / —</small><div className="grow" /><div className="speed-group"><span>0.25×</span><span>0.5×</span><span className="selected">1×</span></div><button disabled>Lanes⌄</button></div><p>No clip loaded</p></section>}
          {!wid && panel && <aside className="empty-inspector"><div className="detail-tabs"><button>Frame</button><button>Clip</button><button>Quality</button></div><p>Load a clip to see values at the playhead.</p></aside>}
        </>}
        {page === "dataset" && <Dataset id={wid} />}
        {page === "library" && <div className="library"><div className="library-actions"><p>{jobs.length} workspaces · {clips.length} saved results</p><button className="btn-small" onClick={() => void remove()}>Delete all data</button></div>
          {!jobs.length && !clips.length && <div className="card-soft"><h2>Your library starts with one video.</h2><p>Upload a clip to keep an interactive workspace here.</p></div>}
          {jobs.map(j => <article className="library-row" key={j.id}><a href={`#/workspace/${j.id}`}><b>{j.name}</b><small>{new Date(j.created * 1000).toLocaleString()} · {j.video?.duration_s.toFixed(1) ?? "—"} s</small></a><span>{j.quality?.grade ?? "—"}</span><span className="pill-sm">{j.status}</span><button className="btn-small" onClick={() => void remove(j.id)}>Delete</button></article>)}
          {clips.map(c => <article className="library-row" key={c.key}><a href={`#/workspace/demo:${encodeURIComponent(c.key)}`}><b>{c.name}</b><small>{c.model_label} · saved result</small></a><span className="pill-sm">Read only</span></article>)}
        </div>}
        {page === "room" && <div className="room-placeholder"><span>⌂</span><h2>A room-fixed view is the next layer.</h2><p>SLAM runs separately using <code>ps5_world_slam.ipynb</code>, about 15–25 minutes per clip. It is outside the 1–2 minute workspace flow.</p><p>The current workspace shows head rotation and hands in the frame-zero camera reference. Room translation needs a separate SLAM run.</p></div>}
        {page === "settings" && <div className="settings-grid"><section className="settings-card"><span className="eyebrow">Colab connection</span><h2>{pill}</h2><p>Discovery: {connection?.source ?? "not found"} · {connection?.url_host ?? "No server"}</p><p>API key in local .env: {connection?.key_present ? "present" : "missing"}</p>
          <label>Manual server URL<input type="url" placeholder="https://….trycloudflare.com" value={manual} onChange={e => { setSettingsDirty(true); setManual(e.target.value); }} /></label>
          <div className="view-tools"><button className="btn-small" onClick={() => void save({ url: manual.trim() })}>Save URL</button><button className="btn-small" onClick={() => void check(true).catch(e => setError(String(e)))}>Test connection</button><button className="btn-small" onClick={() => { setManual(""); void save({ url: "" }); }}>Use discovery</button></div>
          <p>Set the same API key and discovery topic in Colab Secrets and <code>dashboard/.env</code>. Credentials stay in the local gateway.</p>
        </section><section className="settings-card"><span className="eyebrow">Processing</span><p>{model}</p><label>Profile<select value={connection?.settings.profile ?? "fast"} onChange={e => void save({ profile: e.target.value })}><option value="fast">Fast · dashboard</option><option value="full">Full · original pipeline</option></select></label>
          <label><input type="checkbox" checked={connection?.settings.auto_render ?? true} onChange={e => void save({ auto_render: e.target.checked })} /> Prepare rendered MP4 in background</label>
          <label>Local retention (hours)<input type="number" min={1} max={720} defaultValue={connection?.settings.retention_h ?? 24} onBlur={e => void save({ retention_h: +e.target.value })} /></label>
          <p>Videos are processed up to 60 seconds. Remote jobs are deleted after download; inactive server jobs expire after six hours.</p><button className="btn-small" onClick={() => void remove()}>Delete all data</button>
        </section></div>}
      </>}
    </main>
  </div>;
}
