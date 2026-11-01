import { useCallback, useEffect, useRef, useState } from "react";
import type { ClipData } from "../api";
import type { WorkspaceJob } from "../workspaceApi";
import { base, request } from "../workspaceApi";
import VideoOverlay from "./VideoOverlay";
import Timeline from "./Timeline";
import Icon from "./Icon";
import Workspace3D from "./Workspace3D";

export default function WorkspacePlayer({ clip, job, demo, panel, initialFrame = 0 }: {
  clip: ClipData; job?: WorkspaceJob; demo?: string; panel: boolean; initialFrame?: number;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const frame = useRef(initialFrame);
  const [display, setDisplay] = useState(initialFrame);
  const [playing, setPlaying] = useState(false);
  const [follow, setFollow] = useState(true);
  const [path, setPath] = useState(true);
  const [overlay, setOverlay] = useState(true);
  const [preset, setPreset] = useState("Reset:0");
  const [layout, setLayout] = useState("split");
  const [details, setDetails] = useState("frame");
  const [speed, setSpeed] = useState(1);
  const [lanes, setLanes] = useState(true);
  const [timelineHeight, setTimelineHeight] = useState(222);
  const [split, setSplit] = useState(56);
  const [showHands, setShowHands] = useState(true);
  const [labels, setLabels] = useState(true);
  const [head, setHead] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);
  const [error, setError] = useState("");
  const [range, setRange] = useState([0, clip.n_frames - 1]);
  const mounted = useRef(true);
  const cancelRecording = useRef<(() => void) | null>(null);
  const url = demo ? `/api/clips/${encodeURIComponent(demo)}/video` : `${base(job!.id)}/video`;
  const download = (name: string) => demo ? `/api/clips/${encodeURIComponent(demo)}/download/${name}` : `${base(job!.id)}/download/${name}`;
  const onCanvas = useCallback((c: HTMLCanvasElement) => { canvas.current = c; }, []);
  const seek = useCallback((f: number) => {
    const next = Math.max(0, Math.min(clip.n_frames - 1, f));
    frame.current = next; setDisplay(next);
    if (video.current) video.current.currentTime = next / clip.fps;
  }, [clip]);
  const toggle = useCallback(() => {
    const v = video.current;
    if (v) { if (v.paused) void v.play().catch(e => setError(String(e))); else v.pause(); }
  }, []);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; cancelRecording.current?.(); };
  }, []);
  useEffect(() => {
    const v = video.current!;
    let raf = 0, callback = 0, last = 0;
    const tick = (mediaTime = v.currentTime) => {
      frame.current = Math.max(0, Math.min(clip.n_frames - 1, Math.floor(mediaTime * clip.fps + 1e-3)));
      if (performance.now() - last > 80) { setDisplay(frame.current); setPlaying(!v.paused); last = performance.now(); }
    };
    const onVideoFrame: VideoFrameRequestCallback = (_, metadata) => { tick(metadata.mediaTime); callback = v.requestVideoFrameCallback(onVideoFrame); };
    const onAnimation = () => { tick(); raf = requestAnimationFrame(onAnimation); };
    if (v.requestVideoFrameCallback) callback = v.requestVideoFrameCallback(onVideoFrame);
    else raf = requestAnimationFrame(onAnimation);
    const loaded = () => seek(initialFrame);
    v.addEventListener("loadedmetadata", loaded);
    const onSeeked = () => tick();
    v.addEventListener("seeked", onSeeked);
    return () => { cancelAnimationFrame(raf); if (callback) v.cancelVideoFrameCallback(callback); v.removeEventListener("loadedmetadata", loaded); v.removeEventListener("seeked", onSeeked); };
  }, [clip, seek, initialFrame]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (recording || (e.target instanceof HTMLElement && /INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName))) return;
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault(); video.current?.pause();
        seek(frame.current + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? Math.round(clip.fps) : 1));
      }
    };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, [clip, seek, toggle, recording]);

  async function recordView() {
    const v = video.current, gl = canvas.current;
    if (!v || !gl || !job || recording) return;
    setError(""); setRecording(true); setRecordProgress(0);
    const previous = { time: v.currentTime, speed: v.playbackRate };
    let recorder: MediaRecorder | undefined, raf = 0, stream: MediaStream | undefined;
    try {
      if (!window.MediaRecorder) throw new Error("This browser cannot record a canvas; use Chrome or Edge.");
      const out = document.createElement("canvas"); out.width = 1920; out.height = 720;
      const ctx = out.getContext("2d")!;
      stream = out.captureStream(30);
      const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(MediaRecorder.isTypeSupported);
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000 } : undefined);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      const finished = new Promise<void>((resolve, reject) => {
        recorder!.onstop = () => resolve(); recorder!.onerror = () => reject(new Error("Browser recording failed"));
      });
      v.pause(); v.playbackRate = 1;
      await new Promise<void>(resolve => { v.addEventListener("seeked", () => resolve(), { once: true }); v.currentTime = range[0] / clip.fps; if (Math.abs(v.currentTime - range[0] / clip.fps) < .001 && v.readyState >= 2) setTimeout(resolve, 100); });
      const stop = () => { v.pause(); if (recorder?.state === "recording") recorder.stop(); };
      cancelRecording.current = stop;
      const draw = () => {
        if (!mounted.current) { stop(); return; }
        ctx.fillStyle = "#f8f9eb"; ctx.fillRect(0, 0, out.width, out.height);
        const h = Math.min(720, 960 * v.videoHeight / v.videoWidth), w = h * v.videoWidth / v.videoHeight;
        ctx.drawImage(v, (960 - w) / 2, (720 - h) / 2, w, h);
        ctx.drawImage(gl, 960, 0, 960, 720);
        setRecordProgress(Math.min(1, (v.currentTime * clip.fps - range[0]) / Math.max(1, range[1] - range[0])));
        if (v.ended || v.currentTime >= (range[1] + 1) / clip.fps - .01) stop();
        else raf = requestAnimationFrame(draw);
      };
      recorder.start(250); await v.play(); draw(); await finished;
      if (!mounted.current) return;
      setRecordProgress(1);
      const body = new FormData(); body.append("file", new Blob(chunks, { type: recorder.mimeType }), "view.webm");
      const result = await request<{ url: string }>(`${base(job.id)}/export/view`, { method: "POST", body });
      const a = document.createElement("a"); a.href = result.url; a.download = "view.mp4"; a.click();
    } catch (e) { setError(String(e)); }
    finally {
      cancelAnimationFrame(raf); if (recorder?.state === "recording") recorder.stop(); stream?.getTracks().forEach(t => t.stop());
      v.pause(); v.currentTime = previous.time; v.playbackRate = previous.speed;
      cancelRecording.current = null; if (mounted.current) setRecording(false);
    }
  }
  useEffect(() => {
    const exportView = () => { setDetails("exports"); void recordView(); };
    const selectDetails = (e: Event) => setDetails((e as CustomEvent<string>).detail);
    window.addEventListener("workspace-export-view", exportView);
    window.addEventListener("workspace-details", selectDetails);
    return () => { window.removeEventListener("workspace-export-view", exportView); window.removeEventListener("workspace-details", selectDetails); };
  });
  const q = clip.quality;
  const clock = (f: number) => {
    const seconds = Math.floor(f / clip.fps);
    return `00:${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}:${String(f % Math.round(clip.fps)).padStart(2, "0")}`;
  };
  function startResize(e: React.PointerEvent, type: "split" | "timeline") {
    e.preventDefault();
    const initial = { x: e.clientX, y: e.clientY, split, height: timelineHeight };
    const el = e.currentTarget.parentElement;
    const move = (event: PointerEvent) => {
      if (type === "timeline") setTimelineHeight(Math.max(155, Math.min(380, initial.height + initial.y - event.clientY)));
      else if (el) setSplit(Math.max(30, Math.min(72, initial.split + (event.clientX - initial.x) / el.clientWidth * 100)));
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  }
  const left = clip.hands.left, right = clip.hands.right;
  const a = left.kp3d[display], b = right.kp3d[display];
  const distance = a && b ? Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) : null;
  const source = (n: number) => ["out of view", "observed", "infilled", "estimated"][n] ?? "out of view";
  return <div className={`workspace-ready ${panel ? "with-panel" : ""}`}>
    <section className="work-main">
      <div className={`viewer-grid layout-${layout}`}>
        <section className="video-pane" aria-label="Video" style={layout === "split" ? { flexBasis: `calc(${split}% - 7px)` } : undefined}>
          <header className="pane-header"><h2>Video · {clip.name}</h2>{clip.overlay_baked && <span className="pane-badge">Skeleton baked in</span>}<div className="grow" /><button className={labels ? "chip-toggle on" : "chip-toggle"} disabled={recording} aria-pressed={labels} onClick={() => setLabels(!labels)}>Labels</button><button className={overlay ? "chip-toggle on" : "chip-toggle"} disabled={recording} aria-pressed={overlay} onClick={() => setOverlay(!overlay)}>Skeleton</button></header>
          <div className="video-stage"><VideoOverlay clip={clip} videoUrl={url} videoRef={video} frame={display} frameRef={frame} overlay={overlay} handColors={{ left: "#45d16a", right: "#ff9a2e" }} />
            {labels && <div className="video-labels">{(["left", "right"] as const).map(s => <span key={s} className={`hand-chip ${s}`}><i />{s} · {source(clip.hands[s].src[display])} · {(clip.hands[s].conf[display] ?? 0).toFixed(2)}</span>)}</div>}
            <span className="video-clock">{clock(display)}</span>
          </div>
        </section>
        <button className="split-resizer" aria-label="Resize split between video and 3D view" disabled={recording} onPointerDown={e => startResize(e, "split")}><i /></button>
        <section className="scene-pane" aria-label="3D hands"><header className="pane-header"><h2>3D hands · camera space</h2><span className="pane-badge">Head rotation removed</span><div className="grow" />
          <button className={`chip-toggle ${showHands ? "on" : ""}`} aria-pressed={showHands} title="Hand surface fitted from saved joints" disabled={recording} onClick={() => setShowHands(!showHands)}>Hands</button><button className={`chip-toggle ${path ? "on" : ""}`} aria-pressed={path} disabled={recording} onClick={() => setPath(!path)}>Trails</button></header>
          <div className="scene-stage"><Workspace3D clip={clip} frame={frame} path={path} follow={follow} preset={preset} onCanvas={onCanvas} locked={recording} hands={showHands} labels={false} head={head} />
            <div className="scene-controls"><button className={`chip-toggle ${follow ? "on" : ""}`} disabled={recording} onClick={() => { setFollow(!follow); if (preset.startsWith("Wearer")) setPreset(`Reset:${Date.now()}`); }}>Follow</button>
              <select aria-label="Camera view" disabled={recording} value={preset.split(":")[0]} onChange={e => { setFollow(false); setPreset(`${e.target.value}:${Date.now()}`); }}>{["Reset", "Wearer's eye", "Behind", "Top", "Side"].map(p => <option key={p}>{p}</option>)}</select>
              <button className={`chip-toggle ${head ? "on" : ""}`} disabled={recording} onClick={() => setHead(!head)}>Camera</button>
              <select aria-label="Workspace layout" disabled={recording} value={layout} onChange={e => setLayout(e.target.value)}><option value="split">Split view</option><option value="scene">3D large</option><option value="video">Video large</option></select>
            </div><span className="scene-caption">Wrist trail = robot trajectory</span>
          </div>
        </section>
      </div>
      <button className="timeline-resizer" aria-label="Resize timeline" onPointerDown={e => startResize(e, "timeline")}><i /></button>
      <section className="timeline-dock" style={{ height: lanes ? timelineHeight : 118 }} aria-label="Timeline">
        <fieldset disabled={recording} className="transport">
          <button className="play-button" onClick={toggle}><Icon name={playing ? "pause" : "play"} size={14} />{playing ? "Pause" : "Play"}</button>
          <button className="frame-button" aria-label="Previous frame" onClick={() => { video.current?.pause(); seek(frame.current - 1); }}><Icon name="prev" size={16} /></button>
          <button className="frame-button" aria-label="Next frame" onClick={() => { video.current?.pause(); seek(frame.current + 1); }}><Icon name="next" size={16} /></button>
          <output className="transport-time">{clock(display)}</output><span className="transport-frame">frame {display} / {clip.n_frames - 1}</span><div className="grow" />
          <div className="speed-group" role="group" aria-label="Playback speed">{[.25, .5, 1].map(s => <button key={s} className={speed === s ? "selected" : ""} aria-pressed={speed === s} onClick={() => { setSpeed(s); if (video.current) video.current.playbackRate = s; }}>{s}×</button>)}</div>
          <button className="lanes-button" aria-expanded={lanes} onClick={() => setLanes(!lanes)}>Lanes<Icon name="down" size={14} /></button>
        </fieldset>
        <div className="timeline-scroll"><div className="time-ruler"><span /> <div>{Array.from({ length: 11 }, (_, i) => <span key={i} style={{ left: `${i * 10}%` }}>{(i * clip.n_frames / clip.fps / 10).toFixed(0)}s</span>)}</div></div>
          <div className="clip-scrubber"><label htmlFor="video-frame">Clip</label><input id="video-frame" aria-label="Video frame" type="range" disabled={recording} min={0} max={clip.n_frames - 1} step={1} value={display} onChange={e => seek(+e.target.value)} /></div>
          {lanes && <div className="quality-lanes" aria-label="Quality timeline">{["left_hand", "right_hand", "image", "head", "body", "usable"].map(lane => <div className={`quality-lane lane-${lane}`} key={lane}>
            <span><i />{lane === "left_hand" ? "Left" : lane === "right_hand" ? "Right" : lane}</span><div className="lane-track">
              {q?.segments.filter(s => s.lane === lane).map((s, i) => <button key={i} disabled={recording} title={`${s.label} · ${s.start_s.toFixed(2)}–${s.end_s.toFixed(2)} s ${(s.reasons ?? []).join(", ")}`} className={`segment label-${s.label}`} style={{ left: `${s.start_s * clip.fps / clip.n_frames * 100}%`, width: `${(s.end_s - s.start_s) * clip.fps / clip.n_frames * 100}%` }} onClick={e => { const rect = e.currentTarget.getBoundingClientRect(); seek(Math.round((s.start_s + (e.clientX - rect.left) / rect.width * (s.end_s - s.start_s)) * clip.fps)); }} />)}
              {!q && clip.hands[lane === "right_hand" ? "right" : "left"] && <span className="missing-lane">Quality metadata unavailable</span>}
              <i className="lane-cursor" style={{ left: `${display / clip.n_frames * 100}%` }} />
            </div></div>)}<div className="lane-legend"><span><i className="legend-observed" />Observed</span><span><i className="legend-infilled" />Infilled</span><span><i className="legend-estimated" />Estimated</span><span><i className="legend-out" />Out of view / unusable</span></div></div>}
        </div>
      </section>
      {error && <p role="alert" className="error player-error">{error}</p>}
    </section>
    {panel && <aside className="inspector" aria-label="Details">
      <div className="detail-tabs" role="tablist" aria-label="Details">{["frame", "clip", "quality"].map(t => <button key={t} role="tab" aria-selected={details === t} onClick={() => setDetails(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}</div>
      <div className="detail-content">
        {details === "frame" && <><div className="playhead-meta"><span>At playhead</span><span>F {display} · {clock(display)}</span></div>
          {(["left", "right"] as const).map(s => {
            const h = clip.hands[s], k = h.kp3d[display], previous = h.kp3d[Math.max(0, display - 1)];
            const velocity = k && previous ? Math.hypot(k[0] - previous[0], k[1] - previous[1], k[2] - previous[2]) * clip.fps : null;
            return <section className={`hand-card ${s}`} key={s}><header><h2>{s === "left" ? "Left" : "Right"} hand</h2><span>{source(h.src[display])}</span></header><div className="hand-confidence"><strong>{h.src[display] ? (h.conf[display] ?? 0).toFixed(2) : "—"}</strong><span>confidence</span></div><dl>
              <dt>Source</dt><dd>{h.src[display] === 1 ? "21 keypoints observed" : source(h.src[display])}</dd><dt>Wrist x, y, z (m)</dt><dd>{k ? k.slice(0, 3).map(n => `${n >= 0 ? "+" : ""}${n.toFixed(3)}`).join(" ") : "—"}</dd><dt>Wrist speed</dt><dd>{velocity == null ? "—" : `${velocity.toFixed(2)} m/s`}</dd><dt>State</dt><dd>{h.state[display]?.toLowerCase().replace(/_/g, " ")}</dd>
            </dl></section>;
          })}<section className="detail-card"><h3>Both hands</h3><dl><dt>Tracking</dt><dd>{left.src[display] && right.src[display] ? "Both hands" : left.src[display] ? "Left hand" : right.src[display] ? "Right hand" : "Out of view"}</dd><dt>Reference frame</dt><dd>Head-stable</dd><dt>Hand to hand</dt><dd>{distance == null ? "—" : `${distance.toFixed(2)} m`}</dd></dl></section><p className="detail-note">Estimated hands are guesses, not measurements.</p></>}
        {details === "clip" && <><div className="clip-number-grid">{[["Duration", `${(clip.n_frames / clip.fps).toFixed(1)} s`], ["Frames", clip.n_frames], ["Frame rate", `${clip.fps.toFixed(0)} fps`], ["Quality", q?.grade ?? "—"]].map(([k, v]) => <article key={k}><span>{k}</span><strong>{v}</strong></article>)}</div><Timeline clip={clip} frame={display} onSeek={f => { if (!recording) seek(f); }} /><section className="detail-card"><h3>Processing times</h3><dl>{Object.entries(job?.timings ?? {}).map(([k, v]) => <div className="dl-row" key={k}><dt>{k}</dt><dd>{v.toFixed(1)} s</dd></div>)}</dl>{Object.entries(job?.stage_seconds ?? {}).map(([k, v]) => <small className="block" key={k}>{k}: {v.toFixed(1)} s</small>)}</section></>}
        {details === "quality" && <><section className="detail-card"><h3>Clip rating</h3><div className="quality-score">{q ? `${q.grade} · ${q.score.toFixed(0)}` : "Unrated"}<small>/100</small></div><p>{q ? `${Math.round(q.usable.share * 100)}% usable` : "Quality metadata not available"}</p>{q && Object.entries(q.subscores).map(([k, v]) => <label className="rating-row" key={k}>{k.replace(/_/g, " ")}<meter min={0} max={1} value={v} /></label>)}</section><section className="detail-card"><h3>Quality issues</h3>{q && Object.entries(q.issues).map(([k, v]) => <div className="dl-row" key={k}><span>{k.replace(/_/g, " ")}</span><b>{v.toFixed(1)} s</b></div>)}</section><p className="detail-note">Quality thresholds await calibration on real footage.</p></>}
        {details === "exports" && <section className="detail-card export-card"><h3>Export workspace</h3>
          {job && (job.render.status === "done" ? <a className="btn-small" href={`${base(job.id)}/export/render`} download>Rendered video MP4 ↓</a> : <p>Rendered video: {job.render.status}{job.render.error && ` · ${job.render.error}`}</p>)}
          {job && <><label className="range-row">From frame<input type="number" min={0} max={range[1]} disabled={recording} value={range[0]} onChange={e => setRange([Math.min(range[1], Math.max(0, +e.target.value)), range[1]])} /></label><label className="range-row">To frame<input type="number" min={range[0]} max={clip.n_frames - 1} disabled={recording} value={range[1]} onChange={e => setRange([range[0], Math.max(range[0], Math.min(clip.n_frames - 1, +e.target.value))])} /></label><button className="btn-small" disabled={recording} onClick={() => void recordView()}>{recording ? recordProgress < 1 ? `Recording ${Math.round(recordProgress * 100)}%` : "Converting to MP4…" : "Export my 3D view MP4"}</button></>}
          {["hands.jsonl", "robot_wrist_gripper.csv", "stats.json", ...(!demo ? ["quality.json", "scene.json"] : [])].map(n => <a className="download-link" key={n} href={download(n)} download>{n} ↓</a>)}
        </section>}
      </div>
    </aside>}
  </div>;
}
