import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../workspaceApi";
import RoomViewport from "./RoomViewport";
import { RobotScore, RobotTimeline } from "./RobotReplay";
import { disposeRoom, loadRoom, roomBase, type RoomData, type RoomEntry } from "./roomData";
import "../room.css";

const source = ["Out of view", "Observed", "Infilled", "Estimated"];

export default function RoomPage({ id, panel }: { id: string; panel: boolean }) {
  const [rooms, setRooms] = useState<RoomEntry[]>([]);
  const [listed, setListed] = useState(false);
  const [data, setData] = useState<RoomData | null>(null);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [error, setError] = useState("");
  const [videoError, setVideoError] = useState(false);
  const [display, setDisplay] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showVideo, setShowVideo] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [recording, setRecording] = useState(false);
  const frame = useRef(0), clock = useRef(0);
  const handsCanvas = useRef<HTMLCanvasElement | null>(null);
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null), stream = useRef<MediaStream | null>(null);
  const mounted = useRef(true);
  const entry = rooms.find(r => r.id === id);
  const videoClock = !!entry?.has_video && showVideo && !videoError;
  const onCanvas = useCallback((c: HTMLCanvasElement) => { canvas.current = c; }, []);
  const onHandsCanvas = useCallback((c: HTMLCanvasElement) => { handsCanvas.current = c; canvas.current = c; }, []);
  useEffect(() => { if (!showComparison) canvas.current = handsCanvas.current; }, [showComparison]);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; if (recorder.current?.state === "recording") recorder.current.stop();
      stream.current?.getTracks().forEach(t => t.stop()); };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void request<RoomEntry[]>("/api/rooms", { signal: controller.signal }).then(rows => {
      setRooms(rows); setListed(true);
      if (!id && rows.length) location.hash = `/room/${encodeURIComponent(rows[0].id)}`;
    }).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [id]);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController(); let loaded: RoomData | undefined;
    setData(null); setError(""); setProgress({}); setPlaying(false); setDisplay(0); frame.current = clock.current = 0; setVideoError(false);
    void loadRoom(id, controller.signal, (name, fraction) => {
      if (!controller.signal.aborted) setProgress(p => ({ ...p, [name]: fraction }));
    }).then(d => { loaded = d; if (!controller.signal.aborted) setData(d); else { disposeRoom(d.room); if (d.robot) disposeRoom(d.robot.root); } })
      .catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => { controller.abort(); if (loaded) { disposeRoom(loaded.room); if (loaded.robot) disposeRoom(loaded.robot.root); } };
  }, [id]);
  const seek = useCallback((next: number) => {
    if (!data) return;
    frame.current = Math.max(0, Math.min(data.motion.n - 1, Math.round(next)));
    clock.current = frame.current; setDisplay(frame.current);
    for (const v of [video.current]) if (v) v.currentTime = frame.current / data.motion.fps;
  }, [data]);
  const toggle = useCallback(() => {
    if (data && frame.current >= data.motion.n - 1) seek(0);
    setPlaying(p => !p);
  }, [data, seek]);
  useEffect(() => {
    if (!data) return;
    const videos = [videoClock ? video.current : null].filter((v): v is HTMLVideoElement => !!v);
    for (const v of videos) {
      v.playbackRate = speed;
      v.currentTime = frame.current / data.motion.fps;
      if (playing) void v.play().catch(e => { if (e.name !== "AbortError") { setPlaying(false); setError(`Could not start synchronized playback: ${e}`); } });
      else v.pause();
    }
    return () => { videos.forEach(v => v.pause()); };
  }, [data, playing, videoClock, speed]);
  useEffect(() => {
    if (!data) return;
    let raf = 0, last = performance.now(), ui = 0, synced = 0;
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      const master = videoClock ? video.current : null;
      if (playing) {
        if (master) clock.current = master.currentTime * data.motion.fps;
        else clock.current += dt * data.motion.fps * speed;
        frame.current = Math.max(0, Math.min(data.motion.n - 1, Math.floor(clock.current + 1e-4)));
        if (master && now - synced > 250) {
          for (const follower of [video.current]) {
            if (follower && follower !== master && !follower.seeking && Math.abs(follower.currentTime - master.currentTime) > .1)
              follower.currentTime = master.currentTime;
          }
          synced = now;
        }
        if (clock.current >= data.motion.n - 1 || master?.ended) {
          setPlaying(false); setDisplay(frame.current);
          if (recorder.current?.state === "recording") recorder.current.stop();
        }
      }
      if (now - ui >= 100) { setDisplay(frame.current); ui = now; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [data, playing, speed, videoClock]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (!data || recording || (e.target instanceof HTMLElement && /INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName))) return;
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault(); setPlaying(false);
        seek(frame.current + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? Math.round(data.motion.fps) : 1));
      }
    };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, [data, seek, toggle, recording]);
  function recordView() {
    if (recording) { recorder.current?.stop(); return; }
    try {
      if (!canvas.current || !data) return;
      if (!window.MediaRecorder) throw new Error("Use Chrome or Edge to record the 3D canvas.");
      if (frame.current >= data.motion.n - 1) seek(0);
      stream.current = canvas.current.captureStream(30);
      const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(MediaRecorder.isTypeSupported);
      const rec = new MediaRecorder(stream.current, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 6_000_000 });
      recorder.current = rec; const chunks: BlobPart[] = [];
      rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        stream.current?.getTracks().forEach(t => t.stop()); stream.current = null;
        if (!mounted.current) return;
        setRecording(false); setPlaying(false);
        const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
        const a = document.createElement("a"); a.href = url; a.download = `${id}-my-room-view.webm`; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      };
      rec.onerror = () => { setError("Browser recording failed. Try Chrome or Edge."); setRecording(false); setPlaying(false);
        stream.current?.getTracks().forEach(t => t.stop()); };
      rec.start(250); setRecording(true); setPlaying(true);
    } catch (e) { setError(String(e)); stream.current?.getTracks().forEach(t => t.stop()); }
  }
  const expectedFiles = (entry?.has_robot ? 3 : 0) + 2 + (["left", "right"] as const).filter(s => data?.motion.hands[s].mesh ?? true).length;
  const fraction = Math.min(1, Object.values(progress).reduce((a, b) => a + b, 0) / expectedFiles);
  return <div className={`room-page ${showComparison && panel ? "room-with-panel" : ""}`}>
    {rooms.length > 1 && <label className="room-picker">Room <select aria-label="Choose room" value={id}
      onChange={e => { location.hash = `/room/${encodeURIComponent(e.target.value)}`; }}>{rooms.map(r => <option key={r.id}>{r.id}</option>)}</select></label>}
    {error && <p className="error" role="alert">{error}</p>}
    {listed && !rooms.length && <section className="card-soft room-empty"><h2>Add a room twin</h2>
      <p>Download room_web.zip and unzip its contents into <code>dashboard/results/room_twin/&lt;clip&gt;/web/</code>.</p>
      <p>Place room_stats.json and room_video.mp4 beside the web folder, and the original clip in dashboard/raw_clips/.</p></section>}
    {id && !data && !error && <section className="room-loading" role="status"><h2>Loading your workstation…</h2>
      <progress max={1} value={fraction} /><p>{Math.round(fraction * 100)}% · room scan, hands and robot replay</p>
      {Object.entries(progress).map(([name, amount]) => <small key={name}>{name} · {Math.round(amount * 100)}%<br /></small>)}
    </section>}
    {data && <>
      <div className="room-compare-actions"><button className="btn-small" disabled={recording} aria-pressed={showComparison}
        onClick={() => {
          if (!showComparison) {
            if (frame.current >= data.motion.n - 1) seek(0);
            setPlaying(true);
          }
          setShowComparison(!showComparison);
        }}>{showComparison ? "Hide robot replay" : data.robot ? "Show robot replay" : "Show 3D replay"}</button>
        <button className="btn-small" aria-pressed={showVideo} onClick={() => setShowVideo(!showVideo)}>Original video</button></div>
      <div className={`room-main ${showComparison ? "room-split" : "room-slam-only"}`}>
      <RoomViewport data={data} frame={frame} robot={false} onCanvas={onHandsCanvas} overlay={
        showVideo && <aside className="room-video room-original-overlay"><h2>Original · {id}</h2>
        {entry?.has_video && !videoError ? <video ref={video} src={`${roomBase(id)}/video`} muted playsInline preload="auto"
          onLoadedMetadata={() => { if (video.current) video.current.currentTime = frame.current / data.motion.fps; }}
          onError={() => { setVideoError(true); setError(`Could not decode original video. Check dashboard/raw_clips/${id}.mp4.`); }} /> :
          <p>Original clip unavailable. Put <code>{id}.mp4</code> in <code>dashboard/raw_clips/</code> and reload. The 3D replay works independently.</p>}
        <p>Original camera · synchronized playback</p>
      </aside>
      } />
      {showComparison && <RoomViewport data={data} frame={frame} robot onCanvas={onCanvas} />}
      </div>
      <section className="room-transport" aria-label="Room playback">
        <button className="play-button" onClick={toggle} disabled={recording}>{playing ? "Pause" : "Play"}</button>
        <button className="btn-small" aria-label="Previous room frame" disabled={recording} onClick={() => { setPlaying(false); seek(frame.current - 1); }}>←</button>
        <button className="btn-small" aria-label="Next room frame" disabled={recording} onClick={() => { setPlaying(false); seek(frame.current + 1); }}>→</button>
        <span className="room-time">{(display / data.motion.fps).toFixed(2)} s / {((data.motion.n - 1) / data.motion.fps).toFixed(2)} s · f{display} / {data.motion.n - 1}</span>
        <select aria-label="Room playback speed" value={speed} disabled={recording} onChange={e => setSpeed(Number(e.target.value))}>
          {[.25, .5, 1, 2].map(s => <option key={s} value={s}>{s}×</option>)}
        </select>
        <input aria-label="Room frame" type="range" min={0} max={data.motion.n - 1} step={1} value={display} disabled={recording}
          onChange={e => seek(Number(e.target.value))} />
      </section>
      {showComparison && data.robot && <RobotTimeline info={data.robot.info} frame={display} n={data.motion.n} seek={seek} disabled={recording} />}
      {showComparison && panel && <aside className="room-inspector"><span className="eyebrow">At the playhead</span><h2>Room details</h2>
        {(["left", "right"] as const).map(s => { const h = data.motion.hands[s]; return <section className={`room-hand ${s}`} key={s}>
          <h3>{s} hand</h3><p>{source[h.src[display]] ?? "Unknown"}</p>
          <small>{data.bins[s] && h.has_mesh[display] && h.src[display] ? "MANO surface mesh" : h.src[display] ? "Skeleton fallback" : "Hidden · out of view"}</small>
        </section>; })}
        {data.robot && <RobotScore info={data.robot.info} frame={display} download={`${roomBase(id)}/web/robot_traj.json`} />}
        <section><h3>Scan</h3><p>{(data.stats.web_triangles ?? data.stats.triangles ?? 0).toLocaleString()} triangles</p>
          <p>{data.stats.clips_used?.length ?? 0} clips used</p>
          {Object.entries(data.stats.registration ?? {}).map(([name, r]) => <p key={name}>{name}<br />Fitness {(r.fitness * 100).toFixed(1)}% · RMSE {(r.rmse_m * 1000).toFixed(1)} mm</p>)}
        </section>
        <section><h3>Export</h3>{entry?.has_render ? <a className="btn-small" href={`${roomBase(id)}/render`} download>Download fly-through MP4 ↓</a> : <p>room_video.mp4 is not available.</p>}
          <button className="btn-small" onClick={recordView}>{recording ? "Stop recording" : "Record my view"}</button>
          <small>{recording ? "Recording to the end of the clip…" : "Records your current 3D camera from the playhead to the end as WebM."}</small>
        </section>
      </aside>}
    </>}
  </div>;
}
