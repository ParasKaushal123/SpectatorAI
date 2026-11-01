import { useEffect, useRef, useState } from "react";
import { api, COLORS, SRC_NAMES, type ClipData } from "../api";
import HandViewer3D from "./HandViewer3D";
import Timeline from "./Timeline";
import VideoOverlay, { useVideoFrame } from "./VideoOverlay";

export function ModelBadge({ label, variant }: { label: string; variant: string }) {
  return (
    <span className="pill" style={{ background: variant === "finetuned" ? COLORS.mint : "transparent" }}>
      <i className="inline-block h-[8px] w-[8px] rounded-full bg-obsidian" />
      {variant === "finetuned" ? label : "pretrained baseline · WiLoR"}
    </span>
  );
}

/** Video + 2D overlay + synced 3D viewer(s) + timeline. `compare` adds a second 3D viewer on the same frames. */
export default function ClipPlayer({ clip, compare, start }: {
  clip: ClipData; compare?: ClipData | null; start?: { frame: number; nonce: number } | null;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const frame = useVideoFrame(video, clip.fps, clip.n_frames);
  useEffect(() => {
    const v = video.current;
    if (!v || !start) return;
    const go = () => { v.pause(); v.currentTime = (start.frame + 0.5) / clip.fps; };
    if (v.readyState >= 1) go();
    else v.addEventListener("loadedmetadata", go, { once: true });
  }, [start?.nonce, clip.key]);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [overlay, setOverlay] = useState(!clip.overlay_baked);
  const [path, setPath] = useState(true);

  useEffect(() => { setOverlay(!clip.overlay_baked); setPlaying(false); }, [clip.key, clip.overlay_baked]);
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    const on = () => setPlaying(!v.paused);
    v.addEventListener("play", on);
    v.addEventListener("pause", on);
    return () => { v.removeEventListener("play", on); v.removeEventListener("pause", on); };
  }, [clip.key]);
  useEffect(() => { if (video.current) video.current.playbackRate = rate; }, [rate]);

  const seek = (f: number) => {
    const v = video.current;
    if (v) v.currentTime = (Math.max(0, Math.min(clip.n_frames - 1, f)) + 0.5) / clip.fps;
  };
  const toggle = () => {
    const v = video.current;
    if (v) (v.paused ? v.play() : v.pause());
  };
  const L = clip.hands.left, R = clip.hands.right;

  return (
    <div className="flex flex-col gap-13">
      <div className={`grid gap-13 ${compare ? "lg:grid-cols-3" : "lg:grid-cols-[1.35fr_1fr]"}`}>
        <div className="flex flex-col gap-9">
          <VideoOverlay clip={clip} videoUrl={api.videoUrl(clip.key)} videoRef={video} frame={frame} overlay={overlay} />
          {clip.overlay_baked && (
            <p className="text-body-sm text-olive-slate">
              Raw footage for this clip isn't bundled, so this is the video panel of the pipeline's own A100 render (its overlay is baked in;
              grey boxes are candidates it rejected). Toggle “our overlay” to draw the exported keypoints on top.
            </p>
          )}
        </div>
        <div className="flex flex-col gap-9">
          <HandViewer3D clip={clip} frame={frame} showPath={path} height={compare ? 300 : 420} />
          <span className="label">{compare ? `A · ${clip.model_label}` : clip.model_label}</span>
        </div>
        {compare && (
          <div className="flex flex-col gap-9">
            <HandViewer3D clip={compare} frame={Math.min(frame, compare.n_frames - 1)} showPath={path} height={300} />
            <span className="label">B · {compare.model_label}</span>
          </div>
        )}
      </div>

      <div className="card-soft flex flex-wrap items-center gap-13 bg-pure-white py-13">
        <button className="btn min-w-[96px]" onClick={toggle}>{playing ? "Pause" : "Play"}</button>
        <input type="range" min={0} max={clip.n_frames - 1} value={frame} onChange={(e) => seek(Number(e.target.value))}
          className="min-w-[160px] flex-1 accent-[#202020]" aria-label="Scrub" />
        <span className="font-mono text-caption tabular-nums">
          {(frame / clip.fps).toFixed(2)}s · f{frame}/{clip.n_frames - 1}
        </span>
        <div className="flex gap-4">
          {[0.25, 0.5, 1].map((r) => (
            <button key={r} className="btn-ghost px-9 py-6 text-body-sm" aria-pressed={rate === r} onClick={() => setRate(r)}>{r}×</button>
          ))}
        </div>
        <button className="btn-ghost px-13 py-6 text-body-sm" aria-pressed={overlay} onClick={() => setOverlay(!overlay)}>
          {clip.overlay_baked ? "our overlay" : "2D overlay"}
        </button>
        <button className="btn-ghost px-13 py-6 text-body-sm" aria-pressed={path} onClick={() => setPath(!path)}>wrist path</button>
      </div>

      <div className="flex flex-wrap gap-9">
        {(["left", "right"] as const).map((s) => {
          const h = s === "left" ? L : R;
          return (
            <span key={s} className="pill" style={{ background: h.src[frame] === 3 ? COLORS.estimated : h.src[frame] === 2 ? COLORS.infilled : "transparent" }}>
              {s}: {h.src[frame] ? SRC_NAMES[h.src[frame]] : "out of view"} · conf {h.conf[frame].toFixed(2)} · {h.state[frame].toLowerCase().replace(/_/g, " ")}
            </span>
          );
        })}
      </div>

      <Timeline clip={clip} frame={frame} onSeek={seek} />

      <div className="flex flex-wrap gap-9">
        <a className="btn-ghost text-body-sm" href={api.downloadUrl(clip.key, "hands.jsonl")}>↓ hands.jsonl (2D + 3D joints)</a>
        <a className="btn-ghost text-body-sm" href={api.downloadUrl(clip.key, "robot_wrist_gripper.csv")}>↓ robot wrist + gripper CSV</a>
        <a className="btn-ghost text-body-sm" href={api.downloadUrl(clip.key, "stats.json")}>↓ stats.json</a>
      </div>
    </div>
  );
}
