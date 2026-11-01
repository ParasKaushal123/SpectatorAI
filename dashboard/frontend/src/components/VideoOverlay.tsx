import { useEffect, useRef, useState, type RefObject } from "react";
import { BONES, COLORS, type ClipData } from "../api";

/** Current frame index of a <video>, updated every displayed frame. */
export function useVideoFrame(video: RefObject<HTMLVideoElement>, fps: number, n: number) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    let raf = 0;
    let last = -1;
    const tick = () => {
      const v = video.current;
      if (v) {
        const f = Math.max(0, Math.min(n - 1, Math.floor(v.currentTime * fps + 1e-3)));
        if (f !== last) { last = f; setFrame(f); }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [video, fps, n]);
  return frame;
}

/** Video with the pipeline's 2D keypoints drawn on a canvas on top. */
export default function VideoOverlay({ clip, videoUrl, videoRef, frame, overlay, frameRef, handColors }: {
  clip: ClipData; videoUrl: string; videoRef: RefObject<HTMLVideoElement>; frame: number; overlay: boolean; frameRef?: RefObject<number>; handColors?: { left: string; right: string };
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    const draw = () => {
    const currentFrame = frameRef?.current ?? frame;
    const cv = canvas.current;
    if (!cv) return;
    const r = cv.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(r.width * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
    const ctx = cv.getContext("2d")!;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!overlay) return;
    const sx = cv.width / clip.width, sy = cv.height / clip.height;
    for (const side of ["left", "right"] as const) {
      const h = clip.hands[side];
      const k = h.kp2d[currentFrame], src = h.src[currentFrame];
      if (!k || !src) continue;
      const est = src === 3, inf = src === 2;
      const col = handColors?.[side] ?? (side === "left" ? COLORS.left : COLORS.right);
      ctx.globalAlpha = est ? 0.75 : inf ? 0.8 : 1;
      ctx.setLineDash(est ? [6 * dpr, 5 * dpr] : []);
      // light halo so the skeleton reads on any footage
      for (const [w, c] of [[6, "rgba(248,249,235,0.85)"], [2.5, est ? COLORS.estimated : col]] as const) {
        ctx.lineWidth = w * dpr;
        ctx.strokeStyle = c;
        ctx.beginPath();
        for (const [a, b] of BONES) {
          ctx.moveTo(k[2 * a] * sx, k[2 * a + 1] * sy);
          ctx.lineTo(k[2 * b] * sx, k[2 * b + 1] * sy);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      for (let j = 0; j < 21; j++) {
        ctx.beginPath();
        ctx.arc(k[2 * j] * sx, k[2 * j + 1] * sy, (j === 0 ? 5 : 3) * dpr, 0, Math.PI * 2);
        ctx.fillStyle = j === 0 ? COLORS.mint : est ? COLORS.estimated : col;
        ctx.fill();
        ctx.lineWidth = 1.2 * dpr;
        ctx.strokeStyle = "#000";
        ctx.stroke();
      }
      // label
      const tag = `${side.toUpperCase()}${est ? " · ESTIMATED" : inf ? " · INFILLED" : ""}`;
      ctx.globalAlpha = 1;
      ctx.font = `${11 * dpr}px "JetBrains Mono", monospace`;
      const x = k[0] * sx + 10 * dpr, y = k[1] * sy + 18 * dpr;
      const w = ctx.measureText(tag).width + 12 * dpr;
      ctx.fillStyle = est ? COLORS.estimated : inf ? COLORS.infilled : COLORS.mint;
      ctx.strokeStyle = "#000";
      ctx.beginPath();
      ctx.roundRect(x, y - 13 * dpr, w, 18 * dpr, 9 * dpr);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#000";
      ctx.fillText(tag, x + 6 * dpr, y);
    }
    ctx.globalAlpha = 1;
    if (frameRef) raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [clip, frame, overlay, frameRef, handColors]);

  return (
    <div className="relative w-full overflow-hidden rounded-cards border-[1.5px] border-obsidian bg-charcoal"
      style={{ aspectRatio: `${clip.width} / ${clip.height}` }}>
      <video ref={videoRef} src={videoUrl} className="absolute inset-0 h-full w-full" muted playsInline preload="auto" />
      <canvas ref={canvas} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  );
}
