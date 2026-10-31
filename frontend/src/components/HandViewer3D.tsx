import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { BONES, COLORS, type ClipData } from "../api";

type V3 = [number, number, number];
const SIDES = ["left", "right"] as const;
const OPACITY = [0, 1, 0.55, 0.35]; // by source: none, observed, infilled, estimated

/** Clip-level centre (median observed wrist) so the scene stays put while the hands move. */
function useCentre(clip: ClipData): V3 {
  return useMemo(() => {
    const xs: number[] = [], ys: number[] = [], zs: number[] = [];
    for (const s of SIDES) {
      const h = clip.hands[s];
      h.kp3d.forEach((k, i) => {
        if (k && h.src[i] === 1) {
          xs.push(k[0]); ys.push(k[1]); zs.push(k[2]);
        }
      });
    }
    const med = (a: number[]) => (a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 0);
    return [med(xs), med(ys), med(zs)];
  }, [clip]);
}

/** Per-frame view centre: mean wrist position over ±1 s (like the pipeline's render), relative to the clip centre. */
function useFollow(clip: ClipData, c: V3): V3[] {
  return useMemo(() => {
    const n = clip.n_frames, w = Math.round(clip.fps || 30);
    const pts: (V3 | null)[] = Array.from({ length: n }, (_, i) => {
      const ws = SIDES.map((s) => clip.hands[s]).filter((h) => h.kp3d[i] && h.src[i] > 0).map((h) => toV(h.kp3d[i]!, 0, c));
      return ws.length ? [ws.reduce((a, p) => a + p[0], 0) / ws.length, ws.reduce((a, p) => a + p[1], 0) / ws.length,
        ws.reduce((a, p) => a + p[2], 0) / ws.length] : null;
    });
    const out: V3[] = [];
    let last: V3 = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      let sx = 0, sy = 0, sz = 0, k = 0;
      for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) {
        const p = pts[j];
        if (p) { sx += p[0]; sy += p[1]; sz += p[2]; k++; }
      }
      if (k) last = [sx / k, sy / k, sz / k];
      out.push(last);
    }
    return out;
  }, [clip, c]);
}

// head-stable frame (x right, y down, z forward) -> three.js (x right, y up, looking down -z)
const toV = (k: number[], j: number, c: V3): V3 => [k[3 * j] - c[0], -(k[3 * j + 1] - c[1]), -(k[3 * j + 2] - c[2])];

function Hand({ kp, src, color, c }: { kp: number[]; src: number; color: string; c: V3 }) {
  const pts = useMemo(() => BONES.flatMap(([a, b]) => [toV(kp, a, c), toV(kp, b, c)]), [kp, c]);
  const joints = useMemo(() => Array.from({ length: 21 }, (_, j) => toV(kp, j, c)), [kp, c]);
  const op = OPACITY[src];
  const estimated = src === 3;
  return (
    <group>
      <Line points={pts} segments color={estimated ? COLORS.pebble : color} lineWidth={estimated ? 2 : 3.5}
        transparent opacity={op} dashed={estimated} dashSize={0.008} gapSize={0.006} />
      {joints.map((p, j) => (
        <mesh key={j} position={p}>
          <sphereGeometry args={[j === 0 ? 0.009 : 0.0055, 12, 12]} />
          <meshStandardMaterial color={j === 0 ? COLORS.mint : estimated ? COLORS.estimated : color}
            transparent opacity={op} />
        </mesh>
      ))}
    </group>
  );
}

/** Wrist path: the whole clip faint, the last `trailFrames` in the hand's colour. */
function Trajectory({ clip, side, frame, c, trailFrames }: {
  clip: ClipData; side: "left" | "right"; frame: number; c: V3; trailFrames: number;
}) {
  const h = clip.hands[side];
  const segs = useMemo(() => {
    const out: V3[][] = [];
    let cur: V3[] = [];
    h.kp3d.forEach((k, i) => {
      if (k && h.src[i] > 0) cur.push(toV(k, 0, c));
      else if (cur.length) { out.push(cur); cur = []; }
    });
    if (cur.length) out.push(cur);
    return out.filter((s) => s.length > 1);
  }, [h, c]);
  const trail = useMemo(() => {
    const pts: V3[] = [];
    for (let i = Math.max(0, frame - trailFrames); i <= frame; i++) {
      const k = h.kp3d[i];
      if (k && h.src[i] > 0) pts.push(toV(k, 0, c));
    }
    return pts;
  }, [h, frame, c, trailFrames]);
  const color = side === "left" ? COLORS.left : COLORS.right;
  return (
    <group>
      {segs.map((s, i) => <Line key={i} points={s} color={COLORS.sage} lineWidth={1} transparent opacity={0.3} />)}
      {trail.length > 1 && <Line points={trail} color={color} lineWidth={2.5} />}
    </group>
  );
}

export default function HandViewer3D({ clip, frame, showPath = true, height = 420 }: {
  clip: ClipData; frame: number; showPath?: boolean; height?: number;
}) {
  const c = useCentre(clip);
  const follow = useFollow(clip, c)[Math.min(frame, clip.n_frames - 1)] ?? [0, 0, 0];
  const shift: V3 = [-follow[0], -follow[1], -follow[2]];
  const head = toV([0, 0, 0], 0, c);
  const trailFrames = Math.round((clip.fps || 30) * 1.5);
  return (
    <div style={{ height }} className="relative overflow-hidden rounded-cards border-[1.5px] border-obsidian bg-pure-white">
      <Canvas camera={{ position: [0.05, 0.25, 0.68], fov: 40, near: 0.01, far: 20 }} dpr={[1, 2]}>
        <color attach="background" args={[COLORS.cream]} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[1, 2, 2]} intensity={1.2} />
        <gridHelper args={[2, 20, COLORS.sage, "#e3e5cf"]} position={[0, -0.35, 0]} />
        <group position={shift}>
        {/* wearer's head (origin of the head-stable frame) */}
        <mesh position={head}>
          <boxGeometry args={[0.05, 0.035, 0.035]} />
          <meshStandardMaterial color={COLORS.charcoal} />
        </mesh>
        {SIDES.map((s) => {
          const h = clip.hands[s];
          const k = h.kp3d[frame];
          return (
            <group key={s}>
              {showPath && <Trajectory clip={clip} side={s} frame={frame} c={c} trailFrames={trailFrames} />}
              {k && h.src[frame] > 0 && (
                <Hand kp={k} src={h.src[frame]} color={s === "left" ? COLORS.left : COLORS.right} c={c} />
              )}
            </group>
          );
        })}
        </group>
        <OrbitControls makeDefault target={[0, 0, 0]} enableDamping />
      </Canvas>
      <div className="pointer-events-none absolute left-13 top-13 flex flex-wrap gap-6">
        <span className="pill-sm bg-cream-parchment"><i className="inline-block h-[8px] w-[8px] rounded-full" style={{ background: COLORS.left }} />left</span>
        <span className="pill-sm bg-cream-parchment"><i className="inline-block h-[8px] w-[8px] rounded-full" style={{ background: COLORS.right }} />right</span>
        <span className="pill-sm bg-cream-parchment">▪ head</span>
        <span className="pill-sm bg-cream-parchment">metric 3D · metres</span>
      </div>
      <div className="pointer-events-none absolute bottom-9 right-13 font-mono text-[10px] text-pebble">drag to orbit · scroll to zoom</div>
    </div>
  );
}
