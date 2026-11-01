import { useMemo } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { COLORS, SRC_NAMES, type ClipData } from "../api";

const SRC_COLOR = [COLORS.none, COLORS.observed, COLORS.infilled, COLORS.estimated];

function SourceStrip({ src, n, onSeek, label }: { src: number[]; n: number; onSeek: (f: number) => void; label: string }) {
  const runs = useMemo(() => {
    const out: { s: number; a: number; b: number }[] = [];
    src.forEach((s, i) => {
      const last = out[out.length - 1];
      if (last && last.s === s) last.b = i;
      else out.push({ s, a: i, b: i });
    });
    return out;
  }, [src]);
  return (
    <div className="flex items-center gap-9">
      <span className="label w-[44px] shrink-0">{label}</span>
      <div className="relative h-[14px] flex-1 cursor-pointer overflow-hidden rounded-badges border-[1.5px] border-obsidian"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(Math.round(((e.clientX - r.left) / r.width) * (n - 1)));
        }}>
        {runs.map((r, i) => (
          <div key={i} title={`${SRC_NAMES[r.s]} · frames ${r.a}–${r.b}`} className="absolute top-0 h-full"
            style={{ left: `${(r.a / n) * 100}%`, width: `${((r.b - r.a + 1) / n) * 100}%`, background: SRC_COLOR[r.s] }} />
        ))}
      </div>
    </div>
  );
}

export default function Timeline({ clip, frame, onSeek }: { clip: ClipData; frame: number; onSeek: (f: number) => void }) {
  const n = clip.n_frames;
  const data = useMemo(() => {
    const step = Math.max(1, Math.floor(n / 500));
    const out = [];
    for (let i = 0; i < n; i += step) {
      const L = clip.hands.left, R = clip.hands.right;
      out.push({ f: i, left: L.src[i] ? L.conf[i] : null, right: R.src[i] ? R.conf[i] : null });
    }
    return out;
  }, [clip, n]);
  return (
    <div className="card-soft flex flex-col gap-9 bg-pure-white">
      <div className="flex flex-wrap items-center justify-between gap-9">
        <span className="font-medium">Per-frame confidence and source</span>
        <div className="flex flex-wrap gap-6">
          {SRC_NAMES.map((s, i) => (
            <span key={s} className="pill-sm"><i className="inline-block h-[8px] w-[14px] rounded-badges border border-obsidian" style={{ background: SRC_COLOR[i] }} />{s === "none" ? "not tracked" : s}</span>
          ))}
        </div>
      </div>
      <div className="h-[130px] w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -24 }}
            onClick={(e: any) => e?.activeLabel != null && onSeek(Number(e.activeLabel))}>
            <XAxis dataKey="f" type="number" domain={[0, n - 1]} tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: COLORS.pebble }}
              tickFormatter={(f) => `${(f / clip.fps).toFixed(0)}s`} stroke={COLORS.sage} />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: COLORS.pebble }} stroke={COLORS.sage} />
            <Tooltip formatter={(v: any) => (v == null ? "—" : Number(v).toFixed(2))} labelFormatter={(f) => `frame ${f} · ${(Number(f) / clip.fps).toFixed(2)} s`}
              contentStyle={{ background: COLORS.cream, border: "1.5px solid #000", borderRadius: 13, fontFamily: "Space Grotesk" }} />
            <Line type="monotone" dataKey="left" stroke={COLORS.left} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls={false} />
            <Line type="monotone" dataKey="right" stroke={COLORS.right} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls={false} />
            <ReferenceLine x={frame} stroke="#000" strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <SourceStrip src={clip.hands.left.src} n={n} onSeek={onSeek} label="left" />
      <SourceStrip src={clip.hands.right.src} n={n} onSeek={onSeek} label="right" />
    </div>
  );
}
