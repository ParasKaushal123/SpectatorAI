import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, COLORS, pct, type ClipData, type ClipSummary, type Job, type Metrics, type ModelInfo, type Training } from "../api";
import ClipPlayer, { ModelBadge } from "./ClipPlayer";

const tick = { fontSize: 11, fontFamily: "JetBrains Mono", fill: COLORS.pebble };
const tip = { background: COLORS.cream, border: "1.5px solid #000", borderRadius: 13, fontFamily: "Space Grotesk", fontSize: 13 };
const short = (n: string) => n.replace("hand_tracking_", "clip ");

function useClip(key: string | null) {
  const [clip, setClip] = useState<ClipData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!key) return;
    let alive = true;
    setErr(null);
    api.clip(key).then((c) => alive && setClip(c)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [key]);
  return { clip: clip && clip.key === key ? clip : null, err };
}

// ---- Results explorer -------------------------------------------------------------------------------
export function Explorer({ clips, selected, onSelect, start }: {
  clips: ClipSummary[]; selected: string | null; onSelect: (k: string) => void; start: { frame: number; nonce: number } | null;
}) {
  const { clip, err } = useClip(selected);
  const a100 = clips.filter((c) => c.origin === "a100");
  const uploads = clips.filter((c) => c.origin === "upload");
  return (
    <section id="explorer" className="wrap flex flex-col gap-27 py-72">
      <span className="pill self-start">Results explorer · real A100 runs</span>
      <h2 className="display text-[44px] md:text-display">Scrub any clip. Orbit the hands.</h2>
      <div className="flex flex-wrap items-center gap-6">
        {a100.map((c) => (
          <button key={c.key} className="btn-ghost px-13 py-9 text-body-sm" aria-pressed={c.key === selected} onClick={() => onSelect(c.key)}>
            {short(c.name)}{c.variant === "finetuned" ? " · fine-tuned" : ""}
          </button>
        ))}
        {uploads.map((c) => (
          <button key={c.key} className="btn-ghost px-13 py-9 text-body-sm" aria-pressed={c.key === selected} onClick={() => onSelect(c.key)}>
            ↑ {c.name}
          </button>
        ))}
      </div>
      {clip && (
        <div className="flex flex-wrap items-center gap-9">
          <ModelBadge label={clip.model_label} variant={clip.variant} />
          <span className="pill">{clip.origin === "a100" ? "A100 · Colab" : "live upload · CPU"}</span>
          <span className="pill">{clip.n_frames} frames · {clip.fps} fps · {clip.width}×{clip.height}</span>
          <span className="pill">camera: {clip.camera_source}</span>
        </div>
      )}
      {err && <p className="card-soft bg-rose-wash">Couldn’t load this clip: {err}</p>}
      {!clip && !err && <div className="card-soft h-[420px] animate-pulse bg-bone" />}
      {clip && <ClipPlayer clip={clip} start={start} />}
    </section>
  );
}

// ---- Pretrained vs fine-tuned -------------------------------------------------------------------------
export function Compare({ m }: { m: Metrics | null }) {
  const pairs = m?.paired_clips ?? [];
  const [name, setName] = useState<string | null>(null);
  useEffect(() => { if (pairs.length && !name) setName(pairs[0]); }, [pairs.join()]);
  const A = useClip(name ? `pretrained__${name}` : null).clip;
  const B = useClip(name ? `finetuned__${name}` : null).clip;
  const pre = m?.aggregate.pretrained, ft = m?.aggregate.finetuned;
  const rowsOf = (k: string, f: (x: any) => string, better: "up" | "down") => {
    const a = pre ? (pre as any)[k] : null, b = ft ? (ft as any)[k] : null;
    const win = a != null && b != null && (better === "up" ? b > a : b < a);
    return { a: a != null ? f(a) : "—", b: b != null ? f(b) : "awaiting", win };
  };
  const table = [
    ["Hand-frames measured in 3D", rowsOf("observed_mean", (v) => pct(v, 1), "up")],
    ["Out-of-view estimates", rowsOf("estimated_mean", (v) => pct(v, 1), "down")],
    ["Wrist jitter (median, mm/frame²)", rowsOf("jitter_mm_median", (v) => String(v), "down")],
    ["Depth jumps > 10 cm / 1k frames", rowsOf("depth_jumps_per_1k_frames", (v) => String(v), "down")],
    ["Candidate tracks rejected", rowsOf("reject_rate", (v) => pct(v), "up")],
  ] as const;
  return (
    <section id="compare" className="wrap flex flex-col gap-27 pb-72">
      <span className="pill self-start bg-mint-sprout">Headline · pretrained vs fine-tuned</span>
      <h2 className="display text-[44px] md:text-display">Same clips. Two models.<br />Side by side.</h2>
      <p className="max-w-[720px] text-subheading text-charcoal">
        WiLoR fine-tuned on our 40–50 egocentric factory clips, run through the identical PS4 pipeline on the identical clips.
        Only the 3D model changes.
      </p>
      <div className="card overflow-x-auto bg-pure-white">
        <table className="w-full min-w-[520px] text-left">
          <thead className="font-mono text-caption uppercase text-pebble">
            <tr><th className="pb-9">metric (mean over clips)</th><th className="pb-9">A · pretrained WiLoR</th><th className="pb-9">B · fine-tuned (ours)</th></tr>
          </thead>
          <tbody>
            {table.map(([label, r]) => (
              <tr key={label} className="border-t border-bone">
                <td className="py-9 font-medium">{label}</td>
                <td className="py-9">{r.a}</td>
                <td className="py-9"><span className={r.win ? "rounded-badges bg-mint-sprout px-9 py-4 font-bold" : ""}>{r.b}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-13 font-mono text-caption text-pebble">
          A: {pre?.clips ?? 0} clips in results/ · B: {ft?.clips ?? 0} clips in results/finetuned/ · {pairs.length} clips in both
        </p>
      </div>
      {!ft || !pairs.length ? (
        <div className="flex flex-col gap-13 rounded-cards bg-bone p-27 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-col gap-6">
            <span className="text-heading-sm font-bold">Fine-tuned run in progress</span>
            <span className="text-body-sm text-olive-slate">
              Training on a Colab A100. When the checkpoint lands in <code className="font-mono">weights/</code> and its runs in{" "}
              <code className="font-mono">results/finetuned/&lt;clip&gt;/</code>, this section fills in automatically: metrics above, and both
              models’ 3D hands playing in sync below. Until then everything on this page is the pretrained baseline.
            </span>
          </div>
          <span className="pill shrink-0 bg-pure-white">status: awaiting checkpoint</span>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-6">
            {pairs.map((p) => (
              <button key={p} className="btn-ghost px-13 py-9 text-body-sm" aria-pressed={p === name} onClick={() => setName(p)}>{short(p)}</button>
            ))}
          </div>
          {A && B ? <ClipPlayer clip={A} compare={B} /> : <div className="card-soft h-[300px] animate-pulse bg-bone" />}
        </>
      )}
    </section>
  );
}

// ---- Training -------------------------------------------------------------------------------------------
const SERIES_COLORS = [COLORS.left, COLORS.right, COLORS.observed, "#7a5cc8", COLORS.pebble];

export function TrainingSection({ t }: { t: Training | null }) {
  const setup = t?.setup ?? null;
  const ds = t?.dataset ?? null;
  const facts: [string, any][] = [
    ["Base model", setup?.base_model ?? "WiLoR (pretrained)"],
    ["GPU", setup?.gpu ?? setup?.hardware ?? "A100 (Colab)"],
    ["Clips", ds?.clips ?? setup?.clips ?? "40–50"],
    ["Footage", ds?.minutes ? `${ds.minutes} min` : setup?.minutes ? `${setup.minutes} min` : "~3 min per clip"],
    ["Epochs", setup?.epochs ?? "TO VERIFY"],
    ["Learning rate", setup?.lr ?? setup?.learning_rate ?? "TO VERIFY"],
    ["Batch size", setup?.batch_size ?? "TO VERIFY"],
    ["Labels", ds?.labels ?? setup?.labels ?? "TO VERIFY"],
  ];
  return (
    <section id="training" className="wrap flex flex-col gap-27 pb-72">
      <span className="pill self-start">Training · fine-tuning WiLoR on egocentric clips</span>
      <h2 className="display text-[44px] md:text-display">What we trained, and how.</h2>
      <div className="grid gap-13 sm:grid-cols-2 lg:grid-cols-4">
        {facts.map(([k, v]) => (
          <div key={k} className="card-soft flex flex-col gap-6">
            <span className="label">{k}</span>
            <span className={`text-heading-sm font-bold ${String(v).includes("TO VERIFY") ? "text-pebble" : ""}`}>{String(v)}</span>
          </div>
        ))}
      </div>
      {t?.curves?.length ? (
        t.curves.map((c) => (
          <div key={c.file} className="card-soft flex flex-col gap-13 bg-pure-white">
            <div className="flex flex-wrap items-baseline justify-between gap-9">
              <h3 className="text-heading-sm font-bold">Loss curves</h3>
              <span className="font-mono text-caption text-pebble">results/training/{c.file}</span>
            </div>
            <div className="h-[300px]">
              <ResponsiveContainer>
                <LineChart data={c.points} margin={{ left: -6, right: 12 }}>
                  <CartesianGrid stroke={COLORS.none} />
                  <XAxis dataKey="x" tick={tick} stroke={COLORS.sage} label={{ value: c.x, position: "insideBottomRight", offset: -2, style: tick }} />
                  <YAxis tick={tick} stroke={COLORS.sage} />
                  <Tooltip contentStyle={tip} />
                  <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                  {c.series.map((s, i) => (
                    <Line key={s} dataKey={s} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))
      ) : (
        <div className="rounded-cards bg-bone p-27 text-body-sm text-olive-slate">
          Loss curves appear here when the training logs are dropped into <code className="font-mono">results/training/</code>{" "}
          (any CSV with an <code className="font-mono">epoch</code> or <code className="font-mono">step</code> column and one or more
          <code className="font-mono"> *loss*</code> columns, plus optional <code className="font-mono">training.json</code> /{" "}
          <code className="font-mono">dataset.json</code>). Values marked TO VERIFY come from those files.
        </div>
      )}
    </section>
  );
}

// ---- Live upload ----------------------------------------------------------------------------------------
export function Upload({ model, onDone, onDemo }: { model: ModelInfo | null; onDone: (key: string) => void; onDemo: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [camera, setCamera] = useState<"phone" | "fisheye">("phone");
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now() / 1000);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    timer.current = window.setInterval(async () => {
      setNow(Date.now() / 1000);
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (j.status === "done" && j.clip_key) onDone(j.clip_key);
      } catch { /* keep polling */ }
    }, 1000);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [job?.id, job?.status]);

  const start = async () => {
    if (!file) return;
    setErr(null);
    try { setJob(await api.upload(file, camera)); } catch (e: any) { setErr(e.message ?? String(e)); }
  };
  const elapsed = job?.started ? Math.max(0, now - job.started) : 0;
  const etaMin = job?.eta_s ? Math.max(1, Math.round(job.eta_s / 60)) : null;
  const busy = job && (job.status === "queued" || job.status === "running");
  const dropRef = useRef<HTMLLabelElement>(null);

  return (
    <section id="upload" className="wrap flex flex-col gap-27 pb-72">
      <span className="pill self-start">Live · your video, this laptop</span>
      <h2 className="display text-[44px] md:text-display">Try it on your own clip.</h2>
      <div className="grid gap-13 lg:grid-cols-[1.2fr_1fr]">
        <div className="card flex flex-col gap-18">
          <label ref={dropRef} className="flex cursor-pointer flex-col items-center gap-9 rounded-cards border-[1.5px] border-dashed border-sage-mist bg-pure-white p-36 text-center hover:bg-bone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) setFile(f); }}>
            <span className="text-heading-sm font-bold">{file ? file.name : "Drop a video or click to choose"}</span>
            <span className="text-body-sm text-olive-slate">{file ? `${(file.size / 1e6).toFixed(1)} MB` : "mp4 / mov · first 10 s are used · hands in view work best"}</span>
            <input type="file" accept="video/*" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <fieldset className="flex flex-col gap-9">
            <legend className="label mb-6">Camera</legend>
            <div className="flex flex-wrap gap-9">
              <button className="btn-ghost" aria-pressed={camera === "phone"} onClick={() => setCamera("phone")}>Phone / normal camera</button>
              <button className="btn-ghost" aria-pressed={camera === "fisheye"} onClick={() => setCamera("fisheye")}>Fisheye head-cam</button>
            </div>
            <span className="text-caption text-olive-slate">
              {camera === "phone" ? "Uses a pinhole model (~68° horizontal FOV) instead of the fisheye lens model." : "Uses the pipeline's equidistant fisheye model (150° FOV guess; GeoCalib is skipped on CPU)."}
            </span>
          </fieldset>
          <div className="flex flex-wrap items-center gap-13">
            <button className="btn" disabled={!file || !!busy} onClick={start}>{busy ? "Processing…" : "Run the pipeline"}</button>
            <span className="text-caption text-olive-slate">Model: {model?.label ?? "…"}{model?.variant !== "finetuned" ? " (pretrained baseline)" : ""}</span>
          </div>
          {err && <p className="rounded-cards bg-rose-wash p-18 text-body-sm">{err}</p>}
        </div>
        <div className="card-soft flex flex-col gap-13">
          <span className="label">What happens</span>
          <ol className="flex list-decimal flex-col gap-6 pl-18 text-body-sm">
            <li>Trimmed to 10 s, resampled to 8 fps, long side 960 px.</li>
            <li>The full PS4 pipeline runs on this laptop’s CPU — no GPU, no cloud.</li>
            <li>Result opens in the explorer above: video, 2D overlay, 3D hands, timeline.</li>
          </ol>
          {job && (
            <div className="flex flex-col gap-9 rounded-cards bg-pure-white p-18" aria-live="polite">
              <div className="flex items-baseline justify-between gap-9">
                <span className="font-medium">{job.status === "error" ? "Failed" : job.status === "done" ? "Done" : job.stage}</span>
                <span className="font-mono text-caption">{Math.round(job.progress * 100)}%</span>
              </div>
              <div className="h-[12px] overflow-hidden rounded-badges border-[1.5px] border-obsidian">
                <div className="h-full bg-forest-depths transition-all" style={{ width: `${job.progress * 100}%` }} />
              </div>
              {busy && (
                <span className="text-caption text-olive-slate">
                  Processing on CPU{etaMin ? `, about ${etaMin} min` : ""} · {Math.round(elapsed)} s elapsed{job.frames ? ` · ${job.frames} frames` : ""}
                </span>
              )}
              {job.status === "done" && <span className="text-body-sm">Opened in the explorer ↑</span>}
              {job.status === "error" && (
                <div className="flex flex-col gap-9">
                  <p className="text-caption text-wine-shadow">{job.error}</p>
                  <button className="btn self-start" onClick={onDemo}>Show the A100 demo instead</button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function useOnce<T>(fn: () => Promise<T>) {
  const [v, setV] = useState<T | null>(null);
  useEffect(() => { fn().then(setV).catch(() => setV(null)); }, []);
  return v;
}
