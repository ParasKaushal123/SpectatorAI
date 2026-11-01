import { useCallback, useEffect, useState } from "react";
import { api, type ClipSummary } from "./api";
import { HowItWorks, Hero, Results, USPs, type Evidence } from "./components/Story";
import { Compare, Explorer, TrainingSection, Upload, useOnce } from "./components/Sections";
import Safe from "./components/Safe";

const DEFAULT_DEMO = "pretrained__hand_tracking_5"; // best A100 clip: both hands ~90% observed, 0 depth jumps

export default function App() {
  const model = useOnce(api.model);
  const metrics = useOnce(api.metrics);
  const training = useOnce(api.training);
  const evidence = useOnce(() => fetch("/api/evidence").then((r) => r.json() as Promise<Evidence>));
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [start, setStart] = useState<{ frame: number; nonce: number } | null>(null);
  const [bar, setBar] = useState(true);

  const refreshClips = useCallback(() => api.clips().then((cs) => {
    setClips(cs);
    setSelected((s) => s ?? (cs.find((c) => c.key === DEFAULT_DEMO) ?? cs[0])?.key ?? null);
  }), []);
  useEffect(() => { refreshClips(); }, [refreshClips]);

  const jump = (key: string, frame: number) => {
    setSelected(key);
    setStart({ frame, nonce: Date.now() });
    window.location.hash = `/workspace/demo:${encodeURIComponent(key)}?frame=${frame}`;
  };
  const finetuned = model?.variant === "finetuned";
  const modelLabel = finetuned ? model!.label : "Pretrained baseline";

  return (
    <div className="min-h-screen">
      {bar && (
        <div className="flex items-center justify-center gap-13 bg-charcoal px-[16px] py-9 text-center font-mono text-caption uppercase tracking-[0.06em] text-cream-parchment">
          <span>PS4 pipeline · built on pretrained WiLoR · running: {model ? (finetuned ? `fine-tuned checkpoint (${model.wilor_ckpt ?? model.detector})` : "pretrained baseline") : "…"}</span>
          <button aria-label="Close announcement" className="cursor-pointer text-cream-parchment" onClick={() => setBar(false)}>×</button>
        </div>
      )}
      <header className="wrap flex flex-wrap items-center justify-between gap-18 py-18">
        <a href="#" className="text-heading-sm font-bold text-obsidian no-underline">SpectatorAI</a>
        <nav className="hidden gap-27 text-body font-medium text-charcoal md:flex">
          <a href="#how">How it works</a>
          <a href="#results">Results</a>
          <a href="#explorer">Explorer</a>
          <a href="#training">Training</a>
        </nav>
        <div className="flex items-center gap-13">
          <span className={`pill-sm ${finetuned ? "bg-mint-sprout" : ""}`}>{modelLabel}</span>
          <a className="btn" href="#upload">Try your video</a>
        </div>
      </header>

      <main>
        <Safe name="Hero"><Hero m={metrics} modelLabel={modelLabel} /></Safe>
        <HowItWorks />
        <Safe name="USPs"><USPs m={metrics} ev={evidence} onJump={jump} /></Safe>
        <Safe name="Explorer"><Explorer clips={clips} selected={selected} onSelect={(k) => { setSelected(k); setStart(null); }} start={start} /></Safe>
        <Safe name="Comparison"><Compare m={metrics} /></Safe>
        <Safe name="Results"><Results m={metrics} ev={evidence} /></Safe>
        <Safe name="Training"><TrainingSection t={training} /></Safe>
        <Safe name="Upload"><Upload model={model} onDone={(k) => refreshClips().then(() => jump(k, 0))} onDemo={() => jump(DEFAULT_DEMO, 0)} /></Safe>
      </main>

      <footer className="border-t-[1.5px] border-sage-mist">
        <div className="wrap flex flex-wrap items-center justify-between gap-13 py-27 text-body-sm text-olive-slate">
          <span className="font-bold text-obsidian">SpectatorAI</span>
          <span>PS4 pipeline (code/ps4v4) built on pretrained WiLoR + CLIP · numbers read from results/ at load time · “estimated” = out-of-view guess, never a measurement</span>
        </div>
      </footer>
    </div>
  );
}
