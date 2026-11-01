import { useEffect, useRef, useState } from "react";
import { api, type Metrics, type Training } from "./api";
import Icon, { BrandMark } from "./components/Icon";
import "./landing.css";

const preview = "/api/preview-assets/";
const movie = "d319f4fb3eae2c89ba3257c3ea98d3b3.mp4";
const render = "9f6835e33b5ed1915ba957da4e4f9c25.mp4";

export default function Landing({ onUpload }: { onUpload: () => void }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [training, setTraining] = useState<Training | null>(null);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(15);
  const [playing, setPlaying] = useState(true);
  const [variant, setVariant] = useState("pretrained");
  const video = useRef<HTMLVideoElement>(null), scene = useRef<HTMLVideoElement>(null);
  useEffect(() => { void api.metrics().then(setMetrics).catch(() => {}); void api.training().then(setTraining).catch(() => {}); }, []);
  useEffect(() => {
    const timer = setInterval(() => {
      const a = video.current, b = scene.current;
      if (a) { setTime(a.currentTime); setPlaying(!a.paused); if (b && Math.abs(a.currentTime - b.currentTime) > .15) b.currentTime = a.currentTime; }
    }, 100);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    const id = location.hash.slice(1);
    if (["demo", "compare", "training", "usps", "top"].includes(id)) {
      const raf = requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
      return () => cancelAnimationFrame(raf);
    }
  }, []);
  const current = variant === "finetuned" ? metrics?.aggregate.finetuned : metrics?.aggregate.pretrained;
  const baseline = metrics?.aggregate.pretrained, tuned = metrics?.aggregate.finetuned;
  const pct = (n: number | null | undefined) => n == null ? "—" : `${(n * 100).toFixed(1)}%`;
  const kpis = [
    ["Wearer's hands tracked", pct(current?.observed_mean), "Observed hand coverage · saved results"],
    ["False candidates rejected", pct(current?.reject_rate), "Tracklets rejected by the wearer filter"],
    ["Metric 3D output", "21 joints", "Per hand, per frame · wrist + gripper CSV"],
  ];
  const comparisons = [
    ["Hands observed", "per-side coverage · higher is better", pct(tuned?.observed_mean), pct(baseline?.observed_mean)],
    ["Rejected candidates", "tracklet rejection · report with coverage", pct(tuned?.reject_rate), pct(baseline?.reject_rate)],
    ["Wrist jitter", "mm/frame² · lower is better", tuned?.jitter_mm_median?.toFixed(2) ?? "—", baseline?.jitter_mm_median?.toFixed(2) ?? "—"],
    ["Depth jumps", "over 10 cm per 1,000 frames · lower is better", tuned?.depth_jumps_per_1k_frames?.toFixed(2) ?? "—", baseline?.depth_jumps_per_1k_frames?.toFixed(2) ?? "—"],
  ];
  const points = training?.curves?.[0]?.points ?? [];
  const series = training?.curves?.[0]?.series ?? [];
  const values = points.flatMap(p => series.map(s => p[s]).filter((v): v is number => v != null));
  const max = Math.max(...values, 1);
  const curve = (name: string) => points.filter(p => p[name] != null).map((p, i) => `${i ? "L" : "M"}${46 + i / Math.max(1, points.length - 1) * 578},${264 - (p[name] ?? 0) / max * 240}`).join(" ");
  return <div className="landing-page">
    <div className="landing-notice">Built on pretrained WiLoR + CLIP · metrics come from saved results · reference video preview</div>
    <nav className="landing-nav landing-wrap"><a className="landing-brand" href="#/landing"><BrandMark />SpectatorAI</a><div><a href="#demo">Live demo</a><a href="#compare">Results</a><a href="#training">Training</a><a href="#usps">Why us</a></div><a className="landing-button" href="#/workspace">Open workspace</a></nav>
    <header className="landing-hero landing-wrap" id="top"><span className="landing-pill">Egocentric hand tracking · real metric 3D</span>
      <h1>Your hands,<br />in 3D <BrandMark large /></h1>
      <p>SpectatorAI turns head-mounted fisheye video into clean 3D hand tracks — only the wearer's hands, left and right kept straight, ready to drive a robot.</p>
      <div className="landing-actions"><button className="landing-button" onClick={onUpload}>Upload a clip</button><a className="landing-button secondary" href="#compare">See the results</a></div>
      <div className="landing-kpis">{kpis.map(([label, value, note]) => <article key={label}><div>{label}<span>{current ? "results" : "pipeline"}</span></div><strong>{value}</strong><p>{note}</p></article>)}</div>
    </header>
    <section className="landing-demo" id="demo"><div className="demo-decoration"><i /><i /><i /></div><div className="landing-wrap">
      <div className="landing-section-title"><span className="landing-pill">Live demo</span><h2>Drop in a clip.<br />Watch it become 3D.</h2><p>Video, 2D keypoints and the 3D hands stay in sync. Explore the motion in your own workspace.</p></div>
      <div className="landing-demo-grid"><div className="landing-preview"><header>Video · head-camera clip<span>Reference preview</span></header><video ref={video} src={preview + movie} poster={preview + "675bacadb20a1e020f45d1612d7ff790.jpg"} autoPlay loop muted playsInline onLoadedMetadata={e => setDuration(e.currentTarget.duration)} /></div>
        <div className="landing-preview"><header>3D hands · camera space<span>Head rotation removed</span></header><video ref={scene} src={preview + render} poster={preview + "3e1e31e25fc7060a7c8f6021c078532d.jpg"} autoPlay loop muted playsInline /></div></div>
      <div className="landing-demo-controls"><button onClick={() => { for (const v of [video.current, scene.current]) { if (!v) continue; if (playing) v.pause(); else void v.play(); } setPlaying(!playing); }}><Icon name={playing ? "pause" : "play"} size={14} />{playing ? "Pause" : "Play"}</button><span>{time.toFixed(1)} / {duration.toFixed(1)} s</span><input type="range" aria-label="Demo time" min={0} max={duration || 15} step={.0333} value={time} onChange={e => { for (const v of [video.current, scene.current]) if (v) v.currentTime = +e.target.value; setTime(+e.target.value); }} /><a href="#/workspace">Open in workspace ↗</a></div>
      <div className="landing-demo-bottom"><article className="try-clip"><h3>Try your own clip</h3><p>Fisheye or a normal phone video. Opens in the workspace, where the video and 3D hands stay together.</p><button onClick={onUpload}>Drop a video or choose a file <Icon name="plus" /></button></article><article className="demo-facts"><div className="model-switch"><button className={variant === "pretrained" ? "selected" : ""} onClick={() => setVariant("pretrained")}>Pretrained WiLoR</button><button disabled={!tuned} className={variant === "finetuned" ? "selected" : ""} onClick={() => setVariant("finetuned")}>Fine-tuned (ours)</button></div>{[["Model", variant === "finetuned" ? "Fine-tuned" : "Pretrained baseline"], ["Frames tracked", pct(current?.observed_mean)], ["Wrist jitter", current?.jitter_mm_median == null ? "Awaiting results" : `${current.jitter_mm_median} mm/frame²`], ["Data source", "Preview media · saved metrics"]].map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}</article></div>
    </div></section>
    <section className="landing-results landing-wrap" id="compare"><div className="landing-section-title"><span className="landing-pill">Pretrained vs fine-tuned</span><h2>See what fine-tuning<br />on our footage changes.</h2><p>Same pipeline. Same clips. Compare the pretrained baseline with our fine-tuned checkpoint.</p></div><div className="comparison-table"><header><span>Held-out results</span><span>Fine-tuned (ours)</span><span>Pretrained WiLoR</span></header>{comparisons.map(([name, hint, a, b]) => <div className="comparison-row" key={name}><div><h3>{name}</h3><small>{hint}</small></div><strong>{a}</strong><strong>{b}</strong></div>)}<footer>{tuned ? "Metrics read from results/ · inspect paired clips in Showcase" : "Fine-tuned comparison awaits paired results · accuracy against real hand labels is still to verify"}</footer></div></section>
    <section className="landing-training landing-wrap" id="training"><div className="landing-section-title"><span className="landing-pill">Training</span><h2>Trained on what a<br />head camera actually sees.</h2></div><div className="landing-training-grid"><article><header>Loss per epoch <span>{points.length ? "training logs" : "awaiting logs"}</span></header><svg viewBox="0 0 640 300" role="img" aria-label="Training loss"><path d="M46 24H624M46 84H624M46 144H624M46 204H624M46 264H624" stroke="#edeee1" fill="none" />{series.slice(0, 2).map((s, i) => <path key={s} d={curve(s)} stroke={i ? "#c0c2a9" : "#003d21"} strokeWidth="2.5" fill="none" />)}{!points.length && <text x="335" y="150" textAnchor="middle" fill="#7c7d76" fontSize="14">Training curves appear when logs are added.</text>}<text x="46" y="288" fill="#7c7d76" fontSize="11">epoch 1</text><text x="624" y="288" textAnchor="end" fill="#7c7d76" fontSize="11">{points.length ? `epoch ${points.length}` : "—"}</text></svg></article><article><header>Setup <span>{training?.setup ? "from logs" : "to verify"}</span></header>{Object.entries(training?.setup ?? { Backbone: "Pretrained WiLoR", Verifier: "CLIP", Input: "Egocentric fisheye video", Output: "21 joints per hand", "Our pipeline": "Wearer filter + tracking", "Fine-tuned checkpoint": tuned ? "Available" : "Not loaded", "Ground-truth evaluation": "To verify" }).slice(0, 7).map(([k, v]) => <div className="training-fact" key={k}><span>{k}</span><b>{String(v)}</b></div>)}</article></div></section>
    <section className="landing-usps landing-wrap" id="usps"><div className="landing-section-title"><span className="landing-pill">Why SpectatorAI</span><h2>Five things off-the-shelf<br />trackers get wrong.</h2></div><div className="landing-usp-grid">{[
      ["check", "Only the wearer's hands", "Other people's hands, knees and objects are rejected instead of tracked.", "Wearer-first tracking"],
      ["cross", "Left stays left", "Hands keep their identity even when they cross or overlap.", "Track-level identity"],
      ["target", "Fisheye-aware 3D", "Metric 3D from a distorted head camera, with perspective-corrected crops.", "Metres, not pixels"],
      ["eye", "Honest when out of view", "Hands that leave the frame get estimates, clearly marked as guesses.", "Observed / infilled / estimated"],
      ["export", "Robot-ready output", "Wrist poses and gripper trajectories exported per frame for your robot-learning stack.", "JSONL + robot CSV"],
      ["library", "Every number links to a file.", "Explore the evidence, compare models and download the data behind the results.", "Open the workspace ↗"],
    ].map(([icon, title, desc, foot], i) => <article key={title} className={`usp-${i}`}><span className="usp-icon"><Icon name={icon} size={25} /></span><div><h3>{title}</h3><p>{desc}</p></div><a href="#/workspace">{foot}</a></article>)}</div></section>
    <section className="landing-pipeline landing-wrap"><div className="landing-section-title"><span className="landing-pill">How it works</span><h2>From fisheye clip<br />to robot trajectory.</h2></div><ol>{[["01", "Calibrate", "Camera intrinsics and background head motion."], ["02", "Find the wearer", "Detections become tracklets, then left / right / reject decisions."], ["03", "Reconstruct", "WiLoR on perspective-corrected hand crops."], ["04", "Export", "Smooth trajectories, labelled estimates, robot-ready files."]].map(([n, title, desc]) => <li key={n}><span>{n}</span><h3>{title}</h3><p>{desc}</p></li>)}</ol></section>
    <footer className="landing-footer"><div className="landing-wrap"><a className="landing-brand" href="#top"><BrandMark />SpectatorAI</a><span>Our pipeline · pretrained WiLoR + CLIP · estimates stay labelled</span><a href="#/workspace">Open workspace ↗</a></div></footer>
  </div>;
}
