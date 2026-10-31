import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { COLORS, pct, type Metrics } from "../api";
import { ArrowChip, EyeGlyph, HandGlyph } from "./Glyphs";

export interface Evidence {
  reasons: { group: string; count: number }[];
  moments: Record<string, { clip_key: string; clip: string; frame: number; t: number; note: string }>;
}
export type Jump = (clipKey: string, frame: number) => void;

const tick = { fontSize: 11, fontFamily: "JetBrains Mono", fill: COLORS.pebble };
const tip = { background: COLORS.cream, border: "1.5px solid #000", borderRadius: 13, fontFamily: "Space Grotesk", fontSize: 13 };

export function Hero({ m, modelLabel }: { m: Metrics | null; modelLabel: string }) {
  const a = m?.aggregate.pretrained;
  const tiles = [
    { bg: "bg-mint-sprout", ink: "text-forest-depths", tag: "wearer filter", big: a ? pct(a.reject_rate) : "…",
      text: a ? `of ${a.tracklets_total.toLocaleString()} candidate hand tracks rejected as not the wearer's: other people, knees, feet, planks` : "" },
    { bg: "bg-lilac-wash", ink: "text-plum-depth", tag: "3D coverage", big: a ? pct(a.observed_mean) : "…",
      text: a ? `of hand-frames measured in metric 3D; another ${pct(a.estimated_mean, 1)} bridged with estimates that are labelled as estimates` : "" },
    { bg: "bg-sky-wash", ink: "text-cobalt-ink", tag: "real footage", big: a ? a.frames.toLocaleString() : "…",
      text: a ? `frames of head-mounted fisheye factory video (${a.clips} clips, ${a.minutes} min) run end to end on an A100` : "" },
  ];
  return (
    <section className="wrap flex flex-col items-center gap-27 pb-72 pt-63 text-center">
      <span className="pill">Egocentric 3D hand tracking · for robot learning</span>
      <h1 className="display text-[52px] sm:text-display lg:text-[104px]">
        Only the wearer’s <EyeGlyph /> hands.
        <br />
        In metres <HandGlyph />
      </h1>
      <p className="max-w-[640px] text-subheading text-charcoal">
        Head-mounted fisheye video in. Left and right 3D hand skeletons and a robot-ready wrist + gripper trajectory out.
        Other people’s hands, knees and planks are thrown away, and every guess is labelled as one.
      </p>
      <div className="flex flex-wrap justify-center gap-9">
        <a className="btn" href="#explorer">Explore the A100 results</a>
        <a className="btn-ghost" href="#upload">Try your own video</a>
      </div>
      <div className="mt-18 grid w-full max-w-[1000px] gap-13 md:grid-cols-3">
        {tiles.map((t) => (
          <div key={t.tag} className={`${t.bg} flex flex-col gap-9 rounded-cards p-27 text-left`}>
            <span className="font-mono text-caption uppercase">{t.tag}</span>
            <span className={`display text-display ${t.ink}`}>{t.big}</span>
            <span className="text-body-sm">{t.text}</span>
          </div>
        ))}
      </div>
      <p className="font-mono text-caption text-pebble">
        Numbers computed live from results/*/stats.json + tracklets.csv · {modelLabel.toLowerCase()} · PS4 pipeline v4.2
      </p>
    </section>
  );
}

const STEPS = [
  { id: "M1", name: "Fisheye clip", text: "intrinsics (GeoCalib) + head rotation", ours: false },
  { id: "M2", name: "Detection", text: "WiLoR hand detector; weak boxes kept", ours: false },
  { id: "M3", name: "Wearer filter", text: "distance, forearm exit, CLIP veto, on-forearm check", ours: true },
  { id: "M3A", name: "L/R tracking", text: "one labelling over all tracks; crossings cut + relinked", ours: true },
  { id: "M4", name: "WiLoR 3D", text: "pretrained WiLoR on fisheye-corrected crops", ours: false },
  { id: "M5", name: "Infill", text: "smoothing, depth gate, labelled out-of-view estimate", ours: true },
  { id: "M6", name: "Trajectory", text: "wrist pose + gripper CSV for robot learning", ours: true },
];

export function HowItWorks() {
  return (
    <section id="how" className="relative overflow-hidden rounded-t-herocards bg-forest-depths px-[16px] py-72 md:px-27">
      <div aria-hidden className="absolute -right-[60px] top-[40px] h-[360px] w-[360px] rounded-full bg-mint-sprout opacity-70" />
      <div aria-hidden className="absolute right-[150px] top-[250px] h-[200px] w-[200px] rounded-full bg-mint-sprout opacity-60" />
      <div aria-hidden className="absolute right-[40px] top-[330px] h-[120px] w-[120px] rounded-full bg-mint-sprout opacity-80" />
      <div className="relative mx-auto flex max-w-[1200px] flex-col gap-36">
        <span className="pill self-start border-mint-sprout text-mint-sprout">How it works</span>
        <h2 className="max-w-[760px] text-[40px] font-bold leading-[1.02] text-mint-sprout md:text-[54px]">
          Link first. Decide per track, with the whole track’s 3D evidence.
        </h2>
        <p className="max-w-[640px] text-subheading text-cream-parchment/90">
          Frame-by-frame trackers can never undo a decision. PS4 links boxes into short tracks first, then decides
          “wearer’s hand?” and “left or right?” once per track — so a knee, a colleague’s hand or a crossing can’t flip it.
        </p>
        <ol className="grid gap-9 sm:grid-cols-2 lg:grid-cols-7">
          {STEPS.map((s, i) => (
            <li key={s.id} className={`relative flex flex-col gap-6 rounded-cards p-18 ${s.ours ? "bg-mint-sprout" : "bg-cream-parchment"}`}>
              <span className="font-mono text-[10px] uppercase">{s.id}{s.ours ? " · ours" : " · pretrained"}</span>
              <span className="text-[18px] font-bold leading-[1.1]">{s.name}</span>
              <span className="text-caption text-olive-slate">{s.text}</span>
              {i < STEPS.length - 1 && <span aria-hidden className="absolute -right-[8px] top-1/2 hidden text-mint-sprout lg:block">›</span>}
            </li>
          ))}
        </ol>
        <p className="font-mono text-caption uppercase text-mint-sprout/80">
          Mint = what the PS4 pipeline adds · cream = pretrained models (WiLoR detector + WiLoR 3D, CLIP)
        </p>
      </div>
    </section>
  );
}

export function USPs({ m, ev, onJump }: { m: Metrics | null; ev: Evidence | null; onJump: Jump }) {
  const a = m?.aggregate.pretrained;
  const st = m?.synthetic_test;
  const mo = ev?.moments ?? {};
  const other = ev?.reasons.find((r) => r.group.startsWith("Another"))?.count;
  const body = ev?.reasons.find((r) => r.group.startsWith("Body"))?.count;
  const zeroJump = (m?.clips ?? []).filter((c) => c.variant === "pretrained" && c.depth_jumps === 0).length;
  const cards = [
    { bg: "bg-mint-sprout", ink: "text-forest-depths", title: "Only the wearer’s hands",
      stat: a ? pct(a.reject_rate) : "…",
      text: `${a?.tracklets_rejected ?? "…"} of ${a?.tracklets_total ?? "…"} candidate tracks rejected across ${a?.clips ?? "…"} clips: ${other ?? "…"} another person’s hand, ${body ?? "…"} knees, feet or heads.`,
      moment: mo.other_person, cta: "See a rejection" },
    { bg: "bg-lilac-wash", ink: "text-plum-depth", title: "Left stays left",
      stat: st ? `${st.side_swaps}` : "…",
      text: "side swaps outside overlaps (synthetic test). On real clips, tracks are cut at crossings and relinked by forearm direction.",
      moment: mo.crossing, cta: "See a crossing", synthetic: true },
    { bg: "bg-sky-wash", ink: "text-cobalt-ink", title: "Real metric 3D",
      stat: "21 × xyz",
      text: `joints per hand in metres, in a head-stable frame, from fisheye-corrected crops. ${zeroJump} of ${a?.clips ?? "…"} clips have zero depth jumps over 10 cm.`,
      moment: mo.metric3d, cta: "Orbit the hands" },
    { bg: "bg-rose-wash", ink: "text-wine-shadow", title: "Honest about gaps",
      stat: a ? pct(a.estimated_mean, 1) : "…",
      text: "of hand-frames are out-of-view estimates — drawn dashed, labelled “estimated”, never counted as measured.",
      moment: mo.estimated, cta: "See an estimate" },
    { bg: "bg-bone", ink: "text-charcoal", title: "Robot-ready output",
      stat: "6-DoF + grip",
      text: "wrist position + orientation and gripper aperture per frame, as a CSV a robot-learning stack can ingest.",
      moment: mo.metric3d, cta: "Download a trajectory" },
  ];
  return (
    <section id="usps" className="wrap flex flex-col gap-27 py-72">
      <span className="pill self-start">What the pipeline adds · each tied to evidence</span>
      <h2 className="display text-[44px] md:text-display">Five reasons it beats <ArrowChip /> off-the-shelf.</h2>
      <p className="max-w-[720px] text-subheading text-charcoal">
        Off-the-shelf hand models answer “is there a hand?” per frame. They don’t know whose hand it is, which side it is
        when hands cross, or where it went when it left the frame. That’s what this pipeline adds on top of WiLoR.
      </p>
      <div className="grid gap-13 sm:grid-cols-2 lg:grid-cols-5">
        {cards.map((c) => (
          <article key={c.title} className={`${c.bg} flex flex-col gap-13 rounded-cards p-27`}>
            <h3 className={`text-heading-sm font-bold leading-[1.05] ${c.ink}`}>{c.title}</h3>
            <span className={`display text-heading-lg ${c.ink}`}>{c.stat}</span>
            <p className="flex-1 text-body-sm">{c.text}</p>
            {c.synthetic && <span className="pill-sm self-start bg-pure-white">synthetic test</span>}
            {c.title === "Robot-ready output" && c.moment ? (
              <a className="btn self-start px-13 py-9 text-body-sm" href={`/api/clips/${c.moment.clip_key}/download/robot_wrist_gripper.csv`}>
                ↓ Trajectory CSV · {c.moment.clip.replace("hand_tracking_", "clip ")}
              </a>
            ) : c.moment && (
              <button className="btn self-start px-13 py-9 text-body-sm" onClick={() => onJump(c.moment!.clip_key, c.moment!.frame)}>
                {c.cta} · {c.moment.clip.replace("hand_tracking_", "clip ")} @ {c.moment.t}s
              </button>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

export function Results({ m, ev }: { m: Metrics | null; ev: Evidence | null }) {
  const rows = (m?.clips ?? []).filter((c) => c.variant === "pretrained").map((c) => ({
    name: c.name.replace("hand_tracking_", "clip "),
    observed: +(((c.observed_left + c.observed_right) / 2) * 100).toFixed(1),
    infilled: +(c.infilled * 100).toFixed(1),
    estimated: +(c.estimated * 100).toFixed(1),
    accepted: c.tracklets_total - c.tracklets_rejected,
    rejected: c.tracklets_rejected,
    jitter: c.jitter_mm,
    jumps: c.depth_jumps,
  }));
  const reasons = (ev?.reasons ?? []).filter((r) => r.count > 0).sort((a, b) => b.count - a.count);
  return (
    <section id="results" className="wrap flex flex-col gap-27 pb-72">
      <span className="pill self-start">Results · {m?.aggregate.pretrained?.clips ?? "…"} A100 clips · pretrained baseline</span>
      <h2 className="display text-[44px] md:text-display">The numbers, clip by clip.</h2>
      <div className="grid gap-13 lg:grid-cols-2">
        <div className="card-soft flex flex-col gap-13 bg-pure-white">
          <div><h3 className="text-heading-sm font-bold">Where each hand-frame came from</h3>
            <p className="text-body-sm text-olive-slate">Mean of both hands, % of frames. The rest: hand out of view, nothing claimed.</p></div>
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={rows} margin={{ left: -18, right: 6 }}>
                <CartesianGrid vertical={false} stroke={COLORS.none} />
                <XAxis dataKey="name" tick={tick} stroke={COLORS.sage} />
                <YAxis tick={tick} stroke={COLORS.sage} unit="%" domain={[0, 100]} allowDataOverflow />
                <Tooltip contentStyle={tip} formatter={(v: any) => `${v}%`} />
                <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                <Bar dataKey="observed" stackId="a" fill={COLORS.observed} />
                <Bar dataKey="infilled" stackId="a" fill={COLORS.infilled} />
                <Bar dataKey="estimated" stackId="a" fill={COLORS.estimated} stroke="#000" strokeWidth={0.5} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card-soft flex flex-col gap-13 bg-pure-white">
          <div><h3 className="text-heading-sm font-bold">Why candidates were rejected</h3>
            <p className="text-body-sm text-olive-slate">Every rejected track keeps its reason (tracklets.csv), all {m?.aggregate.pretrained?.clips ?? "…"} clips.</p></div>
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={reasons} layout="vertical" margin={{ left: 40, right: 18 }}>
                <CartesianGrid horizontal={false} stroke={COLORS.none} />
                <XAxis type="number" tick={tick} stroke={COLORS.sage} />
                <YAxis type="category" dataKey="group" width={170} tick={{ ...tick, fill: COLORS.charcoal }} stroke={COLORS.sage} />
                <Tooltip contentStyle={tip} />
                <Bar dataKey="count" fill={COLORS.left} radius={[0, 9, 9, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card-soft flex flex-col gap-13 bg-pure-white">
          <div><h3 className="text-heading-sm font-bold">Candidate tracks: kept vs rejected</h3>
            <p className="text-body-sm text-olive-slate">The detector proposes far more “hands” than there are. Most are not the wearer’s.</p></div>
          <div className="h-[260px]">
            <ResponsiveContainer>
              <BarChart data={rows} margin={{ left: -18, right: 6 }}>
                <CartesianGrid vertical={false} stroke={COLORS.none} />
                <XAxis dataKey="name" tick={tick} stroke={COLORS.sage} />
                <YAxis tick={tick} stroke={COLORS.sage} />
                <Tooltip contentStyle={tip} />
                <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                <Bar dataKey="accepted" name="kept (wearer)" stackId="t" fill={COLORS.observed} />
                <Bar dataKey="rejected" stackId="t" fill={COLORS.sage} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card-soft flex flex-col gap-13 overflow-x-auto bg-pure-white">
          <div><h3 className="text-heading-sm font-bold">Smoothness per clip</h3>
            <p className="text-body-sm text-olive-slate">Median wrist acceleration (lower = steadier) and depth jumps over 10 cm, measured frames only.</p></div>
          <table className="w-full min-w-[420px] text-left text-body-sm">
            <thead className="font-mono text-caption uppercase text-pebble">
              <tr><th className="py-6">clip</th><th>observed</th><th>est.</th><th>jitter mm/f²</th><th>depth jumps</th><th>kept / cand.</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className="border-t border-bone">
                  <td className="py-6 font-medium">{r.name}</td><td>{r.observed}%</td><td>{r.estimated}%</td>
                  <td>{r.jitter ?? "—"}</td><td>{r.jumps}</td><td>{r.accepted} / {r.accepted + r.rejected}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {m?.synthetic_test && (
        <div className="card-soft flex flex-wrap items-center gap-13">
          <span className="pill-sm bg-rose-wash">synthetic test</span>
          <span className="text-body-sm">Ground-truth checks on a generated clip with stand-in models (no real labels yet):
            recall {pct(m.synthetic_test.recall, 1)} · side accuracy {pct(m.synthetic_test.side_accuracy)} · false hands {pct(m.synthetic_test.false_hand_rate)} · side swaps {m.synthetic_test.side_swaps}.
            Real-footage accuracy vs hand labels: <b>TO VERIFY</b> (needs a labels CSV).</span>
        </div>
      )}
    </section>
  );
}
