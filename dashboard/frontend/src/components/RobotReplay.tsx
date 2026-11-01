import { useMemo } from "react";
import type { RobotInfo, Side } from "./roomData";

export const ROBOT_GRADES = ["No data", "Copyable", "Out of reach", "Too fast"];
const percent = (v: number) => `${(v * 100).toFixed(1)}%`;

export function RobotScore({ info, frame, download }: { info: RobotInfo; frame: number; download: string }) {
  const f = Math.min(frame, info.n - 1);
  return <section className="robot-score"><h3>Robot replay</h3>
    <strong>{percent(info.summary.copyable_share)}</strong><p>Robot-copyable</p>
    {(["left", "right"] as const).map(side => {
      const summary = info.summary[side], grade = info.grade[side][f];
      return <div className="robot-arm-score" key={side}><h4>{side} arm</h4>
        <p className={`robot-current grade-${grade}`}><i />{ROBOT_GRADES[grade] ?? "Unknown"}</p>
        <p>Copyable {percent(summary.copyable_share)}<br />Out of reach {percent(summary.out_of_reach_share)}<br />Too fast {percent(summary.too_fast_share)}</p>
        <small>Median position error: {summary.median_pos_error_cm.toFixed(1)} cm</small>
      </div>;
    })}
    <a className="btn-small" href={download} download="robot_traj.json">Download robot trajectory ↓</a>
  </section>;
}

export function RobotTimeline({ info, frame, n, seek, disabled }: { info: RobotInfo; frame: number; n: number;
  seek: (frame: number) => void; disabled: boolean }) {
  const lanes = useMemo(() => (["left", "right"] as const).map(side => {
    const segments: { start: number; end: number; grade: number }[] = [];
    for (let i = 0; i < n; i++) {
      const grade = info.grade[side][Math.min(i, info.n - 1)] ?? 0, last = segments[segments.length - 1];
      if (last?.grade === grade) last.end = i + 1; else segments.push({ start: i, end: i + 1, grade });
    }
    return { side, segments };
  }), [info, n]);
  const name = (side: Side) => `Robot ${side === "left" ? "L" : "R"}`;
  return <div className="robot-timeline">
    <div className="robot-legend">{ROBOT_GRADES.map((label, grade) => <span key={grade} className={`grade-${grade}`}><i />{label}</span>)}</div>
    {lanes.map(({ side, segments }) => <div className="robot-lane" key={side}><span>{name(side)}</span>
      <div className="robot-lane-track" aria-label={`${name(side)} grades`}>
        {segments.map(segment => <button key={segment.start} className={`grade-${segment.grade}`} disabled={disabled}
          style={{ left: `${segment.start / n * 100}%`, width: `${(segment.end - segment.start) / n * 100}%` }}
          aria-label={`${name(side)} ${ROBOT_GRADES[segment.grade]} frames ${segment.start}–${segment.end - 1}`}
          title={`${ROBOT_GRADES[segment.grade]} · ${(segment.start / info.fps).toFixed(2)}–${(segment.end / info.fps).toFixed(2)} s`}
          onClick={e => {
            const rect = e.currentTarget.parentElement!.getBoundingClientRect();
            seek(e.detail === 0 ? segment.start : Math.max(0, Math.min(n - 1, Math.floor((e.clientX - rect.left) / rect.width * n))));
          }} />)}
        <i className="robot-lane-playhead" style={{ left: `${frame / Math.max(1, n - 1) * 100}%` }} />
      </div>
    </div>)}
  </div>;
}
