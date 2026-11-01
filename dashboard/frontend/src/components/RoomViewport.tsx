import { useState, type MutableRefObject, type ReactNode } from "react";
import RoomScene from "./RoomScene";
import Safe from "./Safe";
import type { RoomData } from "./roomData";

export default function RoomViewport({ data, frame, robot, onCanvas, overlay }: { data: RoomData;
  frame: MutableRefObject<number>; robot: boolean; overlay?: ReactNode; onCanvas: (canvas: HTMLCanvasElement) => void }) {
  const [pan, setPan] = useState(false), [follow, setFollow] = useState(false), [rotate, setRotate] = useState(false);
  const [trails, setTrails] = useState(true), [showHands, setShowHands] = useState(!robot), [showRobot, setShowRobot] = useState(robot);
  const [preset, setPreset] = useState("shoulder:0");
  const title = robot ? "Robot" : "Hands";
  return <section className={`room-viewer ${robot ? "room-robot-view" : "room-hands-view"}`} aria-label={`${title} 3D view`}>
    <header className="room-tools"><span className="eyebrow">{robot ? "Robot replay" : "SLAM · hands replay"}</span>
      <select aria-label={`${title} camera view`} value={preset.split(":")[0]} onChange={e => { setFollow(false); setRotate(false); setPreset(`${e.target.value}:${Date.now()}`); }}>
        <option value="shoulder">Over shoulder</option><option value="eye">Worker's eye</option><option value="left">Left</option><option value="right">Right</option><option value="top">Top</option>
      </select>
      <button className="chip-toggle" onClick={() => { setFollow(false); setRotate(false); setPreset(`shoulder:${Date.now()}`); }}>Reset view</button>
      <button className={`chip-toggle ${pan ? "on" : ""}`} aria-pressed={pan} onClick={() => { setFollow(false); setPan(!pan); }}>Pan</button>
      <button className={`chip-toggle ${rotate ? "on" : ""}`} aria-pressed={rotate} onClick={() => { setFollow(false); setRotate(!rotate); }}>Rotate camera</button>
      <button className={`chip-toggle ${follow ? "on" : ""}`} aria-pressed={follow} onClick={() => { setRotate(false); setFollow(!follow); }}>Follow head</button>
      <button className={`chip-toggle ${trails ? "on" : ""}`} aria-pressed={trails} onClick={() => setTrails(!trails)}>Trails</button>
      {robot && <button className={`chip-toggle ${showRobot ? "on" : ""}`} aria-pressed={showRobot} disabled={!data.robot} onClick={() => setShowRobot(!showRobot)}>Robot</button>}
      <button className={`chip-toggle ${showHands ? "on" : ""}`} aria-pressed={showHands} onClick={() => setShowHands(!showHands)}>Hands</button>
    </header>
    <div className="room-canvas"><Safe name={`${title} room scene`}><RoomScene data={data} frame={frame} trails={trails} follow={follow} pan={pan}
      rotate={rotate} showRobot={showRobot} showHands={showHands} preset={preset} onCanvas={onCanvas} /></Safe>
      {overlay}
      <span className="room-hint">{pan ? "Drag to pan" : "Drag to orbit · Shift/right-drag to pan"} · scroll to zoom · double-click to focus</span>
    </div>
  </section>;
}
