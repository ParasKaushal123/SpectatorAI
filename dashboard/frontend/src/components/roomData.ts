import * as T from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export type Side = "left" | "right";
export type Vec3 = [number, number, number];
export interface Motion {
  fps: number; n: number; quant: number; focus: Vec3; eye0: Vec3;
  head: Vec3[]; quat: [number, number, number, number][]; gaze: Vec3[]; faces: number[];
  hands: Record<Side, { src: number[]; joints: number[][]; mesh: boolean;
    has_mesh: number[]; anchor: Vec3[]; verts: number }>;
}
export interface RobotSummary { tracked_frames: number; copyable_share: number; out_of_reach_share: number;
  too_fast_share: number; median_pos_error_cm: number }
export interface RobotInfo { fps: number; n: number; links: string[]; grade: Record<Side, number[]>;
  summary: Record<Side, RobotSummary> & { copyable_share: number } }
export interface RoomEntry { id: string; has_video: boolean; has_render: boolean; has_robot?: boolean }
export interface RoomData {
  motion: Motion; room: T.Group; bins: Partial<Record<Side, Int16Array>>;
  robot?: { root: T.Group; nodes: T.Object3D[]; poses: Float32Array; info: RobotInfo };
  stats: { clips_used?: string[]; web_triangles?: number; triangles?: number;
    registration?: Record<string, { fitness: number; rmse_m: number }> };
}
export const roomBase = (id: string) => `/api/rooms/${encodeURIComponent(id)}`;

export function disposeRoom(room: T.Group) {
  room.traverse(o => {
    if (o instanceof T.Mesh) {
      o.geometry.dispose();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
    }
  });
}

export async function loadRoom(id: string, signal: AbortSignal,
  progress: (name: string, fraction: number) => void): Promise<RoomData> {
  const base = roomBase(id);
  async function bytes(name: string) {
    const r = await fetch(`${base}/web/${name}`, { signal });
    if (!r.ok) throw new Error(`Missing ${name}. Expected dashboard/results/room_twin/${id}/web/${name} (${r.status}).`);
    const total = Number(r.headers.get("Content-Length")) || 0;
    if (!r.body) return r.arrayBuffer();
    const reader = r.body.getReader(), chunks: Uint8Array[] = [];
    let count = 0, last = 0;
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      chunks.push(value); count += value.length;
      if (performance.now() - last > 150) { progress(name, total ? Math.min(.98, count / total) : .5); last = performance.now(); }
    }
    const buffer = new Uint8Array(count); let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    progress(name, 1); return buffer.buffer;
  }
  const motion: Motion = JSON.parse(new TextDecoder().decode(await bytes("motion.json")));
  if (!Number.isFinite(motion.fps) || motion.fps <= 0 || !Number.isInteger(motion.n) || motion.n < 1 ||
      motion.head.length !== motion.n || motion.quat.length !== motion.n || motion.gaze.length !== motion.n ||
      !motion.hands?.left || !motion.hands?.right) throw new Error("Invalid room motion.json: missing frames or hand tracks.");
  const bins: RoomData["bins"] = {};
  let robot: RoomData["robot"];
  let room: T.Group | undefined;
  try {
    // Load the scan only after all other files have succeeded, avoiding leaked GPU assets on a missing bin.
    for (const side of ["left", "right"] as const) {
      const h = motion.hands[side];
      if (h.mesh && motion.faces?.length) {
        const raw = await bytes(`hands_${side}.bin`);
        if (raw.byteLength !== motion.n * h.verts * 3 * 2) throw new Error(`Invalid hands_${side}.bin: expected ${motion.n} frames of ${h.verts} vertices.`);
        bins[side] = new Int16Array(raw);
      }
    }
    const sr = await fetch(`${base}/stats`, { signal });
    if (!sr.ok) throw new Error(`Missing room_stats.json. Expected dashboard/results/room_twin/${id}/room_stats.json.`);
    const stats = await sr.json();
    const glb = await bytes("room.glb");
    const gltf = await new GLTFLoader().parseAsync(glb, `${base}/web/`);
    room = gltf.scene;
    room.traverse(o => {
      if (!(o instanceof T.Mesh)) return;
      const colors = o.geometry.getAttribute("color");
      if (colors) {
        // Same one-time sRGB-camera -> linear conversion as supplied viewer.html.
        const linear = new Float32Array(colors.count * 3);
        const convert = (x: number) => x <= .04045 ? x / 12.92 : Math.pow((x + .055) / 1.055, 2.4);
        for (let i = 0; i < colors.count; i++) {
          linear[i * 3] = convert(colors.getX(i)); linear[i * 3 + 1] = convert(colors.getY(i)); linear[i * 3 + 2] = convert(colors.getZ(i));
        }
        o.geometry.setAttribute("color", new T.BufferAttribute(linear, 3));
      }
      (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
      o.material = new T.MeshBasicMaterial({ vertexColors: !!colors, side: T.DoubleSide, toneMapped: false });
    });
    const rr = await fetch(`${base}/web/robot.json`, { signal });
    if (rr.status !== 404) {
      if (!rr.ok) throw new Error(`Could not load robot.json (${rr.status}).`);
      const info: RobotInfo = await rr.json();
      if (!Number.isInteger(info.n) || info.n < 1 || !info.links?.length ||
          !info.grade?.left || !info.grade?.right || info.grade.left.length < info.n || info.grade.right.length < info.n)
        throw new Error("Invalid robot.json: missing link order or grades.");
      progress("robot.json", 1);
      const raw = await bytes("robot_motion.bin");
      if (raw.byteLength !== info.n * info.links.length * 7 * 4) throw new Error("Invalid robot_motion.bin: length does not match robot.json.");
      const poses = new Float32Array(raw);
      const model = await new GLTFLoader().parseAsync(await bytes("robot.glb"), `${base}/web/`);
      const nodes = info.links.map(name => model.scene.getObjectByName(name));
      if (nodes.some(node => !node)) { disposeRoom(model.scene); throw new Error("robot.glb is missing links listed in robot.json."); }
      model.scene.traverse(o => {
        if (!(o instanceof T.Mesh)) return;
        if (!o.geometry.getAttribute("normal")) o.geometry.computeVertexNormals();
        (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
        o.material = new T.MeshStandardMaterial({ vertexColors: !!o.geometry.getAttribute("color"), roughness: .45, metalness: .1 });
      });
      const root = new T.Group();
      for (const node of nodes) root.add(node!);
      robot = { root, nodes: nodes as T.Object3D[], poses, info };
    }
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    return { room, motion, bins, stats, robot };
  } catch (e) { if (room) disposeRoom(room); if (robot) disposeRoom(robot.root); throw e; }
}
