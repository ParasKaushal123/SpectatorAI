import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as Controls } from "three-stdlib";
import * as T from "three";
import { BONES } from "../api";
import type { RoomData, Side } from "./roomData";

type Props = { data: RoomData; frame: MutableRefObject<number>; trails: boolean; follow: boolean; pan: boolean;
  preset: string; showRobot: boolean; showHands: boolean; rotate?: boolean; onCanvas: (canvas: HTMLCanvasElement) => void };

function Hand({ data, side, frame, trails }: { data: RoomData; side: Side;
  frame: MutableRefObject<number>; trails: boolean }) {
  const h = data.motion.hands[side], m = data.motion, bin = data.bins[side];
  const objects = useMemo(() => {
    const color = side === "left" ? "#3cc83c" : "#ff8c00";
    const surface = new T.BufferGeometry();
    surface.setAttribute("position", new T.BufferAttribute(new Float32Array(h.verts * 3), 3).setUsage(T.DynamicDrawUsage));
    surface.setIndex(m.faces ?? []);
    const mesh = new T.Mesh(surface, new T.MeshStandardMaterial({ color, roughness: .55, side: T.DoubleSide, transparent: true }));
    const skeletonGeometry = new T.BufferGeometry();
    skeletonGeometry.setAttribute("position", new T.BufferAttribute(new Float32Array(BONES.length * 6), 3).setUsage(T.DynamicDrawUsage));
    const skeleton = new T.LineSegments(skeletonGeometry, new T.LineDashedMaterial({ color, transparent: true }));
    const trailGeometry = new T.BufferGeometry();
    trailGeometry.setAttribute("position", new T.BufferAttribute(new Float32Array(Math.ceil(m.fps) * 6), 3).setUsage(T.DynamicDrawUsage));
    trailGeometry.setAttribute("color", new T.BufferAttribute(new Float32Array(Math.ceil(m.fps) * 6), 3));
    const trail = new T.LineSegments(trailGeometry, new T.LineBasicMaterial({ vertexColors: true, transparent: true,
      blending: T.AdditiveBlending, depthWrite: false }));
    for (const obj of [mesh, skeleton, trail]) { obj.frustumCulled = false; obj.visible = false; }
    mesh.userData.roomHand = side; trail.userData.roomTrail = side;
    const group = new T.Group(); group.add(mesh, skeleton, trail);
    return { group, mesh, skeleton, trail, color: new T.Color(color) };
  }, [h, m, side]);
  useEffect(() => () => {
    for (const obj of [objects.mesh, objects.skeleton, objects.trail]) { obj.geometry.dispose(); obj.material.dispose(); }
  }, [objects]);
  const last = useRef(-1);
  useEffect(() => { last.current = -1; }, [trails]);
  useFrame(() => {
    const i = frame.current;
    if (i === last.current) return; last.current = i;
    const src = h.src[i], hasMesh = !!bin && h.has_mesh[i] === 1 && src > 0;
    objects.mesh.visible = hasMesh;
    objects.skeleton.visible = !hasMesh && src > 0;
    const opacity = src === 3 ? .4 : src === 2 ? .65 : 1;
    objects.mesh.material.opacity = objects.skeleton.material.opacity = opacity;
    if (hasMesh) {
      const p = objects.mesh.geometry.getAttribute("position"), a = h.anchor[i], off = i * h.verts * 3;
      for (let v = 0; v < h.verts; v++) p.setXYZ(v, a[0] + bin![off + v * 3] * m.quant,
        a[1] + bin![off + v * 3 + 1] * m.quant, a[2] + bin![off + v * 3 + 2] * m.quant);
      p.needsUpdate = true; objects.mesh.geometry.computeVertexNormals();
    } else if (src > 0) {
      const p = objects.skeleton.geometry.getAttribute("position"), j = h.joints[i];
      BONES.forEach(([a, b], k) => { p.setXYZ(k * 2, j[a * 3], j[a * 3 + 1], j[a * 3 + 2]);
        p.setXYZ(k * 2 + 1, j[b * 3], j[b * 3 + 1], j[b * 3 + 2]); });
      p.needsUpdate = true; objects.skeleton.computeLineDistances();
      objects.skeleton.material.dashSize = src === 3 ? .012 : 1000;
      objects.skeleton.material.gapSize = src === 3 ? .008 : 0;
    }
    const p = objects.trail.geometry.getAttribute("position"), c = objects.trail.geometry.getAttribute("color");
    let count = 0;
    for (let t = Math.max(1, i - Math.ceil(m.fps) + 1); t <= i; t++) {
      if (!h.src[t] || !h.src[t - 1]) continue;
      const fade = Math.pow(1 - (i - t) / Math.ceil(m.fps), 2);
      for (const j of [h.joints[t - 1], h.joints[t]]) {
        p.setXYZ(count, j[0], j[1], j[2]); c.setXYZ(count, objects.color.r * fade, objects.color.g * fade, objects.color.b * fade); count++;
      }
    }
    p.needsUpdate = c.needsUpdate = true; objects.trail.geometry.setDrawRange(0, count);
    objects.trail.visible = trails && count > 0;
  });
  return <primitive object={objects.group} dispose={null} />;
}

function Scene({ data, frame, follow, trails, preset, pan, showRobot, showHands, rotate }: Omit<Props, "onCanvas">) {
  const m = data.motion, controls = useRef<Controls>(null);
  // Each viewport needs its own scene nodes; immutable scan geometry/materials can be shared.
  const scan = useMemo(() => data.room.clone(true), [data.room]);
  const { camera } = useThree();
  const focusTarget = useRef<T.Vector3 | null>(null);
  const robotFrame = useRef(-1);
  const correction = useMemo(() => new T.Vector3(), []);
  const limitTarget = (target: T.Vector3) => target.set(
    T.MathUtils.clamp(target.x, m.focus[0] - .9, m.focus[0] + .9),
    T.MathUtils.clamp(target.y, m.focus[1] - .9, m.focus[1] + .9),
    T.MathUtils.clamp(target.z, m.focus[2] - .6, m.focus[2] + .8));
  const head = useMemo(() => {
    const group = new T.Group(), corners = [[-1, -.6, 1], [1, -.6, 1], [1, .6, 1], [-1, .6, 1]].map(p => p.map(v => v * .09));
    const points = corners.flatMap((p, i) => [0, 0, 0, ...p, ...p, ...corners[(i + 1) % 4]]);
    const geo = new T.BufferGeometry(); geo.setAttribute("position", new T.Float32BufferAttribute(points, 3));
    group.add(new T.LineSegments(geo, new T.LineBasicMaterial({ color: "#78dcff" })));
    group.userData.roomHead = true; return group;
  }, []);
  const gaze = useMemo(() => {
    const geo = new T.BufferGeometry(); geo.setAttribute("position", new T.Float32BufferAttribute(new Float32Array(6), 3));
    const line = new T.Line(geo, new T.LineBasicMaterial({ color: "#78dcff" })); line.frustumCulled = false; return line;
  }, []);
  useEffect(() => () => { gaze.geometry.dispose(); gaze.material.dispose(); head.traverse(o => {
    if (o instanceof T.LineSegments) { o.geometry.dispose(); (o.material as T.Material).dispose(); }
  }); }, [gaze, head]);
  const vectors = useMemo(() => ({ hp: new T.Vector3(), g: new T.Vector3(), focus: new T.Vector3(...m.focus),
    back: new T.Vector3(), side: new T.Vector3() }), [m]);
  const eye = () => {
    vectors.hp.fromArray(m.head[frame.current]); vectors.g.fromArray(m.gaze[frame.current]);
    camera.position.copy(vectors.hp).addScaledVector(vectors.g, -.05);
    controls.current?.target.copy(vectors.hp).addScaledVector(vectors.g, .5);
  };
  useEffect(() => {
    camera.up.set(0, 0, 1); camera.position.fromArray(m.eye0);
    controls.current?.target.fromArray(m.focus); controls.current?.update();
  }, [camera, m]);
  useEffect(() => {
    const control = controls.current; if (!control) return;
    focusTarget.current = null;
    const kind = preset.split(":")[0];
    vectors.hp.fromArray(m.head[frame.current]);
    vectors.back.copy(vectors.hp).sub(vectors.focus).setZ(0).normalize();
    if (vectors.back.lengthSq() < .001) vectors.back.set(0, -1, 0);
    vectors.side.set(-vectors.back.y, vectors.back.x, 0);
    control.target.copy(vectors.focus);
    if (kind === "eye") eye();
    else {
      camera.position.copy(vectors.focus);
      if (kind === "left" || kind === "right") camera.position.addScaledVector(vectors.side, kind === "left" ? 1 : -1).addScaledVector(vectors.back, .3).add(new T.Vector3(0, 0, .5));
      else camera.position.addScaledVector(vectors.back, kind === "top" ? .3 : .8).add(new T.Vector3(0, 0, kind === "top" ? 1.3 : .5));
    }
    control.update();
    // Presets apply once; orbiting remains free until Follow head is enabled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, m, camera]);
  useFrame((_, dt) => {
    const i = frame.current;
    if (data.robot && showRobot) {
      const robot = data.robot, f = Math.min(i, robot.info.n - 1);
      robot.root.visible = showRobot;
      if (robotFrame.current !== f) {
        robotFrame.current = f;
        robot.nodes.forEach((node, k) => {
          const off = (f * robot.nodes.length + k) * 7;
          node.position.fromArray(robot.poses, off); node.quaternion.fromArray(robot.poses, off + 3);
        });
      }
    }
    head.position.fromArray(m.head[i]); head.quaternion.fromArray(m.quat[i]);
    const p = gaze.geometry.getAttribute("position"), hp = m.head[i], g = m.gaze[i];
    p.setXYZ(0, ...hp); p.setXYZ(1, hp[0] + g[0] * .6, hp[1] + g[1] * .6, hp[2] + g[2] * .6); p.needsUpdate = true;
    head.visible = !follow;
    if (follow) { eye(); camera.lookAt(controls.current!.target); }
    else if (focusTarget.current && controls.current) {
      controls.current.target.lerp(focusTarget.current, 1 - Math.exp(-12 * dt));
      if (controls.current.target.distanceToSquared(focusTarget.current) < 1e-7) {
        controls.current.target.copy(focusTarget.current); focusTarget.current = null;
      }
      controls.current.update();
    }
  });
  return <>
    <color attach="background" args={["#161817"]} />
    <hemisphereLight args={["#ffffff", "#444444", 1]} /><directionalLight position={[.5, -1, 2]} intensity={1.2} />
    <primitive object={scan} dispose={null} onDoubleClick={(e: { stopPropagation: () => void; point: T.Vector3 }) => {
      if (follow) return; e.stopPropagation(); focusTarget.current = limitTarget(e.point.clone());
    }} />
    <group visible={showHands}><Hand data={data} side="left" frame={frame} trails={trails} /><Hand data={data} side="right" frame={frame} trails={trails} /></group>
    {showRobot && data.robot && <primitive object={data.robot.root} dispose={null} />}
    <primitive object={head} dispose={null} /><primitive object={gaze} dispose={null} />
    <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={.1} enabled={!follow}
      autoRotate={!!rotate && !follow} autoRotateSpeed={.65}
      rotateSpeed={.6} panSpeed={.65} zoomSpeed={.5} zoomToCursor screenSpacePanning
      minDistance={.15} maxDistance={3}
      mouseButtons={{ LEFT: pan ? T.MOUSE.PAN : T.MOUSE.ROTATE, MIDDLE: T.MOUSE.DOLLY, RIGHT: T.MOUSE.PAN }}
      touches={{ ONE: pan ? T.TOUCH.PAN : T.TOUCH.ROTATE, TWO: T.TOUCH.DOLLY_PAN }}
      onStart={() => { focusTarget.current = null; }}
      onChange={() => { const c = controls.current; if (!c || follow) return;
        // Move camera and target together at the pan boundary, preserving the view direction.
        correction.copy(c.target); limitTarget(c.target); correction.subVectors(c.target, correction);
        camera.position.add(correction);
      }} />
  </>;
}

export default function RoomScene(props: Props) {
  return <Canvas dpr={[1, 1.5]} gl={{ preserveDrawingBuffer: true, antialias: true }}
    camera={{ fov: 60, near: .02, far: 50, up: [0, 0, 1] }}
    onCreated={({ gl, scene }) => { gl.outputColorSpace = T.SRGBColorSpace;
      // Exposes only a readiness marker for browser smoke checks, no footage data.
      gl.domElement.dataset.roomScene = "ready"; scene.userData.roomFrame = "Z-up"; props.onCanvas(gl.domElement); }}>
    <Scene {...props} />
  </Canvas>;
}
