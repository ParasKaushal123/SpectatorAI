import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import type { OrbitControls as Controls } from "three-stdlib";
import * as T from "three";
import { BONES, COLORS, type ClipData } from "../api";

type Props = { clip: ClipData; frame: MutableRefObject<number>; follow: boolean; path: boolean;
  preset: string; hands?: boolean; labels?: boolean; head?: boolean; onCanvas: (canvas: HTMLCanvasElement) => void; locked: boolean };
const basis = new T.Quaternion().setFromAxisAngle(new T.Vector3(1, 0, 0), Math.PI);

function Hand({ clip, side, frame, path, labels }: { clip: ClipData; side: "left" | "right";
  frame: MutableRefObject<number>; path: boolean; labels?: boolean }) {
  const h = clip.hands[side];
  const color = side === "left" ? "#45d16a" : "#ff9a2e";
  const spheres = useRef<T.InstancedMesh>(null);
  const labelPosition = useRef<T.Sprite>(null);
  const labelTextures = useMemo(() => ["OUT", "OBSERVED", "INFILLED", "ESTIMATED"].map(state => {
    const canvas = document.createElement("canvas"); canvas.width = 384; canvas.height = 64;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(0, 0, 384, 64, 32); ctx.fill();
    ctx.fillStyle = "#000"; ctx.font = "28px monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`${side.toUpperCase()} · ${state}`, 192, 33);
    const texture = new T.CanvasTexture(canvas); texture.colorSpace = T.SRGBColorSpace; return texture;
  }), [side, color]);
  useEffect(() => () => labelTextures.forEach(t => t.dispose()), [labelTextures]);
  const bones = useRef<T.LineSegments>(null);
  const fingers = useRef<T.InstancedMesh>(null);
  const palm = useRef<T.Mesh>(null);
  const fingerBones = BONES.slice(0, 20);
  const palmOutline = [0, 1, 2, 5, 9, 13, 17];
  const palmPositions = useMemo(() => new Float32Array(16 * 3), []);
  const palmIndices = useMemo(() => {
    const indices: number[] = [];
    for (let j = 0; j < 7; j++) {
      const k = (j + 1) % 7;
      indices.push(7, j, k, 15, k + 8, j + 8, j, j + 8, k, k, j + 8, k + 8);
    }
    return indices;
  }, []);
  const vectors = useMemo(() => ({ a: new T.Vector3(), b: new T.Vector3(), normal: new T.Vector3(),
    centre: new T.Vector3(), up: new T.Vector3(0, 1, 0) }), []);
  const radius = (j: number) => j === 0 ? .013 : .0105 - ((j - 1) % 4) * .0015;
  const trail = useRef<T.LineSegments>(null);
  const lastFrame = useRef(-1);
  const data = useMemo(() => h.kp3d.map(k => k ? new Float32Array(k.map((v, i) => i % 3 ? -v : v)) : null), [h]);
  const boneArray = useMemo(() => new Float32Array(BONES.length * 6), []);
  const trailArray = useMemo(() => new Float32Array(Math.ceil(clip.fps) * 6), [clip]);
  const trailColors = useMemo(() => new Float32Array(trailArray.length), [trailArray]);
  useEffect(() => { lastFrame.current = -1; }, [path, labels]);
  const dummy = useMemo(() => new T.Object3D(), []);
  useEffect(() => {
    for (let j = 0; j < 21; j++) spheres.current?.setColorAt(j, new T.Color(color));
    if (spheres.current?.instanceColor) spheres.current.instanceColor.needsUpdate = true;
  }, [color]);
  useFrame(() => {
    const i = frame.current;
    if (lastFrame.current === i || !spheres.current || !bones.current || !trail.current) return;
    lastFrame.current = i;
    const kp = data[i], source = h.src[i];
    if (labelPosition.current) { labelPosition.current.visible = !!labels && !!kp && !!source; if (kp) labelPosition.current.position.set(kp[0], kp[1] + .07, kp[2]); (labelPosition.current.material as T.SpriteMaterial).map = labelTextures[source] ?? labelTextures[0]; }
    spheres.current.visible = !!kp && !!source;
    bones.current.visible = false;
    if (fingers.current) fingers.current.visible = !!kp && !!source;
    if (palm.current) palm.current.visible = !!kp && !!source;
    if (kp && source) {
      for (let j = 0; j < 21; j++) {
        dummy.position.fromArray(kp, j * 3);
        dummy.scale.setScalar(radius(j));
        dummy.updateMatrix();
        spheres.current.setMatrixAt(j, dummy.matrix);
      }
      spheres.current.instanceMatrix.needsUpdate = true;
      if (fingers.current && palm.current) {
        fingerBones.forEach(([a, b], j) => {
          vectors.a.fromArray(kp, a * 3); vectors.b.fromArray(kp, b * 3);
          dummy.position.copy(vectors.a).add(vectors.b).multiplyScalar(.5);
          dummy.quaternion.setFromUnitVectors(vectors.up, vectors.b.sub(vectors.a).normalize());
          dummy.scale.set(radius(b), vectors.a.distanceTo(new T.Vector3().fromArray(kp, b * 3)), radius(b));
          dummy.updateMatrix(); fingers.current!.setMatrixAt(j, dummy.matrix);
        });
        fingers.current.instanceMatrix.needsUpdate = true;
        vectors.a.fromArray(kp, 5 * 3).sub(vectors.centre.fromArray(kp));
        vectors.b.fromArray(kp, 17 * 3).sub(vectors.centre);
        vectors.normal.crossVectors(vectors.a, vectors.b).normalize().multiplyScalar(.009);
        vectors.centre.set(0, 0, 0);
        for (const j of palmOutline) vectors.centre.add(vectors.a.fromArray(kp, j * 3));
        vectors.centre.divideScalar(7);
        for (let layer = 0; layer < 2; layer++) {
          for (let j = 0; j < 8; j++) {
            if (j === 7) vectors.a.copy(vectors.centre); else vectors.a.fromArray(kp, palmOutline[j] * 3);
            vectors.a.addScaledVector(vectors.normal, layer === 0 ? 1 : -1);
            palmPositions.set(vectors.a.toArray(), (layer * 8 + j) * 3);
          }
        }
        palm.current.geometry.attributes.position.needsUpdate = true;
        palm.current.geometry.computeVertexNormals();
        for (const mesh of [fingers.current, palm.current]) (mesh.material as T.MeshStandardMaterial).opacity = [0, 1, .55, .35][source];
      }
      (spheres.current.material as T.MeshStandardMaterial).opacity = [0, 1, .55, .35][source];
      BONES.forEach(([a, b], j) => { boneArray.set(kp.subarray(a * 3, a * 3 + 3), j * 6); boneArray.set(kp.subarray(b * 3, b * 3 + 3), j * 6 + 3); });
      bones.current.geometry.attributes.position.needsUpdate = true;
      bones.current.computeLineDistances();
      const mat = bones.current.material as T.LineDashedMaterial;
      mat.dashSize = source === 3 ? .008 : 1000; mat.gapSize = source === 3 ? .006 : 0;
      mat.opacity = [0, 1, .55, .35][source];
    }
    let n = 0;
    for (let k = Math.max(1, i - Math.floor(clip.fps * .6) + 1); k <= i; k++) {
      if (data[k] && data[k - 1] && h.src[k] && h.src[k - 1] && n + 6 <= trailArray.length) {
        trailArray.set(data[k - 1]!.subarray(0, 3), n); trailArray.set(data[k]!.subarray(0, 3), n + 3);
        const fade = Math.max(0, 1 - (i - k) / (clip.fps * .6));
        const c = new T.Color(color).multiplyScalar(fade * fade);
        trailColors.set([c.r, c.g, c.b, c.r, c.g, c.b], n); n += 6;
      }
    }
    trail.current.geometry.attributes.position.needsUpdate = true;
    trail.current.geometry.attributes.color.needsUpdate = true;
    trail.current.geometry.setDrawRange(0, n / 3);
  });
  return <group>
    <sprite ref={labelPosition} visible={!!labels && !!h.src[frame.current]} scale={[.25, .042, 1]} renderOrder={10}><spriteMaterial map={labelTextures[1]} transparent depthTest={false} toneMapped={false} /></sprite>
    <mesh ref={palm} frustumCulled={false} userData={{ surface: "fitted-from-joints" }}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[palmPositions, 3]} /><bufferAttribute attach="index" args={[new Uint16Array(palmIndices), 1]} /></bufferGeometry>
      <meshStandardMaterial color={color} roughness={.65} side={T.DoubleSide} transparent />
    </mesh>
    <instancedMesh ref={fingers} args={[undefined, undefined, 20]} frustumCulled={false}>
      <cylinderGeometry args={[.82, 1, 1, 12]} /><meshStandardMaterial color={color} roughness={.65} transparent />
    </instancedMesh>
    <instancedMesh ref={spheres} args={[undefined, undefined, 21]} frustumCulled={false}>
      <sphereGeometry args={[1, 10, 8]} /><meshStandardMaterial color="white" transparent />
    </instancedMesh>
    <lineSegments ref={bones} frustumCulled={false}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[boneArray, 3]} /></bufferGeometry>
      <lineDashedMaterial color={color} transparent dashSize={1000} gapSize={0} />
    </lineSegments>
    <lineSegments ref={trail} visible={path} frustumCulled={false}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[trailArray, 3]} /><bufferAttribute attach="attributes-color" args={[trailColors, 3]} /></bufferGeometry>
      <lineBasicMaterial vertexColors blending={T.AdditiveBlending} transparent depthWrite={false} />
    </lineSegments>

  </group>;
}

function Scene({ clip, frame, follow, path, preset, locked, hands = true, labels = true, head: showHead = false }: Omit<Props, "onCanvas">) {
  const controls = useRef<Controls>(null);
  const head = useRef<T.Group>(null);
  const { camera } = useThree();
  const gravity = useMemo(() => {
    const g = clip.scene?.gravity_ref ?? [0, 1, 0];
    return new T.Quaternion().setFromUnitVectors(new T.Vector3(g[0], -g[1], -g[2]).normalize(), new T.Vector3(0, -1, 0));
  }, [clip]);
  const centres = useMemo(() => {
    let previous = new T.Vector3(0, -.25, -.45).applyQuaternion(gravity);
    return Array.from({ length: clip.n_frames }, (_, i) => {
      const points = (["left", "right"] as const).map(s => clip.hands[s].kp3d[i]).filter((k): k is number[] => !!k);
      if (points.length) {
        const mean = new T.Vector3(); points.forEach(k => mean.add(new T.Vector3(k[0], -k[1], -k[2])));
        previous = mean.divideScalar(points.length).applyQuaternion(gravity);
      }
      return previous.clone();
    });
  }, [clip, gravity]);
  const radii = useMemo(() => Array.from({ length: clip.n_frames }, (_, i) => {
    let radius = .15;
    for (const side of ["left", "right"] as const) {
      const k = clip.hands[side].kp3d[i];
      if (k) for (let j = 0; j < 21; j++) radius = Math.max(radius,
        new T.Vector3(k[j * 3], -k[j * 3 + 1], -k[j * 3 + 2]).applyQuaternion(gravity).distanceTo(centres[i]));
    }
    return radius;
  }), [clip, gravity, centres]);
  const movement = useMemo(() => new T.Vector3(), []);
  const viewDirection = useMemo(() => new T.Vector3(), []);
  const floorY = useMemo(() => {
    const heights: number[] = [];
    for (const side of ["left", "right"] as const) {
      clip.hands[side].kp3d.forEach((k, i) => {
        if (k && clip.hands[side].src[i] === 1) heights.push(new T.Vector3(k[0], -k[1], -k[2]).applyQuaternion(gravity).y);
      });
    }
    heights.sort((a, b) => a - b);
    return (heights[Math.floor(heights.length * .1)] ?? -.4) - .05;
  }, [clip, gravity]);
  const frustum = useMemo(() => {
    const c = clip.scene?.camera;
    const w = c ? Math.min(.4, .12 * Math.tan(Math.min(1.2, c.width / (2 * c.fx)))) : .12;
    const h = w * (clip.height / clip.width);
    const corners = [[-w, -h, -.12], [w, -h, -.12], [w, h, -.12], [-w, h, -.12]];
    return new Float32Array(corners.flatMap((p, i) => [0, 0, 0, ...p, ...p, ...corners[(i + 1) % 4]]));
  }, [clip]);
  const headQ = useMemo(() => new T.Quaternion(), []);
  const eyeQ = useMemo(() => new T.Quaternion(), []);
  useEffect(() => {
    const box = new T.Box3();
    for (const side of ["left", "right"] as const) {
      const k = clip.hands[side].kp3d[frame.current];
      if (k) for (let j = 0; j < 21; j++) box.expandByPoint(new T.Vector3(k[j * 3], -k[j * 3 + 1], -k[j * 3 + 2]).applyQuaternion(gravity));
    }
    const c = box.isEmpty() ? centres[frame.current] ?? new T.Vector3() : box.getCenter(new T.Vector3());
    const radius = box.isEmpty() ? .35 : Math.max(.15, box.getSize(new T.Vector3()).length() / 2 + .04);
    const aspect = camera instanceof T.PerspectiveCamera ? camera.aspect : 1;
    const distance = radius / Math.sin(Math.PI / 8) / Math.min(1, aspect) * 1.12;
    controls.current?.target.copy(c);
    const kind = preset.split(":")[0];
    const offsets: Record<string, [number, number, number]> = { Reset: [.15, .25, 1], Behind: [0, .2, 1], Top: [0, 1, .001], Side: [1, .1, 0] };
    const v = offsets[kind] ?? offsets.Reset;
    camera.position.copy(c).add(new T.Vector3(...v).normalize().multiplyScalar(distance));
    camera.up.set(0, 1, 0); camera.lookAt(c); controls.current?.update();
  }, [preset, centres, camera, frame, clip, gravity]);
  useFrame((_, dt) => {
    if (!head.current) return;
    const q = clip.scene?.head_quat_xyzw[frame.current] ?? [0, 0, 0, 1];
    headQ.fromArray(q).premultiply(basis).multiply(basis.clone().invert());
    head.current.quaternion.copy(headQ);
    if (preset.startsWith("Wearer's eye")) {
      camera.position.set(0, 0, 0);
      eyeQ.copy(gravity).multiply(headQ); camera.quaternion.copy(eyeQ);
    } else if (follow && controls.current && !locked) {
      const target = centres[frame.current];
      if (target) {
        const smoothing = 1 - Math.exp(-dt * 4);
        movement.copy(target).sub(controls.current.target).multiplyScalar(smoothing);
        controls.current.target.add(movement); camera.position.add(movement);
        const aspect = camera instanceof T.PerspectiveCamera ? camera.aspect : 1;
        const needed = (radii[frame.current] + .06) / Math.sin(Math.PI / 8) / Math.min(1, aspect) * 1.12;
        viewDirection.copy(camera.position).sub(controls.current.target);
        if (viewDirection.length() < needed) camera.position.addScaledVector(viewDirection.normalize(),
          (needed - camera.position.distanceTo(controls.current.target)) * smoothing);
      }
      controls.current.update();
    }
  });
  return <>
    <color attach="background" args={["#141414"]} />
    <ambientLight intensity={1.2} /><directionalLight position={[1, 3, 2]} intensity={1.5} />
    <gridHelper args={[3, 30, "#454640", "#2e302b"]} position={[0, floorY, 0]} />
    <group quaternion={gravity} onDoubleClick={e => { e.stopPropagation(); controls.current?.target.copy(e.point); }}>
      <group visible={hands}><Hand clip={clip} side="left" frame={frame} path={path} labels={labels && hands} />
      <Hand clip={clip} side="right" frame={frame} path={path} labels={labels && hands} /></group>
      <group ref={head} visible={showHead}>
        <mesh><boxGeometry args={[.045, .035, .025]} /><meshStandardMaterial color={COLORS.sage} /></mesh>
        <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[frustum, 3]} /></bufferGeometry><lineBasicMaterial color={COLORS.pebble} /></lineSegments>
        <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[new Float32Array([0, 0, 0, 0, 0, -1]), 3]} /></bufferGeometry><lineDashedMaterial color={COLORS.pebble} dashSize={.03} gapSize={.02} /></lineSegments>
      </group>
    </group>
    <OrbitControls ref={controls} makeDefault enableDamping enabled={!locked && !preset.startsWith("Wearer's eye")} minDistance={.02} maxDistance={6} />
    <GizmoHelper alignment="bottom-right" margin={[60, 60]}><GizmoViewport axisColors={[COLORS.right, COLORS.mint, COLORS.left]} labelColor="black" /></GizmoHelper>
  </>;
}

export default function Workspace3D(props: Props) {
  return <Canvas gl={{ preserveDrawingBuffer: true, antialias: true }} dpr={[1, 1.5]}
    camera={{ position: [.4, .2, .6], fov: 45, near: .005, far: 30 }}
    onCreated={({ gl }) => props.onCanvas(gl.domElement)}>
    <Scene {...props} />
  </Canvas>;
}
