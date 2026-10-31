export type Variant = "pretrained" | "finetuned";

export interface ClipSummary {
  key: string;
  name: string;
  variant: Variant;
  origin: "a100" | "upload";
  model_label: string;
  frames: number | null;
  fps: number | null;
  has_raw: boolean;
}

export interface HandTrack {
  src: number[]; // 0 none, 1 observed, 2 infilled, 3 estimated
  conf: number[];
  state: string[];
  kp2d: (number[] | null)[]; // 42 = 21 x (x, y) in source pixels
  kp3d: (number[] | null)[]; // 63 = 21 x (x, y, z) metres, head-stable frame (x right, y down, z forward)
}

export interface ClipData extends ClipSummary {
  n_frames: number;
  fps: number;
  width: number;
  height: number;
  camera_source: string;
  overlay_baked: boolean;
  t: number[];
  hands: { left: HandTrack; right: HandTrack };
  stats: Record<string, any>;
}

export interface ClipMetrics {
  key: string;
  name: string;
  variant: Variant;
  model_label: string;
  frames: number;
  fps: number;
  observed_left: number;
  observed_right: number;
  infilled: number;
  estimated: number;
  out_of_view: number;
  jitter_mm: number | null;
  depth_jumps: number;
  tracklets_total: number;
  tracklets_rejected: number;
  tracklets_left: number;
  tracklets_right: number;
  camera_source: string;
  z_ref_m: number;
  seconds: Record<string, number>;
}

export interface Aggregate {
  clips: number;
  frames: number;
  minutes: number;
  tracklets_total: number;
  tracklets_rejected: number;
  reject_rate: number | null;
  depth_jumps: number;
  depth_jumps_per_1k_frames: number | null;
  jitter_mm_median: number | null;
  observed_mean: number | null;
  estimated_mean: number;
  infilled_mean: number;
}

export interface Metrics {
  clips: ClipMetrics[];
  aggregate: { pretrained: Aggregate | null; finetuned: Aggregate | null };
  paired_clips: string[];
  synthetic_test: Record<string, any> | null;
}

export interface ModelInfo {
  variant: Variant;
  label: string;
  wilor_ckpt: string | null;
  detector: string | null;
}

export interface Job {
  id: string;
  filename: string;
  camera: "phone" | "fisheye";
  status: "queued" | "running" | "done" | "error";
  progress: number;
  stage: string;
  error: string | null;
  eta_s: number | null;
  frames?: number;
  started?: number;
  created: number;
  clip_key?: string;
  model_label: string;
}

export interface Training {
  available: boolean;
  setup?: Record<string, any> | null;
  dataset?: Record<string, any> | null;
  curves?: { file: string; x: string; series: string[]; points: Record<string, number | null>[] }[];
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  model: () => get<ModelInfo>("/api/model"),
  clips: () => get<ClipSummary[]>("/api/clips"),
  clip: (key: string) => get<ClipData>(`/api/clips/${key}`),
  metrics: () => get<Metrics>("/api/metrics"),
  training: () => get<Training>("/api/training"),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),
  videoUrl: (key: string) => `/api/clips/${key}/video`,
  downloadUrl: (key: string, name: string) => `/api/clips/${key}/download/${name}`,
  async upload(file: File, camera: string): Promise<Job> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("camera", camera);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `Upload failed (${r.status})`);
    return r.json();
  },
};

/** Shared palette: hands, sources and charts all use design-system colours. */
export const COLORS = {
  left: "#041668", // Cobalt Ink
  right: "#3f0929", // Wine Shadow
  observed: "#003d21", // Forest Depths
  infilled: "#d3beff", // Lilac Wash
  estimated: "#ffc0e6", // Rose Wash
  none: "#edeee1", // Bone
  mint: "#aafdc0",
  sky: "#b0f4ff",
  sage: "#c0c2a9",
  pebble: "#7c7d76",
  charcoal: "#202020",
  cream: "#f8f9eb",
};
export const SRC_NAMES = ["none", "observed", "infilled", "estimated"] as const;

export const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
