import type { QualityData } from "./api";

export interface WorkspaceJob {
  id: string; name: string; status: string; stage: string; progress: number; created: number;
  started?: number; error?: string; traceback?: string; queue_position?: number;
  stage_seconds: Record<string, number>; timings: Record<string, number>;
  video?: { fps: number; frames: number; duration_s: number; width: number; height: number };
  quality?: QualityData; render: { status: string; progress?: number; error?: string };
}
export interface Connection {
  connected: boolean; url_host: string | null; gpu: string | null; warm: boolean; queued: number;
  source: string | null; key_present: boolean;
  settings: { url: string; profile: string; auto_render: boolean; retention_h: number };
}
export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return r.json();
}
export const base = (wid: string) => `/api/workspaces/${encodeURIComponent(wid)}`;
