export default function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    plus: "M12 5v14M5 12h14", workspace: "M3 4h18v16H3zM12 4v16",
    room: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
    library: "M3 4h7v7H3zM14 4h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
    compare: "M3 17l5-6 4 3 4-7 5 5M3 12l5 3 4-5 4 2 5-4",
    export: "M12 15V3M7 8l5-5 5 5M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5",
    panel: "M3 4h18v16H3zM15 4v16", left: "M15 6l-6 6 6 6", right: "M9 6l6 6-6 6",
    down: "M6 9l6 6 6-6", play: "M7 4.5l13 7.5-13 7.5z", pause: "M8 5v14M16 5v14",
    prev: "M6 5v14M18 5L9 12l9 7z", next: "M18 5v14M6 5l9 7-9 7z",
    settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2",
    check: "M5 12l4 4L19 6", target: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8",
    cross: "M4 7h13l-3-3M20 17H7l3 3", eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] ?? paths.workspace} /></svg>;
}

export function BrandMark({ large = false }: { large?: boolean }) {
  return <span className={`brand-mark ${large ? "large" : ""}`}><Icon name="plus" size={large ? 58 : 13} /></span>;
}
