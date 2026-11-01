/** Black monoline glyphs set inline with display type (design system: "Inline Graphic Glyph"). */
export function EyeGlyph({ size = "0.8em" }: { size?: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" style={{ display: "inline-block", height: size, width: size, verticalAlign: "-0.06em" }}>
      <rect x="2" y="2" width="44" height="44" rx="14" fill="#aafdc0" stroke="#000" strokeWidth="3.5" />
      <path d="M9 24c4-7 9-10 15-10s11 3 15 10c-4 7-9 10-15 10S13 31 9 24z" fill="none" stroke="#000" strokeWidth="3.5" />
      <circle cx="24" cy="24" r="4.5" fill="#000" />
    </svg>
  );
}

export function HandGlyph({ size = "0.8em" }: { size?: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" style={{ display: "inline-block", height: size, width: size, verticalAlign: "-0.06em" }}>
      <circle cx="24" cy="24" r="22" fill="#d3beff" stroke="#000" strokeWidth="3.5" />
      <g stroke="#000" strokeWidth="3.5" strokeLinecap="round" fill="none">
        <path d="M24 36V22M24 22l-7-8M24 22l-2-11M24 22l4-10M24 22l9-6M24 30l-8-3" />
      </g>
      <circle cx="24" cy="36" r="3" fill="#000" />
    </svg>
  );
}

export function ArrowChip({ size = "0.8em" }: { size?: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 72 48" style={{ display: "inline-block", height: size, width: `calc(${size} * 1.5)`, verticalAlign: "-0.06em" }}>
      <rect x="2" y="2" width="68" height="44" rx="22" fill="#b0f4ff" stroke="#000" strokeWidth="3.5" />
      <path d="M18 24h34M40 13l12 11-12 11" fill="none" stroke="#000" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
