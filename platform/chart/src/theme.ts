export type ThemeName = "dark" | "light";
export type Palette = {
  bg: string; grid: string; text: string; muted: string; border: string;
  up: string; down: string; accent: string; crosshair: string; font: string;
};

// Teal vs. orange: separable under deuteranopia/protanopia, unlike red/green.
const DEFAULTS: Record<ThemeName, Omit<Palette, "font">> = {
  dark: {
    bg: "#0b0e14", grid: "rgba(160,170,190,0.06)", text: "#9aa3b5", muted: "#5c6578", border: "#1c2230",
    up: "#26b5c4", down: "#f08a3c", accent: "#e8c35a", crosshair: "#7d879c",
  },
  light: {
    bg: "#fbfbfc", grid: "rgba(20,30,50,0.06)", text: "#3d4554", muted: "#8a92a2", border: "#e3e6ec",
    up: "#0e8796", down: "#d4601f", accent: "#b7860b", crosshair: "#6b7385",
  },
};

/** Defaults overridden by CSS custom properties on (or inherited by) `el`: --chart-up, --chart-bg, ... --font-mono. */
export function readPalette(el: HTMLElement, theme: ThemeName): Palette {
  const cs = getComputedStyle(el);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  const p = { ...DEFAULTS[theme] } as Palette;
  for (const k of Object.keys(p) as (keyof Palette)[]) p[k] = v(`--chart-${k}`) || p[k];
  p.font = v("--font-mono") || "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  return p;
}

/** "#rrggbb" | "rgb(...)" -> rgba with alpha. Other formats are returned unchanged. */
export function alpha(c: string, a: number): string {
  if (c[0] === "#" && c.length === 7) {
    const n = parseInt(c.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  const m = c.match(/^rgba?\(([^,]+),([^,]+),([^,)]+)/);
  return m ? `rgba(${m[1]},${m[2]},${m[3]},${a})` : c;
}

export const dirColor = (p: Palette, dir: number) => (dir > 0 ? p.up : dir < 0 ? p.down : p.accent);
