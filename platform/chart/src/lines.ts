// Polylines: stock-pattern pivots (rule) and support/resistance trendlines (lines).
import { Prim, label, type Scope } from "./prim";
import { alpha, dirColor } from "./theme";
import type { Line } from "./types";

export class LinesPrimitive extends Prim {
  private lines: Line[] = [];
  private width = 1;
  set(lines: Line[], width: number) { this.lines = lines; this.width = width; this.update(); }
  protected draw(s: Scope) {
    const c = s.context, hr = s.horizontalPixelRatio, vr = s.verticalPixelRatio;
    c.lineWidth = Math.max(1, Math.round(this.width * hr));
    c.lineJoin = "round";
    const placed: { x: number; y: number; w: number; h: number }[] = [];
    for (const l of this.lines) {
      const col = dirColor(this.pal, l.dir);
      c.strokeStyle = this.width > 1 ? col : alpha(col, 0.8);
      c.beginPath();
      let lx = 0, ly = 0, ok = false;
      for (const p of l.points) {
        const x = this.x(p.time), y = this.y(p.value);
        if (x == null || y == null) continue;
        lx = x * hr; ly = y * vr;
        ok ? c.lineTo(lx, ly) : c.moveTo(lx, ly);
        ok = true;
      }
      c.stroke();
      if (!ok || !l.label) continue;
      // labels of lines ending at similar prices (OR low, VWAP) step down until they no longer overlap
      c.font = `${11 * vr}px ${this.pal.font}`;
      const w = c.measureText(l.label).width + 8 * vr, h = 15 * vr;
      const x = lx + 4 * hr + w <= s.bitmapSize.width ? lx + 4 * hr : lx - 4 * hr - w;  // flip left at the price axis
      let y = ly - 7.5 * vr;
      while (placed.some((p) => x < p.x + p.w && p.x < x + w && y < p.y + p.h && p.y < y + h)) y += h + 2 * vr;
      placed.push(label(s, this.pal, l.label, x, y, alpha(col, 0.85), this.pal.bg));
    }
  }
}
