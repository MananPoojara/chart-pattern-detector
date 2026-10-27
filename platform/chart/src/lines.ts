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
      if (ok && l.label) label(s, this.pal, l.label, lx + 4 * hr, ly - 7.5 * vr, alpha(col, 0.85), this.pal.bg);
    }
  }
}
