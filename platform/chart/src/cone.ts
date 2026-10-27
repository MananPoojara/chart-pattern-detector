// Forecast: shaded lo..hi cone + median line, starting at the last candle.
import type { AutoscaleInfo, Logical } from "lightweight-charts";
import { Prim, type Scope } from "./prim";
import { alpha } from "./theme";
import type { Forecast, Pt } from "./types";

export class ConePrimitive extends Prim {
  private f: Forecast | null = null;
  set(f: Forecast | null) { this.f = f; this.update(); }
  autoscaleInfo(_from: Logical, to: Logical): AutoscaleInfo | null {
    // only while the cone is on screen, so scrolling back to a 2019 analog rescales to 2019 prices
    if (!this.f?.lo.length || to < this.ti.idx(this.f.lo[0].time)) return null;
    return { priceRange: { minValue: Math.min(...this.f.lo.map((p) => p.value)), maxValue: Math.max(...this.f.hi.map((p) => p.value)) } };
  }
  private pts(s: Scope, ps: Pt[]) {
    return ps.flatMap((p) => {
      const x = this.x(p.time), y = this.y(p.value);
      return x == null || y == null ? [] : [[x * s.horizontalPixelRatio, y * s.verticalPixelRatio]];
    });
  }
  protected draw(s: Scope) {
    if (!this.f) return;
    const c = s.context, col = this.pal.accent;
    const hi = this.pts(s, this.f.hi), lo = this.pts(s, this.f.lo).reverse(), med = this.pts(s, this.f.median);
    if (hi.length && lo.length) {
      c.beginPath();
      [...hi, ...lo].forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
      c.closePath();
      c.fillStyle = alpha(col, 0.13); c.fill();
      c.lineWidth = Math.max(1, Math.round(s.horizontalPixelRatio));
      c.strokeStyle = alpha(col, 0.4); c.stroke();
    }
    c.beginPath();
    med.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
    c.lineWidth = Math.round(2 * s.horizontalPixelRatio);
    c.strokeStyle = col; c.stroke();
  }
}
