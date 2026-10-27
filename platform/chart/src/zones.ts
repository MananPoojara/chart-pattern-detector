// Trade plan zones: entry->target and entry->stop shaded from the signal bar to the right edge,
// plus a dashed vertical marker at the signal bar (paper trades).
import type { AutoscaleInfo, Logical } from "lightweight-charts";
import { Prim, type Scope } from "./prim";
import { alpha } from "./theme";
import type { Time } from "./types";

export type ZoneData = { signal: Time; entry?: number; stop?: number; target?: number; shade: boolean; vline: boolean };

export class ZonesPrimitive extends Prim {
  private z: ZoneData | null = null;
  set(z: ZoneData | null) { this.z = z; this.update(); }
  autoscaleInfo(_from: Logical, to: Logical): AutoscaleInfo | null {
    if (!this.z || to < this.ti.idx(this.z.signal)) return null; // zone off screen (scrolled back in history)
    const v = [this.z?.entry, this.z?.stop, this.z?.target].filter((x): x is number => x != null);
    return v.length ? { priceRange: { minValue: Math.min(...v), maxValue: Math.max(...v) } } : null;
  }
  protected draw(s: Scope) {
    const z = this.z; if (!z) return;
    const c = s.context, hr = s.horizontalPixelRatio, vr = s.verticalPixelRatio;
    const x0 = this.x(z.signal); if (x0 == null) return;
    const X = Math.round(x0 * hr), W = s.bitmapSize.width;
    const band = (a?: number, b?: number, col?: string) => {
      if (a == null || b == null || !col) return;
      const ya = this.y(a), yb = this.y(b); if (ya == null || yb == null) return;
      c.fillStyle = col;
      c.fillRect(X, Math.round(Math.min(ya, yb) * vr), W - X, Math.round(Math.abs(yb - ya) * vr));
    };
    if (z.shade) {
      band(z.entry, z.target, alpha(this.pal.up, 0.12));
      band(z.entry, z.stop, alpha(this.pal.down, 0.12));
    }
    if (!z.vline) return;
    c.strokeStyle = alpha(this.pal.accent, 0.8);
    c.lineWidth = Math.max(1, Math.round(hr));
    c.setLineDash([4 * hr, 3 * hr]);
    c.beginPath(); c.moveTo(X + 0.5, 0); c.lineTo(X + 0.5, s.bitmapSize.height); c.stroke();
    c.setLineDash([]);
  }
}
