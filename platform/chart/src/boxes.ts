// YOLO-style labelled rectangles (vision kind), like ChartScanAI's own output.
// Edge analogs reuse them: past windows dashed, no conf suffix, plus a thin bracket over the H bars that followed.
import type { Logical } from "lightweight-charts";
import { Prim, label, type Scope } from "./prim";
import { alpha, dirColor } from "./theme";
import type { Box } from "./types";

type R = { x: number; y: number; w: number; h: number };
const hit = (a: R, b: R) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

export class BoxesPrimitive extends Prim {
  private boxes: Box[] = [];
  private analog = false;
  private after = 0;
  /** analog: edge style; after: bars covered by the bracket after each past box (0 = none) */
  set(boxes: Box[], { analog = false, after = 0 } = {}) {
    this.analog = analog; this.after = after;
    // live boxes last so they paint (and claim label space) on top of history
    this.boxes = [...boxes].sort((a, b) => +a.live - +b.live);
    this.update();
  }
  protected draw(s: Scope) {
    const c = s.context, hr = s.horizontalPixelRatio, vr = s.verticalPixelRatio, hb = this.halfBar();
    const W = s.bitmapSize.width, H = s.bitmapSize.height;
    const placed: R[] = [];
    const geo = this.boxes.map((b) => {
      const x1 = this.x(b.start), x2 = this.x(b.end), y1 = this.y(b.top), y2 = this.y(b.bottom);
      if (x1 == null || x2 == null || y1 == null || y2 == null) return null;
      const r = { x: Math.round((x1 - hb) * hr), y: Math.round(y1 * vr), w: Math.round((x2 - x1 + 2 * hb) * hr), h: Math.round((y2 - y1) * vr) };
      return r.x > W || r.x + r.w < 0 || r.y > H || r.y + r.h < 0 ? null : r;
    });
    this.boxes.forEach((b, i) => {
      const r = geo[i]; if (!r) return;
      const col = dirColor(this.pal, b.dir);
      c.lineWidth = Math.max(1, Math.round((b.live ? 2 : 1) * hr));
      c.strokeStyle = b.live ? col : alpha(col, 0.45);
      c.fillStyle = alpha(col, b.live ? 0.16 : 0.04);
      c.fillRect(r.x, r.y, r.w, r.h);
      c.setLineDash(this.analog && !b.live ? [4 * hr, 3 * hr] : []);
      c.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
      c.setLineDash([]);
      if (this.analog && !b.live && this.after > 0) {
        // bracket ⌊__⌋ under the box's bottom edge, from its end to end + H bars: "what happened next"
        const xe = this.chart!.timeScale().logicalToCoordinate((this.ti.idx(b.end) + this.after) as Logical);
        if (xe == null) return;
        const x0 = r.x + r.w + 2 * hr, x1 = Math.round(xe * hr), y = r.y + r.h, t = 4 * vr;
        c.lineWidth = Math.max(1, Math.round(hr));
        c.strokeStyle = alpha(col, 0.6);
        c.beginPath(); c.moveTo(x0, y - t); c.lineTo(x0, y); c.lineTo(x1, y); c.lineTo(x1, y - t); c.stroke();
      }
    });
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      const r = geo[i], b = this.boxes[i]; if (!r) continue;
      const text = this.analog ? b.label : `${b.label} ${b.conf.toFixed(2)}`;
      const col = dirColor(this.pal, b.dir);
      c.font = `${11 * vr}px ${this.pal.font}`;
      const w = c.measureText(text).width + 8 * vr, h = 15 * vr;
      // try above-left, then below-left; skip if both collide with an already placed tab
      const spot = [{ x: r.x, y: r.y - h, w, h }, { x: r.x, y: r.y + r.h, w, h }].find((t) => t.y >= 0 && t.y + h <= H && !placed.some((p) => hit(p, t)));
      if (!spot) continue;
      placed.push(spot);
      label(s, this.pal, text, spot.x, spot.y, b.live ? col : alpha(col, 0.55), b.live ? this.pal.bg : this.pal.text);
    }
  }
}
