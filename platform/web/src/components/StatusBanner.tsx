import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";
import { api } from "../lib/api";
import { nseSessionOpen } from "../lib/format";
import type { Health } from "../lib/types";

/** Polls /api/health; shows a banner only when something is actually wrong. */
export function StatusBanner() {
  const [h, setH] = useState<Health | "down" | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () => api.health().then((x) => alive && setH(x), () => alive && setH("down"));
    check();
    const id = setInterval(check, 30_000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  let msg: string | null = null, level = "error";
  if (h === "down") msg = "Can't reach the PatternEdge server. Charts and prices will load once it is back; retrying every 30 s.";
  else if (h && !h.redis) msg = "The server is up but its data store is offline. Model runs and live prices are unavailable.";
  else if (h && nseSessionOpen() && (h.scanner_age_s == null || h.scanner_age_s > 120)) {
    msg = "The live scanner is not reporting. Prices may be stale until it reconnects.";
    level = "warn";
  }
  if (!msg) return null;
  return (
    <div className="banner" role="status" data-level={level}>
      <Icon icon="solar:danger-triangle-linear" width={16} aria-hidden="true" />
      {msg}
    </div>
  );
}
