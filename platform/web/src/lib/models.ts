import type { Kind, Model } from "./types";

export const GROUPS: { label: string; kinds: Kind[] }[] = [
  { label: "Our models", kinds: ["plan", "edge"] },
  { label: "Rule detectors", kinds: ["rule", "lines"] },
  { label: "Vision (YOLO)", kinds: ["vision"] },
  { label: "Forecasters", kinds: ["forecast"] },
];

export function groupModels(models: Model[]) {
  return GROUPS.map((g) => ({ label: g.label, models: models.filter((m) => g.kinds.includes(m.kind)) }))
    .filter((g) => g.models.length);
}

// 1D opens on TradePlan; 5m live opens on ChartScanAI (TradePlan/PatternEdge are daily-only)
export const defaultModel = (models: Model[], intraday = false) =>
  (intraday ? models.find((m) => m.id === "ChartScanAI") ?? models.find((m) => m.intraday)
            : models.find((m) => m.kind === "plan")) ?? models[0];
