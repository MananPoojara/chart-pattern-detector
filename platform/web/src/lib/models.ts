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

export const defaultModel = (models: Model[]) => models.find((m) => m.kind === "plan") ?? models[0];
