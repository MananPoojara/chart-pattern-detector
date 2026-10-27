// Shapes from platform/CONTRACT.md. The chart-facing types (Candle, RunResult, ...)
// are owned by @cpd/chart; app-only ones live here.
export type {
  Time, Candle, Kind, Model, Signal, Board, BoardRow, PaperTrade, RunResult, Analog,
} from "@cpd/chart";

export type Interval = "1D" | "FIVE_MINUTE";
export type Universe = Record<string, string[]>;
export type Health = { ok: boolean; redis: boolean; scanner_age_s: number | null; serve_age_s: number | null };
export type LiveBar = import("@cpd/chart").Candle & { ticker: string; interval: string; final: boolean };
