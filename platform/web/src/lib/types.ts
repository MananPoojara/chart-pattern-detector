// Shapes from platform/CONTRACT.md. The chart-facing types (Candle, RunResult, ...)
// are owned by @cpd/chart; app-only ones live here.
export type {
  Time, Candle, Kind, Model, Signal, Board, BoardRow, PaperTrade, RunResult, Analog, YdTrade,
} from "@cpd/chart";

export type Interval = "1D" | "FIVE_MINUTE";
export type Universe = Record<string, string[]>;
export type Health = { ok: boolean; redis: boolean; scanner_age_s: number | null; serve_age_s: number | null };
export type LiveBar = import("@cpd/chart").Candle & { ticker: string; interval: string; final: boolean };
export type YdRow = {
  ticker: string; bar: number; vs_time: number | null; vs_raw: number | null; vs_eod: number | null; eod_day: string | null;
  close: number; turnover: number | null; ok: boolean; why: string; results: boolean; rank: number | null;
};
export type YdScreen = {
  asof: number; session: string; slot: string; universe: number; covered: number; top: string[];
  lists: { asm: boolean; gsm: boolean; band: string }; rows: YdRow[]; error?: string;
};
