// Mirrors platform/CONTRACT.md "Shared shapes". Keep in sync with the contract, never ahead of it.
export type Time = string | number; // daily "YYYY-MM-DD" | intraday epoch s of IST wall clock as UTC
export type Pt = { time: Time; value: number };
export type Candle = { time: Time; open: number; high: number; low: number; close: number; volume: number };
export type Kind = "rule" | "lines" | "vision" | "forecast" | "edge" | "plan";
export type Model = { id: string; name: string; kind: Kind; url: string; how: string; intraday: boolean };

export type Marker = { time: Time; dir: 1 | -1 | 0; text: string };
export type Box = { start: Time; end: Time; top: number; bottom: number; label: string; dir: 1 | -1 | 0; conf: number; live: boolean };
export type Line = { points: Pt[]; dir: 1 | -1 | 0; label?: string };
export type Level = { title: "entry" | "stop" | "target" | string; price: number };
export type Forecast = { median: Pt[]; lo: Pt[]; hi: Pt[] };
export type Score = Pt[];
export type PaperTrade = {
  signal: Time; why: string; entry: number; stop: number; target: number;
  status: string; detail: string; R: number | null; entryTime: Time | null;
};

export type Analog = {
  start: Time; end: Time;          // the past window (end = its "today")
  sim: number; ctx: number;        // shape similarity / context agreement with today, 0..1
  fwd: number;                     // return over the next H bars after `end` (fraction)
  mfe: number; mae: number;        // best / worst move within H bars (fractions)
  outcome: "target" | "stop" | "time";
  ma50Slope: number;               // 50-DMA slope at `end` (% per 10 bars)
  results?: { q: string; salesYoY: number | null; profitYoY: number | null } | null;
  patterns: string[];
};

export type RunResult = {
  id: string; model: string; kind: Kind; ticker: string; interval: string; asof: Time;
  candles: Candle[];
  markers: Marker[]; boxes: Box[]; lines: Line[]; levels: Level[];
  forecast: Forecast | null; score: Score | null;
  paperTrade: PaperTrade | null;
  note: string;
  stats?: Record<string, number | string>;
  drivers?: { name: string; share: number; value: number }[];
  analogs?: Analog[];   // edge: best first, <= 10
  flags?: string[];     // edge: plain-English context mismatches
  error?: string;
};

export type Signal = {
  id: string; time: Time; ticker: string; interval: string; model: string; signal: string;
  dir: 1 | -1; close: number; stop: number; target: number;
};
export type BoardRow = {
  ticker: string; rank: number; group: string; p_win: number; eR: number; action: "BUY" | "NO TRADE";
  close: number; chg: number; entry: number; stop: number; target: number; spark: number[];
};
export type Board = { asof: string; rows: BoardRow[]; universe: number };
