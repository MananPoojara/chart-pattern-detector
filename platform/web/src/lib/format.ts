import type { Time } from "./types";

const inr = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const price = (n: number | null | undefined) => (n == null || !isFinite(n) ? "—" : inr.format(n));
export const pct = (n: number, digits = 2) => `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(digits)}%`;
export const signed = (n: number, digits = 2) => `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(digits)}`;
export const dirClass = (n: number) => (n > 0 ? "up" : n < 0 ? "down" : "flat");

/** Contract times: "YYYY-MM-DD" (daily) or epoch seconds of IST wall clock treated as UTC. */
export function fmtTime(t: Time | null | undefined): string {
  if (t == null) return "—";
  if (typeof t === "string") return t;
  if (typeof t === "number") {
    const d = new Date(t * 1000);
    return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)}`;
  }
  return String(t);
}

/** NSE cash session in IST, Mon–Fri 09:15–15:30 (exchange holidays not known client-side). */
export function nseSessionOpen(now = new Date()): boolean {
  const ist = new Date(now.getTime() + (5.5 * 60 + now.getTimezoneOffset()) * 60_000);
  const day = ist.getDay();
  const m = ist.getHours() * 60 + ist.getMinutes();
  return day >= 1 && day <= 5 && m >= 9 * 60 + 15 && m < 15 * 60 + 30;
}
