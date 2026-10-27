import { useSyncExternalStore } from "react";

export type Theme = "dark" | "light";
const listeners = new Set<() => void>();
const KEY = "cpd-theme";

export function setTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(KEY, t); } catch { /* storage blocked: keep in-memory */ }
  listeners.forEach((l) => l());
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; },
    () => (document.documentElement.dataset.theme === "light" ? "light" : "dark"),
  );
}
