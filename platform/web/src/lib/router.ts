// Tiny history router: path + query live in the URL so every view is shareable.
import { useSyncExternalStore, type MouseEvent } from "react";

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
addEventListener("popstate", notify);

export function navigate(to: string, { replace = false } = {}) {
  if (to === location.pathname + location.search) return;
  history[replace ? "replaceState" : "pushState"](null, "", to);
  notify();
  if (!replace) scrollTo(0, 0);
}

export function useLocation() {
  const href = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; },
    () => location.pathname + location.search,
  );
  const url = new URL(href, location.origin);
  return { path: url.pathname, query: url.searchParams };
}

/** Anchor that routes client-side on plain left-click. */
export function linkProps(to: string) {
  return {
    href: to,
    onClick(e: MouseEvent) {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      navigate(to);
    },
  };
}
