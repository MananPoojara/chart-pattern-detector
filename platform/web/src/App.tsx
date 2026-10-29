import { lazy, Suspense } from "react";
import { useLocation } from "./lib/router";
import { StatusBanner } from "./components/StatusBanner";
import { Home } from "./pages/Home";

const loadTerminal = () => import("./pages/Terminal");
const Terminal = lazy(loadTerminal);
const Yd = lazy(() => import("./pages/Yd"));
// Warm the terminal chunk once the home page is idle so "Open the terminal" paints at once.
if (!location.pathname.startsWith("/t/")) (window.requestIdleCallback ?? setTimeout)(() => void loadTerminal());

export function App() {
  const { path } = useLocation();
  const m = path.match(/^\/t\/([^/]+)\/?$/);
  const yd = path.match(/^\/yd(?:\/([^/]+))?\/?$/);
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <StatusBanner />
      {yd ? (
        <Suspense fallback={<div className="term-skel" aria-busy="true" aria-label="Loading Yd"><header /><div><span /><span /><span /></div></div>}>
          <Yd ticker={decodeURIComponent(yd[1] ?? "RELIANCE").toUpperCase()} />
        </Suspense>
      ) : m ? (
        <Suspense fallback={<div className="term-skel" aria-busy="true" aria-label="Loading terminal"><header /><div><span /><span /><span /></div></div>}>
          <Terminal ticker={decodeURIComponent(m[1]).toUpperCase()} />
        </Suspense>
      ) : (
        <Home />
      )}
    </>
  );
}
