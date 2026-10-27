import { useEffect, useState } from "react";

/** Overlay for a model run; owns its own timer so the page doesn't re-render each second. */
export function RunLoading({ model, slowKind }: { model: string; slowKind: boolean }) {
  const [s, setS] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setS(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="run-loading" role="status">
      <span className="spinner" aria-hidden="true" />
      <div>
        <p>Running {model}… {s > 0 && <span className="muted">{s}s</span>}</p>
        <p className="small muted">
          {s < 4 ? "Cached runs come back instantly." : slowKind ? "First run of a vision or forecast model can take up to a minute." : "First run on this stock takes a few seconds; later views are cached."}
        </p>
      </div>
    </div>
  );
}
