import { useEffect, useRef, type RefObject } from "react";
import { createPatternChart, type PatternChart } from "@cpd/chart";
import { useTheme } from "../lib/theme";
import type { RunResult } from "../lib/types";

/** Mounts @cpd/chart once; data flows in through the ref, never through React re-renders. */
export function Chart({ result, chartRef }: { result: RunResult | null; chartRef: RefObject<PatternChart | null> }) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  useEffect(() => {
    const c = createPatternChart(el.current!, { theme });
    chartRef.current = c;
    return () => { c.destroy(); chartRef.current = null; };
    // theme changes go through setTheme below, not re-creation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { if (result?.candles?.length) chartRef.current?.setResult(result); }, [result, chartRef]);
  useEffect(() => { chartRef.current?.setTheme(theme); }, [theme, chartRef]);

  return <div ref={el} className="chart-host" />;
}
