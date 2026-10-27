import { useId, useMemo, useState } from "react";
import { Icon } from "@iconify/react";

/** Accessible combobox over the symbol universe (arrow keys, Enter, Escape). */
export function SymbolSearch({ symbols, onPick }: { symbols: string[]; onPick: (t: string) => void }) {
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const id = useId();

  const results = useMemo(() => {
    const s = q.trim().toUpperCase();
    if (!s) return [];
    const starts = symbols.filter((t) => t.startsWith(s));
    const has = symbols.filter((t) => !t.startsWith(s) && t.includes(s));
    return [...starts, ...has].slice(0, 30);
  }, [q, symbols]);

  const open = results.length > 0;
  const pick = (t: string) => { setQ(""); setActive(0); onPick(t); };

  return (
    <div className="search">
      <label htmlFor={`${id}-in`} className="sr-only">Search NSE 500 symbols</label>
      <Icon icon="solar:magnifer-linear" width={16} aria-hidden="true" className="search-icon" />
      <input
        id={`${id}-in`}
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-autocomplete="list"
        aria-activedescendant={open ? `${id}-${active}` : undefined}
        placeholder={symbols.length ? `Search ${symbols.length} symbols` : "Search symbols"}
        autoComplete="off"
        spellCheck={false}
        value={q}
        onChange={(e) => { setQ(e.target.value); setActive(0); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && open) { e.preventDefault(); setActive((a) => (a + 1) % results.length); }
          else if (e.key === "ArrowUp" && open) { e.preventDefault(); setActive((a) => (a - 1 + results.length) % results.length); }
          else if (e.key === "Enter") {
            const t = results[active] ?? (symbols.includes(q.trim().toUpperCase()) ? q.trim().toUpperCase() : undefined);
            if (t) pick(t);
          } else if (e.key === "Escape") setQ("");
        }}
      />
      {open && (
        <ul id={`${id}-list`} role="listbox" className="search-list" aria-label="Matching symbols">
          {results.map((t, i) => (
            <li key={t} id={`${id}-${i}`} role="option" aria-selected={i === active}
              onMouseDown={(e) => { e.preventDefault(); pick(t); }} onMouseEnter={() => setActive(i)}>
              {t}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
