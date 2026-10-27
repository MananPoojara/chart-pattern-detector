import { groupModels } from "../lib/models";
import type { Model } from "../lib/types";

/** Compact grouped native radios: arrow keys move between models, Tab leaves the group. */
export function ModelPicker({ models, value, onChange }: { models: Model[]; value?: string; onChange: (id: string) => void }) {
  if (!models.length) return <div className="picker-skel" aria-busy="true" aria-label="Loading models" />;
  const current = models.find((m) => m.id === value);
  return (
    <fieldset className="picker" aria-describedby="picker-how">
      <legend className="sr-only">Model</legend>
      {groupModels(models).map((g) => (
        <div className="picker-group" key={g.label} role="group" aria-label={g.label}>
          <span className="picker-label" aria-hidden="true">{g.label}</span>
          <div className="picker-grid">
            {g.models.map((m) => (
              <label key={m.id} className="picker-item" data-checked={m.id === value} title={m.how}>
                <input type="radio" name="model" value={m.id} checked={m.id === value} onChange={() => onChange(m.id)} />
                <span className="pi-name">{m.name}</span>
                {!m.intraday && <span className="pi-tag" aria-label="daily only">1D</span>}
              </label>
            ))}
          </div>
        </div>
      ))}
      {current && <p id="picker-how" className="pi-how">{current.how}</p>}
    </fieldset>
  );
}
