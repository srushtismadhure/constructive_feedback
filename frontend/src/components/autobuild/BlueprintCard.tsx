import BlueprintVisual from "@/components/autobuild/BlueprintVisual";
import { Blueprint } from "@/types/blueprint";

interface BlueprintCardProps {
  blueprint: Blueprint;
  selected: boolean;
  onSelect: (id: string) => void;
}

export default function BlueprintCard({ blueprint, selected, onSelect }: BlueprintCardProps) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={() => onSelect(blueprint.id)}
      className={`autobuild-card group ${selected ? "autobuild-card--selected" : ""}`}
    >
      <div className="flex items-center justify-between px-4 pt-4 font-mono text-[10px] tracking-[0.18em]">
        <span className={selected ? "text-regolith" : "text-muted-text"}>{blueprint.icon}</span>
        <span className="text-muted-text">{blueprint.category.toUpperCase()}</span>
      </div>

      <div className="px-3 pt-3">
        <BlueprintVisual variant={blueprint.visual} compact />
      </div>

      <div className="p-4 pt-3">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h3 className="text-left text-base font-bold uppercase tracking-[-0.02em] text-off-white">
              {blueprint.name}
            </h3>
            <p className="mt-1 text-left text-xs leading-5 text-muted-text">{blueprint.purpose}</p>
          </div>
          <span className={`selection-node ${selected ? "selection-node--active" : ""}`} aria-hidden="true" />
        </div>

        <dl className="grid grid-cols-2 gap-x-3 gap-y-3 border-t border-white/10 pt-4 font-mono text-[10px] uppercase tracking-[0.08em]">
          <div>
            <dt className="text-muted-text">Area</dt>
            <dd className="mt-1 text-soft-gray">{blueprint.area}</dd>
          </div>
          <div>
            <dt className="text-muted-text">Capacity</dt>
            <dd className="mt-1 text-soft-gray">{blueprint.capacity}</dd>
          </div>
          <div>
            <dt className="text-muted-text">Build time</dt>
            <dd className="mt-1 text-soft-gray">{blueprint.estimatedDuration}</dd>
          </div>
          <div>
            <dt className="text-muted-text">Material</dt>
            <dd className="mt-1 truncate text-soft-gray">{blueprint.material}</dd>
          </div>
        </dl>
      </div>
    </button>
  );
}
