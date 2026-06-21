"use client";

import { useRouter } from "next/navigation";
import { Blueprint } from "@/types/blueprint";

interface BlueprintCardProps {
  blueprint: Blueprint;
}

const CATEGORY_LABEL: Record<Blueprint["category"], string> = {
  habitat: "Habitat",
  infrastructure: "Infrastructure",
  research: "Research",
  production: "Production",
};

const COMPLEXITY_DOTS: Record<Blueprint["complexity"], number> = {
  basic: 1,
  intermediate: 2,
  advanced: 3,
};

export default function BlueprintCard({ blueprint }: BlueprintCardProps) {
  const router = useRouter();

  async function handleSelect() {
    try {
      await fetch("/api/simulation/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blueprint_id: blueprint.id }),
      });
    } catch {
      // Non-blocking — navigate regardless; backend may not be up yet
    }
    router.push(`/simulation/${blueprint.id}`);
  }

  const dots = COMPLEXITY_DOTS[blueprint.complexity];

  return (
    <button
      onClick={handleSelect}
      className="group relative w-full text-left bg-blueprint-bg border border-blueprint-line rounded-sm p-5 card-hover-glow hover:border-blueprint-accent/60 focus:outline-none focus:ring-2 focus:ring-blueprint-accent/40"
    >
      {/* Corner accents */}
      <span className="corner-accent absolute top-1.5 left-1.5 w-2.5 h-2.5 border-t-2 border-l-2 border-blueprint-accent/70" />
      <span className="corner-accent absolute top-1.5 right-1.5 w-2.5 h-2.5 border-t-2 border-r-2 border-blueprint-accent/70" />
      <span className="corner-accent absolute bottom-1.5 left-1.5 w-2.5 h-2.5 border-b-2 border-l-2 border-blueprint-accent/70" />
      <span className="corner-accent absolute bottom-1.5 right-1.5 w-2.5 h-2.5 border-b-2 border-r-2 border-blueprint-accent/70" />

      {/* Header row */}
      <div className="flex items-start justify-between mb-3">
        <span className="text-3xl leading-none select-none text-blueprint-accent/80 group-hover:text-blueprint-accent transition-colors">
          {blueprint.icon}
        </span>
        <span className={`text-xs px-2 py-0.5 rounded-sm font-mono font-medium badge-${blueprint.category}`}>
          {CATEGORY_LABEL[blueprint.category]}
        </span>
      </div>

      {/* Name */}
      <h3 className="font-mono text-sm font-semibold text-white/90 mb-1 group-hover:text-blueprint-accent transition-colors">
        {blueprint.name}
      </h3>

      {/* Description */}
      <p className="text-xs text-blue-200/50 leading-relaxed mb-4 line-clamp-3">
        {blueprint.description}
      </p>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-1.5">
          <span className="text-blue-400/50 uppercase tracking-wider text-[10px]">Complexity</span>
          <span className="flex gap-0.5">
            {[1, 2, 3].map((n) => (
              <span
                key={n}
                className={`w-1.5 h-1.5 rounded-full ${
                  n <= dots
                    ? blueprint.complexity === "advanced"
                      ? "bg-red-400"
                      : blueprint.complexity === "intermediate"
                      ? "bg-yellow-400"
                      : "bg-green-400"
                    : "bg-blue-900"
                }`}
              />
            ))}
          </span>
        </div>
        <span className="text-blue-400/50 text-[10px] uppercase tracking-wider">
          EST. {blueprint.estimatedDuration}
        </span>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mt-3">
        {blueprint.tags.map((tag) => (
          <span
            key={tag}
            className="text-[10px] font-mono px-1.5 py-0.5 bg-blue-950/60 text-blue-400/70 rounded-sm border border-blue-900/40"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Deploy button overlay */}
      <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-sm bg-blueprint-bg/60 backdrop-blur-[1px]">
        <span className="font-mono text-sm font-semibold text-blueprint-accent border border-blueprint-accent/60 px-5 py-2 rounded-sm bg-blueprint-bg/80">
          DEPLOY SIMULATION &rarr;
        </span>
      </div>
    </button>
  );
}
