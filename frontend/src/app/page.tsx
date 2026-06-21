import BlueprintCard from "@/components/BlueprintCard";
import { BLUEPRINTS } from "@/data/blueprints";

export default function HomePage() {
  return (
    <main className="min-h-screen blueprint-grid relative">
      {/* Radial vignette */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_center,_transparent_40%,_rgba(0,0,0,0.7)_100%)]" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-16">
        {/* Header */}
        <header className="mb-16 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 border border-blueprint-line rounded-sm mb-6 bg-blueprint-bg/60">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse-slow" />
            <span className="font-mono text-xs text-blue-400/70 uppercase tracking-widest">
              Mars Surface Operations — AI Construction Unit
            </span>
          </div>

          <h1 className="font-mono text-4xl md:text-5xl font-bold text-white/90 mb-4 tracking-tight">
            Select a{" "}
            <span className="text-blueprint-accent">Blueprint</span>
          </h1>
          <p className="text-blue-200/50 max-w-xl mx-auto text-sm leading-relaxed">
            Choose a construction preset. An AI agent will configure and run the
            build simulation in the MuJoCo environment. Live feed will stream
            once the simulation initializes.
          </p>
        </header>

        {/* Filter hint */}
        <div className="flex flex-wrap gap-3 justify-center mb-10 font-mono text-xs">
          {(["All", "Habitat", "Infrastructure", "Research", "Production"] as const).map(
            (label) => (
              <span
                key={label}
                className="px-3 py-1 border border-blueprint-line/60 rounded-sm text-blue-400/60 hover:text-blueprint-accent hover:border-blueprint-accent/40 cursor-pointer transition-colors"
              >
                {label.toUpperCase()}
              </span>
            )
          )}
        </div>

        {/* Blueprint grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {BLUEPRINTS.map((bp) => (
            <BlueprintCard key={bp.id} blueprint={bp} />
          ))}
        </div>

        {/* Footer */}
        <footer className="mt-20 text-center font-mono text-[11px] text-blue-900/60 space-y-1">
          <p>MARS CONSTRUCTION SYSTEM v0.1.0 — SIMULATION ENGINE: MUJOCO</p>
          <p>AGENT ORCHESTRATION: MODAL // BACKEND: FASTAPI</p>
        </footer>
      </div>
    </main>
  );
}
