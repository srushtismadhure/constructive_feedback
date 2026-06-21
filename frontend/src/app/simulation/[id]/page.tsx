import Link from "next/link";
import { BLUEPRINTS } from "@/data/blueprints";

interface SimulationPageProps {
  params: Promise<{ id: string }>;
}

export default async function SimulationPage({ params }: SimulationPageProps) {
  const { id } = await params;
  const blueprint = BLUEPRINTS.find((bp) => bp.id === id);

  return (
    <main className="min-h-screen blueprint-grid relative flex flex-col">
      {/* Radial vignette */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_center,_transparent_40%,_rgba(0,0,0,0.7)_100%)]" />

      {/* Top nav bar */}
      <nav className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-blueprint-line/40 bg-blueprint-bg/70 backdrop-blur-sm">
        <Link
          href="/"
          className="font-mono text-xs text-blue-400/70 hover:text-blueprint-accent transition-colors flex items-center gap-2"
        >
          &larr; BLUEPRINTS
        </Link>

        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
          <span className="font-mono text-xs text-blue-400/60 uppercase tracking-widest">
            Simulation Initializing
          </span>
        </div>

        <div className="font-mono text-xs text-blue-900/50">
          {blueprint ? blueprint.name.toUpperCase() : id.toUpperCase()}
        </div>
      </nav>

      {/* Main area */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-6 py-16 gap-8">
        {/* Blueprint info banner */}
        {blueprint && (
          <div className="border border-blueprint-line/50 bg-blueprint-bg/60 rounded-sm px-6 py-4 max-w-lg w-full font-mono">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-blue-400/50 uppercase tracking-widest">
                Selected Blueprint
              </span>
              <span className={`text-xs badge-${blueprint.category} px-2 py-0.5 rounded-sm`}>
                {blueprint.category}
              </span>
            </div>
            <p className="text-sm text-white/80 font-semibold">{blueprint.name}</p>
            <p className="text-xs text-blue-200/40 mt-1">{blueprint.description}</p>
          </div>
        )}

        {/* Placeholder stream area */}
        <div className="relative w-full max-w-4xl aspect-video border border-blueprint-line/60 rounded-sm bg-[#040d18] flex flex-col items-center justify-center overflow-hidden">
          {/* Animated corner brackets */}
          <span className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-blueprint-accent/40 corner-accent" />
          <span className="absolute top-3 right-3 w-5 h-5 border-t-2 border-r-2 border-blueprint-accent/40 corner-accent" />
          <span className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-blueprint-accent/40 corner-accent" />
          <span className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-blueprint-accent/40 corner-accent" />

          {/* Scan line */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div
              className="absolute w-full h-px bg-gradient-to-r from-transparent via-blueprint-accent/20 to-transparent animate-scan"
              style={{ top: 0 }}
            />
          </div>

          {/* Center content */}
          <div className="text-center font-mono space-y-4 px-8">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="w-3 h-3 rounded-full bg-blueprint-accent/60 animate-pulse" />
              <div
                className="w-3 h-3 rounded-full bg-blueprint-accent/60 animate-pulse"
                style={{ animationDelay: "0.3s" }}
              />
              <div
                className="w-3 h-3 rounded-full bg-blueprint-accent/60 animate-pulse"
                style={{ animationDelay: "0.6s" }}
              />
            </div>

            <p className="text-blueprint-accent text-lg font-semibold tracking-widest uppercase">
              Awaiting Simulation Stream
            </p>
            <p className="text-blue-400/50 text-xs max-w-xs mx-auto leading-relaxed">
              The AI agent is configuring the MuJoCo environment. Live video
              feed will appear here once the simulation initializes.
            </p>

            <div className="mt-6 space-y-1 text-left max-w-xs mx-auto text-[11px] text-blue-900/70">
              <div className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                <span>Blueprint received by orchestration agent</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-yellow-500 animate-pulse">◌</span>
                <span>Spawning MuJoCo simulation worker</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-900/40">○</span>
                <span>Establishing video stream connection</span>
              </div>
            </div>
          </div>

          {/* Bottom status bar */}
          <div className="absolute bottom-0 left-0 right-0 h-7 border-t border-blueprint-line/30 bg-blueprint-bg/80 flex items-center px-4 gap-4">
            <span className="font-mono text-[10px] text-blue-900/60">
              SIM_ID: {id.toUpperCase()}-{Math.random().toString(36).slice(2, 8).toUpperCase()}
            </span>
            <span className="font-mono text-[10px] text-blue-900/60 ml-auto">
              AGENT: MODAL // ENGINE: MUJOCO // STREAM: PENDING
            </span>
          </div>
        </div>

        {/* Agent log placeholder */}
        <div className="w-full max-w-4xl border border-blueprint-line/40 rounded-sm bg-blueprint-bg/50 p-4 font-mono text-xs">
          <div className="flex items-center gap-2 mb-3 text-blue-400/50 uppercase tracking-widest text-[10px]">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Agent Log
          </div>
          <div className="space-y-1 text-blue-400/40">
            <p>
              <span className="text-blue-900/60">[00:00:00]</span> Received blueprint selection:{" "}
              <span className="text-blueprint-accent/70">{id}</span>
            </p>
            <p>
              <span className="text-blue-900/60">[00:00:01]</span> Dispatching task to Modal agent...
            </p>
            <p>
              <span className="text-blue-900/60">[00:00:02]</span> Waiting for MuJoCo worker to spin up...
            </p>
            <p className="animate-pulse">
              <span className="text-blue-900/60">[——:——:——]</span>{" "}
              <span className="text-yellow-500/60">_</span>
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
