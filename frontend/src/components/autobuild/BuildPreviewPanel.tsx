import BlueprintVisual from "@/components/autobuild/BlueprintVisual";
import StatusMetric from "@/components/autobuild/StatusMetric";
import { Blueprint } from "@/types/blueprint";

interface BuildPreviewPanelProps {
  blueprint: Blueprint;
  starting: boolean;
  onStart: () => void;
}

export default function BuildPreviewPanel({ blueprint, starting, onStart }: BuildPreviewPanelProps) {
  const progress = Math.round((blueprint.currentLayer / blueprint.totalLayers) * 100);

  return (
    <aside id="build-feed" className="industrial-panel overflow-hidden lg:sticky lg:top-6" key={blueprint.id}>
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
        <div>
          <p className="section-kicker">Live build simulation</p>
          <h3 className="mt-1 text-xl font-black uppercase tracking-[-0.03em] text-off-white">
            {blueprint.name}
          </h3>
        </div>
        <div className="flex items-center gap-2 font-mono text-[9px] tracking-[0.16em] text-soft-gray">
          <span className="live-dot" /> LIVE
        </div>
      </div>

      <div className="live-feed relative overflow-hidden">
        <div className="absolute left-4 top-4 z-10 flex gap-2 font-mono text-[9px] uppercase tracking-[0.14em] text-soft-gray">
          <span className="border border-white/15 bg-black/40 px-2 py-1">CAM 01</span>
          <span className="border border-regolith/30 bg-black/40 px-2 py-1 text-regolith">SURFACE OPS</span>
        </div>
        <div className="feed-scanline" />
        <div className="feed-horizon" />
        <div className="construction-arm" aria-hidden="true">
          <span className="construction-arm__base" />
          <span className="construction-arm__boom" />
          <span className="construction-arm__tool" />
        </div>
        <div className="relative z-[2] mx-auto w-[88%] pt-14 opacity-75">
          <BlueprintVisual variant={blueprint.visual} />
        </div>
        <div className="absolute bottom-4 left-4 z-10 font-mono text-[9px] uppercase leading-5 tracking-[0.14em] text-soft-gray/70">
          <p>Terrain lock: sector 07</p>
          <p>Dust interference: low</p>
        </div>
      </div>

      <div className="border-t border-white/10 p-5">
        <div className="mb-3 flex items-end justify-between font-mono">
          <div>
            <p className="text-[9px] uppercase tracking-[0.16em] text-muted-text">Layer progress</p>
            <p className="mt-1 text-sm font-semibold text-off-white">
              {blueprint.currentLayer} / {blueprint.totalLayers}
            </p>
          </div>
          <span className="text-xl font-bold text-regolith">{progress}%</span>
        </div>
        <div className="h-1.5 overflow-hidden bg-white/10">
          <div className="progress-fill h-full bg-regolith" style={{ width: `${progress}%` }} />
        </div>

        <dl className="mt-5 grid grid-cols-2 gap-x-5">
          <StatusMetric label="Robots active" value={blueprint.robotsActive} tone="success" />
          <StatusMetric label="Regolith used" value={blueprint.regolithUsed} />
          <StatusMetric label="Print stability" value={blueprint.printStability} tone="success" />
          <StatusMetric label="Signal lock" value={blueprint.signalLock} />
          <StatusMetric label="Nozzle temp" value={blueprint.nozzleTemperature} tone="warning" />
          <StatusMetric label="Material flow" value="Nominal" tone="success" />
        </dl>

        <button type="button" className="command-button mt-5 w-full" onClick={onStart} disabled={starting}>
          <span>{starting ? "INITIALIZING PROJECT" : "START PROJECT"}</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </aside>
  );
}
