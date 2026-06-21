"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { startSimulation, HABITAT_ID_TO_BACKEND } from "@/lib/api";

type HabitatId = "regolith-shielded-dome" | "vertical-ellipsoid-pressure";

interface HabitatSystem {
  id: HabitatId;
  drawingId: string;
  name: string;
  specificationName: string;
  subtitle: string;
  description: string;
  metrics: ReadonlyArray<{
    label: string;
    value: string;
  }>;
  visual: "dome" | "ellipsoid";
}

const HABITATS: HabitatSystem[] = [
  {
    id: "regolith-shielded-dome",
    drawingId: "AHS-01",
    name: "Regolith Dome",
    specificationName: "Regolith-Shielded Dome Habitat",
    subtitle: "Low-profile, radiation-shielded habitat",
    description:
      "Robotic layer-by-layer construction using compacted Martian regolith.",
    metrics: [
      { label: "Crew", value: "4–6" },
      { label: "Footprint", value: "12.6 m" },
      { label: "Build cycle", value: "42 sols" },
    ],
    visual: "dome",
  },
  {
    id: "vertical-ellipsoid-pressure",
    drawingId: "AHS-02",
    name: "Ellipsoid Habitat",
    specificationName: "Vertical Ellipsoid Pressure Habitat",
    subtitle: "Vertical multi-level pressure vessel",
    description:
      "Compact pressurized structure designed for vertically organized crew operations.",
    metrics: [
      { label: "Crew", value: "8–10" },
      { label: "Height", value: "17.2 m" },
      { label: "Levels", value: "4" },
    ],
    visual: "ellipsoid",
  },
];

function DomeBlueprint() {
  return (
    <svg viewBox="0 0 760 330" role="img" aria-label="Elevation and section blueprint of a regolith-shielded dome habitat">
      <g className="habitat-blueprint__grid">
        {Array.from({ length: 19 }, (_, index) => (
          <path key={`vx-${index}`} d={`M${index * 40 + 20} 20v270`} />
        ))}
        {Array.from({ length: 7 }, (_, index) => (
          <path key={`hy-${index}`} d={`M20 ${index * 40 + 30}h720`} />
        ))}
      </g>

      <g className="habitat-blueprint__fine">
        <path d="M24 28h22M24 28v22M736 28h-22M736 28v22M24 286h22M24 286v-22M736 286h-22M736 286v-22" />
        <path d="M380 26v260" strokeDasharray="3 8" />
      </g>

      <g className="habitat-blueprint__labels">
        <text x="32" y="48">ELEVATION</text>
        <text x="400" y="48">RADIAL SECTION</text>
      </g>

      <g className="habitat-blueprint__geometry">
        <path d="M58 242Q184 58 330 242" />
        <path d="M66 242Q186 72 322 242" />
        <path d="M75 242Q188 86 313 242" />
        <path d="M48 242h292" />
        <path d="M105 242v-39q0-18 18-18h34q18 0 18 18v39" />
        <path d="M119 242v-36q0-8 8-8h25q8 0 8 8v36" />
        <path d="M184 75v167" strokeDasharray="5 7" />
        <path d="M88 211q96-40 202 0" strokeDasharray="4 6" />

        <path d="M415 242Q525 64 706 242" />
        <path d="M426 242Q532 84 694 242" />
        <path d="M446 242Q538 111 674 242" />
        <path d="M405 242h312" />
        <path d="M455 230h198" />
        <path d="M466 230v-31h176v31" />
        <path d="M478 199q77-109 152 0" />
        <path d="M488 199q67-88 132 0" />
        <path d="M554 116v114" strokeDasharray="4 6" />
        <path d="M505 230v-20h42v20M566 230v-20h42v20" />
      </g>

      <g className="habitat-blueprint__layers">
        {[0, 1, 2, 3, 4, 5].map((layer) => (
          <path key={layer} d={`M${430 + layer * 4} ${232 - layer * 4}Q532 ${92 + layer * 4} ${690 - layer * 5} ${232 - layer * 4}`} />
        ))}
      </g>

      <g className="habitat-blueprint__dimensions">
        <path d="M58 272h272M58 262v20M330 262v20" />
        <path d="M350 242V68M340 242h20M340 68h20" />
        <text x="164" y="268">Ø 12.6 M</text>
        <text x="356" y="164" transform="rotate(-90 356 164)">5.4 M</text>
        <path d="M438 274h266M438 264v20M704 264v20" />
        <text x="548" y="270">SCALE 1:100</text>
      </g>

      <g className="habitat-blueprint__callouts">
        <path d="M87 112h-36v-18" />
        <text x="30" y="89">650 MM REGOLITH / BASALT SHELL</text>
        <path d="M608 184h99" />
        <text x="594" y="177">PRESSURE VOLUME 460 M³</text>
      </g>
    </svg>
  );
}

function EllipsoidBlueprint() {
  return (
    <svg viewBox="0 0 760 330" role="img" aria-label="Elevation and section blueprint of a vertical ellipsoid pressure habitat">
      <g className="habitat-blueprint__grid">
        {Array.from({ length: 19 }, (_, index) => (
          <path key={`vx-${index}`} d={`M${index * 40 + 20} 20v270`} />
        ))}
        {Array.from({ length: 7 }, (_, index) => (
          <path key={`hy-${index}`} d={`M20 ${index * 40 + 30}h720`} />
        ))}
      </g>

      <g className="habitat-blueprint__fine">
        <path d="M24 28h22M24 28v22M736 28h-22M736 28v22M24 286h22M24 286v-22M736 286h-22M736 286v-22" />
        <path d="M380 26v260" strokeDasharray="3 8" />
      </g>

      <g className="habitat-blueprint__labels">
        <text x="32" y="48">ELEVATION</text>
        <text x="400" y="48">LONGITUDINAL SECTION</text>
      </g>

      <g className="habitat-blueprint__geometry">
        <path d="M186 58C105 80 96 190 146 242h80c50-52 41-162-40-184z" />
        <path d="M186 69c-67 20-74 112-33 163h66c41-51 34-143-33-163z" />
        <path d="M186 58v184" strokeDasharray="4 7" />
        <path d="M132 242h108M146 252h80" />
        <path d="M121 119h130M108 159h156M116 199h140" strokeDasharray="4 6" />

        <path d="M554 57c-83 24-88 139-37 185h74c51-46 46-161-37-185z" />
        <path d="M554 70c-65 22-67 122-28 160h56c39-38 37-138-28-160z" />
        <path d="M554 58v184" />
        <path d="M514 107h80M502 146h104M500 185h108M516 224h76" />
        <path d="M523 107v117M585 107v117" strokeDasharray="3 5" />
        <path d="M490 242h128M505 253h98" />
      </g>

      <g className="habitat-blueprint__layers">
        <path d="M148 85q38-21 76 0M123 122q63-31 126 0M111 162q76-35 152 0M119 202q67-30 134 0" />
        <path d="M528 81q26-17 52 0M511 118q43-24 86 0M502 157q52-27 104 0M507 197q47-23 94 0" />
      </g>

      <g className="habitat-blueprint__dimensions">
        <path d="M85 58v184M75 58h20M75 242h20" />
        <text x="82" y="174" transform="rotate(-90 82 174)">17.2 M</text>
        <path d="M132 274h108M132 264v20M240 264v20" />
        <text x="166" y="270">Ø 7.2 M</text>
        <path d="M642 58v184M632 58h20M632 242h20" />
        <text x="648" y="172" transform="rotate(-90 648 172)">SCALE 1:100</text>
      </g>

      <g className="habitat-blueprint__callouts">
        <path d="M224 91h89v-15" />
        <text x="236" y="71">REGOLITH POLYMER / ALUMINUM RIBS</text>
        <path d="M522 176h-58v-22" />
        <text x="400" y="149">PRESSURE VOLUME 980 M³ / 4 LEVELS</text>
      </g>
    </svg>
  );
}

function HabitatBlueprint({ habitat }: { habitat: HabitatSystem }) {
  return (
    <div className="habitat-blueprint">
      {habitat.visual === "dome" ? <DomeBlueprint /> : <EllipsoidBlueprint />}
      <div className="habitat-blueprint__footer" aria-hidden="true">
        <span>ATOMZ ENGINEERING</span>
        <span>{habitat.drawingId} / SCALE 1:100 / REV 03.7</span>
      </div>
    </div>
  );
}

export default function HabitatSelectionScreen() {
  const router = useRouter();
  const [selectedId, setSelectedId] = useState<HabitatId>(HABITATS[0].id);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"llm" | "greedy">("llm");
  const selected = HABITATS.find((habitat) => habitat.id === selectedId) ?? HABITATS[0];

  async function launch() {
    setError(null);
    setLaunching(true);
    try {
      const backendType = HABITAT_ID_TO_BACKEND[selectedId];
      const res = await startSimulation(backendType, mode);
      router.push(`/live-construction?sim=${res.simulation_id}&habitat=${selectedId}&mode=${mode}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start simulation");
      setLaunching(false);
    }
  }

  return (
    <main className="habitat-selection-shell min-h-screen">
      <header className="habitat-selection-header">
        <nav className="mx-auto flex h-[92px] max-w-[1440px] items-center justify-between px-5 sm:px-8 lg:px-12" aria-label="Primary">
          <Link href="/" className="flex items-center gap-2.5" aria-label="Atomz home">
            <img src="/references/atomz_logo_transparent.png" alt="Atomz logo" width={32} height={32} className="h-7 w-auto object-contain sm:h-8" />
            <img src="/references/atomz_title.svg" alt="Atomz" width={131} height={24} className="h-5 w-auto object-contain sm:h-6" />
          </Link>

          <Link href="/" className="habitat-selection-back">Back</Link>
        </nav>
      </header>

      <div className="relative z-10 mx-auto w-full max-w-[1440px] px-5 pb-12 pt-10 sm:px-8 sm:pt-12 lg:px-12 lg:pt-14">
        <header className="habitat-selection-intro">
          <div>
            <p className="habitat-selection-eyebrow">Habitat configuration</p>
            <h1>Select a construction system</h1>
          </div>
        </header>

        <section className="habitat-card-grid" aria-label="Habitat systems">
          {HABITATS.map((habitat) => {
            const isSelected = habitat.id === selectedId;

            return (
              <button
                key={habitat.id}
                type="button"
                className={`habitat-system-card ${isSelected ? "habitat-system-card--selected" : ""}`}
                aria-pressed={isSelected}
                onClick={() => setSelectedId(habitat.id)}
              >
                <div className="habitat-system-card__meta">
                  <span>{habitat.drawingId} / {habitat.specificationName}</span>
                  <span className="habitat-system-card__selection">
                    <span aria-hidden="true" />
                    {isSelected ? "Selected" : "Select system"}
                  </span>
                </div>

                <HabitatBlueprint habitat={habitat} />

                <div className="habitat-system-card__content">
                  <h2>{habitat.name}</h2>
                  <p className="habitat-system-card__subtitle">{habitat.subtitle}</p>
                  <p className="habitat-system-card__description">{habitat.description}</p>

                  <dl>
                    {habitat.metrics.map((metric) => (
                      <div key={metric.label}>
                        <dt>{metric.label}</dt>
                        <dd>{metric.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </button>
            );
          })}
        </section>

        <aside className="habitat-selection-summary" aria-label="Selected habitat summary">
          <div>
            <span>Selected habitat</span>
            <strong>{selected.name}</strong>
          </div>
          <div className="habitat-selection-summary__metrics">
            {selected.metrics.map((metric) => (
              <span key={metric.label}>{metric.label}: {metric.value}</span>
            ))}
          </div>
          <div
            role="radiogroup"
            aria-label="Coordinator mode"
            style={{ display: "flex", gap: "0.4rem", marginBottom: "0.75rem" }}
          >
            {(["llm", "greedy"] as const).map((m) => {
              const active = mode === m;
              return (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => setMode(m)}
                  title={m === "llm" ? "MiniMax M3 coordinator (falls back to greedy)" : "Nearest-idle-robot heuristic, no LLM"}
                  style={{
                    flex: 1,
                    padding: "0.4rem 0.6rem",
                    fontSize: "0.62rem",
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                    borderRadius: 6,
                    border: active ? "1px solid rgba(255,140,60,0.7)" : "1px solid rgba(120,130,150,0.3)",
                    background: active ? "rgba(255,120,40,0.15)" : "transparent",
                    color: active ? "rgba(255,200,160,0.95)" : "rgba(180,190,205,0.6)",
                  }}
                >
                  {m === "llm" ? "LLM · MiniMax M3" : "Greedy"}
                </button>
              );
            })}
          </div>
          <button type="button" onClick={launch} disabled={launching}>
            {launching ? "Dispatching to orchestrator…" : `Continue with ${selected.name}`}
            <span aria-hidden="true">→</span>
          </button>
          {error && (
            <p role="alert" style={{ color: "#ff6b6b", fontSize: "0.75rem", marginTop: "0.5rem" }}>
              {error}
            </p>
          )}
        </aside>
      </div>
    </main>
  );
}
