import { Blueprint } from "@/types/blueprint";

interface BlueprintVisualProps {
  variant: Blueprint["visual"];
  compact?: boolean;
}

function HabitatGeometry({ variant }: { variant: Blueprint["visual"] }) {
  switch (variant) {
    case "dome":
      return (
        <>
          <path d="M68 136C76 72 111 42 160 42s84 30 92 94" />
          <path d="M91 136c8-45 32-69 69-69s61 24 69 69" />
          <path d="M68 136h184M92 103h136M117 70h86M160 42v94" />
          <ellipse cx="160" cy="136" rx="100" ry="13" />
        </>
      );
    case "cylinder":
      return (
        <>
          <ellipse cx="92" cy="91" rx="33" ry="49" />
          <ellipse cx="228" cy="91" rx="33" ry="49" />
          <path d="M92 42h136M92 140h136M92 67h136M92 115h136" />
          <path d="M82 43c32 14 32 82 0 96M102 43c32 14 32 82 0 96" />
          <path d="M218 43c-32 14-32 82 0 96M238 43c-32 14-32 82 0 96" />
        </>
      );
    case "vault":
      return (
        <>
          <path d="M54 137h212M68 137V99c0-35 28-63 63-63h58c35 0 63 28 63 63v38" />
          <path d="M92 137V98c0-22 18-39 39-39h58c21 0 39 17 39 39v39" />
          <path d="M68 100h184M92 82h136M160 36v101" />
          <path d="M128 137v-25h64v25" />
        </>
      );
    case "command":
      return (
        <>
          <path d="M72 137V72l31-25h114l31 25v65z" />
          <path d="M103 47l18 25h78l18-25M72 72h176M105 137V91h110v46" />
          <path d="M160 47V22M147 22h26M125 102h70M125 115h70" />
          <circle cx="160" cy="22" r="5" />
        </>
      );
    case "greenhouse":
      return (
        <>
          <path d="M51 137h218M66 137V88l34-43h120l34 43v49" />
          <path d="M100 45l22 92M130 45l10 92M160 45v92M190 45l-10 92M220 45l-22 92" />
          <path d="M66 88h188M84 66h152M54 113h212" />
          <path d="M142 137v-27h36v27" />
        </>
      );
    case "modular":
      return (
        <>
          <path d="M50 137V82h68v55M126 137V53h72v84M206 137V72h64v65" />
          <path d="M50 82l34-20 34 20M126 53l36-22 36 22M206 72l32-19 32 19" />
          <path d="M84 62v75M162 31v106M238 53v84M50 106h68M126 82h72M206 98h64" />
          <path d="M118 112h8M198 105h8" />
        </>
      );
  }
}

export default function BlueprintVisual({ variant, compact = false }: BlueprintVisualProps) {
  return (
    <div className={`blueprint-visual ${compact ? "blueprint-visual--compact" : ""}`}>
      <svg viewBox="0 0 320 180" role="img" aria-label={`${variant} habitat wireframe`}>
        <g className="blueprint-grid-lines">
          {[40, 80, 120, 160, 200, 240, 280].map((x) => (
            <path key={`x-${x}`} d={`M${x} 18v144`} />
          ))}
          {[36, 72, 108, 144].map((y) => (
            <path key={`y-${y}`} d={`M20 ${y}h280`} />
          ))}
        </g>
        <g className="blueprint-geometry">
          <HabitatGeometry variant={variant} />
        </g>
        <g className="blueprint-ticks">
          <path d="M22 25h18M22 25v18M298 25h-18M298 25v18M22 155h18M22 155v-18M298 155h-18M298 155v-18" />
        </g>
        <text x="28" y="170">GRID M-24 // SCALE 1:240</text>
      </svg>
    </div>
  );
}
