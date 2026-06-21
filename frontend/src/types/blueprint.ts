export interface Blueprint {
  id: string;
  name: string;
  description: string;
  category: "habitat" | "infrastructure" | "research" | "production";
  complexity: "basic" | "intermediate" | "advanced";
  estimatedDuration: string;
  icon: string;
  tags: string[];
  area: string;
  capacity: string;
  material: string;
  purpose: string;
  visual: "dome" | "cylinder" | "vault" | "command" | "greenhouse" | "modular";
  robotsActive: number;
  currentLayer: number;
  totalLayers: number;
  regolithUsed: string;
  printStability: string;
  signalLock: string;
  nozzleTemperature: string;
}
