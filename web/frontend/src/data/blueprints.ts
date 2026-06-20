import { Blueprint } from "@/types/blueprint";

export const BLUEPRINTS: Blueprint[] = [
  {
    id: "habitat-dome",
    name: "Habitat Dome",
    description:
      "Pressurized geodesic dome for crew living quarters. Supports up to 12 colonists with full life support systems.",
    category: "habitat",
    complexity: "intermediate",
    estimatedDuration: "72 hrs",
    icon: "⬡",
    tags: ["pressurized", "crew", "life-support"],
  },
  {
    id: "research-lab",
    name: "Research Laboratory",
    description:
      "Modular scientific facility with radiation shielding. Equipped for geology, biology, and atmospheric analysis.",
    category: "research",
    complexity: "advanced",
    estimatedDuration: "96 hrs",
    icon: "◈",
    tags: ["science", "modular", "shielded"],
  },
  {
    id: "greenhouse-module",
    name: "Greenhouse Module",
    description:
      "Hydroponic food production unit with UV grow lights and controlled atmosphere for crop cultivation.",
    category: "production",
    complexity: "basic",
    estimatedDuration: "48 hrs",
    icon: "⬟",
    tags: ["food", "hydroponic", "agriculture"],
  },
  {
    id: "solar-array",
    name: "Solar Array Station",
    description:
      "High-efficiency photovoltaic array with dust-resistant coating and automated panel tilting system.",
    category: "infrastructure",
    complexity: "basic",
    estimatedDuration: "36 hrs",
    icon: "◎",
    tags: ["power", "renewable", "automated"],
  },
  {
    id: "underground-shelter",
    name: "Underground Shelter",
    description:
      "Radiation-protected subterranean habitat excavated beneath the Martian surface. Passive thermal regulation.",
    category: "habitat",
    complexity: "advanced",
    estimatedDuration: "120 hrs",
    icon: "⬡",
    tags: ["radiation-safe", "underground", "thermal"],
  },
  {
    id: "comm-tower",
    name: "Communication Tower",
    description:
      "High-gain antenna array for long-range communications with Earth and orbital relays.",
    category: "infrastructure",
    complexity: "intermediate",
    estimatedDuration: "24 hrs",
    icon: "△",
    tags: ["comms", "antenna", "relay"],
  },
  {
    id: "water-extractor",
    name: "Water Extraction Plant",
    description:
      "Atmospheric water vapor collector and subsurface ice melting facility with purification system.",
    category: "production",
    complexity: "intermediate",
    estimatedDuration: "60 hrs",
    icon: "◇",
    tags: ["water", "ice", "purification"],
  },
  {
    id: "landing-pad",
    name: "Landing Pad",
    description:
      "Reinforced concrete launch and landing platform with fuel storage and vehicle maintenance bay.",
    category: "infrastructure",
    complexity: "basic",
    estimatedDuration: "30 hrs",
    icon: "⊕",
    tags: ["landing", "launch", "vehicles"],
  },
];
