export interface Blueprint {
  id: string;
  name: string;
  description: string;
  category: "habitat" | "infrastructure" | "research" | "production";
  complexity: "basic" | "intermediate" | "advanced";
  estimatedDuration: string;
  icon: string;
  tags: string[];
}
