const PRODUCTION_API_URL =
  "https://constructive-feedback--mars-construction-fastapi-app.modal.run";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL?.trim() ||
  (process.env.NODE_ENV === "production"
    ? PRODUCTION_API_URL
    : "http://localhost:8000")
).replace(/\/+$/, "");

export const HABITAT_ID_TO_BACKEND = {
  "regolith-shielded-dome": "regolith_dome",
  "vertical-ellipsoid-pressure": "ellipsoid_habitat",
} as const;

type HabitatType = (typeof HABITAT_ID_TO_BACKEND)[keyof typeof HABITAT_ID_TO_BACKEND];
type CoordinatorMode = "llm" | "greedy";

export interface StartSimulationResponse {
  simulation_id: string;
  blueprint_id?: string | null;
  habitat_type?: HabitatType | null;
  status: string;
  message: string;
  status_url?: string | null;
  frame_url?: string | null;
  stream_url?: string | null;
  cancel_url?: string | null;
}

function simulationUrl(simulationId: string, suffix = "") {
  return `${API_BASE_URL}/simulation/${encodeURIComponent(simulationId)}${suffix}`;
}

async function apiError(response: Response) {
  const payload = (await response.json().catch(() => null)) as {
    detail?: unknown;
    message?: unknown;
  } | null;

  if (typeof payload?.detail === "string") return payload.detail;
  if (typeof payload?.message === "string") return payload.message;
  return `API request failed with status ${response.status}`;
}

export async function startSimulation(
  habitatType: HabitatType,
  coordinatorMode: CoordinatorMode = "llm",
): Promise<StartSimulationResponse> {
  const response = await fetch(`${API_BASE_URL}/simulation/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      habitat_type: habitatType,
      coordinator_mode: coordinatorMode,
    }),
  });

  if (!response.ok) {
    throw new Error(await apiError(response));
  }

  return (await response.json()) as StartSimulationResponse;
}

export function statusUrl(simulationId: string) {
  return simulationUrl(simulationId);
}

export function frameUrl(simulationId: string) {
  return simulationUrl(simulationId, "/frames");
}

export function streamUrl(simulationId: string) {
  return simulationUrl(simulationId, "/stream");
}
