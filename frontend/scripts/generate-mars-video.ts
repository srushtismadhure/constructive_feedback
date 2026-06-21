import { config } from "dotenv";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

config({ path: path.resolve(process.cwd(), ".env.local") });

const API_BASE = "https://api.minimax.io/v1";
const API_KEY = process.env.MINIMAX_API_KEY;

const INPUT_IMAGE = path.resolve(
  process.cwd(),
  "public/reference-images/mars_landing.png",
);

const OUTPUT_VIDEO = path.resolve(
  process.cwd(),
  "public/videos/mars_landing_dark_smooth.mp4",
);

const POLL_INTERVAL_MS = 10_000;
const MAX_WAIT_MS = 30 * 60 * 1000;

const PROMPT = `
Use the reference image mars_landing as the primary grounding for composition, robot scale, Mars terrain, lighting direction, and cinematic mood.

Create an extremely slow, smooth, cinematic landing-page background video of an autonomous robotic construction machine on Mars actively 3D-printing a structure. The result should be a darker, more premium, more polished version of the current scene, closer to the second visual direction: darker surroundings, cleaner composition, and a smaller bright glowing rectangular backlight behind the robot.

The environment is darker overall, with richer shadows and higher contrast, while preserving the warm orange Mars glow. The glowing rectangular backlight is smaller, more contained, and cleaner, framing the robot without overpowering the scene.

The robotic arm and nozzle must look precise, smooth, and believable. The nozzle clearly contacts the build surface and visibly traces the 3D print path. The printing area looks neat and intentional, with clearly visible horizontal layered rings and ridges showing the additive process; deposited material appears organized, smooth, and structured, never rough or messy.

The machine feels heavy, industrial, and realistic, moving very slowly and deliberately. Arm motion is minimal and smooth; the nozzle stays close to the surface as if carefully following a print path.

The camera stays almost stationary, with only an extremely subtle cinematic push-in if needed: no fast panning, no cuts, no shake, no sudden motion. Dust drifts gently and slowly.

Keep it cinematic, atmospheric, and suitable for a futuristic landing-page hero background. Avoid flicker, morphing, deformation, or unstable motion. Make the beginning and ending visually similar for seamless looping.

No text, no UI, no logos, no overlays.

Prioritize smooth temporal consistency, clean silhouette readability, visible printing behavior, and a premium sci-fi industrial Mars aesthetic.
`.trim();

type BaseResponse = {
  status_code?: number;
  status_msg?: string;
};

type CreateTaskResponse = {
  task_id?: string;
  base_resp?: BaseResponse;
};

type QueryTaskResponse = {
  task_id?: string;
  status?: "Preparing" | "Queueing" | "Processing" | "Success" | "Fail";
  file_id?: string;
  video_width?: number;
  video_height?: number;
  base_resp?: BaseResponse;
};

type RetrieveFileResponse = {
  file?: {
    file_id?: string;
    filename?: string;
    download_url?: string;
  };
  base_resp?: BaseResponse;
};

function assertApiKey(): string {
  if (!API_KEY) {
    throw new Error(
      [
        "MINIMAX_API_KEY was not found.",
        "Add this line to frontend/.env.local:",
        "MINIMAX_API_KEY=your_actual_key",
      ].join("\n"),
    );
  }

  return API_KEY;
}

async function fetchJson<T>(
  url: string,
  options: RequestInit,
  label: string,
): Promise<T> {
  const response = await fetch(url, options);
  const rawBody = await response.text();

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawBody);
  } catch {
    parsed = rawBody;
  }

  if (!response.ok) {
    throw new Error(
      `${label} failed with HTTP ${response.status}:\n${rawBody}`,
    );
  }

  const baseResponse =
    typeof parsed === "object" && parsed !== null && "base_resp" in parsed
      ? (parsed as { base_resp?: BaseResponse }).base_resp
      : undefined;

  if (
    baseResponse?.status_code !== undefined &&
    baseResponse.status_code !== 0
  ) {
    throw new Error(
      `${label} failed: ${baseResponse.status_msg ?? "Unknown MiniMax error"} ` +
        `(status code ${baseResponse.status_code})`,
    );
  }

  return parsed as T;
}

function getMimeType(filename: string): string {
  const extension = path.extname(filename).toLowerCase();

  const mimeTypes: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
  };

  const mimeType = mimeTypes[extension];

  if (!mimeType) {
    throw new Error(
      `Unsupported image type "${extension}". Use PNG, JPG, JPEG, or WebP.`,
    );
  }

  return mimeType;
}

async function imageToDataUrl(imagePath: string): Promise<string> {
  const imageStats = await stat(imagePath).catch(() => null);

  if (!imageStats) {
    throw new Error(
      `Reference image not found:\n${imagePath}\n\n` +
        "Expected file: public/reference-images/mars_landing.png",
    );
  }

  const maxBytes = 20 * 1024 * 1024;

  if (imageStats.size >= maxBytes) {
    throw new Error(
      `Reference image is ${(imageStats.size / 1024 / 1024).toFixed(2)} MB. ` +
        "MiniMax requires an image smaller than 20 MB.",
    );
  }

  const imageBuffer = await readFile(imagePath);
  const mimeType = getMimeType(imagePath);
  const base64 = imageBuffer.toString("base64");

  // This Data URL exists only in memory. No Base64 file is written to disk.
  return `data:${mimeType};base64,${base64}`;
}

async function createVideoTask(
  apiKey: string,
  imageDataUrl: string,
): Promise<string> {
  if (PROMPT.length > 2000) {
    throw new Error(`Prompt is ${PROMPT.length} characters; MiniMax allows 2000.`);
  }

  console.log("Submitting 1080P image-to-video task to MiniMax...");

  const result = await fetchJson<CreateTaskResponse>(
    `${API_BASE}/video_generation`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "MiniMax-Hailuo-2.3",
        first_frame_image: imageDataUrl,
        prompt: PROMPT,
        duration: 6,
        resolution: "1080P",
        prompt_optimizer: false,
      }),
    },
    "Video task creation",
  );

  if (!result.task_id) {
    throw new Error(
      `MiniMax did not return a task_id:\n${JSON.stringify(result, null, 2)}`,
    );
  }

  return result.task_id;
}

async function waitForVideo(
  apiKey: string,
  taskId: string,
): Promise<QueryTaskResponse> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < MAX_WAIT_MS) {
    const queryUrl = new URL(`${API_BASE}/query/video_generation`);
    queryUrl.searchParams.set("task_id", taskId);

    const result = await fetchJson<QueryTaskResponse>(
      queryUrl.toString(),
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
      },
      "Video status query",
    );

    console.log(`MiniMax status: ${result.status ?? "Unknown"}`);

    if (result.status === "Success") {
      if (!result.file_id) {
        throw new Error("Generation succeeded, but MiniMax returned no file_id.");
      }

      return result;
    }

    if (result.status === "Fail") {
      throw new Error(
        `MiniMax video generation failed:\n${JSON.stringify(result, null, 2)}`,
      );
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  throw new Error("Timed out after 30 minutes waiting for MiniMax.");
}

async function getDownloadUrl(
  apiKey: string,
  fileId: string,
): Promise<string> {
  const retrieveUrl = new URL(`${API_BASE}/files/retrieve`);
  retrieveUrl.searchParams.set("file_id", fileId);

  const result = await fetchJson<RetrieveFileResponse>(
    retrieveUrl.toString(),
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    },
    "Video file retrieval",
  );

  const downloadUrl = result.file?.download_url;

  if (!downloadUrl) {
    throw new Error(
      `MiniMax returned no download URL:\n${JSON.stringify(result, null, 2)}`,
    );
  }

  return downloadUrl;
}

async function downloadVideo(
  downloadUrl: string,
  outputPath: string,
): Promise<void> {
  console.log("Downloading generated MP4...");

  const response = await fetch(downloadUrl);

  if (!response.ok) {
    throw new Error(
      `Video download failed with HTTP ${response.status}: ${response.statusText}`,
    );
  }

  const videoBuffer = Buffer.from(await response.arrayBuffer());

  if (videoBuffer.length === 0) {
    throw new Error("The downloaded MP4 was empty.");
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, videoBuffer);
}

async function main(): Promise<void> {
  const apiKey = assertApiKey();

  console.log(`Reference image: ${INPUT_IMAGE}`);
  console.log("Converting image to an in-memory Base64 Data URL...");

  const imageDataUrl = await imageToDataUrl(INPUT_IMAGE);
  const taskId = await createVideoTask(apiKey, imageDataUrl);

  console.log(`Task created: ${taskId}`);
  console.log("Waiting for MiniMax. Status will be checked every 10 seconds...");

  const completedTask = await waitForVideo(apiKey, taskId);
  const downloadUrl = await getDownloadUrl(apiKey, completedTask.file_id!);

  await downloadVideo(downloadUrl, OUTPUT_VIDEO);

  console.log("\nVideo generation complete.");
  console.log(
    `Resolution reported by MiniMax: ` +
      `${completedTask.video_width ?? "?"}x${completedTask.video_height ?? "?"}`,
  );
  console.log(`Saved to: ${OUTPUT_VIDEO}`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`\nGeneration stopped:\n${message}`);
  process.exitCode = 1;
});
