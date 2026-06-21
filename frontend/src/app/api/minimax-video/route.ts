import { NextRequest, NextResponse } from "next/server";

// Keep MiniMax credentials and upstream calls on the server.

export const runtime = "nodejs";

const MINIMAX_API_BASE = "https://api.minimax.io/v1";
const MINIMAX_TIMEOUT_MS = 30_000;
const BASE64_IMAGE_DATA_URL =
  /^data:image\/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}$/;

type VideoRequest = {
  prompt?: unknown;
  model?: unknown;
  duration?: unknown;
  resolution?: unknown;
  firstFrameImage?: unknown;
  lastFrameImage?: unknown;
};

function getApiKey() {
  return process.env.MINIMAX_API_KEY?.trim();
}

function getFirstFrameImage(value: unknown) {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  const image = value.trim();
  return BASE64_IMAGE_DATA_URL.test(image) ? image : null;
}

async function readJson(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();

  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { error: "MiniMax returned a non-JSON response." };
  }
}

function upstreamError(
  data: Record<string, unknown>,
  status: number,
  fallback: string,
) {
  return NextResponse.json(
    {
      error: fallback,
      details: data,
    },
    { status },
  );
}

function getMiniMaxError(data: Record<string, unknown>) {
  const baseResponse = data.base_resp;

  if (
    baseResponse &&
    typeof baseResponse === "object" &&
    "status_code" in baseResponse &&
    baseResponse.status_code !== 0
  ) {
    const statusMessage =
      "status_msg" in baseResponse ? baseResponse.status_msg : undefined;

    return typeof statusMessage === "string"
      ? statusMessage
      : "MiniMax returned an error.";
  }

  return typeof data.error_message === "string" ? data.error_message : null;
}

function connectionError(error: unknown) {
  const timedOut = error instanceof Error && error.name === "TimeoutError";

  return NextResponse.json(
    {
      error: timedOut
        ? "MiniMax took too long to respond. Please try again."
        : "Could not connect to MiniMax.",
    },
    { status: 502 },
  );
}

export async function POST(request: NextRequest) {
  const apiKey = getApiKey();

  if (!apiKey) {
    return NextResponse.json(
      { error: "MINIMAX_API_KEY is not configured on the server." },
      { status: 500 },
    );
  }

  let body: VideoRequest;

  try {
    body = (await request.json()) as VideoRequest;
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";

  if (!prompt) {
    return NextResponse.json(
      { error: "A non-empty prompt is required." },
      { status: 400 },
    );
  }

  const duration = body.duration ?? 6;
  const resolution = body.resolution ?? "768P";

  if (duration !== 6 && duration !== 10) {
    return NextResponse.json(
      { error: "duration must be either 6 or 10 seconds." },
      { status: 400 },
    );
  }

  if (!["720P", "768P", "1080P"].includes(String(resolution))) {
    return NextResponse.json(
      { error: "resolution must be 720P, 768P, or 1080P." },
      { status: 400 },
    );
  }

  const payload: Record<string, unknown> = {
    model:
      typeof body.model === "string" && body.model.trim()
        ? body.model.trim()
        : "MiniMax-Hailuo-2.3",
    prompt,
    duration,
    resolution,
  };

  const imageToVideoRequested = Object.prototype.hasOwnProperty.call(
    body,
    "firstFrameImage",
  );

  if (imageToVideoRequested) {
    const firstFrameImage = getFirstFrameImage(body.firstFrameImage);

    if (!firstFrameImage) {
      return NextResponse.json(
        {
          error:
            "firstFrameImage is required for image-to-video generation and must be a valid JPEG, PNG, or WebP base64 data URL.",
        },
        { status: 400 },
      );
    }

    payload.first_frame_image = firstFrameImage;
  }

  if (typeof body.lastFrameImage === "string" && body.lastFrameImage.trim()) {
    payload.last_frame_image = body.lastFrameImage.trim();
  }

  try {
    const response = await fetch(`${MINIMAX_API_BASE}/video_generation`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: AbortSignal.timeout(MINIMAX_TIMEOUT_MS),
    });
    const data = await readJson(response);
    const miniMaxError = getMiniMaxError(data);

    if (!response.ok || miniMaxError || typeof data.task_id !== "string") {
      return upstreamError(
        data,
        response.ok ? 502 : response.status,
        miniMaxError ?? "MiniMax did not create the video task.",
      );
    }

    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    return connectionError(error);
  }
}

export async function GET(request: NextRequest) {
  const apiKey = getApiKey();

  if (!apiKey) {
    return NextResponse.json(
      { error: "MINIMAX_API_KEY is not configured on the server." },
      { status: 500 },
    );
  }

  const taskId = request.nextUrl.searchParams.get("taskId")?.trim();

  if (!taskId) {
    return NextResponse.json(
      { error: "The taskId query parameter is required." },
      { status: 400 },
    );
  }

  try {
    const statusUrl = new URL(`${MINIMAX_API_BASE}/query/video_generation`);
    statusUrl.searchParams.set("task_id", taskId);

    const statusResponse = await fetch(statusUrl, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store",
      signal: AbortSignal.timeout(MINIMAX_TIMEOUT_MS),
    });
    const statusData = await readJson(statusResponse);
    const miniMaxError = getMiniMaxError(statusData);

    if (!statusResponse.ok || miniMaxError) {
      return upstreamError(
        statusData,
        statusResponse.ok ? 502 : statusResponse.status,
        miniMaxError ?? "MiniMax did not return the video task status.",
      );
    }

    if (statusData.status === "Fail") {
      return upstreamError(
        statusData,
        502,
        "MiniMax video generation failed.",
      );
    }

    if (statusData.status !== "Success" || typeof statusData.file_id !== "string") {
      return NextResponse.json(statusData);
    }

    const fileUrl = new URL(`${MINIMAX_API_BASE}/files/retrieve`);
    fileUrl.searchParams.set("file_id", statusData.file_id);

    const fileResponse = await fetch(fileUrl, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store",
      signal: AbortSignal.timeout(MINIMAX_TIMEOUT_MS),
    });
    const fileData = await readJson(fileResponse);

    if (!fileResponse.ok) {
      return upstreamError(
        fileData,
        fileResponse.status,
        "The video completed, but MiniMax did not return its file data.",
      );
    }

    return NextResponse.json({ ...statusData, ...fileData });
  } catch (error) {
    return connectionError(error);
  }
}
