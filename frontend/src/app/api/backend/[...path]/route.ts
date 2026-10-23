import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const noStoreHeaders = { "Cache-Control": "no-store" };
const prefix = "/api/backend";

async function forward(request: Request) {
  const url = new URL(request.url);
  const backendPath = url.pathname.slice(prefix.length) || "/";

  // Explicit allowlist: this proxy cannot be used as a general-purpose URL proxy.
  const allowed = new Set([
    "/health",
    "/query",
    "/ingest",
    "/regression/latest",
    "/regression/history",
  ]);

  if (!allowed.has(backendPath)) {
    return NextResponse.json(
      { detail: "Backend route not found." },
      { status: 404, headers: noStoreHeaders },
    );
  }

  const requiredMethod =
    backendPath === "/query" || backendPath === "/ingest" ? "POST" : "GET";

  if (request.method !== requiredMethod) {
    return NextResponse.json(
      { detail: "Method not allowed." },
      { status: 405, headers: noStoreHeaders },
    );
  }

  if (request.method !== "GET") {
    const origin = request.headers.get("origin");
    const host = request.headers.get("host")?.toLowerCase();
    let originUrl: URL | null = null;

    try {
      originUrl = new URL(origin || "");
    } catch {
      originUrl = null;
    }

    if (
      !originUrl ||
      !host ||
      originUrl.host.toLowerCase() !== host ||
      originUrl.protocol !== url.protocol
    ) {
      return NextResponse.json(
        { detail: "Cross-origin request rejected." },
        { status: 403, headers: noStoreHeaders },
      );
    }
  }

  const apiKey = process.env.API_ACCESS_KEY?.trim();
  if (backendPath !== "/health" && !apiKey) {
    return NextResponse.json(
      { detail: "Backend API access is not configured on the server." },
      { status: 503, headers: noStoreHeaders },
    );
  }

  const backendBase = (
    process.env.INTERNAL_API_URL || "http://127.0.0.1:8000"
  ).replace(/\/+$/, "");

  try {
    const body = request.method === "GET" ? undefined : await request.text();

    const upstream = await fetch(
      `${backendBase}${backendPath}${url.search}`,
      {
        method: request.method,
        headers: {
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
          ...(body !== undefined
            ? {
                "Content-Type":
                  request.headers.get("content-type") || "application/json",
              }
            : {}),
        },
        ...(body !== undefined ? { body } : {}),
        cache: "no-store",
      },
    );

    return new NextResponse(await upstream.text(), {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") || "application/json",
        ...noStoreHeaders,
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "Backend service is unavailable." },
      { status: 502, headers: noStoreHeaders },
    );
  }
}

export async function GET(request: Request) {
  return forward(request);
}

export async function POST(request: Request) {
  return forward(request);
}
