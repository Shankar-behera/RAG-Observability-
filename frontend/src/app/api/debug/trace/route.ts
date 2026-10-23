import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const noStoreHeaders = { "Cache-Control": "no-store" };

export async function POST(request: Request) {
  const requestUrl = new URL(request.url);
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
    originUrl.protocol !== requestUrl.protocol
  ) {
    return NextResponse.json(
      { detail: "Cross-origin debug request rejected." },
      { status: 403, headers: noStoreHeaders },
    );
  }

  const apiKey = process.env.DEBUG_API_KEY?.trim();

  if (!apiKey) {
    return NextResponse.json(
      { detail: "Debug tracing is not configured on the server." },
      { status: 503, headers: noStoreHeaders },
    );
  }

  const backendUrl = (
    process.env.INTERNAL_API_URL || "http://127.0.0.1:8000"
  ).replace(/\/+$/, "");

  try {
    const body = await request.text();
    const upstream = await fetch(`${backendUrl}/debug/trace`, {
      method: "POST",
      headers: {
        "Content-Type": request.headers.get("content-type") || "application/json",
        "X-Debug-API-Key": apiKey,
      },
      body,
      cache: "no-store",
    });

    return new NextResponse(await upstream.text(), {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json",
        ...noStoreHeaders,
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "Debug trace service is unavailable." },
      { status: 502, headers: noStoreHeaders },
    );
  }
}
