import type { RetrievedChunk } from "@/lib/api";

const ROLE_COLORS: Record<string, string> = {
  Engineering: "bg-blue-950 text-blue-300 border-blue-800",
  HR: "bg-purple-950 text-purple-300 border-purple-800",
  Executive: "bg-yellow-950 text-yellow-300 border-yellow-800",
  General: "bg-zinc-800 text-zinc-300 border-zinc-700",
};

export default function ChunkCard({
  chunk,
  flagged,
  flagReason,
}: {
  chunk: RetrievedChunk;
  flagged?: boolean;
  flagReason?: string;
}) {
  const roleClasses = ROLE_COLORS[chunk.acl_role] ?? ROLE_COLORS.General;
  return (
    <div
      className={`rounded-lg border p-3 text-sm ${
        flagged ? "border-rose-700 bg-rose-950/40" : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`rounded border px-2 py-0.5 text-xs font-medium ${roleClasses}`}>{chunk.acl_role}</span>
          <span className="text-xs text-zinc-500">{chunk.source}</span>
        </div>
        <span className="text-xs font-mono text-zinc-500">score {chunk.score.toFixed(3)}</span>
      </div>
      <p className="text-zinc-300">{chunk.text}</p>
      {flagged && flagReason && <p className="mt-2 text-xs text-rose-400">⚠ {flagReason}</p>}
    </div>
  );
}
