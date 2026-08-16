"use client";

import { useState } from "react";
import { api, ApiError, type DebugTraceResponse, type Role } from "@/lib/api";
import RoleSelector from "@/components/RoleSelector";
import ChunkCard from "@/components/ChunkCard";
import RootCauseBadge from "@/components/RootCauseBadge";
import MetricBadge from "@/components/MetricBadge";

export default function DebuggerPage() {
  const [query, setQuery] = useState("What approvals are needed before a production deploy?");
  const [role, setRole] = useState<Role>("Engineering");
  const [groundTruth, setGroundTruth] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<DebugTraceResponse | null>(null);

  async function runTrace() {
    setLoading(true);
    setError(null);
    setTrace(null);
    try {
      const result = await api.debugTrace(query, role, groundTruth || undefined);
      setTrace(result);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.message}` : String(e));
    } finally {
      setLoading(false);
    }
  }

  const leakedIds = new Set(trace?.permission_leaked_doc_ids ?? []);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold">Query & Trace Debugger</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Run a query as a given role and trace exactly why the response passed or failed — retriever,
          generator, or permission layer.
        </p>
      </div>

      <div className="grid gap-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
        <div className="grid gap-4 sm:grid-cols-[1fr_180px]">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-zinc-400">Query</label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm focus:border-zinc-500 focus:outline-none"
              placeholder="Ask something..."
            />
          </div>
          <RoleSelector value={role} onChange={setRole} />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-zinc-400">
            Ground truth (optional — enables context recall scoring)
          </label>
          <input
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm focus:border-zinc-500 focus:outline-none"
            placeholder="Reference answer, if you have one..."
          />
        </div>
        <button
          onClick={runTrace}
          disabled={loading || !query.trim()}
          className="w-fit rounded-lg bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Running trace…" : "Run trace"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {trace && (
        <div className="flex flex-col gap-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <RootCauseBadge rootCause={trace.root_cause} />
            <div className="flex gap-3">
              <MetricBadge label="Faithfulness" value={trace.faithfulness} threshold={0.7} />
              <MetricBadge label="Answer Relevancy" value={trace.answer_relevancy} threshold={0.7} />
              <MetricBadge label="Context Precision" value={trace.context_precision} threshold={0.5} />
              <MetricBadge label="Context Recall" value={trace.context_recall} threshold={0.5} />
            </div>
          </div>

          {trace.reasons.length > 0 && (
            <ul className="list-inside list-disc rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-sm text-zinc-300">
              {trace.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}

          <div>
            <h2 className="mb-2 text-sm font-semibold text-zinc-300">Generated Answer</h2>
            <p className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-sm text-zinc-200">
              {trace.answer || <span className="text-zinc-500">(empty response)</span>}
            </p>
          </div>

          <div>
            <h2 className="mb-2 text-sm font-semibold text-zinc-300">
              Retrieved Chunks ({trace.retrieved_chunks.length}) — ACL-filtered for role &quot;{trace.role}&quot;
            </h2>
            <div className="grid gap-2 sm:grid-cols-2">
              {trace.retrieved_chunks.map((c) => (
                <ChunkCard key={c.id} chunk={c} flagged={leakedIds.has(c.id)} flagReason="Permission leak: role should not see this ACL tier" />
              ))}
              {trace.retrieved_chunks.length === 0 && (
                <p className="text-sm text-zinc-500">No chunks retrieved for this role/query.</p>
              )}
            </div>
          </div>

          <details className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
            <summary className="cursor-pointer text-sm font-semibold text-zinc-300">
              Unfiltered top-k (permission audit only — never used to answer the query)
            </summary>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {trace.unfiltered_top_chunks.map((c) => (
                <ChunkCard key={c.id} chunk={c} />
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
