"use client";

import { useEffect, useState } from "react";
import { api, ApiError, type RegressionHistoryItem, type RegressionSummary } from "@/lib/api";
import MetricBadge from "@/components/MetricBadge";

export default function EvalPage() {
  const [latest, setLatest] = useState<RegressionSummary | null>(null);
  const [history, setHistory] = useState<RegressionHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [latestData, historyData] = await Promise.all([
          api.regressionLatest().catch((e) => {
            if (e instanceof ApiError && e.status === 404) return null;
            throw e;
          }),
          api.regressionHistory().catch(() => []),
        ]);
        setLatest(latestData);
        setHistory(historyData);
      } catch (e) {
        setError(e instanceof ApiError ? `${e.status}: ${e.message}` : String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold">CI Regression History</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Results from <code className="text-zinc-300">backend/ci/run_regression.py</code> against the 50-pair
          golden dataset. The pipeline blocks deploy when average faithfulness drops below the gate.
        </p>
      </div>

      {loading && <p className="text-sm text-zinc-500">Loading…</p>}
      {error && (
        <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {!loading && !error && !latest && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-400">
          No regression runs yet. Run <code className="text-zinc-300">python backend/ci/run_regression.py</code>{" "}
          to generate the first report.
        </div>
      )}

      {latest && (
        <div className="flex flex-col gap-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <span
                className={`rounded-full border px-3 py-1 text-xs font-bold ${
                  latest.passed
                    ? "border-emerald-800 bg-emerald-950 text-emerald-300"
                    : "border-rose-800 bg-rose-950 text-rose-300"
                }`}
              >
                {latest.passed ? "GATE PASSED" : "GATE FAILED — DEPLOY BLOCKED"}
              </span>
              <p className="mt-2 text-xs text-zinc-500">
                {latest.total_samples} samples · provider {latest.llm_provider} · store {latest.vector_store} ·{" "}
                {new Date(latest.timestamp).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-3">
              <MetricBadge label="Avg Faithfulness" value={latest.avg_faithfulness} threshold={latest.faithfulness_gate} />
              <MetricBadge label="Avg Relevancy" value={latest.avg_answer_relevancy} threshold={0.7} />
              <MetricBadge label="Avg Ctx Precision" value={latest.avg_context_precision} threshold={0.5} />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-xs text-zinc-500">
                  <th className="py-2 pr-4">Role</th>
                  <th className="py-2 pr-4">Query</th>
                  <th className="py-2 pr-4">Faithfulness</th>
                  <th className="py-2 pr-4">Ctx Precision</th>
                  <th className="py-2 pr-4">Root Cause</th>
                </tr>
              </thead>
              <tbody>
                {latest.per_sample.map((s, i) => (
                  <tr key={i} className="border-b border-zinc-900">
                    <td className="py-2 pr-4 text-zinc-400">{s.role}</td>
                    <td className="py-2 pr-4 text-zinc-200">{s.query}</td>
                    <td className={`py-2 pr-4 tabular-nums ${s.faithfulness >= 0.85 ? "text-emerald-400" : "text-rose-400"}`}>
                      {s.faithfulness.toFixed(2)}
                    </td>
                    <td className="py-2 pr-4 tabular-nums text-zinc-300">{s.context_precision.toFixed(2)}</td>
                    <td className="py-2 pr-4 text-zinc-400">{s.root_cause}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div>
          <h2 className="mb-2 text-sm font-semibold text-zinc-300">Run history</h2>
          <div className="overflow-x-auto rounded-xl border border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900/50 text-xs text-zinc-500">
                  <th className="px-4 py-2">Timestamp</th>
                  <th className="px-4 py-2">Samples</th>
                  <th className="px-4 py-2">Avg Faithfulness</th>
                  <th className="px-4 py-2">Result</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.timestamp} className="border-b border-zinc-900">
                    <td className="px-4 py-2 text-zinc-400">{h.timestamp}</td>
                    <td className="px-4 py-2 text-zinc-300">{h.total_samples}</td>
                    <td className="px-4 py-2 tabular-nums text-zinc-300">{h.avg_faithfulness.toFixed(3)}</td>
                    <td className="px-4 py-2">
                      <span className={h.passed ? "text-emerald-400" : "text-rose-400"}>{h.passed ? "PASS" : "FAIL"}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
