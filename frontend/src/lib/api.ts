const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Role = "Engineering" | "HR" | "Executive" | "General";

export interface RetrievedChunk {
  id: string;
  text: string;
  acl_role: string;
  source: string;
  score: number;
}

export interface QueryResponse {
  query: string;
  role: string;
  answer: string;
  retrieved_chunks: RetrievedChunk[];
}

export interface DebugTraceResponse {
  query: string;
  role: string;
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  unfiltered_top_chunks: RetrievedChunk[];
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number | null;
  root_cause: "pass" | "retriever_failure" | "generator_failure" | "permission_failure";
  reasons: string[];
  permission_blocked_doc_ids: string[];
  permission_leaked_doc_ids: string[];
}

export interface RegressionSummary {
  timestamp: string;
  llm_provider: string;
  vector_store: string;
  total_samples: number;
  avg_faithfulness: number;
  avg_answer_relevancy: number;
  avg_context_precision: number;
  faithfulness_gate: number;
  passed: boolean;
  per_sample: Array<{
    query: string;
    role: string;
    expected_source: string | null;
    answer: string;
    retrieved_sources: string[];
    faithfulness: number;
    answer_relevancy: number;
    context_precision: number;
    context_recall: number | null;
    root_cause: string;
  }>;
}

export interface RegressionHistoryItem {
  timestamp: string;
  avg_faithfulness: number;
  avg_answer_relevancy: number;
  avg_context_precision: number;
  passed: boolean;
  total_samples: number;
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore parse failure */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string; llm_provider: string; vector_store: string }>("/health"),

  ingest: (documents: { text: string; acl_role: string; source?: string }[]) =>
    request<{ ingested: number; total_in_store: number }>("/ingest", {
      method: "POST",
      body: JSON.stringify({ documents }),
    }),

  query: (query: string, role: Role, top_k?: number) =>
    request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify({ query, role, top_k }),
    }),

  debugTrace: (query: string, role: Role, ground_truth?: string, top_k?: number) =>
    request<DebugTraceResponse>("/debug/trace", {
      method: "POST",
      body: JSON.stringify({ query, role, ground_truth, top_k }),
    }),

  regressionLatest: () => request<RegressionSummary>("/regression/latest"),

  regressionHistory: (limit = 20) => request<RegressionHistoryItem[]>(`/regression/history?limit=${limit}`),
};

export { ApiError };
