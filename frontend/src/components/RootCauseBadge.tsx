const STYLES: Record<string, { label: string; classes: string; description: string }> = {
  pass: {
    label: "PASS",
    classes: "bg-emerald-950 text-emerald-300 border-emerald-800",
    description: "Faithfulness and context precision both cleared their thresholds.",
  },
  retriever_failure: {
    label: "RETRIEVER FAILURE",
    classes: "bg-amber-950 text-amber-300 border-amber-800",
    description: "The retriever did not surface relevant chunks (low context precision).",
  },
  generator_failure: {
    label: "GENERATOR FAILURE",
    classes: "bg-orange-950 text-orange-300 border-orange-800",
    description: "Relevant chunks were retrieved but the LLM ignored or contradicted them.",
  },
  permission_failure: {
    label: "PERMISSION FAILURE",
    classes: "bg-rose-950 text-rose-300 border-rose-800",
    description: "ACL filtering blocked an allowed document or leaked a restricted one.",
  },
};

export default function RootCauseBadge({ rootCause }: { rootCause: string }) {
  const style = STYLES[rootCause] ?? STYLES.pass;
  return (
    <div className={`inline-flex flex-col gap-1 rounded-lg border px-3 py-2 ${style.classes}`}>
      <span className="text-xs font-bold tracking-wide">{style.label}</span>
      <span className="text-xs opacity-80">{style.description}</span>
    </div>
  );
}
