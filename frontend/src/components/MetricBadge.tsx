export default function MetricBadge({
  label,
  value,
  threshold = 0.7,
}: {
  label: string;
  value: number | null;
  threshold?: number;
}) {
  const color =
    value === null
      ? "text-zinc-500 border-zinc-800"
      : value >= threshold
      ? "text-emerald-400 border-emerald-800"
      : "text-rose-400 border-rose-800";

  return (
    <div className={`flex flex-col items-center rounded-lg border px-4 py-3 ${color}`}>
      <span className="text-2xl font-semibold tabular-nums">{value === null ? "—" : value.toFixed(2)}</span>
      <span className="mt-1 text-xs text-zinc-500">{label}</span>
    </div>
  );
}
