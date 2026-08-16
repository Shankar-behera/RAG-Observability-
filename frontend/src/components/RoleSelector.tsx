import type { Role } from "@/lib/api";

const ROLES: Role[] = ["Engineering", "HR", "Executive", "General"];

export default function RoleSelector({
  value,
  onChange,
}: {
  value: Role;
  onChange: (role: Role) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-zinc-400">Query as role</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as Role)}
        className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 focus:border-zinc-500 focus:outline-none"
      >
        {ROLES.map((r) => (
          <option key={r} value={r}>
            {r}
          </option>
        ))}
      </select>
    </div>
  );
}
