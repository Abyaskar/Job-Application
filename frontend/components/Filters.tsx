"use client";

import type { SearchMode } from "@/lib/types";
import { cn } from "@/lib/cn";

const MODES: { value: SearchMode; label: string }[] = [
  { value: "hybrid", label: "Hybrid" },
  { value: "vector", label: "Vector" },
  { value: "keyword", label: "Keyword" },
];

export function Filters({
  mode,
  onModeChange,
  location,
  onLocationChange,
  domain,
  onDomainChange,
  locations,
  domains,
}: {
  mode: SearchMode;
  onModeChange: (m: SearchMode) => void;
  location: string;
  onLocationChange: (v: string) => void;
  domain: string;
  onDomainChange: (v: string) => void;
  locations: string[];
  domains: string[];
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center rounded-xl border border-base-700 bg-base-900 p-1">
        {MODES.map((m) => (
          <button
            key={m.value}
            onClick={() => onModeChange(m.value)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
              mode === m.value ? "bg-accent text-white" : "text-base-400 hover:text-base-100"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      <select
        value={location}
        onChange={(e) => onLocationChange(e.target.value)}
        className="rounded-xl border border-base-700 bg-base-900 px-3 py-2 text-xs text-base-300 focus:border-accent focus:outline-none"
      >
        <option value="">All locations</option>
        {locations.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>

      <select
        value={domain}
        onChange={(e) => onDomainChange(e.target.value)}
        className="rounded-xl border border-base-700 bg-base-900 px-3 py-2 text-xs text-base-300 focus:border-accent focus:outline-none"
      >
        <option value="">All domains</option>
        {domains.map((d) => (
          <option key={d} value={d}>
            {d.replace(/_/g, " ")}
          </option>
        ))}
      </select>
    </div>
  );
}
