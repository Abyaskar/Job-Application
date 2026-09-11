"use client";

import { cn } from "@/lib/cn";

/** LinkedIn discovery freshness windows (maps to f_TPR seconds). */
export type FreshnessFilter = "1" | "7" | "30";

/** Discovery location scope — never dataset job locations. */
export type DiscoveryLocationMode = "resume" | "global";

const FRESHNESS: { value: FreshnessFilter; label: string }[] = [
  { value: "1", label: "Last 24 hours" },
  { value: "7", label: "Last 7 days" },
  { value: "30", label: "Last 1 month" },
];

export function Filters({
  freshness,
  onFreshnessChange,
  discoveryLocation,
  onDiscoveryLocationChange,
  resumeLocationLabel,
}: {
  freshness: FreshnessFilter;
  onFreshnessChange: (v: FreshnessFilter) => void;
  discoveryLocation: DiscoveryLocationMode;
  onDiscoveryLocationChange: (v: DiscoveryLocationMode) => void;
  /** Dynamic label from the uploaded resume, e.g. "Mumbai" — never hardcoded. */
  resumeLocationLabel: string | null;
}) {
  const resumeOptionLabel = resumeLocationLabel
    ? `Resume Location (${resumeLocationLabel})`
    : "Resume Location";

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center rounded-xl border border-base-700 bg-base-900 p-1">
        {FRESHNESS.map((f) => (
          <button
            key={f.value}
            type="button"
            onClick={() => onFreshnessChange(f.value)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
              freshness === f.value
                ? "bg-accent text-white"
                : "text-base-400 hover:text-base-100"
            )}
            title={`Apply LinkedIn freshness filter: ${f.label}`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <select
        value={discoveryLocation}
        onChange={(e) =>
          onDiscoveryLocationChange(e.target.value as DiscoveryLocationMode)
        }
        className="rounded-xl border border-base-700 bg-base-900 px-3 py-2 text-xs text-base-300 focus:border-accent focus:outline-none"
      >
        <option value="resume">{resumeOptionLabel}</option>
        <option value="global">Global</option>
      </select>
    </div>
  );
}
