import { ACTION_META, RecommendedAction } from "@/lib/types";

export function ActionBadge({ action }: { action: RecommendedAction }) {
  const meta = ACTION_META[action];
  return (
    <span className={`pill ${meta.colorClass}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${meta.dotClass}`} />
      {meta.label}
    </span>
  );
}

export function UncertaintyBadge({ uncertainty }: { uncertainty: number }) {
  if (uncertainty < 0.35) return null;
  const level = uncertainty >= 0.65 ? "High uncertainty" : "Moderate uncertainty";
  return (
    <span className="pill border-base-600 bg-base-800 text-base-300" title="Confidence in this score is reduced — likely due to thin resume or job-description evidence.">
      <span className="h-1.5 w-1.5 rounded-full bg-base-400" />
      {level}
    </span>
  );
}
