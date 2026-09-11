"use client";

import { FileText, Briefcase } from "lucide-react";
import type { EvidenceSnippet } from "@/lib/types";

function highlightMatches(text: string, terms: string[]): React.ReactNode {
  if (terms.length === 0) return text;
  const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    terms.some((t) => t.toLowerCase() === part.toLowerCase()) ? (
      <mark key={i} className="rounded bg-accent/25 px-0.5 text-accent-light">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

export function EvidencePanel({
  evidence,
  highlightTerms,
}: {
  evidence: EvidenceSnippet[];
  highlightTerms: string[];
}) {
  const resumeEvidence = evidence.filter((e) => e.source === "resume");
  const jobEvidence = evidence.filter((e) => e.source === "job_description");

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <EvidenceColumn
        icon={<FileText className="h-4 w-4" />}
        title="Resume evidence"
        items={resumeEvidence}
        highlightTerms={highlightTerms}
        emptyText="No directly matching resume evidence was retrieved for this job's focus skills."
      />
      <EvidenceColumn
        icon={<Briefcase className="h-4 w-4" />}
        title="Job description evidence"
        items={jobEvidence}
        highlightTerms={highlightTerms}
        emptyText="No directly matching job description evidence was retrieved."
      />
    </div>
  );
}

function EvidenceColumn({
  icon,
  title,
  items,
  highlightTerms,
  emptyText,
}: {
  icon: React.ReactNode;
  title: string;
  items: EvidenceSnippet[];
  highlightTerms: string[];
  emptyText: string;
}) {
  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-base-200">
        {icon}
        {title}
      </div>
      {items.length === 0 ? (
        <p className="text-sm text-base-500">{emptyText}</p>
      ) : (
        <ul className="space-y-3">
          {items.map((item, i) => (
            <li key={i} className="rounded-lg border border-base-700/60 bg-base-900/60 p-3 text-sm leading-relaxed text-base-300">
              {highlightMatches(item.text, highlightTerms)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
