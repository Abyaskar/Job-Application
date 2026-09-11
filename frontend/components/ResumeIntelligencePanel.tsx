"use client";

import { motion } from "framer-motion";
import {
  FileCheck,
  FileText,
  Brain,
  Briefcase,
  GraduationCap,
  MapPin,
  Mail,
  Phone,
  BadgeCheck,
  AlertTriangle,
  XCircle,
  Loader2,
  CheckCircle2,
} from "lucide-react";
import type { ParsedResume } from "@/lib/types";

interface ResumeIntelligencePanelProps {
  resume: ParsedResume | null;
  isLoading?: boolean;
}

export function ResumeIntelligencePanel({
  resume,
  isLoading = false,
}: ResumeIntelligencePanelProps) {
  if (isLoading) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-base-100 mb-4">
          Resume Intelligence
        </h3>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center gap-3 animate-pulse">
              <Loader2 className="h-4 w-4 text-base-500 animate-spin" />
              <span className="text-sm text-base-400">Processing...</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!resume) {
    return (
      <div className="card p-6 text-center text-base-400">
        <FileText className="h-12 w-12 mx-auto mb-3 text-base-600" />
        <p>No resume uploaded yet.</p>
        <p className="text-sm mt-1">Upload your resume to see intelligent analysis.</p>
      </div>
    );
  }

  // Validation state indicators
  const validationStateConfig = {
    valid_resume: {
      icon: CheckCircle2,
      color: "text-signal-apply",
      bgColor: "bg-signal-apply/10",
      borderColor: "border-signal-apply/30",
      title: "Valid Resume",
      message: "Your resume was successfully processed.",
    },
    resume_requires_ocr: {
      icon: AlertTriangle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "OCR Required",
      message: "This appears to be a scanned/image document.",
    },
    low_extraction_quality: {
      icon: AlertTriangle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Low Quality Extraction",
      message: "Text extraction quality was poor.",
    },
    not_a_resume: {
      icon: XCircle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Not a Resume",
      message: "The uploaded document is not a resume.",
    },
    corrupted_file: {
      icon: XCircle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Corrupted File",
      message: "The file could not be read.",
    },
    unsupported_document: {
      icon: XCircle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Unsupported Format",
      message: "This file format is not supported.",
    },
    insufficient_information: {
      icon: AlertTriangle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Insufficient Information",
      message: "The resume lacks required information.",
    },
    extraction_failed: {
      icon: XCircle,
      color: "text-signal-danger",
      bgColor: "bg-signal-danger/10",
      borderColor: "border-signal-danger/30",
      title: "Extraction Failed",
      message: "An error occurred during processing.",
    },
  };

  const stateConfig = validationStateConfig[resume.validation_state as keyof typeof validationStateConfig] || validationStateConfig.valid_resume;
  const StateIcon = stateConfig.icon;

  // Processing steps visualization
  const processingSteps = [
    { id: "file_uploaded", label: "File Recognized", completed: true },
    { id: "type_detected", label: "Content Extracted", completed: (resume.extraction_quality_score ?? 0) > 0.3 },
    { id: "resume_detected", label: "Resume Detected", completed: resume.document_type === "resume" },
    { id: "information_identified", label: "Information Identified", completed: !!resume.detected_name || !!resume.detected_email },
    { id: "skills_identified", label: "Skills Identified", completed: resume.skills.length > 0 },
    { id: "experience_identified", label: "Experience Identified", completed: resume.experience.length > 0 || resume.total_experience_years > 0 },
    { id: "career_domain_identified", label: "Career Domain Identified", completed: true },
  ];

  return (
    <div className="card p-6 space-y-6">
      {/* Validation Status */}
      <div className={`flex items-start gap-4 p-4 rounded-xl border ${stateConfig.bgColor} ${stateConfig.borderColor}`}>
        <StateIcon className={`h-6 w-6 shrink-0 ${stateConfig.color}`} />
        <div className="flex-1">
          <h3 className={`font-semibold ${stateConfig.color}`}>{stateConfig.title}</h3>
          <p className="text-sm text-base-400 mt-1">{stateConfig.message}</p>
          {resume.extraction_quality_score !== undefined && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-base-500">Extraction Quality:</span>
              <div className="flex-1 h-2 bg-base-800 rounded-full overflow-hidden max-w-[150px]">
                <motion.div
                  className={`h-full rounded-full ${
                    resume.extraction_quality_score > 0.7
                      ? "bg-signal-apply"
                      : resume.extraction_quality_score > 0.4
                      ? "bg-accent"
                      : "bg-signal-danger"
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${resume.extraction_quality_score * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <span className="text-xs text-base-400 tabular-nums">
                {Math.round(resume.extraction_quality_score * 100)}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Processing Steps Timeline */}
      <div>
        <h4 className="text-sm font-medium text-base-300 mb-3">Processing Pipeline</h4>
        <div className="space-y-2">
          {processingSteps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center gap-3"
            >
              {step.completed ? (
                <CheckCircle2 className="h-4 w-4 text-signal-apply shrink-0" />
              ) : (
                <div className="h-4 w-4 rounded-full border-2 border-base-700 shrink-0" />
              )}
              <span className={`text-sm ${step.completed ? "text-base-200" : "text-base-500"}`}>
                {step.label}
              </span>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Extracted Personal Information */}
      {(resume.detected_name || resume.detected_email || resume.detected_phone || resume.current_location) && (
        <div>
          <h4 className="text-sm font-medium text-base-300 mb-3">Detected Information</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {resume.detected_name && (
              <InfoItem icon={BadgeCheck} label="Name" value={resume.detected_name} />
            )}
            {resume.detected_email && (
              <InfoItem icon={Mail} label="Email" value={resume.detected_email} />
            )}
            {resume.detected_phone && (
              <InfoItem icon={Phone} label="Phone" value={resume.detected_phone} />
            )}
            {resume.current_location && (
              <InfoItem icon={MapPin} label="Location" value={resume.current_location} />
            )}
          </div>
        </div>
      )}

      {/* Skills Summary */}
      {resume.skills && resume.skills.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-base-300 mb-3">
            Identified Skills ({resume.skills.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {resume.skills.slice(0, 15).map((skill) => (
              <span
                key={skill}
                className="pill bg-accent/10 text-accent-light border-accent/20 text-xs"
              >
                {skill.replace(/_/g, " ")}
              </span>
            ))}
            {resume.skills.length > 15 && (
              <span className="pill bg-base-800 text-base-400 border-base-700 text-xs">
                +{resume.skills.length - 15} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Experience Summary */}
      {(resume.experience.length > 0 || resume.total_experience_years > 0) && (
        <div>
          <h4 className="text-sm font-medium text-base-300 mb-3">Experience</h4>
          <div className="space-y-2">
            {resume.total_experience_years > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <Briefcase className="h-4 w-4 text-base-500" />
                <span className="text-base-200">
                  {resume.total_experience_years.toFixed(1)} years total experience
                </span>
              </div>
            )}
            {resume.experience.slice(0, 3).map((exp, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                <Briefcase className="h-4 w-4 text-base-500 mt-0.5 shrink-0" />
                <div>
                  <span className="text-base-200">{exp.title}</span>
                  {exp.company && (
                    <span className="text-base-400"> at {exp.company}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Education Summary */}
      {resume.education && resume.education.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-base-300 mb-3">Education</h4>
          <div className="space-y-2">
            {resume.education.map((edu, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                <GraduationCap className="h-4 w-4 text-base-500 mt-0.5 shrink-0" />
                <div>
                  <span className="text-base-200">{edu.degree}</span>
                  {edu.field && <span className="text-base-400"> in {edu.field}</span>}
                  {edu.institution && (
                    <span className="text-base-400"> — {edu.institution}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Career Domains */}
      {resume.career_domains && resume.career_domains.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-base-300 mb-3">Career Domains</h4>
          <div className="space-y-2">
            {resume.career_domains.map((domain, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="text-base-200">{domain.domain_name}</span>
                <span className="text-xs text-base-500">
                  {Math.round(domain.confidence * 100)}% confidence
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function InfoItem({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg bg-base-900/50 border border-base-800">
      <Icon className="h-4 w-4 text-base-500 shrink-0" />
      <div>
        <div className="text-xs text-base-500">{label}</div>
        <div className="text-sm text-base-200">{value}</div>
      </div>
    </div>
  );
}
