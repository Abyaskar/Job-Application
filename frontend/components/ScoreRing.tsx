"use client";

import { motion } from "framer-motion";

export function ScoreRing({
  value,
  size = 64,
  strokeWidth = 6,
  label,
}: {
  value: number; // 0..1
  size?: number;
  strokeWidth?: number;
  label?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(1, value));
  const offset = circumference * (1 - pct);

  const color = pct >= 0.75 ? "#3ddc97" : pct >= 0.5 ? "#f5b942" : pct >= 0.3 ? "#5fb0ff" : "#6b7186";

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#232733"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-sm font-semibold tabular-nums text-base-100">{Math.round(pct * 100)}%</span>
        {label && <span className="text-[9px] uppercase tracking-wide text-base-400">{label}</span>}
      </div>
    </div>
  );
}
