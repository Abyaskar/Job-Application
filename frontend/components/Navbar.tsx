"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, Command } from "lucide-react";
import { cn } from "@/lib/cn";

const links = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/compare", label: "Compare" },
];

export function Navbar({ onOpenPalette }: { onOpenPalette?: () => void }) {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-base-800/80 bg-base-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
            <Sparkles className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold tracking-tight text-base-100">
            Job Application Strategy AI
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "rounded-lg px-3 py-2 text-sm transition-colors",
                pathname?.startsWith(l.href)
                  ? "bg-base-800 text-base-100"
                  : "text-base-400 hover:text-base-100"
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>

        <button
          onClick={onOpenPalette}
          className="flex items-center gap-2 rounded-lg border border-base-700 bg-base-900 px-3 py-1.5 text-xs text-base-400 transition-colors hover:border-base-500 hover:text-base-200"
        >
          <Command className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Search jobs</span>
          <span className="kbd">⌘K</span>
        </button>
      </div>
    </header>
  );
}
