"use client";

import { useEffect, useRef, useState } from "react";

import { ActiveFilters, FilterOptions } from "@/lib/api";

interface FilterBarProps {
  options: FilterOptions;
  filters: ActiveFilters;
  onFilterChange: (filters: ActiveFilters) => void;
}

type SingleKey = "season_path";
type MultiKey = "garments" | "colors" | "materials";

const SINGLE_CATEGORIES: { key: SingleKey; label: string; optionsKey: keyof FilterOptions }[] = [
  { key: "season_path", label: "COLLECTION", optionsKey: "seasons" },
];

const MULTI_CATEGORIES: { key: MultiKey; label: string; optionsKey: keyof FilterOptions }[] = [
  { key: "garments", label: "ITEM", optionsKey: "garments" },
  { key: "colors", label: "COLOUR", optionsKey: "colors" },
  { key: "materials", label: "MATERIAL", optionsKey: "materials" },
];

function chipLabel(label: string, selected: string | string[] | undefined): string {
  if (!selected || (Array.isArray(selected) && selected.length === 0)) return label;
  if (typeof selected === "string") return selected.length > 18 ? selected.slice(0, 17) + "…" : selected;
  if (selected.length === 1) return selected[0].length > 18 ? selected[0].slice(0, 17) + "…" : selected[0];
  return `${label} (${selected.length})`;
}

function isActive(selected: string | string[] | undefined): boolean {
  if (!selected) return false;
  if (Array.isArray(selected)) return selected.length > 0;
  return true;
}

export function FilterBar({ options, filters, onFilterChange }: FilterBarProps) {
  const [openCategory, setOpenCategory] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenCategory(null);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const hasActiveFilters =
    Boolean(filters.season_path) ||
    (filters.garments?.length ?? 0) > 0 ||
    (filters.colors?.length ?? 0) > 0 ||
    (filters.materials?.length ?? 0) > 0;

  const toggleSingle = (key: SingleKey, value: string) => {
    const next = { ...filters };
    if (next[key] === value) delete next[key];
    else next[key] = value;
    onFilterChange(next);
    setOpenCategory(null);
  };

  const toggleMulti = (key: MultiKey, value: string) => {
    const current = filters[key] ?? [];
    const next = { ...filters };
    if (current.includes(value)) {
      const updated = current.filter((v) => v !== value);
      if (updated.length === 0) delete next[key];
      else next[key] = updated;
    } else {
      next[key] = [...current, value];
    }
    onFilterChange(next);
  };

  const clearAll = () => {
    onFilterChange({});
    setOpenCategory(null);
  };

  const chipClass = (active: boolean, isOpen: boolean) =>
    `flex items-center gap-1.5 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.12em] transition-colors ${
      active
        ? "border-foreground bg-foreground text-background"
        : isOpen
          ? "border-foreground/40 text-foreground"
          : "border-border text-foreground/55 hover:border-foreground/40 hover:text-foreground"
    }`;

  return (
    <div
      ref={containerRef}
      className="mx-auto flex w-full max-w-[1440px] flex-wrap items-center gap-2 px-4 pb-3 pt-1 sm:px-8"
    >
      {/* Single-select: COLLECTION */}
      {SINGLE_CATEGORIES.map(({ key, label, optionsKey }) => {
        const active = isActive(filters[key]);
        const isOpen = openCategory === key;
        const opts = options[optionsKey] as string[];
        return (
          <div key={key} className="relative">
            <button
              onClick={() => setOpenCategory(isOpen ? null : key)}
              className={chipClass(active, isOpen)}
            >
              <span className="max-w-[160px] truncate">{chipLabel(label, filters[key])}</span>
              <span className={`text-[7px] transition-transform ${isOpen ? "rotate-180" : ""}`}>▼</span>
            </button>
            {isOpen && (
              <div className="absolute left-0 top-full z-50 mt-1.5 max-h-64 w-60 overflow-y-auto rounded-xl border border-border bg-[#f7f7f5] py-1 shadow-md">
                {opts.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => toggleSingle(key, opt)}
                    className={`w-full px-3 py-1.5 text-left text-[11px] transition-colors ${
                      filters[key] === opt
                        ? "bg-foreground/8 font-semibold text-foreground"
                        : "text-foreground/70 hover:bg-foreground/5 hover:text-foreground"
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Multi-select: ITEM, COLOUR, MATERIAL */}
      {MULTI_CATEGORIES.map(({ key, label, optionsKey }) => {
        const selected = filters[key] ?? [];
        const active = selected.length > 0;
        const isOpen = openCategory === key;
        const opts = options[optionsKey] as string[];
        return (
          <div key={key} className="relative">
            <button
              onClick={() => setOpenCategory(isOpen ? null : key)}
              className={chipClass(active, isOpen)}
            >
              <span className="max-w-[160px] truncate">{chipLabel(label, selected)}</span>
              <span className={`text-[7px] transition-transform ${isOpen ? "rotate-180" : ""}`}>▼</span>
            </button>
            {isOpen && (
              <div className="absolute left-0 top-full z-50 mt-1.5 max-h-64 w-56 overflow-y-auto rounded-xl border border-border bg-[#f7f7f5] py-1 shadow-md">
                {opts.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => toggleMulti(key, opt)}
                    className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] transition-colors ${
                      selected.includes(opt)
                        ? "bg-foreground/8 text-foreground"
                        : "text-foreground/70 hover:bg-foreground/5 hover:text-foreground"
                    }`}
                  >
                    <span
                      className={`flex h-3 w-3 shrink-0 items-center justify-center rounded-sm border ${
                        selected.includes(opt)
                          ? "border-foreground bg-foreground text-background"
                          : "border-foreground/30"
                      }`}
                    >
                      {selected.includes(opt) && (
                        <svg viewBox="0 0 8 8" className="h-2 w-2" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <polyline points="1,4 3,6.5 7,1.5" />
                        </svg>
                      )}
                    </span>
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {hasActiveFilters && (
        <button
          onClick={clearAll}
          className="ml-1 text-[10px] uppercase tracking-[0.12em] text-foreground/40 underline underline-offset-2 hover:text-foreground/70"
        >
          Clear
        </button>
      )}
    </div>
  );
}
