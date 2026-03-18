"use client";

import { Search } from "lucide-react";
import { useState } from "react";

import { Input } from "@/components/ui/input";

interface SearchBarProps {
  onSubmit: (value: string) => void;
}

export function SearchBar({ onSubmit }: SearchBarProps) {
  const [value, setValue] = useState("");

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }
    onSubmit(trimmed);
  };

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40">
      <div className="relative mx-auto w-full max-w-[680px] px-4 pb-6 sm:px-8">
        <div className="pointer-events-auto relative">
          <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
          <Input
            className="pl-10 text-base"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                handleSubmit();
              }
            }}
            placeholder="Ask the Archive"
            aria-label="Search the archive"
          />
        </div>
      </div>
    </div>
  );
}
