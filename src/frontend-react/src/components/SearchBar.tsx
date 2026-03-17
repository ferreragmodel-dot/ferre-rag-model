"use client";

import { Search } from "lucide-react";

import { Input } from "@/components/ui/input";

export function SearchBar() {
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40">
      <div className="relative mx-auto w-full max-w-[680px] px-4 pb-6 sm:px-8">
        <div className="pointer-events-auto relative">
          <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
          <Input
            className="pl-10 text-base"
            placeholder="Ask the Archive"
            aria-label="Search the archive"
            disabled
          />
        </div>
      </div>
    </div>
  );
}
