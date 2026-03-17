"use client";

import Image from "next/image";
import { ArrowLeft, Search } from "lucide-react";

import { Input } from "@/components/ui/input";
import { ConversationResponse } from "@/lib/types";

interface ConversationPopupProps {
  query: string;
  data: ConversationResponse | null;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:9000";

export function ConversationPopup({ query, data, isLoading, error, onClose }: ConversationPopupProps) {
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/20 p-4" role="dialog" aria-modal="true">
      <div className="grid h-[90vh] w-full max-w-4xl grid-cols-[140px,1fr] overflow-hidden rounded-lg border border-border bg-[#f2f2f2]">
        <aside className="border-r border-border bg-[#d9d9d9] p-4">
          <button
            type="button"
            onClick={onClose}
            className="mb-6 inline-flex items-center gap-2 text-xs text-foreground/80 hover:text-foreground"
            aria-label="Close conversations"
          >
            <ArrowLeft className="h-3 w-3" />
          </button>
          <p className="text-xs uppercase tracking-wide text-foreground/90">Conversation</p>
        </aside>

        <section className="relative flex flex-col overflow-y-auto px-6 pb-6 pt-6">
          <h2
            className="mb-6 text-center"
            style={{
              fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif',
              fontWeight: 700,
              fontStyle: "normal",
              fontSize: "24px",
              lineHeight: "100%",
              letterSpacing: "0px",
              textTransform: "uppercase",
            }}
          >
            Gianfranco Ferré Archive
          </h2>

          <p className="mb-6 text-right text-sm text-foreground/85">{query}</p>

          {isLoading ? (
            <div className="py-6 text-xs text-foreground/70">Loading conversation...</div>
          ) : error ? (
            <div className="py-6 text-xs text-red-700">Failed to load conversation: {error}</div>
          ) : data ? (
            <>
              <div className="mb-6 flex items-center gap-3">
                <div className="relative mt-0.5 h-10 w-10 flex-shrink-0 overflow-hidden rounded-full bg-muted">
                  <Image
                    src={`${API_BASE_URL}/images/ferre.png`}
                    alt="Archive"
                    fill
                    className="object-cover"
                    sizes="40px"
                  />
                </div>

                <div className="flex-1 text-sm leading-relaxed text-foreground/90">
                  <p>{data.response}</p>
                </div>
              </div>

              <div className="mb-6 grid grid-cols-3 gap-4">
                {data.images.map((image) => (
                  <div key={image.source_path} className="relative aspect-[4/5] overflow-hidden rounded-lg bg-muted">
                    <Image src={image.image_url} alt={image.source_path} fill className="object-cover" sizes="200px" />
                  </div>
                ))}
              </div>
            </>
          ) : null}

          <div className="mt-auto">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
              <Input className="pl-10 text-sm" placeholder="Ask the Archive" aria-label="Ask the archive" disabled />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
