"use client";

import { useState, useEffect, useRef } from "react";
import { flushSync } from "react-dom";
import Image from "next/image";
import { Search } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Input } from "@/components/ui/input";
import {
  fetchArchiveItemDetail,
  startItemChat,
  continueItemChat,
  ChatMessage as APIChatMessage,
  ChatSource,
} from "@/lib/api";
import { ArchiveItemDetailResponse } from "@/lib/types";

interface ItemDetailModalProps {
  sourcePath: string;
  imageUrl?: string;
  onClose: () => void;
}

function renderWithCitations(
  text: string,
  onCitationClick: (num: number) => void,
  activeCitation: number | null
) {
  const parts: (string | JSX.Element)[] = [];
  const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const citationNumbers = match[1].split(",").map((n) => parseInt(n.trim(), 10));
    const citationButtons = citationNumbers.map((num, idx) => (
      <button
        key={`citation-${match!.index}-${idx}`}
        onClick={() => onCitationClick(num)}
        className={`inline font-semibold text-xs cursor-pointer transition-colors mx-0.5 ${
          activeCitation === num
            ? "text-blue-600"
            : "text-blue-500 hover:text-blue-700 hover:underline"
        }`}
      >
        [{num}]{idx < citationNumbers.length - 1 ? ", " : ""}
      </button>
    ));
    parts.push(...citationButtons);
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

export function ItemDetailModal({ sourcePath, imageUrl, onClose }: ItemDetailModalProps) {
  // Detail state
  const [detail, setDetail] = useState<ArchiveItemDetailResponse | null>(null);
  const [isDetailLoading, setIsDetailLoading] = useState(true);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Chat state
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<APIChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [sources, setSources] = useState<ChatSource[]>([]);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchArchiveItemDetail(sourcePath)
      .then(setDetail)
      .catch((e: Error) => setDetailError(e.message))
      .finally(() => setIsDetailLoading(false));
  }, [sourcePath]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const getTextValue = (value: unknown): string | null => {
    if (typeof value !== "string") return null;
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
  };

  const handleSubmit = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isChatLoading) return;

    setInputValue("");
    flushSync(() => {
      setMessages((prev) => [
        ...prev,
        { message_id: `temp-${Date.now()}`, role: "user" as const, content: trimmed },
      ]);
    });

    setIsChatLoading(true);
    try {
      const chatData = chatId
        ? await continueItemChat(chatId, trimmed)
        : await startItemChat(sourcePath, trimmed);
      setChatId(chatData.chat_id);
      setMessages(chatData.messages);
      if (chatData.sources) setSources(chatData.sources);
      setChatError(null);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setIsChatLoading(false);
    }
  };

  // Compute the 4 metadata fields shown on the left
  const season = detail
    ? (getTextValue(detail.metadata.season_path) ?? getTextValue(detail.metadata.season))
    : null;
  const year = detail
    ? (getTextValue(detail.metadata.year_path) ?? getTextValue(detail.metadata.year))
    : null;
  const collectionLine = detail ? getTextValue(detail.metadata.collection_line) : null;
  const archivalDescription = detail ? getTextValue(detail.metadata.description) : null;
  const llmDescription = detail ? getTextValue(detail.metadata.llm_description) : null;
  const description = archivalDescription ?? llmDescription;
  const isAIGeneratedDescription = !archivalDescription && Boolean(llmDescription);

  const heading =
    getTextValue(detail?.metadata.object) ??
    getTextValue(detail?.metadata.file) ??
    getTextValue(detail?.metadata.look) ??
    getTextValue(detail?.metadata.label) ??
    ([getTextValue(detail?.metadata.collection_line), getTextValue(detail?.metadata.season_path)]
      .filter(Boolean)
      .join(" ") || null);

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div
        className="relative grid w-full max-w-5xl overflow-hidden rounded-2xl bg-[#f2f2f2] shadow-2xl"
        style={{
          fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif',
          gridTemplateColumns: "280px 1fr",
          height: "85vh",
        }}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-3 z-10 text-xl text-foreground/65 hover:text-foreground"
          aria-label="Close"
        >
          ×
        </button>

        {/* LEFT: image + 4 metadata fields */}
        <div className="flex flex-col gap-4 overflow-y-auto border-r border-border p-6">
          <div className="relative aspect-[4/5] flex-shrink-0 overflow-hidden rounded-xl border border-border bg-card">
            {isDetailLoading ? (
              <div className="h-full w-full animate-pulse bg-muted" />
            ) : detailError ? (
              <div className="flex h-full items-center justify-center p-4 text-center text-xs text-red-700">
                {detailError}
              </div>
            ) : (
              <Image
                src={detail?.image_url ?? imageUrl ?? ""}
                alt={heading ?? "Archive item"}
                fill
                className="object-cover"
                sizes="280px"
              />
            )}
          </div>

          {!isDetailLoading && detail && (
            <div>
              <h2 className="mb-3 text-xl tracking-tight">{heading}</h2>
              <div className="space-y-1.5 text-sm leading-6 text-foreground/85">
                {season && <p><strong>Season:</strong> {season}</p>}
                {year && <p><strong>Year:</strong> {year}</p>}
                {collectionLine && <p><strong>Collection line:</strong> {collectionLine}</p>}
                {description && (
                  <p>
                    <strong>Description:</strong> {description}{" "}
                    {isAIGeneratedDescription && (
                      <span className="ml-1 inline-flex items-center rounded-full border border-black/10 bg-white/60 px-2 py-0.5 text-[10px] font-medium tracking-[0.04em] text-foreground/65">
                        AI generated
                      </span>
                    )}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: embedded chat */}
        <div className="flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            {chatError && (
              <div className="mb-4 rounded border border-red-200 bg-red-50 p-3 text-xs text-red-700">
                {chatError}
              </div>
            )}

            {messages.length === 0 && !isChatLoading && (
              <div className="flex h-full items-center justify-center text-sm text-foreground/35">
                Ask a question about this item
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.message_id}
                className={`mb-6 flex ${msg.role === "user" ? "justify-end" : "items-start gap-3"}`}
              >
                <div
                  className={`max-w-sm rounded-lg p-3 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-blue-500 text-white"
                      : "border border-border bg-white text-foreground/90"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <ReactMarkdown
                      className="prose prose-sm max-w-none"
                      components={{
                        p: ({ children }) => (
                          <p className="mb-2 last:mb-0">
                            {Array.isArray(children)
                              ? renderWithCitations(
                                  children.map((c) => (typeof c === "string" ? c : "")).join(""),
                                  setActiveCitation,
                                  activeCitation
                                )
                              : renderWithCitations(String(children), setActiveCitation, activeCitation)}
                          </p>
                        ),
                        ul: ({ children }) => (
                          <ul className="mb-2 list-inside list-disc">{children}</ul>
                        ),
                        li: ({ children }) => (
                          <li className="mb-1">
                            {Array.isArray(children)
                              ? renderWithCitations(
                                  children.map((c) => (typeof c === "string" ? c : "")).join(""),
                                  setActiveCitation,
                                  activeCitation
                                )
                              : renderWithCitations(String(children), setActiveCitation, activeCitation)}
                          </li>
                        ),
                        strong: ({ children }) => (
                          <strong className="font-semibold">{children}</strong>
                        ),
                        em: ({ children }) => <em className="italic">{children}</em>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}

            {isChatLoading && (
              <div className="mb-6 flex items-start gap-3">
                <div className="rounded-lg border border-border bg-white p-4">
                  <div className="flex gap-2">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50 delay-100" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50 delay-200" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Citation popup */}
          {activeCitation !== null && sources[activeCitation - 1] && (
            <div
              className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40"
              onClick={() => setActiveCitation(null)}
              role="dialog"
              aria-modal="true"
            >
              <div
                className="flex max-h-[70vh] w-11/12 max-w-md flex-col rounded-lg border border-border bg-white p-6 shadow-lg"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="mb-4 flex flex-shrink-0 items-start justify-between">
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-wide text-foreground/50">
                      Document
                    </div>
                    <h3
                      className={`text-lg font-semibold ${
                        sources[activeCitation - 1].document === "Unknown"
                          ? "italic text-foreground/60"
                          : "text-foreground"
                      }`}
                    >
                      [{activeCitation}] {sources[activeCitation - 1].document}
                    </h3>
                  </div>
                  <button
                    onClick={() => setActiveCitation(null)}
                    className="ml-4 flex-shrink-0 text-xl leading-none text-foreground/50 hover:text-foreground"
                    aria-label="Close"
                  >
                    ✕
                  </button>
                </div>
                <div className="overflow-y-auto text-sm leading-relaxed text-foreground/80">
                  {sources[activeCitation - 1].excerpt}
                </div>
              </div>
            </div>
          )}

          {/* Input */}
          <div className="border-t border-border bg-white px-6 py-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
              <Input
                className="pl-10 text-sm"
                placeholder="Ask the Archive"
                aria-label="Ask the archive"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isChatLoading) handleSubmit();
                }}
                disabled={isChatLoading}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
