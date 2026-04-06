"use client";

import { useState, useEffect, useRef } from "react";
import { flushSync } from "react-dom";
import Image from "next/image";
import { ArrowLeft, Search } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Input } from "@/components/ui/input";
import { startChat, continueChat, ChatMessage as APIChatMessage, ChatSource } from "@/lib/api";

interface ConversationImageItem {
  source_path: string;
  image_url: string;
}

interface ConversationPopupProps {
  initialQuery: string;
  onClose: () => void;
}


// Parse citation markers and make them interactive
function renderWithCitations(
  text: string,
  onCitationClick: (num: number) => void,
  activeCitation: number | null
) {
  const parts: (string | JSX.Element)[] = [];
  // Updated regex to match [1], [1, 2], [1,2], etc.
  const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    // Add text before citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    // Parse multiple citation numbers from the match
    const citationNumbers = match[1]
      .split(',')
      .map(n => n.trim())
      .map(n => parseInt(n, 10));

    // Create buttons for each citation number
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
        [{num}]{idx < citationNumbers.length - 1 ? ', ' : ''}
      </button>
    ));

    parts.push(...citationButtons);
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

export function ConversationPopup({ initialQuery, onClose }: ConversationPopupProps) {
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<APIChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<ConversationImageItem[]>([]);
  const [sources, setSources] = useState<ChatSource[]>([]);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize chat on mount
  useEffect(() => {
    const initChat = async () => {
      try {
        // Add user message immediately with forced flush
        flushSync(() => {
          const newUserMessage: APIChatMessage = {
            message_id: `temp-${Date.now()}`,
            role: "user",
            content: initialQuery,
          };
          setMessages([newUserMessage]);
        });

        setIsLoading(true);
        setError(null);

        const chatData = await startChat(initialQuery);
        setChatId(chatData.chat_id);
        setMessages(chatData.messages);

        // Use images and sources from the chat response
        if (chatData.images) {
          setImages(chatData.images);
        }
        if (chatData.sources) {
          setSources(chatData.sources);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to start chat");
      } finally {
        setIsLoading(false);
      }
    };

    initChat();
  }, [initialQuery]);

  const handleSubmit = async () => {
    if (!inputValue.trim() || !chatId || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue("");

    // Add user message to chat immediately with forced flush
    flushSync(() => {
      const newUserMessage: APIChatMessage = {
        message_id: `temp-${Date.now()}`,
        role: "user",
        content: userMessage,
      };
      setMessages((prev) => [...prev, newUserMessage]);
    });

    setIsLoading(true);
    try {
      const chatData = await continueChat(chatId, userMessage);
      setMessages(chatData.messages);
      // Update images and sources from response
      if (chatData.images) {
        setImages(chatData.images);
      }
      if (chatData.sources) {
        setSources(chatData.sources);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && inputValue.trim() && !isLoading) {
      handleSubmit();
    }
  };

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

        <section className="relative flex flex-col overflow-hidden">
          <h2
            className="border-b border-border px-6 py-4 text-center"
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

          {/* Main content area */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            {error && (
              <div className="mb-4 rounded bg-red-50 p-3 text-xs text-red-700 border border-red-200">
                {error}
              </div>
            )}

            {isLoading && messages.length === 0 ? (
              <div className="py-12 text-center text-sm text-foreground/60">
                Loading conversation...
              </div>
            ) : (
              <>
                {/* All messages */}
                {messages.map((msg) => (
                  <div
                    key={msg.message_id}
                    className={`mb-6 flex ${
                      msg.role === "user" ? "justify-end" : "items-start gap-3"
                    }`}
                  >
                    {msg.role === "assistant" && (
                      <div className="relative mt-0.5 h-10 w-10 flex-shrink-0 overflow-hidden rounded-full bg-muted">
                        <Image
                          src="/ferre.png"
                          alt="Archive"
                          fill
                          className="object-cover"
                          sizes="40px"
                        />
                      </div>
                    )}
                    <div
                      className={`rounded-lg p-3 text-sm leading-relaxed max-w-sm ${
                        msg.role === "user"
                          ? "bg-blue-500 text-white"
                          : "bg-white border border-border text-foreground/90"
                      }`}
                    >
                      {msg.role === "assistant" ? (
                        <div>
                          <ReactMarkdown
                            className="prose prose-sm max-w-none"
                            components={{
                              p: ({ children }) => (
                                <p className="mb-2 last:mb-0">
                                  {Array.isArray(children) ? renderWithCitations(
                                    children.join(""),
                                    setActiveCitation,
                                    activeCitation
                                  ) : renderWithCitations(
                                    String(children),
                                    setActiveCitation,
                                    activeCitation
                                  )}
                                </p>
                              ),
                              ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
                              li: ({ children }) => <li className="mb-1">{children}</li>,
                              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                              em: ({ children }) => <em className="italic">{children}</em>,
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <ReactMarkdown
                          className="prose prose-sm max-w-none"
                          components={{
                            p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                            ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
                            li: ({ children }) => <li className="mb-1">{children}</li>,
                            strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                            em: ({ children }) => <em className="italic">{children}</em>,
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      )}
                    </div>
                  </div>
                ))}

                {/* Images after first response */}
                {images.length > 0 && !isLoading && (
                  <div className="mb-6">
                    <div className="grid grid-cols-3 gap-3">
                      {images.map((img, idx) => (
                        <div
                          key={idx}
                          className="relative aspect-[3/5] overflow-hidden rounded-lg border border-border bg-muted"
                        >
                          <Image
                            src={img.image_url}
                            alt="Archive item"
                            fill
                            className="object-cover object-top"
                            sizes="(max-width: 768px) 100px, 150px"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Sources popup modal */}
                {activeCitation !== null && sources[activeCitation - 1] && (
                  <div
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40"
                    onClick={() => setActiveCitation(null)}
                    role="dialog"
                    aria-modal="true"
                  >
                    <div
                      className="bg-white rounded-lg p-6 max-w-md w-11/12 shadow-lg border border-border"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <div className="text-xs uppercase tracking-wide text-foreground/50 mb-1">Document</div>
                          <h3 className={`text-lg font-semibold ${
                            sources[activeCitation - 1].document === "Unknown"
                              ? "italic text-foreground/60"
                              : "text-foreground"
                          }`}>
                            [{activeCitation}] {sources[activeCitation - 1].document}
                          </h3>
                        </div>
                        <button
                          onClick={() => setActiveCitation(null)}
                          className="text-foreground/50 hover:text-foreground text-xl leading-none"
                          aria-label="Close"
                        >
                          ✕
                        </button>
                      </div>

                      <div className="mb-4">
                        <div className="text-xs uppercase tracking-wide text-foreground/50 mb-1">Year</div>
                        <div className={`text-sm ${
                          sources[activeCitation - 1].year === "Unknown"
                            ? "italic text-foreground/60"
                            : "text-foreground/70"
                        }`}>
                          {sources[activeCitation - 1].year}
                        </div>
                      </div>

                      <div className="text-sm text-foreground/80 leading-relaxed">
                        {sources[activeCitation - 1].excerpt}
                      </div>
                    </div>
                  </div>
                )}

                {/* Loading indicator for follow-ups */}
                {isLoading && messages.length > 0 && (
                  <div className="mb-6 flex items-center gap-3">
                    <div className="relative mt-0.5 h-10 w-10 flex-shrink-0 overflow-hidden rounded-full bg-muted">
                      <Image
                        src="/ferre.png"
                        alt="Archive"
                        fill
                        className="object-cover"
                        sizes="40px"
                      />
                    </div>
                    <div className="rounded-lg bg-white border border-border p-4">
                      <div className="flex gap-2">
                        <div className="h-2 w-2 rounded-full bg-foreground/50 animate-bounce"></div>
                        <div className="h-2 w-2 rounded-full bg-foreground/50 animate-bounce delay-100"></div>
                        <div className="h-2 w-2 rounded-full bg-foreground/50 animate-bounce delay-200"></div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input area */}
          <div className="border-t border-border bg-white px-6 py-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
              <Input
                className="pl-10 text-sm"
                placeholder="Ask the Archive"
                aria-label="Ask the archive"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
