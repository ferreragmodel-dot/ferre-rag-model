"use client";

import { useState, useEffect, useRef } from "react";
import { flushSync } from "react-dom";
import Image from "next/image";
import { ArrowLeft, Search } from "lucide-react";

import { Input } from "@/components/ui/input";
import { startChat, continueChat, ChatMessage as APIChatMessage, fetchArchiveConversation } from "@/lib/api";

interface ConversationImageItem {
  source_path: string;
  image_url: string;
}

interface ConversationPopupProps {
  initialQuery: string;
  onClose: () => void;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:9000";

export function ConversationPopup({ initialQuery, onClose }: ConversationPopupProps) {
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<APIChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<ConversationImageItem[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize chat on mount
  useEffect(() => {
    const initChat = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Add initial user message immediately
        const initialUserMessage: APIChatMessage = {
          message_id: `init-user-${Date.now()}`,
          role: "user",
          content: initialQuery,
        };
        flushSync(() => {
          setMessages([initialUserMessage]);
        });

        const chatData = await startChat(initialQuery);
        setChatId(chatData.chat_id);

        // Merge: keep our initial user message, add only the assistant response
        const responseMessages = chatData.messages;
        const assistantMessages = responseMessages.filter((msg) => msg.role === "assistant");

        flushSync(() => {
          setMessages((prev) => [...prev, ...assistantMessages]);
        });

        // Fetch images for the initial query
        try {
          const conversationData = await fetchArchiveConversation(initialQuery);
          setImages(conversationData.images);
        } catch {
          // If images fetch fails, continue without images
          setImages([]);
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
                {/* Find first assistant message and split */}
                {messages.length > 0 && (() => {
                  const firstAssistantIdx = messages.findIndex((msg) => msg.role === "assistant");
                  const initialMessages = messages.slice(0, Math.max(1, firstAssistantIdx + 1));
                  const followUpMessages = messages.slice(firstAssistantIdx + 1);

                  return (
                    <>
                      {/* Initial exchange */}
                      {initialMessages.map((msg) => (
                        <div
                          key={msg.message_id}
                          className={`mb-6 flex ${
                            msg.role === "user" ? "justify-end" : "items-start gap-3"
                          }`}
                        >
                          {msg.role === "assistant" && (
                            <div className="relative mt-0.5 h-10 w-10 flex-shrink-0 overflow-hidden rounded-full bg-muted">
                              <Image
                                src={`${API_BASE_URL}/images/ferre.png`}
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
                            <p>{msg.content}</p>
                          </div>
                        </div>
                      ))}

                      {/* Images after initial exchange */}
                      {images.length > 0 && (
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

                      {/* Follow-up messages */}
                      {followUpMessages.map((msg) => (
                        <div
                          key={msg.message_id}
                          className={`mb-6 flex ${
                            msg.role === "user" ? "justify-end" : "items-start gap-3"
                          }`}
                        >
                          {msg.role === "assistant" && (
                            <div className="relative mt-0.5 h-10 w-10 flex-shrink-0 overflow-hidden rounded-full bg-muted">
                              <Image
                                src={`${API_BASE_URL}/images/ferre.png`}
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
                            <p>{msg.content}</p>
                          </div>
                        </div>
                      ))}

                      {/* Loading indicator for follow-ups */}
                      {isLoading && followUpMessages.length > 0 && (
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
                  );
                })()}
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
