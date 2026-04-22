import {
  ArchiveItemDetailResponse,
  ConversationResponse,
  LandingFeedResponse,
} from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:9000";

export interface ChatMessage {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  image_path?: string;
}

export interface ChatImage {
  source_path: string;
  image_url: string;
}

export interface ChatSource {
  document: string;
  excerpt: string;
}

export interface ChatData {
  chat_id: string;
  title: string;
  dts: number;
  messages: ChatMessage[];
  images?: ChatImage[];
  sources?: ChatSource[];
}

export async function fetchLandingFeed(offset = 0, limit = 24): Promise<LandingFeedResponse> {
  const url = new URL("/archive/landing-feed", API_BASE_URL);
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("limit", String(limit));

  const response = await fetch(url.toString(), {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch landing feed: ${response.status}`);
  }

  return response.json() as Promise<LandingFeedResponse>;
}

export async function fetchArchiveItemDetail(
  sourcePath: string,
): Promise<ArchiveItemDetailResponse> {
  const url = new URL("/archive/item-detail", API_BASE_URL);
  url.searchParams.set("source_path", sourcePath);

  const response = await fetch(url.toString(), {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch archive item detail: ${response.status}`);
  }

  return response.json() as Promise<ArchiveItemDetailResponse>;
}

export async function fetchArchiveConversation(message: string): Promise<ConversationResponse> {
  const url = new URL("/archive/conversation", API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch archive conversation: ${response.status}`);
  }

  return response.json() as Promise<ConversationResponse>;
}

// Chat endpoints for multi-turn conversations
export async function startChat(message: string): Promise<ChatData> {
  const url = new URL("/llm-agent/chats", API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ content: message }),
  });

  if (!response.ok) {
    throw new Error(`Failed to start chat: ${response.status}`);
  }

  return response.json() as Promise<ChatData>;
}

export async function continueChat(chatId: string, message: string): Promise<ChatData> {
  const url = new URL(`/llm-agent/chats/${chatId}`, API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ content: message }),
  });

  if (!response.ok) {
    throw new Error(`Failed to continue chat: ${response.status}`);
  }

  return response.json() as Promise<ChatData>;
}

export async function getChat(chatId: string): Promise<ChatData> {
  const url = new URL(`/llm-agent/chats/${chatId}`, API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch chat: ${response.status}`);
  }

  return response.json() as Promise<ChatData>;
}
