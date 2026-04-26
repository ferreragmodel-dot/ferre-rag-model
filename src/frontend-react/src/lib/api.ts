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

export interface ActiveFilters {
  season_path?: string;
  garments?: string[];
  colors?: string[];
  materials?: string[];
}

export interface FilterOptions {
  seasons: string[];
  garments: string[];
  colors: string[];
  materials: string[];
}

export async function fetchFilterOptions(): Promise<FilterOptions> {
  const url = new URL("/archive/filter-options", API_BASE_URL);
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error(`Failed to fetch filter options: ${response.status}`);
  return response.json() as Promise<FilterOptions>;
}

export async function fetchLandingFeed(
  offset = 0,
  limit = 24,
  filters: ActiveFilters = {},
): Promise<LandingFeedResponse> {
  const url = new URL("/archive/landing-feed", API_BASE_URL);
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("limit", String(limit));
  if (filters.season_path) url.searchParams.set("season_path", filters.season_path);
  for (const g of filters.garments ?? []) url.searchParams.append("garments", g);
  for (const c of filters.colors ?? []) url.searchParams.append("colors", c);
  for (const m of filters.materials ?? []) url.searchParams.append("materials", m);

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

export async function startItemChat(sourcePath: string, message: string): Promise<ChatData> {
  const url = new URL("/llm-agent/item-chats", API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_path: sourcePath, content: message }),
  });

  if (!response.ok) {
    throw new Error(`Failed to start item chat: ${response.status}`);
  }

  return response.json() as Promise<ChatData>;
}

export async function continueItemChat(chatId: string, message: string): Promise<ChatData> {
  const url = new URL(`/llm-agent/item-chats/${chatId}`, API_BASE_URL);

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: message }),
  });

  if (!response.ok) {
    throw new Error(`Failed to continue item chat: ${response.status}`);
  }

  return response.json() as Promise<ChatData>;
}

export interface ClusterItem {
  source_path: string;
  image_url: string;
}

export interface ItemClusterResponse {
  cluster_id: string;
  items: ClusterItem[];
}

export async function fetchItemCluster(sourcePath: string): Promise<ItemClusterResponse> {
  const url = new URL("/archive/item-cluster", API_BASE_URL);
  url.searchParams.set("source_path", sourcePath);
  const response = await fetch(url.toString());
  if (!response.ok) throw new Error(`Failed to fetch item cluster: ${response.status}`);
  return response.json() as Promise<ItemClusterResponse>;
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
