"use client";

import { useState } from "react";

import { ConversationPopup } from "@/components/ConversationPopup";
import { ImageGrid } from "@/components/ImageGrid";
import { SearchBar } from "@/components/SearchBar";
import { TitleBar } from "@/components/TitleBar";
import { fetchArchiveConversation } from "@/lib/api";
import { ConversationResponse } from "@/lib/types";

export default function HomePage() {
  const [conversationQuery, setConversationQuery] = useState<string | null>(null);
  const [conversationData, setConversationData] = useState<ConversationResponse | null>(null);
  const [isConversationLoading, setIsConversationLoading] = useState(false);
  const [conversationError, setConversationError] = useState<string | null>(null);

  const handleConversationSubmit = (query: string) => {
    setConversationQuery(query);
    setConversationData(null);
    setConversationError(null);
    setIsConversationLoading(true);

    fetchArchiveConversation(query)
      .then((response) => {
        setConversationData(response);
      })
      .catch((error: Error) => {
        setConversationError(error.message);
      })
      .finally(() => {
        setIsConversationLoading(false);
      });
  };

  const closeConversation = () => {
    setConversationQuery(null);
    setConversationData(null);
    setConversationError(null);
    setIsConversationLoading(false);
  };

  return (
    <main>
      <TitleBar />
      <SearchBar onSubmit={handleConversationSubmit} />
      <ImageGrid />

      {conversationQuery ? (
        <ConversationPopup
          query={conversationQuery}
          data={conversationData}
          isLoading={isConversationLoading}
          error={conversationError}
          onClose={closeConversation}
        />
      ) : null}
    </main>
  );
}
