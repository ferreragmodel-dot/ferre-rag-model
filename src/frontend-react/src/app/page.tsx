"use client";

import { useState } from "react";

import { ConversationPopup } from "@/components/ConversationPopup";
import { ImageGrid } from "@/components/ImageGrid";
import { SearchBar } from "@/components/SearchBar";
import { TitleBar } from "@/components/TitleBar";

export default function HomePage() {
  const [conversationQuery, setConversationQuery] = useState<string | null>(null);
  const [conversationError, setConversationError] = useState<string | null>(null);

  const handleConversationSubmit = (query: string) => {
    setConversationQuery(query);
    setConversationError(null);
  };

  const closeConversation = () => {
    setConversationQuery(null);
    setConversationError(null);
  };

  return (
    <main>
      <TitleBar />
      <SearchBar onSubmit={handleConversationSubmit} />
      <ImageGrid />

      {conversationQuery ? (
        <ConversationPopup
          initialQuery={conversationQuery}
          onClose={closeConversation}
        />
      ) : null}
    </main>
  );
}
