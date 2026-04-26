"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { ConversationPopup } from "@/components/ConversationPopup";
import { FilterBar } from "@/components/FilterBar";
import { ImageGrid } from "@/components/ImageGrid";
import { SearchBar } from "@/components/SearchBar";
import { TitleBar } from "@/components/TitleBar";
import { ActiveFilters, fetchFilterOptions } from "@/lib/api";

export default function HomePage() {
  const [conversationQuery, setConversationQuery] = useState<string | null>(null);
  const [filters, setFilters] = useState<ActiveFilters>({});

  const { data: filterOptions } = useQuery({
    queryKey: ["filter-options"],
    queryFn: fetchFilterOptions,
    staleTime: Infinity,
  });

  const handleConversationSubmit = (query: string) => {
    setConversationQuery(query);
  };

  const closeConversation = () => {
    setConversationQuery(null);
  };

  return (
    <main className="pt-14">
      <TitleBar />
      <SearchBar onSubmit={handleConversationSubmit} />

      {filterOptions ? (
        <FilterBar options={filterOptions} filters={filters} onFilterChange={setFilters} />
      ) : null}

      <ImageGrid filters={filters} />

      {conversationQuery ? (
        <ConversationPopup
          initialQuery={conversationQuery}
          onClose={closeConversation}
        />
      ) : null}
    </main>
  );
}
