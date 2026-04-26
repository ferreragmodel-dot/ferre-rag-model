"use client";

import { InfiniteData, useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { ImageCard } from "@/components/ImageCard";
import { ItemDetailModal } from "@/components/ItemDetailModal";
import { ActiveFilters, fetchLandingFeed } from "@/lib/api";
import { ArchiveImageItem, LandingFeedResponse } from "@/lib/types";

const PAGE_SIZE = 24;
const GRID_GAP_PX = 16;
const CARD_ASPECT_HEIGHT_OVER_WIDTH = 5 / 4;

function getColumnCount(width: number): number {
  if (width >= 1280) return 5;
  if (width >= 1024) return 4;
  if (width >= 640) return 3;
  return 2;
}

interface ImageGridProps {
  filters?: ActiveFilters;
}

export function ImageGrid({ filters = {} }: ImageGridProps) {
  const gridRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const [columnCount, setColumnCount] = useState(2);
  const [staggerOffsetPx, setStaggerOffsetPx] = useState(0);
  const [selectedItem, setSelectedItem] = useState<ArchiveImageItem | null>(null);

  const { data, error, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteQuery<
      LandingFeedResponse,
      Error,
      InfiniteData<LandingFeedResponse, number>,
      ["landing-feed", ActiveFilters],
      number
    >({
      queryKey: ["landing-feed", filters],
      queryFn: ({ pageParam }) => fetchLandingFeed(pageParam, PAGE_SIZE, filters),
      initialPageParam: 0,
      getNextPageParam: (lastPage) =>
        lastPage.pagination.has_more ? lastPage.pagination.next_offset : undefined,
    });

  useEffect(() => {
    const update = () => setColumnCount(getColumnCount(window.innerWidth));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;
    const updateStagger = () => {
      const gridWidth = grid.clientWidth;
      if (!gridWidth || columnCount <= 0) { setStaggerOffsetPx(0); return; }
      const totalGap = GRID_GAP_PX * (columnCount - 1);
      const columnWidth = (gridWidth - totalGap) / columnCount;
      const tileHeight = columnWidth * CARD_ASPECT_HEIGHT_OVER_WIDTH;
      setStaggerOffsetPx(Math.round(tileHeight * 0.5));
    };
    updateStagger();
    const ro = new ResizeObserver(updateStagger);
    ro.observe(grid);
    return () => ro.disconnect();
  }, [columnCount]);

  const getColumnPaddingTop = (columnIndex: number) =>
    columnIndex % 2 === 1 ? staggerOffsetPx : 0;

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const firstEntry = entries[0];
        if (firstEntry.isIntersecting && hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { rootMargin: "400px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  const columns = useMemo(() => {
    const items = data?.pages.flatMap((page) => page.items) ?? [];
    const nextColumns = Array.from({ length: columnCount }, () => [] as typeof items);
    items.forEach((item, index) => {
      nextColumns[index % columnCount].push(item);
    });
    return nextColumns;
  }, [columnCount, data]);

  const skeletonColumns = useMemo(() => {
    return Array.from({ length: columnCount }, (_, columnIndex) =>
      Array.from({ length: 6 }, (_, rowIndex) => ({
        key: `skeleton-${columnIndex}-${rowIndex}`,
      })),
    );
  }, [columnCount]);

  if (isLoading) {
    return (
      <section className="mx-auto w-full max-w-[1440px] px-4 pb-10 sm:px-8">
        <div
          ref={gridRef}
          className="grid gap-4"
          style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
        >
          {skeletonColumns.map((column, columnIndex) => (
            <div
              key={`skeleton-column-${columnIndex}`}
              className="flex flex-col gap-4"
              style={{ paddingTop: `${getColumnPaddingTop(columnIndex)}px` }}
            >
              {column.map((tile) => (
                <div key={tile.key} className="aspect-[4/5] rounded-xl bg-muted" />
              ))}
            </div>
          ))}
        </div>
      </section>
    );
  }

  if (error instanceof Error) {
    return (
      <section className="mx-auto w-full max-w-[1440px] px-4 pb-10 sm:px-8">
        <div className="rounded-xl border border-border bg-card p-6 text-sm text-red-700">
          Failed to load archive feed: {error.message}
        </div>
      </section>
    );
  }

  const isEmpty = !isLoading && (data?.pages.flatMap((p) => p.items).length ?? 0) === 0;

  if (isEmpty) {
    return (
      <section className="mx-auto w-full max-w-[1440px] px-4 pb-10 sm:px-8">
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <p className="text-base text-foreground/50">No items found.</p>
          <p className="mt-1.5 text-sm text-foreground/35">Please try different filters.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto w-full max-w-[1440px] px-4 pb-10 sm:px-8">
      <div
        ref={gridRef}
        className="grid gap-4"
        style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
      >
        {columns.map((column, columnIndex) => (
          <div
            key={`column-${columnIndex}`}
            className="flex flex-col gap-4"
            style={{ paddingTop: `${getColumnPaddingTop(columnIndex)}px` }}
          >
            {column.map((item) => (
              <ImageCard key={item.id} item={item} onClick={setSelectedItem} />
            ))}
          </div>
        ))}
      </div>

      <div ref={sentinelRef} className="h-8" />

      {isFetchingNextPage ? (
        <div className="py-4 text-center text-sm text-foreground/55">Loading more artifacts...</div>
      ) : null}

      {selectedItem ? (
        <ItemDetailModal
          sourcePath={selectedItem.source_path}
          imageUrl={selectedItem.image_url}
          onClose={() => setSelectedItem(null)}
        />
      ) : null}
    </section>
  );
}
