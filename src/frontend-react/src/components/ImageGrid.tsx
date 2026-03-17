"use client";

import Image from "next/image";
import { Search } from "lucide-react";
import { InfiniteData, useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { ImageCard } from "@/components/ImageCard";
import { Input } from "@/components/ui/input";
import { fetchArchiveItemDetail, fetchLandingFeed } from "@/lib/api";
import { ArchiveImageItem, ArchiveItemDetailResponse, LandingFeedResponse } from "@/lib/types";

const PAGE_SIZE = 24;
const GRID_GAP_PX = 16;
const CARD_ASPECT_HEIGHT_OVER_WIDTH = 5 / 4;

function getColumnCount(width: number): number {
  if (width >= 1280) {
    return 5;
  }
  if (width >= 1024) {
    return 4;
  }
  if (width >= 640) {
    return 3;
  }
  return 2;
}


export function ImageGrid() {
  const gridRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const [columnCount, setColumnCount] = useState(2);
  const [staggerOffsetPx, setStaggerOffsetPx] = useState(0);
  const [selectedItem, setSelectedItem] = useState<ArchiveImageItem | null>(null);
  const [detail, setDetail] = useState<ArchiveItemDetailResponse | null>(null);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const { data, error, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useInfiniteQuery<
      LandingFeedResponse,
      Error,
      InfiniteData<LandingFeedResponse, number>,
      ["landing-feed"],
      number
    >({
      queryKey: ["landing-feed"],
      queryFn: ({ pageParam }) => fetchLandingFeed(pageParam, PAGE_SIZE),
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

  const closeDetail = () => {
    setSelectedItem(null);
    setDetail(null);
    setDetailError(null);
    setIsDetailLoading(false);
  };

  const handleImageClick = (item: ArchiveImageItem) => {
    setSelectedItem(item);
    setDetail(null);
    setDetailError(null);
    setIsDetailLoading(true);

    fetchArchiveItemDetail(item.source_path)
      .then((response) => {
        setDetail(response);
      })
      .catch((error: Error) => {
        setDetailError(error.message);
      })
      .finally(() => {
        setIsDetailLoading(false);
      });
  };

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) {
      return;
    }

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
              <ImageCard key={item.id} item={item} onClick={handleImageClick} />
            ))}
          </div>
        ))}
      </div>

      <div ref={sentinelRef} className="h-8" />

      {isFetchingNextPage ? (
        <div className="py-4 text-center text-sm text-foreground/55">Loading more artifacts...</div>
      ) : null}

      {selectedItem ? (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" role="dialog" aria-modal="true">
          <div
            className="relative w-full max-w-5xl rounded-2xl bg-[#f2f2f2] p-6 shadow-2xl"
            style={{ fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif' }}
          >
            <button
              type="button"
              onClick={closeDetail}
              className="absolute right-4 top-3 text-xl text-foreground/65 hover:text-foreground"
              aria-label="Close"
            >
              ×
            </button>

            {isDetailLoading ? (
              <div className="py-16 text-center text-sm text-foreground/70">Loading archive detail...</div>
            ) : detailError ? (
              <div className="py-16 text-center text-sm text-red-700">Failed to load detail: {detailError}</div>
            ) : detail ? (
              <div className="grid gap-6 md:grid-cols-[320px,1fr]">
                <div className="relative aspect-[4/5] overflow-hidden rounded-xl border border-border bg-card">
                  <Image
                    src={detail.image_url}
                    alt={selectedItem.title}
                    fill
                    className="object-cover"
                    sizes="320px"
                  />
                </div>
                <div className="flex max-h-[70vh] flex-col">
                  <div className="flex-1 overflow-y-auto pr-2">
                    <h2 className="mb-4 text-3xl tracking-tight">{detail.metadata.object ?? selectedItem.title}</h2>
                    <div className="space-y-2 text-sm leading-6 text-foreground/85">
                      <p><strong>Season:</strong> {detail.metadata.season ?? "-"}</p>
                      <p><strong>Collection line:</strong> {detail.metadata.collection_line ?? "-"}</p>
                      <p><strong>Look:</strong> {detail.metadata.look ?? "-"}</p>
                      <p><strong>Year:</strong> {detail.metadata.year ?? "-"}</p>
                      <p><strong>Description:</strong> {detail.metadata.description ?? "-"}</p>
                      <p><strong>Materials:</strong> {detail.metadata.materials ?? "-"}</p>
                      <p><strong>Working process:</strong> {detail.metadata.working_process ?? "-"}</p>
                    </div>
                  </div>

                  <div className="mt-4 border-t border-border/60 pt-4">
                    <div className="relative">
                      <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/45" />
                      <Input
                        className="pl-10 text-base"
                        placeholder="Ask the Archive"
                        aria-label="Ask the archive"
                        disabled
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}
