import Image from "next/image";

import { ArchiveImageItem } from "@/lib/types";

interface ImageCardProps {
  item: ArchiveImageItem;
}

export function ImageCard({ item }: ImageCardProps) {

  return (
    <article className="group relative overflow-hidden rounded-xl border border-border bg-card shadow-museum transition-transform duration-300 hover:-translate-y-1">
      <div className="relative aspect-[4/5] w-full">
        <Image
          src={item.image_url}
          alt={item.title}
          fill
          className="object-cover"
          sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 20vw"
          priority={false}
        />
      </div>
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent px-3 pb-3 pt-10 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
        <p className="line-clamp-2 text-xs text-white/95">{item.title}</p>
      </div>
    </article>
  );
}
