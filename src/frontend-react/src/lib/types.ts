export interface ArchiveImageItem {
  id: string;
  title: string;
  image_url: string;
  source_path: string;
}

export interface ArchiveItemMetadata {
  [key: string]: string | null;
}

export interface ArchiveItemDetailResponse {
  id: string;
  image_url: string;
  metadata: ArchiveItemMetadata;
}

export interface LandingFeedResponse {
  items: ArchiveImageItem[];
  pagination: {
    offset: number;
    limit: number;
    next_offset: number;
    has_more: boolean;
  };
}
