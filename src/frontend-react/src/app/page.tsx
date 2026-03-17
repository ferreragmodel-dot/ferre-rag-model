import { ImageGrid } from "@/components/ImageGrid";
import { SearchBar } from "@/components/SearchBar";
import { TitleBar } from "@/components/TitleBar";

export default function HomePage() {
  return (
    <main>
      <TitleBar />
      <SearchBar />
      <ImageGrid />
    </main>
  );
}
