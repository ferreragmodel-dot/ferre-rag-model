import Link from "next/link";

export function TitleBar() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-border bg-[#f7f7f5]">
      <div className="flex h-14 items-center justify-between px-6 sm:px-10">
        <Link href="/">
          <h1
            style={{
              fontFamily: 'Didot, "Bodoni 72", "Times New Roman", serif',
              fontWeight: 700,
              fontSize: "15px",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            Gianfranco Ferré Archive
          </h1>
        </Link>
        <nav className="flex items-center">
          <span className="text-[11px] uppercase tracking-[0.12em] text-foreground/50">
            Data Exploration
          </span>
        </nav>
      </div>
    </header>
  );
}
