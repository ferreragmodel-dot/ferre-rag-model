"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function TitleBar() {
  const pathname = usePathname();

  const navLink = (href: string, label: string) => (
    <Link
      href={href}
      className={`text-[11px] uppercase tracking-[0.12em] transition-colors ${
        pathname === href
          ? "text-foreground underline underline-offset-4"
          : "text-foreground/50 hover:text-foreground"
      }`}
    >
      {label}
    </Link>
  );

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
        <nav className="flex items-center gap-6">
          {navLink("/", "Data Exploration")}
          {navLink("/about", "About")}
        </nav>
      </div>
    </header>
  );
}
