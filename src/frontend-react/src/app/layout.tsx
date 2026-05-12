import type { Metadata } from "next";

import { QueryProvider } from "@/providers/query-provider";
import PasswordGate from "@/components/PasswordGate";

import "./globals.css";

export const metadata: Metadata = {
  title: "Gianfranco Ferré Archive",
  description: "Visual archive interface for Gianfranco Ferre.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans text-foreground antialiased">
        <QueryProvider>
          <PasswordGate>{children}</PasswordGate>
        </QueryProvider>
      </body>
    </html>
  );
}
