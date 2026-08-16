import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Observability Platform",
  description: "Offline/online RAG evaluation, failure localization, and CI regression gating.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        <header className="border-b border-zinc-800">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-sm font-semibold tracking-tight">
              RAG Observability
            </Link>
            <nav className="flex gap-6 text-sm text-zinc-400">
              <Link href="/" className="hover:text-zinc-100">
                Debugger
              </Link>
              <Link href="/eval" className="hover:text-zinc-100">
                CI Regression History
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
