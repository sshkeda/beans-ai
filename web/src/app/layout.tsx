import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Play Beans AI",
  description: "Play Beans AI, an LLM-powered chess engine.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={cn(
          geistSans.variable,
          geistMono.variable,
          "antialiased dark font-sans"
        )}
      >
        <nav className="absolute top-4 left-1/2 -translate-x-1/2">
          <div className="flex items-center">
            <Button variant="ghost" size="sm">
              <Link href="/">
                <h1 className="text-2xl font-bold italic">beans-ai</h1>
              </Link>
            </Button>
          </div>
        </nav>
        {children}
        <Link
          href="https://github.com/sshkeda/beans-ai"
          target="_blank"
          className="text-xs absolute bottom-4 left-1/2 -translate-x-1/2 text-muted-foreground/40"
        >
          Built by Stephen Shkeda
        </Link>
        <Toaster />
      </body>
    </html>
  );
}
