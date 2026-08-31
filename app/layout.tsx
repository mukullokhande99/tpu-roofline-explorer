import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TPU Roofline Explorer",
  description:
    "Interactive SRAM-aware roofline modeling for TPU-style systolic accelerators.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "./favicon.svg",
    shortcut: "./favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
