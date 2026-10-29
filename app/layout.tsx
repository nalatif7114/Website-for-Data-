import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial NLP Lab — Indonesian Financial Sentiment Intelligence",
  description: "Compare fine-tuned language models on Indonesian financial news.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
