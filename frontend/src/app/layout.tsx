import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Atomz — Autonomous Habitat Construction",
  description:
    "Robotic construction systems for extreme terrain, off-world habitats, and autonomous infrastructure.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased scanlines">{children}</body>
    </html>
  );
}
