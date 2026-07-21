import type { ReactNode } from "react";

export const metadata = {
  title: "Open Edge Speech Benchmark",
  description: "Objective, reproducible benchmarks for local (edge) speech AI.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
