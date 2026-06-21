import type { Metadata } from "next";
import { Suspense } from "react";
import LiveConstructionView from "@/components/autobuild/LiveConstructionView";

export const metadata: Metadata = {
  title: "Live Construction Feed — Atomz",
  description: "Live autonomous habitat construction on the Martian surface.",
};

export default function LiveConstructionPage() {
  return (
    <Suspense fallback={null}>
      <LiveConstructionView />
    </Suspense>
  );
}
