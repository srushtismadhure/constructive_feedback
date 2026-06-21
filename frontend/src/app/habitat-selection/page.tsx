import type { Metadata } from "next";
import HabitatSelectionScreen from "@/components/autobuild/HabitatSelectionScreen";

export const metadata: Metadata = {
  title: "Select Habitat System — Atomz",
  description: "Choose an autonomous shelter architecture for Martian deployment.",
};

export default function HabitatSelectionPage() {
  return <HabitatSelectionScreen />;
}
