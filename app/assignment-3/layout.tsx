import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NVIDIA GPU Roofline Explorer · Assignment 3",
  description: "Interactive hierarchical roofline modeling for NVIDIA GPUs.",
};

export default function AssignmentThreeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
