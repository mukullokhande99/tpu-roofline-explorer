import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Flex-PE Research Explorer",
  description:
    "Interactive research explorer for SIMD multiprecision Flex-PE hardware and TPU-level design-space exploration.",
};

export default function AssignmentTwoLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
