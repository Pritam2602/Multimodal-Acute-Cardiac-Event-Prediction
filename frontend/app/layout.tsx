import type { Metadata } from "next";
import "./globals.css";
import AuthGuard from "@/components/ui/AuthGuard";

export const metadata: Metadata = {
  title: "AMI Prediction Platform — Temporal Multimodal Cardiac AI",
  description:
    "Research-grade clinical AI reasoning workstation for acute myocardial injury prediction using multimodal temporal analysis of 12-lead ECG and troponin trajectories.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-base text-text-primary antialiased">
        <AuthGuard>{children}</AuthGuard>
      </body>
    </html>
  );
}
