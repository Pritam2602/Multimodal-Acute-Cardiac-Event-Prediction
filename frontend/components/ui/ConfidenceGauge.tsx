"use client";
import { useEffect, useRef } from "react";

interface ConfidenceGaugeProps {
  probability: number;
  size?: number;
  label?: string;
}

export default function ConfidenceGauge({ probability, size = 160, label = "AMI Probability" }: ConfidenceGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2 + 10;
    const r = size * 0.38;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const fillAngle = startAngle + (endAngle - startAngle) * probability;

    ctx.clearRect(0, 0, size, size);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.stroke();

    // Color based on probability
    let color: string;
    if (probability >= 0.75) color = "#dc2626";
    else if (probability >= 0.6494) color = "#ea580c";
    else if (probability >= 0.25) color = "#d97706";
    else color = "#1a7a45";

    // Fill arc (no glow on light theme)
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.stroke();

    // Threshold marker at 0.6494
    const thresholdAngle = startAngle + (endAngle - startAngle) * 0.6494;
    const tx = cx + (r + 8) * Math.cos(thresholdAngle);
    const ty = cy + (r + 8) * Math.sin(thresholdAngle);
    ctx.beginPath();
    ctx.arc(tx, ty, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#f59e0b";
    ctx.fill();

    // Center text
    ctx.textAlign = "center";
    ctx.fillStyle = color;
    ctx.font = `bold ${size * 0.175}px 'JetBrains Mono', monospace`;
    ctx.fillText(`${(probability * 100).toFixed(1)}%`, cx, cy + 4);

    ctx.fillStyle = "#6b9580";
    ctx.font = `${size * 0.075}px 'DM Sans', sans-serif`;
    ctx.fillText(label, cx, cy + size * 0.22);
  }, [probability, size, label]);

  return <canvas ref={canvasRef} />;
}
