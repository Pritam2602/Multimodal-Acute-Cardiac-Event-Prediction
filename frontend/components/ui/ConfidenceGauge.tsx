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
    ctx.strokeStyle = "#162845";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.stroke();

    // Color gradient based on probability
    let color: string;
    if (probability >= 0.75) color = "#f43f5e";
    else if (probability >= 0.48) color = "#f97316";
    else if (probability >= 0.25) color = "#eab308";
    else color = "#10b981";

    // Glow effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;

    // Fill arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.stroke();

    ctx.shadowBlur = 0;

    // Threshold marker at 0.48
    const thresholdAngle = startAngle + (endAngle - startAngle) * 0.48;
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

    ctx.fillStyle = "#4a6a94";
    ctx.font = `${size * 0.075}px 'DM Sans', sans-serif`;
    ctx.fillText(label, cx, cy + size * 0.22);
  }, [probability, size, label]);

  return <canvas ref={canvasRef} />;
}
