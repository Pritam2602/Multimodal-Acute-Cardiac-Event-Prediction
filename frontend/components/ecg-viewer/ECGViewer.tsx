"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { ZoomIn, ZoomOut, Maximize2, Play, Pause, RotateCcw } from "lucide-react";
import { generateAll12Leads, downsample, LEAD_NAMES, ECGPattern } from "@/lib/utils/ecgGenerator";
import { Admission } from "@/lib/utils/mockData";
import { getPatternForAdmission } from "@/lib/utils/ecgGenerator";

const GRID_COLS = 4;
const GRID_ROWS = 3;
// Layout: Row0: I,II,III,aVR | Row1: aVL,aVF,V1,V2 | Row2: V3,V4,V5,V6
const LEAD_LAYOUT: string[][] = [
  ["I", "II", "III", "aVR"],
  ["aVL", "aVF", "V1", "V2"],
  ["V3", "V4", "V5", "V6"],
];

const DISPLAY_SAMPLES = 1000;

interface ECGViewerProps {
  admission: Admission;
  activeTimestep: number;
  attentionWeights?: Record<string, number>;
}

function getLeadColor(lead: string, attentionWeights?: Record<string, number>): string {
  if (!attentionWeights) return "#0369a1";
  const w = attentionWeights[lead] ?? 0;
  if (w >= 0.80) return "#dc2626";
  if (w >= 0.60) return "#ea580c";
  if (w >= 0.40) return "#d97706";
  if (w >= 0.20) return "#0284c7";
  return "#64748b";
}

function getLeadGlow(lead: string, attentionWeights?: Record<string, number>): number {
  if (!attentionWeights) return 0;
  const w = attentionWeights[lead] ?? 0;
  return w * 4;
}

function drawECGChannel(
  ctx: CanvasRenderingContext2D,
  signal: number[],
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  glowStrength: number,
  zoom: number,
  offset: number,
  label: string,
  attentionVal: number,
) {
  // Background cell (ECG paper — warm white)
  ctx.fillStyle = "#fffbf9";
  ctx.fillRect(x, y, w, h);

  // Grid lines (ECG paper style — light red/pink)
  ctx.strokeStyle = "rgba(239, 68, 68, 0.12)";
  ctx.lineWidth = 0.5;
  const gridSpacing = 20;
  for (let gx = x; gx <= x + w; gx += gridSpacing) {
    ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
  }
  for (let gy = y; gy <= y + h; gy += gridSpacing) {
    ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
  }
  // Major grid every 5 small
  ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
  ctx.lineWidth = 0.5;
  for (let gx = x; gx <= x + w; gx += gridSpacing * 5) {
    ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
  }
  for (let gy = y; gy <= y + h; gy += gridSpacing * 5) {
    ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
  }

  // Baseline
  ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(x, y + h / 2);
  ctx.lineTo(x + w, y + h / 2);
  ctx.stroke();

  // Determine visible window
  const visibleSamples = Math.floor(DISPLAY_SAMPLES / zoom);
  const startIdx = Math.min(Math.floor(offset * DISPLAY_SAMPLES), DISPLAY_SAMPLES - visibleSamples);
  const endIdx = Math.min(startIdx + visibleSamples, signal.length);
  const visible = signal.slice(startIdx, endIdx);

  if (visible.length === 0) return;

  // Scale
  let minV = Infinity, maxV = -Infinity;
  for (const v of signal) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
  const range = Math.max(maxV - minV, 0.001);
  const padding = 0.12;

  // Glow
  if (glowStrength > 0) {
    ctx.shadowColor = color;
    ctx.shadowBlur = glowStrength;
  }

  // Draw signal
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.lineJoin = "round";
  for (let i = 0; i < visible.length; i++) {
    const px = x + (i / (visible.length - 1)) * w;
    const norm = (visible[i] - minV) / range;
    const py = y + h * padding + norm * h * (1 - 2 * padding);
    const screenY = y + h - (py - y);
    if (i === 0) ctx.moveTo(px, screenY);
    else ctx.lineTo(px, screenY);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Lead label
  ctx.fillStyle = color;
  ctx.font = "bold 11px 'JetBrains Mono', monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, x + 5, y + 15);

  // Attention bar
  if (attentionVal > 0) {
    const barW = w * 0.6;
    const barH = 2;
    ctx.fillStyle = "rgba(0,0,0,0.07)";
    ctx.fillRect(x + 5, y + h - 8, barW, barH);
    ctx.fillStyle = color;
    ctx.fillRect(x + 5, y + h - 8, barW * attentionVal, barH);
  }
}

export default function ECGViewer({ admission, activeTimestep, attentionWeights }: ECGViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isolatedLead, setIsolatedLead] = useState<string | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const offsetRef = useRef(0);

  const { pattern, hr } = getPatternForAdmission(admission.hadm_id, activeTimestep);
  const [signals, setSignals] = useState<Record<string, number[]>>({});

  useEffect(() => {
    const raw = generateAll12Leads(hr, pattern as ECGPattern, activeTimestep);
    const down: Record<string, number[]> = {};
    for (const lead of LEAD_NAMES) {
      down[lead] = downsample(raw[lead], DISPLAY_SAMPLES);
    }
    setSignals(down);
    setOffset(0);
    setIsPlaying(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
  }, [activeTimestep, admission.hadm_id, pattern, hr]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || Object.keys(signals).length === 0) return;

    const W = container.clientWidth;
    const H = container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== W * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#f8f7f5";
    ctx.fillRect(0, 0, W, H);

    if (isolatedLead) {
      const signal = signals[isolatedLead];
      if (!signal) return;
      const color = getLeadColor(isolatedLead, attentionWeights);
      const glow = getLeadGlow(isolatedLead, attentionWeights);
      const attnVal = attentionWeights?.[isolatedLead] ?? 0;
      drawECGChannel(ctx, signal, 4, 4, W - 8, H - 8, color, glow, zoom, offset, isolatedLead, attnVal);
    } else {
      const cellW = Math.floor(W / GRID_COLS);
      const cellH = Math.floor((H - 40) / GRID_ROWS);

      for (let row = 0; row < GRID_ROWS; row++) {
        for (let col = 0; col < GRID_COLS; col++) {
          const lead = LEAD_LAYOUT[row][col];
          const signal = signals[lead];
          if (!signal) continue;
          const color = getLeadColor(lead, attentionWeights);
          const glow = getLeadGlow(lead, attentionWeights);
          const attnVal = attentionWeights?.[lead] ?? 0;
          drawECGChannel(
            ctx, signal,
            col * cellW + 1,
            row * cellH + 1,
            cellW - 2, cellH - 2,
            color, glow, zoom, offset, lead, attnVal,
          );
        }
      }

      // Rhythm strip (Lead II full width at bottom)
      const rhythmSignal = signals["II"];
      if (rhythmSignal) {
        const rhythmY = GRID_ROWS * cellH + 2;
        const rhythmH = H - rhythmY - 2;
        if (rhythmH > 10) {
          const color = getLeadColor("II", attentionWeights);
          drawECGChannel(ctx, rhythmSignal, 1, rhythmY, W - 2, rhythmH, color, 0, zoom, offset, "II (Rhythm Strip)", attentionWeights?.["II"] ?? 0);
          ctx.fillStyle = "#64748b";
          ctx.font = "10px 'DM Sans', sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(`${hr} bpm · 10s · ${DISPLAY_SAMPLES}pt`, W - 6, rhythmY + 14);
        }
      }
    }
  }, [signals, zoom, offset, isolatedLead, attentionWeights, hr]);

  useEffect(() => { draw(); }, [draw]);

  // Playback animation
  useEffect(() => {
    if (!isPlaying) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }
    const speed = 0.0003;
    const animate = () => {
      offsetRef.current = (offsetRef.current + speed) % 1;
      setOffset(offsetRef.current);
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [isPlaying]);

  const reset = () => { setZoom(1); setOffset(0); offsetRef.current = 0; setIsPlaying(false); };

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-default bg-surface">
        <div className="flex items-center gap-1">
          <button onClick={() => setZoom(z => Math.min(z * 1.5, 8))} className="p-1.5 rounded hover:bg-elevated text-text-muted hover:text-text-primary transition-colors">
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setZoom(z => Math.max(z / 1.5, 1))} className="p-1.5 rounded hover:bg-elevated text-text-muted hover:text-text-primary transition-colors">
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-[10px] font-mono text-text-muted px-1">{zoom.toFixed(1)}x</span>
        </div>

        <div className="w-px h-4 bg-border-default" />

        <button
          onClick={() => setIsPlaying(p => !p)}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
            isPlaying ? "bg-cyan/10 text-cyan border border-cyan/30" : "bg-elevated text-text-secondary hover:text-text-primary border border-border-default"
          }`}
        >
          {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          {isPlaying ? "Pause" : "Play"}
        </button>

        <button onClick={reset} className="p-1.5 rounded hover:bg-elevated text-text-muted hover:text-text-primary transition-colors">
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <div className="flex-1 flex items-center gap-2 px-2">
          <input
            type="range"
            min={0} max={100} step={1}
            value={Math.round(offset * 100)}
            onChange={e => { const v = parseInt(e.target.value) / 100; setOffset(v); offsetRef.current = v; }}
            className="flex-1 accent-cyan h-1"
            disabled={isPlaying}
          />
        </div>

        <div className="w-px h-4 bg-border-default" />

        {/* Lead isolation */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsolatedLead(null)}
            className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${!isolatedLead ? "bg-cyan/10 text-cyan" : "text-text-muted hover:text-text-primary"}`}
          >
            12-Lead
          </button>
          <select
            value={isolatedLead ?? ""}
            onChange={e => setIsolatedLead(e.target.value || null)}
            className="bg-elevated border border-border-default rounded px-2 py-1 text-[10px] text-text-primary focus:outline-none"
          >
            <option value="">Isolate lead...</option>
            {LEAD_NAMES.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>

        <Maximize2 className="w-3.5 h-3.5 text-text-muted ml-1" />

        {/* Timestep indicator */}
        <div className="ml-auto flex items-center gap-1.5 px-2 py-1 rounded bg-cyan/5 border border-cyan/20">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan animate-pulse-slow" />
          <span className="text-[10px] font-mono font-semibold text-cyan">
            T{activeTimestep} · {pattern.replace(/_/g, " ").replace("stemi", "STEMI")} · {hr} BPM
          </span>
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="flex-1 relative ecg-grid overflow-hidden">
        <canvas ref={canvasRef} className="absolute inset-0" />
        {Object.keys(signals).length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-8 h-8 border-2 border-cyan border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="text-xs text-text-muted">Rendering ECG signals...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
