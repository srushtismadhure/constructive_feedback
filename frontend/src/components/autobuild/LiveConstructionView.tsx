"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { frameUrl, streamUrl, statusUrl } from "@/lib/api";

interface LogEntry {
  node: string;
  text: string;
  warning?: boolean;
}

// Shape of events emitted by run_construction_agent onto the queue (core/main.py).
interface AgentEvent {
  type: "step" | "warning" | "heartbeat" | "done";
  node?: string;
  event?: string;
  step?: number;
  warning?: string;
  status?: string;
  completion_pct?: number;
}

export default function LiveConstructionView() {
  const params = useSearchParams();
  const simId = params.get("sim");

  const [log, setLog] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState(0);
  const [state, setState] = useState<string>("starting");
  const [done, setDone] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const logEndRef = useRef<HTMLDivElement>(null);

  // SSE: orchestration Brain plan/assign/dispatch log.
  useEffect(() => {
    if (!simId) return;
    const es = new EventSource(streamUrl(simId));
    es.onmessage = (e) => {
      let evt: AgentEvent;
      try {
        evt = JSON.parse(e.data);
      } catch {
        return;
      }
      if (evt.type === "heartbeat") return;
      if (evt.type === "done") {
        setLog((prev) => [...prev, { node: "done", text: `Build ${evt.status} — ${evt.completion_pct ?? 0}% complete` }]);
        es.close();
        return;
      }
      if (evt.event) {
        setLog((prev) => [
          ...prev,
          { node: evt.node ?? "agent", text: evt.event!, warning: evt.type === "warning" || Boolean(evt.warning) },
        ]);
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [simId]);

  // Poll live MuJoCo worker status for progress + completion.
  useEffect(() => {
    if (!simId) return;
    let active = true;
    const poll = async () => {
      try {
        const res = await fetch(statusUrl(simId));
        if (!res.ok || !active) return;
        const data = await res.json();
        setProgress(data.progress ?? data.completion_pct ?? 0);
        setState(data.status ?? "running");
        if (["complete", "cancelled", "error"].includes(data.status)) {
          setDone(true);
          active = false;
        }
      } catch {
        /* transient — keep polling */
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [simId]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  return (
    <main className="live-construction-shell">
      <header className="live-construction-header">
        <Link href="/" className="live-construction-logo" aria-label="Atomz home">
          <img src="/references/atomz_logo_transparent.png" alt="" width={28} height={28} />
          <img src="/references/atomz_title.svg" alt="Atomz" width={105} height={20} />
        </Link>
        <Link href="/habitat-selection" className="live-construction-back">
          <span aria-hidden="true">←</span>
          Back
        </Link>
      </header>

      <div className="live-construction-stage">
        <section className="live-construction-frame" aria-label="Live Mars construction feed">
          {simId ? (
            // Multipart JPEG feed straight from the MuJoCo worker.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={frameUrl(simId)}
              alt="Live autonomous construction feed"
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          ) : (
            <video autoPlay loop muted playsInline preload="auto" poster="/reference-images/mars_landing.png">
              <source src="/videos/mars-robot-hero.mp4" type="video/mp4" />
            </video>
          )}

          <div className="live-construction-video-overlay" aria-hidden="true" />

          <div className="live-construction-indicator">
            <span aria-hidden="true" />
            {done ? `${state} · ${progress}%` : `Live feed · ${progress}%`}
          </div>

          <div className="live-construction-glass" aria-hidden="true">
            <div className="live-construction-glass__panel live-construction-glass__panel--left" />
            <div className="live-construction-glass__panel live-construction-glass__panel--right" />
            <div className="live-construction-glass__seam" />
          </div>

          {simId && panelOpen && (
            <aside
              aria-label="Orchestration agent log"
              style={{
                position: "absolute",
                zIndex: 5,
                top: "4%",
                right: "3%",
                width: "min(360px, 42%)",
                maxHeight: "82%",
                overflowY: "auto",
                fontFamily: "var(--font-mono, ui-monospace, monospace)",
                fontSize: "0.7rem",
                lineHeight: 1.55,
                background: "rgba(4, 11, 20, 0.72)",
                backdropFilter: "blur(8px)",
                border: "1px solid rgba(255, 140, 60, 0.28)",
                borderRadius: 10,
                padding: "0.85rem 1rem",
                color: "rgba(255, 220, 195, 0.88)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.6rem" }}>
                <span style={{ textTransform: "uppercase", letterSpacing: "0.16em", color: "rgba(255,170,110,0.75)", fontSize: "0.62rem" }}>
                  ● Orchestrator · plan / assign / dispatch
                </span>
                <button
                  type="button"
                  onClick={() => setPanelOpen(false)}
                  aria-label="Minimize orchestrator log"
                  title="Minimize"
                  style={{
                    cursor: "pointer",
                    background: "transparent",
                    border: "1px solid rgba(255,170,110,0.4)",
                    borderRadius: 5,
                    color: "rgba(255,200,160,0.9)",
                    lineHeight: 1,
                    padding: "0.1rem 0.4rem",
                    fontSize: "0.8rem",
                  }}
                >
                  –
                </button>
              </div>
              {log.length === 0 && <p style={{ opacity: 0.5 }}>Awaiting Brain… spinning up coordinator.</p>}
              {log.map((entry, i) => (
                <p key={i} style={{ color: entry.warning ? "#e8b84b" : undefined, margin: "0.15rem 0" }}>
                  {entry.text}
                </p>
              ))}
              <div ref={logEndRef} />
            </aside>
          )}

          {simId && !panelOpen && (
            <button
              type="button"
              onClick={() => setPanelOpen(true)}
              aria-label="Show orchestrator log"
              title="Show orchestrator log"
              style={{
                position: "absolute",
                zIndex: 5,
                top: "4%",
                right: "3%",
                cursor: "pointer",
                fontFamily: "var(--font-mono, ui-monospace, monospace)",
                fontSize: "0.62rem",
                textTransform: "uppercase",
                letterSpacing: "0.16em",
                background: "rgba(4, 11, 20, 0.72)",
                backdropFilter: "blur(8px)",
                border: "1px solid rgba(255, 140, 60, 0.28)",
                borderRadius: 10,
                padding: "0.5rem 0.8rem",
                color: "rgba(255,170,110,0.9)",
              }}
            >
              ● Orchestrator ▸
            </button>
          )}
        </section>
      </div>
    </main>
  );
}
