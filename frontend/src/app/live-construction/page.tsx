import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Live Construction Feed — Atomz",
  description: "Live autonomous habitat construction on the Martian surface.",
};

export default function LiveConstructionPage() {
  return (
    <main className="live-construction-shell">
      <header className="live-construction-header">
        <Link href="/" className="live-construction-logo" aria-label="Atomz home">
          <img
            src="/references/atomz_logo_transparent.png"
            alt=""
            width={28}
            height={28}
          />
          <img
            src="/references/atomz_title.svg"
            alt="Atomz"
            width={105}
            height={20}
          />
        </Link>

        <Link href="/habitat-selection" className="live-construction-back">
          <span aria-hidden="true">←</span>
          Back
        </Link>
      </header>

      <div className="live-construction-stage">
        <section className="live-construction-frame" aria-label="Live Mars construction video">
          <video
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            poster="/reference-images/mars_landing.png"
            aria-label="Autonomous construction robot operating on the Martian surface"
          >
            <source src="/videos/mars-robot-hero.mp4" type="video/mp4" />
          </video>

          <div className="live-construction-video-overlay" aria-hidden="true" />

          <div className="live-construction-indicator">
            <span aria-hidden="true" />
            Live construction feed
          </div>

          <div className="live-construction-glass" aria-hidden="true">
            <div className="live-construction-glass__panel live-construction-glass__panel--left" />
            <div className="live-construction-glass__panel live-construction-glass__panel--right" />
            <div className="live-construction-glass__seam" />
          </div>
        </section>
      </div>
    </main>
  );
}
