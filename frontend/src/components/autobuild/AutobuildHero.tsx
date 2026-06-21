import Link from "next/link";

export default function AutobuildHero() {
  return (
    <section id="technology" className="hero" aria-label="Autonomous Mars construction">
      <video
        className="hero-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="/reference-images/mars_landing.png"
        aria-hidden="true"
      >
        <source src="/videos/mars_landing_dark_smooth.mp4" type="video/mp4" />
      </video>
      {/* Stable fallback for reduced-motion users (see globals.css). */}
      <img
        className="hero-poster-fallback"
        src="/reference-images/mars_landing.png"
        alt="Autonomous construction robot before a glowing doorway on the Martian surface"
      />

      <div className="hero-overlay hero-overlay--left" aria-hidden="true" />
      <div className="hero-overlay hero-overlay--top" aria-hidden="true" />
      <div className="hero-overlay hero-overlay--bottom" aria-hidden="true" />
      <div className="hero-overlay hero-overlay--portal" aria-hidden="true" />

      <div className="hero-content">
        <div className="hero-copy">
          <h1 className="hero-title">
            Autonomous
            <br />
            Habitats.
            <span>Built by machine.</span>
          </h1>

          <div className="mt-10">
            <Link href="/habitat-selection" className="command-button hero-primary-cta">
              Get Started
            </Link>
          </div>
        </div>
      </div>

    </section>
  );
}
