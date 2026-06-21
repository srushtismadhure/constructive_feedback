"use client";

import { useState } from "react";
import AutobuildHero from "@/components/autobuild/AutobuildHero";

const NAV_LINKS: ReadonlyArray<readonly [string, string]> = [
  ["Solutions", "#designs"],
  ["Technology", "#technology"],
  ["Projects", "#projects"],
  ["About Us", "#about"],
];

function MenuIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
    </svg>
  );
}

function AutobuildNav() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header
      className="site-header"
      onKeyDown={(event) => {
        if (event.key === "Escape") setMenuOpen(false);
      }}
    >
      <nav
        className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between px-5 sm:px-8 lg:px-12"
        aria-label="Primary"
      >
        <a href="#top" className="flex items-center gap-2.5" aria-label="Atomz home">
          <img
            src="/references/atomz_logo_transparent.png"
            alt="Atomz logo"
            width={32}
            height={32}
            className="h-7 w-auto object-contain sm:h-8"
          />
          <img
            src="/references/atomz_title.svg"
            alt="Atomz"
            width={131}
            height={24}
            className="h-5 w-auto object-contain sm:h-6"
          />
        </a>

        <div className="hidden items-center gap-8 font-mono text-[10px] uppercase tracking-[0.18em] text-soft-gray md:flex">
          {NAV_LINKS.map(([label, href]) => (
            <a key={href} className="nav-link" href={href}>{label}</a>
          ))}
        </div>

        <button
          type="button"
          className="nav-toggle md:hidden"
          aria-expanded={menuOpen}
          aria-controls="mobile-nav"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span className="sr-only">{menuOpen ? "Close menu" : "Open menu"}</span>
          <MenuIcon open={menuOpen} />
        </button>
      </nav>

      <div id="mobile-nav" className="mobile-nav md:hidden" hidden={!menuOpen}>
        {NAV_LINKS.map(([label, href]) => (
          <a key={href} href={href} onClick={() => setMenuOpen(false)}>{label}</a>
        ))}
      </div>
    </header>
  );
}

export default function AutobuildExperience() {
  return (
    <main id="top" className="autobuild-shell min-h-screen">
      <AutobuildNav />
      <AutobuildHero />
    </main>
  );
}
