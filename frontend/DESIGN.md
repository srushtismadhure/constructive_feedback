# AUTOBUILD Design System

## 1. Brand Concept

**Product name:** AUTOBUILD  
**Product type:** Futuristic web app for autonomous construction robotics  
**Core mood:** Mars terrain, machine-god robotics, industrial precision, off-world habitat construction

AUTOBUILD should feel like a serious mission-control interface for machines building the first off-world civilization. The visual direction should be powerful, cinematic, technical, and restrained.

The interface should not look like a generic AI dashboard. It should feel like premium robotics software built for autonomous construction systems.

---

## 2. Visual Identity

### Design Keywords

- Autonomous construction
- Mars infrastructure
- Machine-god robotics
- Industrial mission control
- Off-world habitats
- Precision engineering
- Regolith printing
- Blueprint selection
- Live robotic build feed

### Visual Feeling

The UI should feel like:

> A dark industrial Mars interface where autonomous construction robots feel less like tools and more like ancient machines building the first off-world civilization.

### Avoid

Do not make the UI look like:
- Generic AI SaaS dashboard
- Neon cyberpunk clutter
- Random hologram interface
- Overly blue futuristic UI
- Cartoon robot app
- Startup pastel landing page
- Fake “AI brain” dashboard
- Over-glowing sci-fi template

---

## 3. Color System

### Primary Brand Colors

```css
--regolith-orange: #FF6B00;
--burnt-mars: #B94300;
--solar-amber: #FFB000;
```

Use **Regolith Orange `#FF6B00`** as the main brand color.

Use orange for:
- Primary buttons
- Active blueprint cards
- Key highlights
- Progress bars
- Robot/build status indicators
- Selected states
- Small glow accents

Do not use orange everywhere. It should feel powerful because it is controlled.

---

### Background Colors

```css
--deep-space: #07090B;
--graphite: #11161B;
--panel: #171C22;
--panel-dark: #0B0D0F;
```

Use these for:
- Main app background
- Section backgrounds
- Dashboard panels
- Blueprint cards
- Preview containers

The background should be dark graphite, not pure black everywhere.

---

### Text Colors

```css
--off-white: #F2F0EA;
--muted-text: #8B918F;
--soft-gray: #C6C7C2;
```

Use:
- `#F2F0EA` for main headings
- `#C6C7C2` for body copy
- `#8B918F` for secondary metadata and technical labels

---

### Supporting Colors

```css
--dust-beige: #C49A6C;
--blueprint-cyan: #7DE3FF;
--danger-red: #FF3B30;
--success-green: #4ADE80;
--warning-yellow: #FACC15;
```

Use sparingly:
- Cyan only for scan/blueprint hints
- Red only for serious warnings
- Green only for safe/confirmed states
- Yellow only for caution states

---

## 4. Typography

### Font Direction

Use a clean sans-serif for most UI and a monospace font for technical values.

Suggested fonts:

```css
font-family: Inter, Arial, sans-serif;
font-mono: "JetBrains Mono", "SFMono-Regular", monospace;
```

### Heading Style

Headings should be:
- Uppercase
- Bold or black weight
- Condensed-feeling if possible
- Tight line height
- Large and cinematic in the hero

Example:

```text
AUTONOMOUS HABITATS.
BUILT BY MACHINE.
```

### Technical Labels

Technical labels should feel like mission-control metadata.

Examples:

```text
SITE: MARS TERRA
STATUS: ACTIVE
LAYER 42 OF 68
SIGNAL LOCK: 97%
NOZZLE TEMP: 184°C
```

Use monospace for these.

---

## 5. UI Layout

The web app should have one main flow:

```text
1. Landing hero
2. Blueprint selection gallery
3. Selected blueprint construction preview
4. Placeholder live robot construction video area
```

---

## 6. Page Structure

### Top Navigation

Include:
- AUTOBUILD logo
- Navigation links:
  - Designs
  - Technology
  - Build Feed
  - Contact

The logo should feel industrial and minimal. Avoid cute robot logos.

Suggested logo treatment:

```text
A AUTOBUILD
Constructing the Future
```

---

### Hero Section

The hero should communicate power and scale.

#### Hero Visual Direction

Use a visual style inspired by:
- Massive robotic construction arm
- Mars terrain
- Orange solar backlight
- Machine-god silhouette
- Dusty atmosphere
- Monumental scale

The robot should feel powerful, calm, and inevitable.

#### Hero Headline

```text
AUTONOMOUS HABITATS.
BUILT BY MACHINE.
```

#### Hero Subtext

```text
Robotic construction systems for extreme terrain, off-world habitats, and autonomous infrastructure.
```

#### Hero Buttons

Primary CTA:

```text
CHOOSE BLUEPRINT
```

Secondary CTA:

```text
WATCH BUILD SEQUENCE
```

---

### Blueprint Selection Section

Section label:

```text
DESIGNS
```

Section title:

```text
SELECT BUILD DESIGN
```

Section subtext:

```text
Choose a habitat blueprint and preview the autonomous construction sequence.
```

Blueprint cards should be placeholders for now. They should be controlled from one editable data array in code.

#### Placeholder Blueprint Names

```text
Dome Habitat
Cylindrical Habitat
Vault Habitat
Command Module
Greenhouse Module
Modular Shelter
```

Each blueprint card should include:
- Blueprint name
- Area
- Crew capacity or purpose
- Build time
- Material type
- Orange wireframe-style placeholder visual

Example metadata:

```text
Dome Habitat
120m²
3–5 Crew
42h Build
Regolith Composite
```

---

### Build Preview Panel

The build preview panel should update when the user selects a blueprint card.

Panel title:

```text
LIVE BUILD SIMULATION
```

Include:
- Selected blueprint name
- Placeholder live video area
- Progress bar
- Current layer
- Robots active
- Material used
- Print stability
- Signal lock
- Nozzle temperature
- Start Project button

Suggested metrics:

```text
ROBOTS ACTIVE: 3
LAYER PROGRESS: 42 / 68
REGOLITH USED: 18.4 tons
PRINT STABILITY: 98%
SIGNAL LOCK: 97%
NOZZLE TEMP: 184°C
```

Button:

```text
START PROJECT
```

---

## 7. Component Rules

### Cards

Cards should use:
- Dark panel background
- Thin borders
- Slight orange border on hover
- Orange border when selected
- Subtle glow only when active
- Rounded corners, but not too soft

Good style:

```css
background: rgba(0, 0, 0, 0.35);
border: 1px solid rgba(255, 107, 0, 0.2);
box-shadow: 0 0 35px rgba(255, 107, 0, 0.12);
```

Avoid giant shadows or cartoon glow.

---

### Buttons

Primary button:
- Regolith Orange background
- Off-white text
- Uppercase
- Slight letter spacing
- Sharp or lightly rounded corners

Secondary button:
- Transparent background
- Thin off-white or orange border
- Hover state changes border to orange

Button labels should feel command-like.

Examples:

```text
CHOOSE BLUEPRINT
WATCH BUILD SEQUENCE
START PROJECT
VIEW BUILD FEED
```

---

### Blueprint Visuals

Blueprint visuals should be:
- CSS or SVG-based placeholders
- Orange wireframe
- Dark background
- Subtle grid texture
- Simple geometric habitat shapes

Do not use random AI-generated blueprint images at first. Placeholders are fine and cleaner.

Blueprint visual effects:
- Thin grid lines
- Wireframe outlines
- Small coordinate ticks
- Layer rings
- Minimal cyan accents only if needed

---

### Live Video Placeholder

Since real video may come later, the placeholder should still feel intentional.

Include:
- Dark video container
- Orange-tinted robotic construction icon or silhouette
- “LIVE BUILD FEED”
- Camera label like `CAM 01`
- Small red live indicator
- Optional scanline overlay

Do not leave it as a blank gray box.

---

## 8. Motion and Interaction

Use restrained motion.

Good interactions:
- Blueprint card hover lift
- Selected card orange glow
- Smooth progress bar animation
- Subtle scanline movement
- Slow hero background glow
- Button hover shift
- Preview panel fade when selection changes

Avoid:
- Too many bouncing elements
- Excessive parallax
- Random animated particles everywhere
- Fast flashy cyberpunk movement

Motion should feel heavy, slow, and industrial.

---

## 9. Texture and Backgrounds

Use:
- Subtle grid lines
- Mars dust gradient
- Radial orange glow
- Dark metallic panels
- Thin dividers
- Small technical labels

Recommended background idea:

```css
background:
  radial-gradient(circle at 70% 35%, rgba(255,107,0,0.28), transparent 35%),
  linear-gradient(180deg, #120704, #07090B 70%);
```

Subtle grid overlay:

```css
background-image:
  linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
  linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
background-size: 64px 64px;
```

---

## 10. Copy System

### Brand Voice

The copy should sound:
- Technical
- Confident
- Cinematic
- Industrial
- Minimal

Do not sound:
- Cute
- Overly startup-y
- Too explanatory
- Too “AI-powered everything”

### Good Copy Examples

```text
Robotic construction systems for extreme terrain.
```

```text
Select a habitat blueprint and preview the autonomous build sequence.
```

```text
Live simulation of autonomous regolith deposition.
```

```text
Machine-calibrated construction for terrain where humans cannot build first.
```

```text
From terrain scan to sealed habitat shell.
```

### Bad Copy Examples

```text
Unlock the power of AI to revolutionize your construction journey!
```

```text
Welcome to the future of amazing smart robot innovation!
```

```text
AI-powered magical construction experience!
```

Too cheesy. Do not use.

---

## 11. Realistic Robotics / Construction Terms

Use realistic terms throughout the interface.

Recommended terms:
- Terrain scan
- Regolith composite
- Extrusion nozzle
- Layer height
- Print stability
- Path deviation
- Material flow
- Signal lock
- Autonomous build sequence
- Foundation ring
- Habitat shell
- Thermal check
- Arm torque
- Dust interference
- Emergency stop
- Manual override

Example UI labels:

```text
PATH DEVIATION
MATERIAL FLOW
ARM TORQUE
FOUNDATION RING
THERMAL CHECK
DUST INTERFERENCE
```

These make the UI feel less fake.

---

## 12. Responsive Design Rules

Desktop:
- Hero: two-column layout
- Blueprint gallery: 3-column grid
- Build preview: sticky or right-side panel

Tablet:
- Hero stacks slightly
- Blueprint gallery: 2-column grid
- Preview below or beside cards

Mobile:
- Single-column
- Navigation simplified
- Blueprint cards stacked
- Build preview below blueprint selection
- Buttons full-width

---

## 13. Implementation Rules for Codex

When building this UI:

1. Build directly inside the existing frontend repo.
2. Do not create a new repo.
3. Do not overwrite backend/API logic.
4. Do not change auth, env files, database files, or existing server routes unless explicitly needed.
5. Keep blueprint data in one editable array.
6. Use reusable components.
7. Use placeholders for blueprint images and build videos.
8. Make the UI responsive.
9. Run build/lint after changes.
10. Fix any TypeScript or styling errors.

---

## 14. Suggested Component Structure

Recommended structure:

```text
src/
  components/
    autobuild/
      AutobuildHero.tsx
      BlueprintCard.tsx
      BlueprintGallery.tsx
      BuildPreviewPanel.tsx
      BlueprintVisual.tsx
      StatusMetric.tsx
  data/
    blueprints.ts
  app/
    page.tsx
```

Alternative route if this should not replace the homepage:

```text
src/
  app/
    autobuild/
      page.tsx
```

---

## 15. Final Design Test

Before accepting the design, ask:

1. Does this feel like autonomous construction robotics?
2. Does it feel premium and industrial?
3. Does the orange feel intentional, not sprayed everywhere?
4. Does the UI still make sense without real blueprint/video assets?
5. Does it avoid generic AI dashboard clichés?
6. Would a user understand the flow: choose blueprint → preview build → start project?

If yes, the design direction is correct.
