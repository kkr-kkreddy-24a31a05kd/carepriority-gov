"""CarePriority Gov - UI Styling & 3D Glassmorphism Design System
==============================================================
"Midnight Glass Command Center" theme featuring:
- Deep navy canvas (#050E1F to #081528) with CSS-only animated drifting aurora orbs (#087F8C, #66E3C4)
- Faint perspective grid floor (CSS 3D transform)
- Subtle noise texture via inline SVG feTurbulence at low opacity
- Soft vignette overlay
- Medical-themed abstract SVG motif (care cross, ECG pulse line, hex grid) at low opacity (offline, no remote images)
- Glassmorphism cards (backdrop-filter: blur(18px) saturate(140%), 1px gradient borders, layered shadows)
- 3D depth effects (CSS perspective 900px, hover tilt 2-4 deg, translateZ lift, diagonal shine sweep)
- Raised 3D buttons with top highlight and pressed state moving down 2px
- Metric numbers with gradient text and soft glow; Level 1 pulsing red ring badge
- Staggered page-load fade/slide-in under 600ms total
- Frosted glass sidebar with polished SVG logo and active-item highlight bar
- Calmer, dimmer version of background and cards for Intake and Queue pages
- @media (prefers-reduced-motion: reduce) compliance (static background, no animation)
- WCAG AA contrast, visible mint focus rings, no horizontal overflow on mobile
- Helper render_html() to prevent Markdown code-block indentation bugs
"""

import textwrap
import streamlit as st
from typing import Optional

# High-fidelity Medical Command Center SVG Brand Logo
SVG_LOGO = """<svg width="46" height="46" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#087F8C" />
      <stop offset="45%" stop-color="#0E4A6F" />
      <stop offset="100%" stop-color="#050E1F" />
    </linearGradient>
    <linearGradient id="mintPulse" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#66E3C4" />
      <stop offset="50%" stop-color="#38BDF8" />
      <stop offset="100%" stop-color="#66E3C4" />
    </linearGradient>
    <filter id="coreGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3.5" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <!-- Outer Shield Shell -->
  <path d="M50 6 L88 22 C88 60 50 94 50 94 C50 94 12 60 12 22 Z" 
        fill="url(#shieldGrad)" stroke="#66E3C4" stroke-width="2.5" />
  <!-- Inner Frosted Glass Inset -->
  <path d="M50 16 L78 28 C78 55 50 81 50 81 C50 81 22 55 22 28 Z" 
        fill="rgba(255, 255, 255, 0.06)" stroke="rgba(102, 227, 196, 0.35)" stroke-width="1.2" />
  <!-- Medical Cross Accent Behind Pulse -->
  <path d="M50 32 V68 M32 50 H68" stroke="rgba(255, 255, 255, 0.12)" stroke-width="6" stroke-linecap="round" />
  <!-- Dynamic Clinical ECG Waveform -->
  <path d="M26 50 L38 50 L42 38 L47 62 L51 28 L56 70 L60 50 L74 50" 
        stroke="url(#mintPulse)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#coreGlow)" />
  <!-- Central Resuscitation Node -->
  <circle cx="51" cy="50" r="3.5" fill="#FFFFFF" />
</svg>"""

APP_CSS = """
<style>
/* --------------------------------------------------------------------------
   GOOGLE FONTS IMPORT: Sora (Headings) + Inter (Body)
   -------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

:root {
  --bg-deep: #050E1F;
  --bg-canvas: #081528;
  --bg-surface: #0A1C35;
  --brand-teal: #087F8C;
  --brand-mint: #66E3C4;
  --brand-cyan: #38BDF8;
  --accent-blue: #0284C7;
  --text-main: #F1F5F9;
  --text-subtle: #CBD5E1;
  --text-muted: #94A3B8;
  
  --tier-l1: #FF2A55;
  --tier-l2: #FF7A00;
  --tier-l3: #FFC700;
  --tier-l4: #00A3FF;
  --tier-l5: #00D68F;
  --tier-abstain: #A78BFA;

  --border-glass: rgba(102, 227, 196, 0.22);
  --border-glass-subtle: rgba(255, 255, 255, 0.08);
  --card-glass: rgba(10, 26, 47, 0.72);
  --shadow-command: 0 14px 38px 0 rgba(0, 0, 0, 0.55), 0 2px 8px 0 rgba(0, 0, 0, 0.25);
}

/* --------------------------------------------------------------------------
   CRITICAL FIX: REMOVE STRAY RED DASH & STREAMLIT DECORATION HEADER
   -------------------------------------------------------------------------- */
[data-testid="stDecoration"],
div[data-testid="stDecoration"],
header[data-testid="stHeader"]::before {
  display: none !important;
  visibility: hidden !important;
  height: 0px !important;
  width: 0px !important;
  opacity: 0 !important;
  background: transparent !important;
  border: none !important;
}

header[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: none !important;
}

/* --------------------------------------------------------------------------
   BASE CANVAS & ANIMATED DRIFTING AURORA ORBS (STAGE 1 - CSS ONLY)
   -------------------------------------------------------------------------- */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-deep) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
  font-size: 14.5px !important;
  line-height: 1.55 !important;
  overflow-x: hidden !important;
  max-width: 100vw !important;
}

/* Drifting Aurora Orbs Layer */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -4;
  pointer-events: none;
  background:
    radial-gradient(circle at 18% 22%, rgba(8, 127, 140, 0.28) 0%, transparent 48%),
    radial-gradient(circle at 82% 78%, rgba(102, 227, 196, 0.20) 0%, transparent 48%),
    radial-gradient(circle at 50% 50%, rgba(8, 21, 40, 0.94) 0%, var(--bg-deep) 100%);
  animation: auroraDrift 28s ease-in-out infinite alternate;
}

/* Faint Perspective Grid Floor (CSS 3D Transform) */
.cpg-grid-floor {
  position: fixed;
  bottom: -15vh;
  left: -20vw;
  width: 140vw;
  height: 55vh;
  background-image: 
    linear-gradient(to top, rgba(8, 127, 140, 0.25) 0%, transparent 85%),
    linear-gradient(0deg, rgba(102, 227, 196, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(102, 227, 196, 0.08) 1px, transparent 1px);
  background-size: 100% 100%, 42px 42px, 42px 42px;
  transform: perspective(600px) rotateX(65deg);
  transform-origin: bottom center;
  pointer-events: none;
  z-index: -3;
  opacity: 0.7;
  mask-image: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 85%);
  -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 85%);
}

@keyframes auroraDrift {
  0% {
    filter: hue-rotate(0deg);
    transform: scale(1) translate(0, 0);
  }
  50% {
    filter: hue-rotate(15deg);
    transform: scale(1.05) translate(-2%, 1.5%);
  }
  100% {
    filter: hue-rotate(-12deg);
    transform: scale(1) translate(2%, -1.5%);
  }
}

/* --------------------------------------------------------------------------
   STAGGERED PAGE-LOAD FADE/SLIDE-IN (<600ms TOTAL)
   -------------------------------------------------------------------------- */
@keyframes pageFadeSlide {
  0% {
    opacity: 0;
    transform: translateY(12px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

.main .block-container {
  padding-top: 1.8rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1480px !important;
}

.main .block-container > div:nth-child(1) {
  animation: pageFadeSlide 280ms cubic-bezier(0.16, 1, 0.3, 1) both;
}
.main .block-container > div:nth-child(2) {
  animation: pageFadeSlide 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
}
.main .block-container > div:nth-child(3) {
  animation: pageFadeSlide 560ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* --------------------------------------------------------------------------
   TYPOGRAPHY
   -------------------------------------------------------------------------- */
h1, h2, h3, h4, .sora-heading {
  font-family: 'Sora', 'Inter', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em !important;
  color: #FFFFFF !important;
}

p, span, div, label {
  letter-spacing: -0.005em;
}

.label-caps {
  font-size: 0.76rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text-muted) !important;
}

/* --------------------------------------------------------------------------
   GLASSMORPHISM CARDS & CSS 3D EFFECTS (STAGE 2)
   -------------------------------------------------------------------------- */
.glass-card {
  perspective: 900px;
  background: var(--card-glass);
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  border: 1px solid var(--border-glass);
  border-radius: 20px;
  padding: 22px 26px;
  box-shadow: var(--shadow-command), inset 0 1px 1px 0 rgba(255, 255, 255, 0.16);
  margin-bottom: 20px;
  overflow: hidden;
  position: relative;
  transition: transform 260ms cubic-bezier(0.16, 1, 0.3, 1), 
              box-shadow 260ms ease, 
              border-color 260ms ease;
  animation: cardFadeIn 450ms ease-out backwards;
}

/* Diagonal Shine Sweep */
.glass-card::before {
  content: "";
  position: absolute;
  top: -60%;
  left: -60%;
  width: 220%;
  height: 220%;
  background: linear-gradient(60deg, transparent 40%, rgba(255, 255, 255, 0.08) 50%, transparent 60%);
  transform: rotate(25deg) translateY(-100%);
  transition: transform 0.75s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
  z-index: 1;
}

/* 3D Perspective Tilt & Lift on Hover */
.glass-card:hover {
  transform: perspective(900px) rotateX(2.5deg) rotateY(-1deg) translateZ(8px);
  box-shadow: 0 22px 48px 0 rgba(0, 0, 0, 0.65), 
              0 0 24px rgba(102, 227, 196, 0.22),
              inset 0 1px 1px 0 rgba(255, 255, 255, 0.3);
  border-color: rgba(102, 227, 196, 0.45);
}

.glass-card:hover::before {
  transform: rotate(25deg) translateY(100%);
}

.glass-card-sm {
  background: rgba(10, 26, 47, 0.60);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--border-glass-subtle);
  border-radius: 14px;
  padding: 14px 18px;
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
}

@keyframes cardFadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* --------------------------------------------------------------------------
   CALMER VERSION FOR INTAKE & QUEUE PAGES (Minimal tilt, no shine sweep)
   -------------------------------------------------------------------------- */
.calm-mode [data-testid="stAppViewContainer"]::before {
  opacity: 0.45 !important;
  animation: none !important;
}

.calm-mode .cpg-grid-floor {
  opacity: 0.28 !important;
}

.calm-mode .cpg-medical-motif {
  opacity: 0.035 !important;
}

.calm-mode .glass-card:hover,
.calm-card:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5) !important;
}

.calm-mode .glass-card::before,
.calm-card::before {
  display: none !important;
}

/* --------------------------------------------------------------------------
   COMMAND CENTER HERO & LIVE STATUS STRIP
   -------------------------------------------------------------------------- */
.hero-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 18px;
  padding: 20px 24px;
  margin-bottom: 22px;
  background: linear-gradient(135deg, rgba(14, 42, 75, 0.7) 0%, rgba(8, 22, 42, 0.8) 100%);
  border: 1px solid rgba(102, 227, 196, 0.25);
  border-radius: 20px;
  box-shadow: var(--shadow-command), inset 0 1px 1px 0 rgba(255, 255, 255, 0.2);
}

.hero-brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.hero-title {
  font-family: 'Sora', sans-serif;
  font-size: 1.7rem;
  font-weight: 800;
  line-height: 1.1;
  color: #FFFFFF;
}

.hero-title-accent {
  background: linear-gradient(135deg, #66E3C4 0%, #38BDF8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-tagline {
  font-size: 0.86rem;
  color: var(--text-muted);
  margin-top: 4px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.live-status-strip {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  background: rgba(10, 26, 47, 0.7);
  border: 1px solid var(--border-glass);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.status-dot-green {
  background: #00D68F;
  box-shadow: 0 0 8px #00D68F;
  animation: dotPulseGreen 2s infinite;
}

.status-dot-orange {
  background: #FF7A00;
  box-shadow: 0 0 8px #FF7A00;
  animation: dotPulseOrange 2s infinite;
}

.status-dot-blue {
  background: #38BDF8;
  box-shadow: 0 0 8px #38BDF8;
}

@keyframes dotPulseGreen {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.85); }
}

@keyframes dotPulseOrange {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.85); }
}

/* --------------------------------------------------------------------------
   METRIC KPI CARDS WITH GRADIENT NUMBERS & SOFT GLOW (STAGE 2)
   -------------------------------------------------------------------------- */
.kpi-container {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 16px 18px;
  min-height: 138px;
  border-radius: 18px;
  background: rgba(10, 26, 47, 0.74);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-glass);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.12);
  transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
  position: relative;
  overflow: hidden;
}

.kpi-container:hover {
  transform: translateY(-4px);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.55), 0 0 18px rgba(102, 227, 196, 0.2);
  border-color: rgba(102, 227, 196, 0.45);
}

.kpi-top-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.kpi-title {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-muted);
}

.kpi-icon-chip {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.82rem;
}

.kpi-val {
  font-family: 'Sora', sans-serif;
  font-size: 2.15rem;
  font-weight: 800;
  line-height: 1.1;
  background: linear-gradient(135deg, #FFFFFF 20%, #66E3C4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 2px 10px rgba(102, 227, 196, 0.3));
  margin: 4px 0;
}

.kpi-sub {
  font-size: 0.74rem;
  font-weight: 500;
  color: var(--brand-mint);
}

.kpi-accent-bar {
  height: 3px;
  width: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, #087F8C 0%, #66E3C4 100%);
  margin-top: 10px;
}

.kpi-accent-bar-orange {
  background: linear-gradient(90deg, #B85000 0%, #FF7A00 100%);
}

.kpi-accent-bar-purple {
  background: linear-gradient(90deg, #5B21B6 0%, #A78BFA 100%);
}

.kpi-accent-bar-cyan {
  background: linear-gradient(90deg, #0284C7 0%, #38BDF8 100%);
}

.kpi-accent-bar-green {
  background: linear-gradient(90deg, #007A52 0%, #00D68F 100%);
}

.kpi-accent-bar-yellow {
  background: linear-gradient(90deg, #B38600 0%, #FFC700 100%);
}

/* --------------------------------------------------------------------------
   TRIAGE PRIORITY TIER BADGES (LEVEL 1 PULSING RED RING)
   -------------------------------------------------------------------------- */
@keyframes pulseRedRing {
  0% {
    box-shadow: 0 0 0 0 rgba(255, 42, 85, 0.8), 0 0 16px rgba(255, 42, 85, 0.5);
    border-color: #FF2A55;
  }
  70% {
    box-shadow: 0 0 0 12px rgba(255, 42, 85, 0), 0 0 20px rgba(255, 42, 85, 0.1);
    border-color: #FF8099;
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 42, 85, 0), 0 0 16px rgba(255, 42, 85, 0.5);
    border-color: #FF2A55;
  }
}

.tier-badge-l1 {
  background: linear-gradient(135deg, #FF2A55 0%, #A00025 100%);
  color: #FFFFFF !important;
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  animation: pulseRedRing 2.2s infinite;
  border: 1px solid #FF8099;
  letter-spacing: 0.02em;
}

.tier-badge-l2 {
  background: linear-gradient(135deg, #FF7A00 0%, #B85000 100%);
  color: #FFFFFF !important;
  font-family: 'Sora', sans-serif;
  font-weight: 700;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(255, 122, 0, 0.6);
  box-shadow: 0 0 12px rgba(255, 122, 0, 0.3);
  letter-spacing: 0.02em;
}

.tier-badge-l3 {
  background: linear-gradient(135deg, #FFC700 0%, #B38600 100%);
  color: #081528 !important; /* WCAG AA High Contrast */
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(255, 199, 0, 0.7);
  box-shadow: 0 0 12px rgba(255, 199, 0, 0.25);
  letter-spacing: 0.02em;
}

.tier-badge-l4 {
  background: linear-gradient(135deg, #00A3FF 0%, #005F99 100%);
  color: #FFFFFF !important;
  font-family: 'Sora', sans-serif;
  font-weight: 700;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(0, 163, 255, 0.55);
  box-shadow: 0 0 12px rgba(0, 163, 255, 0.25);
  letter-spacing: 0.02em;
}

.tier-badge-l5 {
  background: linear-gradient(135deg, #00D68F 0%, #007A52 100%);
  color: #050E1F !important; /* High Contrast */
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(0, 214, 143, 0.6);
  box-shadow: 0 0 12px rgba(0, 214, 143, 0.25);
  letter-spacing: 0.02em;
}

.tier-badge-abstain {
  background: linear-gradient(135deg, #8B5CF6 0%, #4C1D95 100%);
  color: #FFFFFF !important;
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 0.82rem;
  padding: 5px 13px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(167, 139, 250, 0.6);
  box-shadow: 0 0 14px rgba(139, 92, 246, 0.35);
  letter-spacing: 0.02em;
}

/* --------------------------------------------------------------------------
   GLASS PATIENT CARD FOR QUEUE
   -------------------------------------------------------------------------- */
.patient-card {
  background: rgba(10, 26, 47, 0.72);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-glass);
  border-radius: 18px;
  padding: 18px 22px;
  margin-bottom: 16px;
  box-shadow: var(--shadow-command);
  transition: transform 200ms ease, box-shadow 200ms ease;
  position: relative;
}

.patient-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.6), 0 0 18px rgba(102, 227, 196, 0.15);
}

.patient-card-l1 { border-left: 6px solid var(--tier-l1) !important; }
.patient-card-l2 { border-left: 6px solid var(--tier-l2) !important; }
.patient-card-l3 { border-left: 6px solid var(--tier-l3) !important; }
.patient-card-l4 { border-left: 6px solid var(--tier-l4) !important; }
.patient-card-l5 { border-left: 6px solid var(--tier-l5) !important; }
.patient-card-abstain { border-left: 6px solid var(--tier-abstain) !important; }

.vitals-tile-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  background: rgba(4, 14, 28, 0.5);
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.07);
}

.vitals-tile {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.82rem;
  color: var(--text-subtle);
}

.vitals-tile strong {
  color: #FFFFFF;
}

.red-flag-chip {
  background: rgba(255, 42, 85, 0.16);
  border: 1px solid rgba(255, 42, 85, 0.45);
  color: #FF8099;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin: 2px 4px 2px 0;
  box-shadow: 0 0 8px rgba(255, 42, 85, 0.2);
}

/* Re-Triage Due Pulsing Alert */
@keyframes retriagePulse {
  0% {
    box-shadow: 0 0 0 0 rgba(255, 122, 0, 0.75);
    border-color: #FF7A00;
  }
  70% {
    box-shadow: 0 0 0 8px rgba(255, 122, 0, 0);
    border-color: #FFA336;
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 122, 0, 0);
    border-color: #FF7A00;
  }
}

.retriage-badge {
  background: linear-gradient(135deg, rgba(255, 122, 0, 0.25) 0%, rgba(255, 42, 85, 0.25) 100%);
  border: 1.5px solid #FF7A00;
  color: #FFA336 !important;
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 0.76rem;
  padding: 4px 10px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  animation: retriagePulse 1.8s infinite;
  letter-spacing: 0.04em;
}

/* --------------------------------------------------------------------------
   BANNERS (SURGE, PLACEHOLDER, ESCALATION, SAFETY)
   -------------------------------------------------------------------------- */
@keyframes surgeGlowPulse {
  0% { box-shadow: 0 8px 24px rgba(255, 122, 0, 0.35); }
  100% { box-shadow: 0 12px 36px rgba(255, 42, 85, 0.55); }
}

.placeholder-banner {
  background: linear-gradient(135deg, rgba(255, 122, 0, 0.15) 0%, rgba(13, 34, 63, 0.8) 100%);
  border: 1.5px solid #FF7A00;
  border-left: 7px solid #FF7A00;
  border-radius: 16px;
  padding: 16px 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-command);
}

.safety-strip-banner {
  background: linear-gradient(90deg, rgba(8, 127, 140, 0.35) 0%, rgba(10, 26, 47, 0.85) 100%);
  border-left: 5px solid #66E3C4;
  border-top: 1px solid rgba(102, 227, 196, 0.3);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 14px 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-command);
}

.escalation-banner {
  background: linear-gradient(90deg, rgba(255, 42, 85, 0.22) 0%, rgba(10, 26, 47, 0.85) 100%);
  border-left: 5px solid #FF2A55;
  border-top: 1px solid rgba(255, 42, 85, 0.35);
  border-right: 1px solid rgba(255, 42, 85, 0.2);
  border-bottom: 1px solid rgba(255, 42, 85, 0.2);
  border-radius: 16px;
  padding: 14px 22px;
  margin: 14px 0 20px 0;
  color: #FFE6EB;
  box-shadow: 0 6px 20px rgba(255, 42, 85, 0.15);
}

.abstention-banner {
  background: linear-gradient(90deg, rgba(139, 92, 246, 0.22) 0%, rgba(10, 26, 47, 0.85) 100%);
  border-left: 5px solid #A78BFA;
  border-top: 1px solid rgba(139, 92, 246, 0.35);
  border-right: 1px solid rgba(139, 92, 246, 0.2);
  border-bottom: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 16px;
  padding: 14px 22px;
  margin: 14px 0 20px 0;
  color: #F5EEFF;
  box-shadow: 0 6px 20px rgba(139, 92, 246, 0.15);
}

/* --------------------------------------------------------------------------
   SIDEBAR STYLING (GLASS PANEL & ACTIVE-ITEM HIGHLIGHT)
   -------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: rgba(6, 17, 34, 0.92) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
  border-right: 1px solid rgba(102, 227, 196, 0.18) !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.55) !important;
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem !important;
  padding-left: 1.2rem !important;
  padding-right: 1.2rem !important;
}

[data-testid="stSidebar"] .stRadio > div {
  gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio label {
  background: rgba(255, 255, 255, 0.03) !important;
  border: 1px solid rgba(255, 255, 255, 0.06) !important;
  border-radius: 10px !important;
  padding: 9px 14px !important;
  transition: all 180ms ease !important;
  cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(8, 127, 140, 0.25) !important;
  border-color: rgba(102, 227, 196, 0.3) !important;
  transform: translateX(3px) !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked),
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
  background: linear-gradient(90deg, rgba(8, 127, 140, 0.4) 0%, rgba(10, 26, 47, 0.7) 100%) !important;
  border-left: 4px solid #66E3C4 !important;
  border-color: rgba(102, 227, 196, 0.4) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 14px rgba(8, 127, 140, 0.25) !important;
}

/* --------------------------------------------------------------------------
   STREAMLIT WIDGET OVERRIDES: 3D RAISED BUTTONS & DARK GLASS INPUTS
   -------------------------------------------------------------------------- */
/* Buttons: 3D Raised Look with Top Highlight and Pressed State */
.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
  background: linear-gradient(180deg, #098A98 0%, #065B66 100%) !important;
  color: #FFFFFF !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  border: 1px solid rgba(102, 227, 196, 0.45) !important;
  border-radius: 12px !important;
  padding: 8px 20px !important;
  box-shadow: 0 4px 14px rgba(8, 127, 140, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
  transition: all 180ms ease !important;
  letter-spacing: 0.01em !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  background: linear-gradient(180deg, #0B9EAE 0%, #087380 100%) !important;
  box-shadow: 0 8px 20px rgba(102, 227, 196, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
  border-color: var(--brand-mint) !important;
}

.stButton > button:active {
  transform: translateY(2px) !important; /* Pressed state moves down 2px */
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5), inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}

/* Form Container */
[data-testid="stForm"] {
  background: rgba(10, 26, 47, 0.68) !important;
  backdrop-filter: blur(18px) !important;
  -webkit-backdrop-filter: blur(18px) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: 20px !important;
  padding: 26px !important;
  box-shadow: var(--shadow-command), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
}

/* Text Inputs & Textareas */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
  background: rgba(6, 18, 34, 0.78) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 10px !important;
  color: #FFFFFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
  padding: 9px 12px !important;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
  transition: border-color 180ms ease, box-shadow 180ms ease !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--brand-mint) !important;
  outline: 2px solid rgba(102, 227, 196, 0.4) !important;
  box-shadow: 0 0 12px rgba(102, 227, 196, 0.25) !important;
}

/* Selectboxes & Dropdowns */
[data-testid="stSelectbox"] > div > div {
  background: rgba(6, 18, 34, 0.78) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 10px !important;
  color: #FFFFFF !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px !important;
  background: rgba(6, 18, 34, 0.5) !important;
  padding: 6px !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 10px !important;
  color: var(--text-muted) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.86rem !important;
  padding: 8px 16px !important;
  border: none !important;
  transition: all 180ms ease !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  background: rgba(8, 127, 140, 0.4) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 12px rgba(8, 127, 140, 0.3) !important;
  border: 1px solid rgba(102, 227, 196, 0.35) !important;
}

/* Expanders */
[data-testid="stExpander"] {
  background: rgba(10, 26, 47, 0.6) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: 14px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
  margin-bottom: 12px !important;
}

/* Tables, DataFrames & Charts */
div[data-testid="stAltairChart"],
[data-testid="stTable"],
[data-testid="stDataFrame"] {
  background: var(--card-glass) !important;
  backdrop-filter: blur(18px) saturate(140%) !important;
  -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: 20px !important;
  padding: 18px 22px !important;
  box-shadow: var(--shadow-command) !important;
  margin-bottom: 18px !important;
}

table {
  width: 100% !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  border: 1px solid var(--border-glass-subtle) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

th {
  background: rgba(8, 26, 48, 0.85) !important;
  color: #66E3C4 !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  padding: 12px 16px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
}

td {
  background: rgba(6, 18, 34, 0.4) !important;
  color: #F1F5F9 !important;
  padding: 10px 16px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
  font-size: 0.88rem !important;
}

tr:hover td {
  background: rgba(8, 127, 140, 0.12) !important;
}

/* --------------------------------------------------------------------------
   FOOTER
   -------------------------------------------------------------------------- */
.app-footer {
  text-align: center;
  padding: 28px 16px 12px 16px;
  margin-top: 48px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* --------------------------------------------------------------------------
   ACCESSIBILITY & PREFERS-REDUCED-MOTION
   -------------------------------------------------------------------------- */
:focus-visible {
  outline: 2px solid var(--brand-mint) !important;
  outline-offset: 2px !important;
}

@media (prefers-reduced-motion: reduce) {
  *, ::before, ::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
    transform: none !important;
  }
}

@media (max-width: 768px) {
  .hero-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .kpi-container {
    min-height: auto;
  }
}
</style>
"""

BACKGROUND_LAYER_HTML = """
<div class="cpg-fixed-background" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: -2; overflow: hidden;">
  <!-- Perspective Grid Floor (CSS 3D Transform) -->
  <div class="cpg-grid-floor"></div>

  <!-- Medical Abstract Motif SVG (Care Cross, ECG Pulse, Hex Grid) -->
  <svg class="cpg-medical-motif" viewBox="0 0 1000 600" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.06; pointer-events: none;">
    <defs>
      <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#087F8C" stop-opacity="0" />
        <stop offset="25%" stop-color="#087F8C" stop-opacity="0.8" />
        <stop offset="50%" stop-color="#66E3C4" stop-opacity="1" />
        <stop offset="75%" stop-color="#38BDF8" stop-opacity="0.8" />
        <stop offset="100%" stop-color="#087F8C" stop-opacity="0" />
      </linearGradient>
    </defs>
    <!-- Hex Grid Network (Top Right & Left) -->
    <path d="M80 60 L110 77 L110 112 L80 129 L50 112 L50 77 Z" stroke="#66E3C4" stroke-width="1.2" stroke-dasharray="3 3" />
    <path d="M140 60 L170 77 L170 112 L140 129 L110 112 L110 77 Z" stroke="#087F8C" stroke-width="1.2" />
    <path d="M110 112 L140 129 L140 164 L110 181 L80 164 L80 129 Z" stroke="#66E3C4" stroke-width="1.2" />
    <path d="M860 80 L890 97 L890 132 L860 149 L830 132 L830 97 Z" stroke="#66E3C4" stroke-width="1.2" stroke-dasharray="4 2" />
    <path d="M920 80 L950 97 L950 132 L920 149 L890 132 L890 97 Z" stroke="#087F8C" stroke-width="1.2" />
    <path d="M890 132 L920 149 L920 184 L890 201 L860 184 L860 149 Z" stroke="#66E3C4" stroke-width="1.2" />

    <!-- Medical Care Cross Abstract Nodes -->
    <path d="M910 380 V450 M875 415 H945" stroke="#66E3C4" stroke-width="6" stroke-linecap="round" />
    <circle cx="910" cy="415" r="30" stroke="#087F8C" stroke-width="1.5" stroke-dasharray="5 5" />
    <path d="M90 390 V450 M60 420 H120" stroke="#087F8C" stroke-width="5" stroke-linecap="round" />

    <!-- Clinical ECG Pulse Waveform Running Horizontally -->
    <path d="M0 320 L280 320 L295 285 L310 365 L325 240 L342 395 L358 320 L410 320 L425 295 L435 345 L450 320 L1000 320" stroke="url(#ecgGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
  </svg>

  <!-- Inline SVG feTurbulence Noise Texture -->
  <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.035; pointer-events: none;">
    <filter id="noiseFilter">
      <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch" />
    </filter>
    <rect width="100%" height="100%" filter="url(#noiseFilter)" />
  </svg>

  <!-- Soft Vignette Gradient -->
  <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(ellipse at 50% 50%, transparent 45%, rgba(5, 14, 31, 0.78) 100%); pointer-events: none;"></div>
</div>
"""


def clean_html(markup: str) -> str:
    """Safely cleans HTML markup:
    Dedents each line, removes blank lines, and strips leading and trailing whitespace.
    Guarantees that no line has leading whitespace (preventing CommonMark from parsing 4+ space
    indents as code blocks) and eliminates blank lines between tags.
    """
    if not markup:
        return ""
    lines = [line.strip() for line in markup.splitlines()]
    return "\n".join([l for l in lines if l])


def render_html(markup: str):
    """Safely renders HTML markup without markdown converting indented lines into code blocks.
    Dedents each line, removes blank lines, strips whitespace, then renders via st.markdown(..., unsafe_allow_html=True).
    """
    cleaned = clean_html(markup)
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)


def apply_theme(calm: bool = False):
    """Injects custom CSS styles, 3D depth, and the offline SVG background layer.
    If calm is True (e.g. on Intake and Queue pages), a calmer, dimmer version with
    minimal tilt and no shine sweep is applied.
    """
    render_html(APP_CSS)
    render_html(BACKGROUND_LAYER_HTML)
    if calm:
        render_html("""
        <style>
          [data-testid="stAppViewContainer"]::before {
            opacity: 0.42 !important;
            animation: none !important;
          }
          .cpg-grid-floor {
            opacity: 0.28 !important;
          }
          .cpg-medical-motif {
            opacity: 0.03 !important;
          }
          .glass-card:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5) !important;
          }
          .glass-card::before {
            display: none !important;
          }
        </style>
        """)


def render_brand_header(page_title: str, subtitle: Optional[str] = None):
    """Renders the top branding header with SVG logo and title without indentation bugs."""
    sub_html = f"<div style='color: #94A3B8; font-size: 0.88rem; margin-top: 3px;'>{subtitle}</div>" if subtitle else ""
    header_html = f"""<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px; padding-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.08);">
  <div>{SVG_LOGO}</div>
  <div>
    <div style="font-family: 'Sora', sans-serif; font-size: 1.65rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">
      CarePriority <span style="background: linear-gradient(135deg, #66E3C4, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Gov</span>
    </div>
    <div style="font-size: 0.82rem; font-weight: 700; color: #66E3C4; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
      {page_title}
    </div>
    {sub_html}
  </div>
</div>"""
    render_html(header_html)


def render_tier_badge(tier: Optional[int], is_abstained: bool = False) -> str:
    """Returns accessible HTML for a tier badge (color is always paired with icon and clear text)."""
    if is_abstained or tier is None:
        badge = '<span class="tier-badge-abstain">⚠️ Needs Nurse Review</span>'
    elif tier == 1:
        badge = '<span class="tier-badge-l1">🚨 L1: Resuscitation</span>'
    elif tier == 2:
        badge = '<span class="tier-badge-l2">⚡ L2: Emergent</span>'
    elif tier == 3:
        badge = '<span class="tier-badge-l3">⏳ L3: Urgent</span>'
    elif tier == 4:
        badge = '<span class="tier-badge-l4">🔷 L4: Less Urgent</span>'
    elif tier == 5:
        badge = '<span class="tier-badge-l5">🟢 L5: Non-Urgent</span>'
    else:
        badge = '<span class="tier-badge-l4">🔷 Unclassified</span>'
    return clean_html(badge)


def render_footer():
    """Renders the mandatory clinical disclaimer footer."""
    footer_html = """<div class="app-footer">
  <div><strong>Demo only — not validated for clinical use.</strong> Independent demonstration project, not an official government service.</div>
  <div style="margin-top: 4px; font-size: 0.74rem; opacity: 0.75;">CarePriority Gov &bull; OPD Severity Triage & Guardrail System &bull; Designed for Overloaded Healthcare Facilities</div>
</div>"""
    render_html(footer_html)
