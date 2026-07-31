# -*- coding: utf-8 -*-
# Apple-style theme for the UnQTube2 Streamlit UI (v2).
#
# Design goals, in priority order:
#   1. Fast. Blur is limited to three surfaces, no web fonts, no rerun animations.
#   2. Stable. Streamlit re-runs the whole script on every click, so nothing here
#      may animate on mount or the page appears to jump on each interaction.
#   3. Responsive. Fluid type and spacing, real breakpoints, 44px touch targets.
#   4. Apple. System font stack, system colours, hairlines, calm easing.
#
# Injected by webui/App.py. No application logic lives here.

APPLE_CSS = r'''
<style>
/* =========================================================
   1. TOKENS
   ========================================================= */
:root {
  /* Native Apple UI font on Apple devices, best native match elsewhere.
     No @import: a web font would block first paint over the tunnel. */
  --font: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display',
          'Segoe UI Variable Text', 'Segoe UI', Roboto, 'Helvetica Neue',
          Arial, sans-serif;
  --font-mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace;

  --bg: #07080c;
  --surface-1: rgba(255,255,255,0.045);
  --surface-2: rgba(255,255,255,0.075);
  --surface-3: rgba(255,255,255,0.11);
  --line: rgba(255,255,255,0.10);
  --line-2: rgba(255,255,255,0.18);

  --text: rgba(255,255,255,0.95);
  --text-2: rgba(235,235,245,0.62);
  --text-3: rgba(235,235,245,0.36);

  --blue: #0a84ff;
  --blue-2: #409cff;
  --indigo: #5e5ce6;
  --green: #30d158;
  --orange: #ff9f0a;
  --red: #ff453a;

  --r-xs: 8px;
  --r-sm: 10px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 22px;
  --r-pill: 980px;

  --shadow: 0 6px 24px rgba(0,0,0,0.32);
  --shadow-lg: 0 18px 48px rgba(0,0,0,0.44);

  --ease: cubic-bezier(0.32, 0.72, 0, 1);
  --fast: 0.15s;

  /* Fluid page gutter and rhythm */
  --gutter: clamp(0.9rem, 3.2vw, 3rem);
  --stack: clamp(0.85rem, 1.6vw, 1.25rem);
  --blur: saturate(170%) blur(20px);
}

/* =========================================================
   2. BASE
   ========================================================= */
html, body, .stApp, [data-testid='stAppViewContainer'], button, input, textarea, select {
  font-family: var(--font) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

html, body { overflow-x: hidden; }

/* One painted layer instead of a fixed pseudo-element: cheaper on scroll. */
.stApp {
  color: var(--text) !important;
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(52rem 34rem at 8% -10%, rgba(10,132,255,0.16), transparent 60%),
    radial-gradient(42rem 30rem at 104% 0%, rgba(94,92,230,0.14), transparent 60%),
    radial-gradient(40rem 30rem at 50% 112%, rgba(191,90,242,0.10), transparent 60%);
  background-attachment: fixed;
  background-repeat: no-repeat;
}

.stApp::before { content: none !important; }

/* =========================================================
   3. CHROME AND PAGE CONTAINER
   ========================================================= */
[data-testid='stHeader'] {
  background: rgba(7,8,12,0.72) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  height: 3rem !important;
}

[data-testid='stDecoration'],
[data-testid='stAppDeployButton'] { display: none !important; }

[data-testid='stToolbar'] { right: 0.6rem !important; }

.main .block-container,
[data-testid='stAppViewBlockContainer'] {
  max-width: 1160px !important;
  padding-left: var(--gutter) !important;
  padding-right: var(--gutter) !important;
  padding-top: clamp(3.4rem, 7vh, 5rem) !important;
  padding-bottom: clamp(3rem, 10vh, 6rem) !important;
  padding-bottom: calc(clamp(3rem, 10vh, 6rem) + env(safe-area-inset-bottom)) !important;
}

/* Vertical rhythm between top level blocks */
[data-testid='stAppViewBlockContainer'] [data-testid='stVerticalBlock'] {
  gap: var(--stack) !important;
}

/* =========================================================
   4. TYPOGRAPHY AND HERO
   ========================================================= */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font) !important;
  color: var(--text) !important;
  -webkit-text-fill-color: currentColor !important;
  background: none !important;
  animation: none !important;
  text-shadow: none !important;
  letter-spacing: -0.022em !important;
  line-height: 1.15 !important;
}

h1 { font-size: clamp(1.75rem, 3.4vw, 2.6rem) !important; font-weight: 700 !important; letter-spacing: -0.034em !important; }
h2 { font-size: clamp(1.25rem, 2.2vw, 1.5rem) !important; font-weight: 640 !important; }
h3 { font-size: clamp(1.05rem, 1.8vw, 1.18rem) !important; font-weight: 600 !important; }

p, span, label, li, .stMarkdown, [data-testid='stMarkdownContainer'] p {
  font-family: var(--font) !important;
  color: var(--text-2) !important;
  font-size: clamp(0.9rem, 1.1vw, 0.97rem) !important;
  letter-spacing: -0.01em !important;
  line-height: 1.55 !important;
}

a { color: var(--blue-2) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }
code, a, [data-testid='stMarkdownContainer'] p { overflow-wrap: break-word; }

[data-testid='stWidgetLabel'] label p,
[data-testid='stWidgetLabel'] label {
  color: var(--text) !important;
  font-weight: 560 !important;
  font-size: 0.92rem !important;
  margin-bottom: 0.3rem !important;
}

/* Hero rendered by App.py */
.uq-hero {
  margin: 0 0 clamp(1.5rem, 4vw, 2.75rem) !important;
  padding: clamp(1.4rem, 3.5vw, 2.4rem) clamp(1.2rem, 3vw, 2.2rem) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-xl) !important;
  background:
    linear-gradient(140deg, rgba(10,132,255,0.16) 0%, rgba(94,92,230,0.10) 45%, rgba(255,255,255,0.03) 100%) !important;
  box-shadow: var(--shadow) !important;
}

.uq-hero-badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 620;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--blue-2);
  background: rgba(10,132,255,0.14);
  border: 1px solid rgba(10,132,255,0.28);
  border-radius: var(--r-pill);
  padding: 0.26rem 0.7rem;
  margin-bottom: 0.85rem;
}

.uq-hero-title {
  margin: 0 0 0.45rem !important;
  font-size: clamp(1.9rem, 5.2vw, 3rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.04em !important;
  line-height: 1.05 !important;
  color: #fff !important;
}

.uq-hero-sub {
  margin: 0 !important;
  max-width: 46ch;
  font-size: clamp(0.95rem, 1.5vw, 1.08rem) !important;
  line-height: 1.5 !important;
  color: var(--text-2) !important;
}

/* =========================================================
   5. LAYOUT: COLUMNS THAT ACTUALLY REFLOW
   ========================================================= */
[data-testid='stHorizontalBlock'] {
  flex-wrap: wrap !important;
  gap: var(--stack) !important;
  align-items: flex-start !important;
}

[data-testid='stHorizontalBlock'] > [data-testid='stColumn'],
[data-testid='stHorizontalBlock'] > [data-testid='column'] {
  min-width: 0 !important;
}

[data-testid='stColumn'] > div,
[data-testid='column'] > div { min-width: 0 !important; }

hr, [data-testid='stDivider'] {
  border-color: var(--line) !important;
  opacity: 1 !important;
  margin: 0.4rem 0 !important;
}

/* =========================================================
   6. BUTTONS
   ========================================================= */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
[data-testid='baseButton-secondary'],
[data-testid='baseButton-primary'] {
  font-family: var(--font) !important;
  font-size: 0.94rem !important;
  font-weight: 590 !important;
  letter-spacing: -0.01em !important;
  text-transform: none !important;
  color: var(--text) !important;
  background: var(--surface-2) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: var(--r-pill) !important;
  padding: 0.6rem 1.3rem !important;
  min-height: 40px !important;
  box-shadow: none !important;
  animation: none !important;
  transition: background var(--fast) var(--ease),
              border-color var(--fast) var(--ease),
              transform var(--fast) var(--ease) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  background: var(--surface-3) !important;
  border-color: rgba(255,255,255,0.28) !important;
  color: #fff !important;
  transform: none !important;
}

.stButton > button:active,
.stDownloadButton > button:active,
.stFormSubmitButton > button:active { transform: scale(0.98) !important; }

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(10,132,255,0.32) !important;
}

.stButton > button[kind='primary'],
.stFormSubmitButton > button[kind='primary'] {
  background: var(--blue) !important;
  border-color: transparent !important;
  color: #fff !important;
  font-weight: 620 !important;
}

.stButton > button[kind='primary']:hover,
.stFormSubmitButton > button[kind='primary']:hover {
  background: var(--blue-2) !important;
  border-color: transparent !important;
}

.stDownloadButton > button {
  background: rgba(48,209,88,0.16) !important;
  border-color: rgba(48,209,88,0.42) !important;
  color: #7ff0a4 !important;
}

.stDownloadButton > button:hover {
  background: rgba(48,209,88,0.24) !important;
  color: #b6ffd0 !important;
}

/* The main call to action from Main.py */
.generate-btn button {
  width: 100% !important;
  font-size: 1.02rem !important;
  min-height: 52px !important;
  animation: none !important;
}

/* =========================================================
   7. INPUTS - style the outer shell only, no doubled borders
   ========================================================= */
.stTextInput [data-baseweb='input'],
.stNumberInput [data-baseweb='input'],
.stTextArea [data-baseweb='textarea'],
.stDateInput [data-baseweb='input'],
.stSelectbox [data-baseweb='select'] > div:first-child,
.stMultiSelect [data-baseweb='select'] > div:first-child {
  background: var(--surface-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  min-height: 44px !important;
  box-shadow: none !important;
  transition: border-color var(--fast) var(--ease),
              background var(--fast) var(--ease),
              box-shadow var(--fast) var(--ease) !important;
}

.stTextInput [data-baseweb='input']:hover,
.stNumberInput [data-baseweb='input']:hover,
.stTextArea [data-baseweb='textarea']:hover,
.stSelectbox [data-baseweb='select'] > div:first-child:hover {
  background: var(--surface-2) !important;
  border-color: var(--line-2) !important;
}

.stTextInput [data-baseweb='input']:focus-within,
.stNumberInput [data-baseweb='input']:focus-within,
.stTextArea [data-baseweb='textarea']:focus-within,
.stSelectbox [data-baseweb='select'] > div:first-child:focus-within {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3.5px rgba(10,132,255,0.22) !important;
}

/* Inner elements stay transparent so only one box is visible */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input,
[data-baseweb='select'] div,
[data-baseweb='input'] > div,
[data-baseweb='input'] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--text) !important;
  font-size: 0.95rem !important;
}

.stTextInput input, .stNumberInput input { padding: 0.55rem 0.85rem !important; }
.stTextArea textarea { padding: 0.7rem 0.85rem !important; line-height: 1.55 !important; }

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: var(--text-3) !important;
  font-style: normal !important;
}

/* Number input steppers */
.stNumberInput button {
  background: transparent !important;
  border: none !important;
  border-left: 1px solid var(--line) !important;
  border-radius: 0 !important;
  color: var(--text-2) !important;
  min-height: 0 !important;
}

.stNumberInput button:hover { background: var(--surface-2) !important; color: #fff !important; }

/* Dropdown menu: one of the three surfaces allowed to blur */
div[data-baseweb='popover'] > div,
div[data-baseweb='popover'] ul {
  background: rgba(26,26,30,0.86) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  box-shadow: var(--shadow-lg) !important;
}

div[data-baseweb='popover'] li {
  color: var(--text-2) !important;
  border-radius: var(--r-xs) !important;
  margin: 2px 6px !important;
  min-height: 40px !important;
  display: flex !important;
  align-items: center !important;
}

div[data-baseweb='popover'] li:hover { background: var(--surface-2) !important; color: #fff !important; }

/* Multiselect chips */
[data-baseweb='tag'] {
  background: rgba(10,132,255,0.18) !important;
  border: 1px solid rgba(10,132,255,0.3) !important;
  border-radius: var(--r-pill) !important;
  color: #cfe6ff !important;
}

/* =========================================================
   8. SIDEBAR
   ========================================================= */
[data-testid='stSidebar'] {
  background: rgba(13,14,19,0.82) !important;
  border-right: 1px solid var(--line) !important;
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  animation: none !important;
}

[data-testid='stSidebar'] > div:first-child {
  background: transparent !important;
  padding: 1.2rem 1rem 2.5rem !important;
}

[data-testid='stSidebar'] h1,
[data-testid='stSidebar'] h2,
[data-testid='stSidebar'] h3 {
  font-size: 0.72rem !important;
  font-weight: 640 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  margin: 1.5rem 0 0.55rem !important;
  padding: 0 0 0.45rem !important;
  border-bottom: 1px solid var(--line) !important;
}

[data-testid='stSidebar'] [data-testid='stVerticalBlock'] { gap: 0.7rem !important; }

[data-testid='stSidebarCollapseButton'] button,
[data-testid='collapsedControl'] button {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
}

/* =========================================================
   9. TABS - segmented control, scrolls instead of overflowing
   ========================================================= */
.stTabs [data-baseweb='tab-list'] {
  background: var(--surface-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-lg) !important;
  padding: 4px !important;
  gap: 2px !important;
  flex-wrap: nowrap !important;
  overflow-x: auto !important;
  scrollbar-width: none !important;
  -webkit-overflow-scrolling: touch;
}

.stTabs [data-baseweb='tab-list']::-webkit-scrollbar { display: none !important; }

.stTabs [data-baseweb='tab-list'] button {
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
  font-weight: 560 !important;
  letter-spacing: -0.01em !important;
  text-transform: none !important;
  white-space: nowrap !important;
  flex: 0 0 auto !important;
  color: var(--text-2) !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  padding: 0.5rem 1.05rem !important;
  min-height: 40px !important;
  transition: background var(--fast) var(--ease), color var(--fast) var(--ease) !important;
}

.stTabs [data-baseweb='tab-list'] button:hover { background: var(--surface-2) !important; color: var(--text) !important; }

.stTabs [data-baseweb='tab-list'] button[aria-selected='true'] {
  background: var(--surface-3) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: none !important;
}

.stTabs [data-baseweb='tab-highlight'],
.stTabs [data-baseweb='tab-border'] { display: none !important; }

.stTabs [data-baseweb='tab-panel'] { padding-top: 1.1rem !important; }

/* =========================================================
   10. CARDS - only real containers, never every nested block
   ========================================================= */
[data-testid='stExpander'],
[data-testid='stForm'],
[data-testid='stStatus'] {
  background: var(--surface-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-lg) !important;
  box-shadow: none !important;
  animation: none !important;
}

[data-testid='stExpander'] summary,
.streamlit-expanderHeader {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.96rem !important;
  color: var(--text) !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-lg) !important;
  padding: 0.9rem 1.05rem !important;
  min-height: 48px !important;
  transition: background var(--fast) var(--ease) !important;
}

[data-testid='stExpander'] summary:hover { background: var(--surface-2) !important; box-shadow: none !important; }
[data-testid='stExpander'] summary:focus-visible { outline: none !important; box-shadow: 0 0 0 3px rgba(10,132,255,0.3) !important; }
[data-testid='stExpander'] details > div { border: none !important; background: transparent !important; padding: 0 1.05rem 1.05rem !important; }
[data-testid='stForm'] { padding: 1.1rem !important; }

/* Bordered st.container only, not every wrapper */
[data-testid='stVerticalBlockBorderWrapper'][data-test-scroll-behavior='normal'] { background: transparent !important; border: none !important; }

/* =========================================================
   11. SLIDER, CHECKBOX, RADIO, TOGGLE
   ========================================================= */
.stSlider [data-baseweb='slider'] div[role='slider'] {
  background: #fff !important;
  border: none !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.45) !important;
  height: 22px !important;
  width: 22px !important;
  animation: none !important;
}

.stSlider [data-baseweb='slider'] > div > div { background: rgba(255,255,255,0.14) !important; height: 4px !important; border-radius: 999px !important; }
.stSlider [data-baseweb='slider'] > div > div > div { background: var(--blue) !important; border-radius: 999px !important; }
.stSlider [data-testid='stTickBar'] { display: none !important; }

[data-testid='stCheckbox'] label,
[data-testid='stRadio'] label { min-height: 32px !important; }

[data-baseweb='checkbox'] div:first-child {
  border-radius: 6px !important;
  border: 1.5px solid var(--line-2) !important;
  background: var(--surface-1) !important;
  transition: background var(--fast) var(--ease), border-color var(--fast) var(--ease) !important;
}

[data-baseweb='checkbox'] input:checked + div {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
}

[data-baseweb='radio'] div[data-checked='true'] { background: var(--blue) !important; border-color: var(--blue) !important; }

/* =========================================================
   12. FEEDBACK
   ========================================================= */
[data-testid='stAlert'], [data-testid='stNotification'], .stAlert {
  background: var(--surface-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-lg) !important;
  box-shadow: none !important;
  color: var(--text) !important;
  padding: 0.85rem 1rem !important;
}

.stSuccess { border-color: rgba(48,209,88,0.4) !important;  background: rgba(48,209,88,0.09) !important; }
.stInfo    { border-color: rgba(10,132,255,0.4) !important; background: rgba(10,132,255,0.09) !important; }
.stWarning { border-color: rgba(255,159,10,0.4) !important; background: rgba(255,159,10,0.09) !important; }
.stError   { border-color: rgba(255,69,58,0.4) !important;  background: rgba(255,69,58,0.09) !important; }

.stProgress > div > div { background: rgba(255,255,255,0.10) !important; border-radius: 999px !important; height: 5px !important; }
.stProgress > div > div > div,
.stProgress > div > div > div > div {
  background: var(--blue) !important;
  border-radius: 999px !important;
  box-shadow: none !important;
  animation: none !important;
}

.stSpinner > div { border-color: var(--blue) transparent transparent transparent !important; }

[data-testid='stToast'] {
  background: rgba(26,26,30,0.9) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  box-shadow: var(--shadow-lg) !important;
}

[data-testid='stStatusWidget'] {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-pill) !important;
}

/* =========================================================
   13. MEDIA, CODE, TABLES, UPLOADER
   ========================================================= */
.stVideo, .stVideo video, [data-testid='stImage'] img {
  width: 100% !important;
  height: auto !important;
  border-radius: var(--r-lg) !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow) !important;
  transform: none !important;
}

[data-testid='stAudio'] { width: 100% !important; }

code, .stCodeBlock, pre {
  font-family: var(--font-mono) !important;
  background: rgba(255,255,255,0.045) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  font-size: 0.85rem !important;
}

pre { overflow-x: auto !important; }

[data-testid='stDataFrame'], [data-testid='stTable'] {
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  overflow: auto !important;
  max-width: 100% !important;
}

[data-testid='stFileUploaderDropzone'] {
  background: var(--surface-1) !important;
  border: 1.5px dashed var(--line-2) !important;
  border-radius: var(--r-lg) !important;
  transition: border-color var(--fast) var(--ease), background var(--fast) var(--ease) !important;
}

[data-testid='stFileUploaderDropzone']:hover {
  border-color: rgba(10,132,255,0.65) !important;
  background: rgba(10,132,255,0.06) !important;
}

/* =========================================================
   14. SCROLLBAR
   ========================================================= */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.16);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); background-clip: content-box; }

/* =========================================================
   15. MOTION AND PERFORMANCE
   Streamlit re-runs the script on every interaction, so entry
   animations would replay on every click. Nothing animates on mount.
   ========================================================= */
[data-testid='stAppViewBlockContainer'] > div,
[data-testid='stVerticalBlock'],
[data-testid='stHorizontalBlock'],
[data-testid='stExpander'],
[data-testid='stSidebar'],
.stAlert, .stButton > button, h1, h2, h3 {
  animation: none !important;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* Blur is the most expensive paint operation. Drop it on small or
   low-power devices, the solid fallback looks nearly identical. */
@media (max-width: 900px), (prefers-reduced-transparency: reduce), (update: slow) {
  [data-testid='stHeader'],
  [data-testid='stSidebar'],
  div[data-baseweb='popover'] > div,
  div[data-baseweb='popover'] ul {
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
  }
  [data-testid='stSidebar'] { background: #0d0e13 !important; }
  [data-testid='stHeader'] { background: #07080c !important; }
  div[data-baseweb='popover'] > div,
  div[data-baseweb='popover'] ul { background: #1a1a1e !important; }
  .stApp { background-attachment: scroll !important; }
}

/* =========================================================
   16. BREAKPOINTS
   ========================================================= */
@media (min-width: 1600px) {
  .main .block-container,
  [data-testid='stAppViewBlockContainer'] { max-width: 1320px !important; }
}

@media (max-width: 1200px) {
  .main .block-container,
  [data-testid='stAppViewBlockContainer'] { max-width: 100% !important; }
}

@media (max-width: 992px) {
  [data-testid='stSidebar'] { max-width: 88vw !important; }
}

/* Tablets and phones: columns become full width */
@media (max-width: 768px) {
  [data-testid='stHorizontalBlock'] > [data-testid='stColumn'],
  [data-testid='stHorizontalBlock'] > [data-testid='column'] {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
  }

  .uq-hero { border-radius: var(--r-lg) !important; }
  .stTabs [data-baseweb='tab-list'] button { padding: 0.5rem 0.85rem !important; font-size: 0.86rem !important; }
  [data-testid='stExpander'] summary { padding: 0.8rem 0.9rem !important; }
}

@media (max-width: 576px) {
  .stButton > button,
  .stDownloadButton > button,
  .stFormSubmitButton > button { width: 100% !important; }

  [data-testid='stSidebar'] { max-width: 92vw !important; }
  [data-testid='stHorizontalBlock'] { gap: 0.7rem !important; }
  h1 { font-size: 1.6rem !important; }
}

@media (max-width: 400px) {
  .uq-hero { padding: 1.1rem 0.95rem !important; }
  .uq-hero-sub { font-size: 0.9rem !important; }
}

/* Short landscape phones */
@media (max-height: 480px) and (orientation: landscape) {
  .main .block-container,
  [data-testid='stAppViewBlockContainer'] { padding-top: 3rem !important; }
  .uq-hero { padding: 0.9rem 1.1rem !important; margin-bottom: 1rem !important; }
  .uq-hero-title { font-size: 1.5rem !important; }
  .uq-hero-sub { display: none !important; }
}

/* Touch devices: Apple HIG minimum hit area */
@media (pointer: coarse) {
  button, [role='tab'], [role='option'], summary { min-height: 44px !important; }
  .stTextInput input, .stNumberInput input, .stTextArea textarea { font-size: 16px !important; }
}
</style>
'''


def inject(st):
    # Apply the theme to the current Streamlit page.
    st.markdown(APPLE_CSS, unsafe_allow_html=True)
