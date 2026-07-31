# -*- coding: utf-8 -*-
# Apple-style glassmorphism theme for the UnQTube2 Streamlit UI.
#
# Design language: macOS Sonoma / visionOS.
#   - SF Pro system font stack with an Inter web fallback
#   - Frosted translucent surfaces (backdrop-filter blur + saturate)
#   - Hairline borders, layered soft shadows, generous corner radii
#   - Apple system colors and calm spring easing
#
# Injected by webui/App.py. Nothing here touches application logic.

APPLE_CSS = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* =========================================================
   DESIGN TOKENS
   ========================================================= */
:root {
  --font-sf: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text',
             'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  --font-mono: 'SF Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;

  --bg-base: #06070a;
  --glass: rgba(255, 255, 255, 0.06);
  --glass-2: rgba(255, 255, 255, 0.10);
  --glass-3: rgba(255, 255, 255, 0.16);
  --hairline: rgba(255, 255, 255, 0.12);
  --hairline-2: rgba(255, 255, 255, 0.22);
  --blur: saturate(180%) blur(30px);

  --text: rgba(255, 255, 255, 0.94);
  --text-2: rgba(235, 235, 245, 0.62);
  --text-3: rgba(235, 235, 245, 0.38);

  --blue: #0a84ff;
  --blue-dark: #0060df;
  --indigo: #5e5ce6;
  --purple: #bf5af2;
  --green: #30d158;
  --orange: #ff9f0a;
  --red: #ff453a;

  --r-sm: 10px;
  --r-md: 14px;
  --r-lg: 20px;
  --r-xl: 28px;
  --r-pill: 980px;

  --shadow-sm: 0 1px 2px rgba(0,0,0,0.28), 0 4px 14px rgba(0,0,0,0.20);
  --shadow-md: 0 10px 34px rgba(0,0,0,0.36);
  --shadow-lg: 0 26px 68px rgba(0,0,0,0.48);

  --ease: cubic-bezier(0.32, 0.72, 0, 1);
}

/* =========================================================
   BASE
   ========================================================= */
html, body, .stApp,
[data-testid='stAppViewContainer'] {
  font-family: var(--font-sf) !important;
  color: var(--text) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.stApp {
  background: var(--bg-base) !important;
}

/* Ambient gradient mesh, the visionOS style backdrop */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(58rem 40rem at 10% -12%, rgba(10, 132, 255, 0.20), transparent 62%),
    radial-gradient(48rem 34rem at 106% 2%, rgba(191, 90, 242, 0.16), transparent 62%),
    radial-gradient(46rem 34rem at 48% 118%, rgba(94, 92, 230, 0.16), transparent 62%);
}

[data-testid='stHeader'] {
  background: rgba(6, 7, 10, 0.55) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid='stDecoration'],
[data-testid='stAppDeployButton'] {
  display: none !important;
}

.main .block-container,
[data-testid='stAppViewBlockContainer'] {
  max-width: 1240px !important;
  padding: 3rem 2.4rem 6rem !important;
  position: relative;
  z-index: 1;
}

/* =========================================================
   TYPOGRAPHY
   ========================================================= */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sf) !important;
  color: var(--text) !important;
  -webkit-text-fill-color: currentColor !important;
  background: none !important;
  animation: none !important;
  text-shadow: none !important;
  letter-spacing: -0.022em !important;
}

h1 {
  font-size: clamp(2.1rem, 4vw, 3.05rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.038em !important;
  line-height: 1.06 !important;
  margin-bottom: 0.35rem !important;
}

h2 { font-size: 1.55rem !important; font-weight: 640 !important; }
h3 { font-size: 1.18rem !important; font-weight: 600 !important; }

p, span, label, li, .stMarkdown, [data-testid='stMarkdownContainer'] p {
  font-family: var(--font-sf) !important;
  color: var(--text-2) !important;
  letter-spacing: -0.01em !important;
  line-height: 1.55 !important;
}

[data-testid='stWidgetLabel'] label p,
.stTextInput label, .stTextArea label, .stSelectbox label,
.stNumberInput label, .stSlider label, .stCheckbox label {
  color: var(--text) !important;
  font-weight: 560 !important;
  font-size: 0.92rem !important;
}

/* Big hero title used by Main.py */
.main-title {
  text-align: center !important;
  font-size: clamp(2.2rem, 5vw, 3.4rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.04em !important;
  margin: 0.5rem 0 2.6rem !important;
  background: linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.58) 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  animation: none !important;
}

/* =========================================================
   BUTTONS
   ========================================================= */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
  font-family: var(--font-sf) !important;
  font-size: 0.95rem !important;
  font-weight: 590 !important;
  letter-spacing: -0.01em !important;
  text-transform: none !important;
  color: var(--text) !important;
  background: var(--glass-2) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-pill) !important;
  padding: 0.62rem 1.4rem !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  box-shadow: var(--shadow-sm) !important;
  animation: none !important;
  transition: transform 0.2s var(--ease), background 0.2s var(--ease),
              border-color 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  background: var(--glass-3) !important;
  border-color: var(--hairline-2) !important;
  color: #ffffff !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-md) !important;
}

.stButton > button:active,
.stDownloadButton > button:active,
.stFormSubmitButton > button:active {
  transform: scale(0.985) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.35) !important;
}

.stButton > button[kind='primary'],
.stFormSubmitButton > button[kind='primary'] {
  background: linear-gradient(180deg, #0a84ff 0%, #0060df 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  color: #ffffff !important;
  font-weight: 620 !important;
  box-shadow: 0 6px 22px rgba(10, 132, 255, 0.38) !important;
}

.stButton > button[kind='primary']:hover {
  background: linear-gradient(180deg, #2a95ff 0%, #0a6fe8 100%) !important;
  box-shadow: 0 12px 34px rgba(10, 132, 255, 0.46) !important;
}

.stDownloadButton > button {
  background: linear-gradient(180deg, rgba(48, 209, 88, 0.92), rgba(38, 170, 72, 0.92)) !important;
  border-color: rgba(255, 255, 255, 0.18) !important;
  color: #05130a !important;
}

/* Full width generate button wrapper from Main.py */
.generate-btn button {
  width: 100% !important;
  font-size: 1.05rem !important;
  padding: 0.95rem 2rem !important;
  animation: none !important;
}

/* =========================================================
   INPUTS
   ========================================================= */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input,
.stDateInput > div > div > input,
[data-baseweb='select'] > div,
[data-baseweb='input'] {
  background: var(--glass) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  color: var(--text) !important;
  font-family: var(--font-sf) !important;
  font-size: 0.95rem !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
  animation: none !important;
  transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease),
              background 0.2s var(--ease) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
  padding: 0.72rem 0.95rem !important;
}

.stTextInput > div > div > input:hover,
.stTextArea > div > div > textarea:hover,
[data-baseweb='select'] > div:hover {
  border-color: var(--hairline-2) !important;
  background: var(--glass-2) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus,
[data-baseweb='select'] > div:focus-within {
  border-color: rgba(10, 132, 255, 0.9) !important;
  box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.22) !important;
  outline: none !important;
  animation: none !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: var(--text-3) !important;
  font-style: normal !important;
}

/* Dropdown menus */
div[data-baseweb='popover'] > div,
div[data-baseweb='menu'] {
  background: rgba(28, 28, 32, 0.82) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  box-shadow: var(--shadow-lg) !important;
}

div[data-baseweb='popover'] li {
  color: var(--text-2) !important;
  border-radius: 8px !important;
  margin: 2px 6px !important;
  transition: background 0.15s var(--ease) !important;
}

div[data-baseweb='popover'] li:hover {
  background: rgba(255, 255, 255, 0.10) !important;
  color: #ffffff !important;
}

/* =========================================================
   SIDEBAR
   ========================================================= */
[data-testid='stSidebar'] {
  background: rgba(14, 16, 22, 0.70) !important;
  border-right: 1px solid var(--hairline) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  animation: none !important;
}

[data-testid='stSidebar'] > div:first-child {
  background: transparent !important;
  padding: 1.6rem 1.15rem !important;
}

[data-testid='stSidebar'] h2,
[data-testid='stSidebar'] h3,
[data-testid='stSidebar'] .stSubheader {
  font-size: 0.74rem !important;
  font-weight: 620 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  border-bottom: none !important;
  margin: 1.6rem 0 0.6rem !important;
}

/* =========================================================
   TABS as a segmented control
   ========================================================= */
.stTabs [data-baseweb='tab-list'] {
  background: var(--glass) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-pill) !important;
  padding: 4px !important;
  gap: 2px !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
}

.stTabs [data-baseweb='tab-list'] button {
  font-family: var(--font-sf) !important;
  font-size: 0.9rem !important;
  font-weight: 560 !important;
  letter-spacing: -0.01em !important;
  text-transform: none !important;
  color: var(--text-2) !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-pill) !important;
  padding: 0.48rem 1.15rem !important;
  transition: background 0.2s var(--ease), color 0.2s var(--ease) !important;
}

.stTabs [data-baseweb='tab-list'] button:hover {
  background: rgba(255, 255, 255, 0.07) !important;
  color: var(--text) !important;
}

.stTabs [data-baseweb='tab-list'] button[aria-selected='true'] {
  background: rgba(255, 255, 255, 0.15) !important;
  color: #ffffff !important;
  box-shadow: var(--shadow-sm) !important;
  border: none !important;
}

.stTabs [data-baseweb='tab-highlight'],
.stTabs [data-baseweb='tab-border'] {
  display: none !important;
}

/* =========================================================
   CARDS, EXPANDERS, FORMS, STATUS
   ========================================================= */
[data-testid='stExpander'],
[data-testid='stForm'],
[data-testid='stStatus'],
[data-testid='stVerticalBlockBorderWrapper'] {
  background: var(--glass) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-lg) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  box-shadow: var(--shadow-md) !important;
  animation: none !important;
}

[data-testid='stExpander'] summary,
.streamlit-expanderHeader {
  font-family: var(--font-sf) !important;
  font-weight: 600 !important;
  font-size: 0.98rem !important;
  color: var(--text) !important;
  background: transparent !important;
  border: none !important;
  padding: 1rem 1.2rem !important;
  transition: background 0.2s var(--ease) !important;
}

[data-testid='stExpander'] summary:hover {
  background: rgba(255, 255, 255, 0.05) !important;
  box-shadow: none !important;
}

[data-testid='stExpander'] details > div {
  border: none !important;
  background: transparent !important;
  padding: 0 1.2rem 1.2rem !important;
}

/* =========================================================
   SLIDERS
   ========================================================= */
.stSlider [data-baseweb='slider'] div[role='slider'] {
  background: #ffffff !important;
  border: none !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.3) !important;
  width: 22px !important;
  height: 22px !important;
  animation: none !important;
  transition: transform 0.15s var(--ease) !important;
}

.stSlider [data-baseweb='slider'] div[role='slider']:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 0 0 6px rgba(10, 132, 255, 0.18) !important;
}

.stSlider [data-baseweb='slider'] > div > div {
  background: rgba(255, 255, 255, 0.14) !important;
  height: 5px !important;
  border-radius: 999px !important;
}

.stSlider [data-baseweb='slider'] > div > div > div {
  background: linear-gradient(90deg, var(--blue), var(--indigo)) !important;
  border-radius: 999px !important;
}

/* =========================================================
   CHECKBOX / TOGGLE
   ========================================================= */
[data-testid='stCheckbox'] [data-baseweb='checkbox'] div:first-child {
  border-radius: 7px !important;
  border: 1.5px solid var(--hairline-2) !important;
  background: var(--glass) !important;
  transition: all 0.2s var(--ease) !important;
}

[data-testid='stCheckbox'] [data-baseweb='checkbox'] input:checked + div {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
  box-shadow: none !important;
}

/* =========================================================
   FEEDBACK
   ========================================================= */
[data-testid='stAlert'],
[data-testid='stNotification'],
.stAlert {
  background: var(--glass) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-lg) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  box-shadow: var(--shadow-sm) !important;
  color: var(--text) !important;
}

.stSuccess { border-color: rgba(48, 209, 88, 0.45) !important; background: rgba(48, 209, 88, 0.10) !important; }
.stInfo    { border-color: rgba(10, 132, 255, 0.45) !important; background: rgba(10, 132, 255, 0.10) !important; }
.stWarning { border-color: rgba(255, 159, 10, 0.45) !important; background: rgba(255, 159, 10, 0.10) !important; }
.stError   { border-color: rgba(255, 69, 58, 0.45) !important; background: rgba(255, 69, 58, 0.10) !important; }

.stProgress > div > div {
  background: rgba(255, 255, 255, 0.10) !important;
  border-radius: 999px !important;
  height: 6px !important;
}

.stProgress > div > div > div,
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--blue), var(--indigo)) !important;
  border-radius: 999px !important;
  box-shadow: none !important;
  animation: none !important;
}

.stSpinner > div {
  border-color: var(--blue) transparent transparent transparent !important;
}

[data-testid='stToast'] {
  background: rgba(28, 28, 32, 0.84) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  backdrop-filter: var(--blur) !important;
  box-shadow: var(--shadow-lg) !important;
}

/* =========================================================
   MEDIA, CODE, TABLES
   ========================================================= */
.stVideo, .stVideo video, [data-testid='stImage'] img {
  border-radius: var(--r-lg) !important;
  border: 1px solid var(--hairline) !important;
  box-shadow: var(--shadow-lg) !important;
  overflow: hidden !important;
  transition: transform 0.3s var(--ease) !important;
}

.stVideo:hover { transform: none !important; }

[data-testid='stAudio'] { border-radius: var(--r-pill) !important; }

code, .stCodeBlock, pre {
  font-family: var(--font-mono) !important;
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid var(--hairline) !important;
  border-radius: var(--r-md) !important;
  font-size: 0.86rem !important;
}

[data-testid='stDataFrame'], [data-testid='stTable'] {
  border-radius: var(--r-md) !important;
  overflow: hidden !important;
  border: 1px solid var(--hairline) !important;
}

[data-testid='stFileUploaderDropzone'] {
  background: var(--glass) !important;
  border: 1.5px dashed var(--hairline-2) !important;
  border-radius: var(--r-lg) !important;
  backdrop-filter: var(--blur) !important;
  transition: all 0.2s var(--ease) !important;
}

[data-testid='stFileUploaderDropzone']:hover {
  border-color: rgba(10, 132, 255, 0.7) !important;
  background: rgba(10, 132, 255, 0.06) !important;
}

hr, [data-testid='stDivider'] {
  border-color: var(--hairline) !important;
  opacity: 1 !important;
}

/* =========================================================
   SCROLLBAR
   ========================================================= */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.28); background-clip: content-box; }

/* =========================================================
   MOTION
   ========================================================= */
@keyframes appleFadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}

[data-testid='stAppViewBlockContainer'] > div {
  animation: appleFadeUp 0.55s var(--ease) both;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
  }
}

/* =========================================================
   RESPONSIVE
   ========================================================= */
@media (max-width: 768px) {
  .main .block-container,
  [data-testid='stAppViewBlockContainer'] {
    padding: 1.6rem 1.1rem 4rem !important;
  }
  h1 { font-size: 2rem !important; }
  .stButton > button { padding: 0.58rem 1.1rem !important; font-size: 0.9rem !important; }
}
</style>
'''


def inject(st):
    # Apply the theme to the current Streamlit page.
    st.markdown(APPLE_CSS, unsafe_allow_html=True)
