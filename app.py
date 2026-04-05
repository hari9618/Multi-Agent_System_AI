"""
app.py — NexusAI Multi-Agent Chatbot v3.1 (Fixed)
==================================================
Fixes applied:
  ✅ st.text_area height raised from 44→100 (Streamlit min is 68px)
  ✅ _elapsed stored as proper session_state key (not attr)
  ✅ reviewer_json passed back cleanly from run_pipeline
  ✅ memory_clear + reset_llm wired to "New chat" button
  ✅ Layout uses st.columns consistently (no container/column mismatch)
  ✅ send guard checks user_query is not None before .strip()
  ✅ prefill pop returns "" default so text_area never gets None
"""

import streamlit as st
import os, time, json
from dotenv import load_dotenv
from agents import run_pipeline, memory_clear, reset_llm, CONFIG, TOOLS, FAISS_AVAILABLE

load_dotenv()

# ── Inject API key from env silently ──────────────────────────────────────────
_env_key = os.getenv("GROQ_API_KEY", "")
if _env_key:
    CONFIG["groq_api_key"] = _env_key

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — PREMIUM DARK CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:ital,wght@1,400;1,600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #07090e;
  --bg2:       #0c0f18;
  --bg3:       #111522;
  --border:    rgba(255,255,255,0.06);
  --border2:   rgba(255,255,255,0.1);
  --text:      #e2e8f4;
  --muted:     #4a5568;
  --muted2:    #2d3748;
  --accent:    #4f8ef7;
  --accent2:   #3b6fd4;
  --green:     #10b981;
  --purple:    #8b5cf6;
  --orange:    #f59e0b;
  --red:       #ef4444;
  --font:      'Outfit', system-ui, sans-serif;
  --mono:      'JetBrains Mono', monospace;
  --serif:     'Playfair Display', Georgia, serif;
}

html, body, [class*="css"] {
  font-family: var(--font) !important;
  -webkit-font-smoothing: antialiased;
}

/* Kill Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp {
  background: var(--bg) !important;
  color: var(--text);
}

.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 10px; }

/* ══════════════════════════════════════════════════════
   TOP NAV BAR
══════════════════════════════════════════════════════ */
.nexus-nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 1000;
  height: 56px;
  background: rgba(7, 9, 14, 0.92);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.nav-logo-mark {
  width: 30px; height: 30px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--purple) 100%);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem; font-weight: 700; color: #fff;
  box-shadow: 0 0 16px rgba(79,142,247,0.35);
  flex-shrink: 0;
}

.nav-name {
  font-family: var(--font);
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: -0.3px;
  color: var(--text);
}

.nav-tag {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 100px;
}

.nav-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--green);
  font-family: var(--mono);
}

.status-pulse {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-dot 2s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(16,185,129,0.6);
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

/* ══════════════════════════════════════════════════════
   MAIN LAYOUT
══════════════════════════════════════════════════════ */
.nexus-layout {
  display: flex;
  height: 100vh;
  padding-top: 56px;
}

/* ── Left panel: chat ── */
.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 720px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

/* ══════════════════════════════════════════════════════
   WELCOME SCREEN (shown when no messages)
══════════════════════════════════════════════════════ */
.welcome-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 1rem;
  text-align: center;
  animation: fade-up 0.6s ease both;
}

@keyframes fade-up {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

.welcome-orb {
  width: 72px; height: 72px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(79,142,247,0.2) 0%, rgba(139,92,246,0.2) 100%);
  border: 1px solid rgba(79,142,247,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 0 40px rgba(79,142,247,0.1), inset 0 1px 0 rgba(255,255,255,0.07);
  animation: orb-float 4s ease-in-out infinite;
}

@keyframes orb-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

.welcome-title {
  font-family: var(--serif);
  font-size: 2rem;
  font-style: italic;
  font-weight: 400;
  color: var(--text);
  margin-bottom: 0.5rem;
  letter-spacing: -0.5px;
  line-height: 1.2;
}

.welcome-sub {
  font-size: 0.9rem;
  color: var(--muted);
  font-weight: 400;
  margin-bottom: 2.5rem;
  line-height: 1.6;
  max-width: 400px;
}

/* ── Agent badges ── */
.agent-badges {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 2.5rem;
  flex-wrap: wrap;
  justify-content: center;
}
.agent-badge {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 5px 12px;
  border-radius: 100px;
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.5px;
  border: 1px solid;
}
.badge-p { color: #60a5fa; border-color: rgba(96,165,250,0.2); background: rgba(59,130,246,0.06); }
.badge-w { color: #a78bfa; border-color: rgba(167,139,250,0.2); background: rgba(139,92,246,0.06); }
.badge-r { color: #34d399; border-color: rgba(52,211,153,0.2); background: rgba(16,185,129,0.06); }
.badge-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

/* ── Example prompts ── */
.examples-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  width: 100%;
  max-width: 560px;
}
.example-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
}
.example-card:hover {
  border-color: rgba(79,142,247,0.25);
  background: rgba(79,142,247,0.04);
  transform: translateY(-1px);
}
.example-icon { font-size: 0.9rem; margin-bottom: 0.3rem; display: block; }
.example-text {
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.4;
  font-weight: 400;
}
.example-card:hover .example-text { color: #8090a8; }

/* ══════════════════════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════════════════════ */
.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 0 1rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* ── User message ── */
.msg-user {
  display: flex;
  justify-content: flex-end;
  animation: msg-appear 0.3s ease both;
}

@keyframes msg-appear {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-user-bubble {
  max-width: 70%;
  background: linear-gradient(135deg, var(--accent2) 0%, var(--accent) 100%);
  color: #fff;
  padding: 0.75rem 1.1rem;
  border-radius: 16px 16px 4px 16px;
  font-size: 0.9rem;
  line-height: 1.6;
  font-weight: 400;
  box-shadow: 0 4px 20px rgba(79,142,247,0.25);
}

/* ── AI message ── */
.msg-ai {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  animation: msg-appear 0.3s ease both;
}

.msg-ai-avatar {
  width: 32px; height: 32px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(139,92,246,0.15));
  border: 1px solid rgba(79,142,247,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.85rem;
  flex-shrink: 0;
  margin-top: 2px;
}

.msg-ai-content { flex: 1; min-width: 0; }

.msg-ai-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.4rem;
  opacity: 0.7;
}

.msg-ai-bubble {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px 16px 16px 16px;
  padding: 1rem 1.2rem;
  font-size: 0.875rem;
  line-height: 1.85;
  color: #bbc5d4;
  white-space: pre-wrap;
  word-break: break-word;
}

/* ══════════════════════════════════════════════════════
   AGENT WORKFLOW CARD (shown during & after run)
══════════════════════════════════════════════════════ */
.workflow-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  animation: msg-appear 0.3s ease both;
}

.workflow-title {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted2);
  margin-bottom: 0.85rem;
}

.workflow-steps {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.wf-step {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  padding: 0.65rem 0.4rem;
  border-radius: 10px;
  border: 1px solid transparent;
  transition: all 0.3s ease;
}
.wf-icon { font-size: 1.1rem; line-height: 1; }
.wf-name {
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.wf-desc { font-size: 0.6rem; font-weight: 400; opacity: 0.6; }

/* Step states */
.wf-step.idle    { color: #1e2d3d; }
.wf-step.idle .wf-icon { opacity: 0.2; }

.wf-step.active {
  background: rgba(79,142,247,0.07);
  border-color: rgba(79,142,247,0.25);
  color: #60a5fa;
  animation: step-pulse 1.2s ease-in-out infinite;
}
.wf-step.done {
  background: rgba(16,185,129,0.05);
  border-color: rgba(16,185,129,0.15);
  color: #34d399;
}
.wf-step.error {
  background: rgba(239,68,68,0.06);
  border-color: rgba(239,68,68,0.15);
  color: #f87171;
}

@keyframes step-pulse {
  0%, 100% { box-shadow: none; }
  50%       { box-shadow: 0 0 14px rgba(79,142,247,0.2); }
}

.wf-connector {
  color: #1a2535;
  font-size: 0.75rem;
  flex-shrink: 0;
  padding-bottom: 0.8rem;
}

/* ── Stats row ── */
.stats-row {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
}
.stat-chip {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 500;
  padding: 3px 9px;
  border-radius: 100px;
  border: 1px solid var(--border);
  color: var(--muted);
  background: rgba(255,255,255,0.02);
}
.stat-chip.highlight { color: var(--accent); border-color: rgba(79,142,247,0.2); }
.stat-chip.green     { color: var(--green);  border-color: rgba(16,185,129,0.2); }
.stat-chip.orange    { color: var(--orange); border-color: rgba(245,158,11,0.2); }

/* ── Plan inline ── */
.plan-section {
  margin-top: 0.75rem;
  padding-top: 0.65rem;
  border-top: 1px solid var(--border);
}
.plan-heading {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted2);
  margin-bottom: 0.5rem;
}
.plan-steps-inline {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.plan-step-inline {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  font-size: 0.75rem;
}
.psi-num {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--accent);
  background: rgba(79,142,247,0.08);
  border: 1px solid rgba(79,142,247,0.15);
  border-radius: 3px;
  padding: 1px 5px;
  flex-shrink: 0;
}
.psi-title { color: #8090a8; font-weight: 500; }
.psi-desc  { color: var(--muted2); font-size: 0.7rem; }

/* ── Score badge ── */
.score-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 100px;
  margin-left: auto;
}
.score-high { color: var(--green);  background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); }
.score-mid  { color: var(--orange); background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); }
.score-low  { color: var(--red);    background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.2);  }

/* ══════════════════════════════════════════════════════
   THINKING INDICATOR
══════════════════════════════════════════════════════ */
.thinking {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  animation: msg-appear 0.3s ease both;
}
.thinking-avatar {
  width: 32px; height: 32px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(139,92,246,0.15));
  border: 1px solid rgba(79,142,247,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.85rem;
  flex-shrink: 0;
}
.thinking-bubble {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px 16px 16px 16px;
  padding: 0.9rem 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.think-text {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.3px;
}
.think-dots { display: flex; gap: 3px; align-items: center; }
.think-dots span {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--accent);
  animation: dot-bounce 1.2s ease-in-out infinite;
}
.think-dots span:nth-child(2) { animation-delay: 0.2s; }
.think-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dot-bounce {
  0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
  40%            { opacity: 1;   transform: scale(1.2); }
}

/* ══════════════════════════════════════════════════════
   INPUT BAR (FIXED BOTTOM)
══════════════════════════════════════════════════════ */
.input-bar-wrapper {
  position: sticky;
  bottom: 0;
  padding: 1rem 0 1.5rem;
  background: linear-gradient(to top, var(--bg) 70%, transparent 100%);
}

.input-bar {
  display: flex;
  align-items: flex-end;
  gap: 0.6rem;
  background: var(--bg2);
  border: 1px solid var(--border2);
  border-radius: 14px;
  padding: 0.65rem 0.65rem 0.65rem 1.1rem;
  box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03) inset;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.input-bar:focus-within {
  border-color: rgba(79,142,247,0.3);
  box-shadow: 0 4px 30px rgba(0,0,0,0.4), 0 0 0 3px rgba(79,142,247,0.06), 0 0 0 1px rgba(255,255,255,0.03) inset;
}

/* Override Streamlit textarea inside our bar */
.input-bar .stTextArea { flex: 1; }
.input-bar .stTextArea > div { background: transparent !important; }
.input-bar .stTextArea > div > div { background: transparent !important; border: none !important; }
.input-bar .stTextArea textarea {
  background: transparent !important;
  border: none !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
  line-height: 1.5 !important;
  padding: 0.3rem 0 !important;
  resize: none !important;
  caret-color: var(--accent) !important;
  box-shadow: none !important;
}
.input-bar .stTextArea textarea:focus { box-shadow: none !important; outline: none !important; }
.input-bar .stTextArea textarea::placeholder { color: var(--muted2) !important; }
.input-bar .stTextArea label { display: none !important; }

/* Send button */
.send-btn .stButton > button {
  width: 40px !important; height: 40px !important;
  min-height: 40px !important;
  border-radius: 10px !important;
  background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
  color: #fff !important;
  border: none !important;
  font-size: 1rem !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 2px 12px rgba(79,142,247,0.35) !important;
  transition: all 0.2s !important;
  flex-shrink: 0 !important;
}
.send-btn .stButton > button:hover {
  transform: scale(1.05) !important;
  box-shadow: 0 4px 20px rgba(79,142,247,0.5) !important;
}
.send-btn .stButton > button:disabled {
  background: var(--bg3) !important;
  box-shadow: none !important;
  transform: none !important;
  opacity: 0.5 !important;
}

/* Hint text below input */
.input-hint {
  text-align: center;
  font-size: 0.65rem;
  color: var(--muted2);
  margin-top: 0.4rem;
  font-family: var(--mono);
  letter-spacing: 0.3px;
}

/* ══════════════════════════════════════════════════════
   DOWNLOAD BUTTON
══════════════════════════════════════════════════════ */
.stDownloadButton > button {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.5px !important;
  color: var(--muted) !important;
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 5px 14px !important;
  height: auto !important;
  transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
  color: var(--green) !important;
  border-color: rgba(16,185,129,0.25) !important;
  background: rgba(16,185,129,0.04) !important;
}

/* Expander */
details > summary {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  color: var(--muted2) !important;
  letter-spacing: 0.5px !important;
}
details > summary:hover { color: var(--muted) !important; }

/* Spinner override */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Streamlit warnings / errors */
.stAlert { border-radius: 10px !important; font-size: 0.82rem !important; }

/* Hide Streamlit form labels globally */
label[data-testid="stWidgetLabel"] { display: none !important; }

/* ══════════════════════════════════════════════════════
   RIGHT PANEL — LIVE DETAILS (shown after run)
══════════════════════════════════════════════════════ */
.detail-panel {
  width: 320px;
  flex-shrink: 0;
  border-left: 1px solid var(--border);
  background: var(--bg2);
  padding: 1.5rem 1.25rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.detail-section { display: flex; flex-direction: column; gap: 0.5rem; }
.detail-heading {
  font-family: var(--mono);
  font-size: 0.55rem;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted2);
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}

/* Metrics grid */
.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
}
.metric-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem 0.75rem;
}
.metric-box-val {
  font-family: var(--mono);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1;
}
.metric-box-label {
  font-size: 0.62rem;
  color: var(--muted2);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 0.2rem;
  font-weight: 500;
}

/* Plan steps in panel */
.panel-step {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.4rem 0.6rem;
  align-items: start;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.panel-step:last-child { border-bottom: none; }
.ps-num {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--accent);
  background: rgba(79,142,247,0.08);
  border: 1px solid rgba(79,142,247,0.15);
  border-radius: 3px;
  padding: 1px 5px;
  margin-top: 1px;
}
.ps-title { font-size: 0.75rem; font-weight: 500; color: #8090a8; }
.ps-desc  { font-size: 0.68rem; color: var(--muted2); line-height: 1.4; }

/* Complexity badge */
.complexity-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 600;
  padding: 2px 9px;
  border-radius: 100px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.cx-low    { color: var(--green);  background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.18); }
.cx-medium { color: var(--orange); background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.18); }
.cx-high   { color: var(--red);    background: rgba(239,68,68,0.07);  border: 1px solid rgba(239,68,68,0.18);  }

/* Issues list in panel */
.issue-item {
  font-size: 0.72rem;
  color: var(--orange);
  padding: 4px 0 4px 1rem;
  position: relative;
  line-height: 1.4;
  border-bottom: 1px solid rgba(245,158,11,0.06);
}
.issue-item:last-child { border-bottom: none; }
.issue-item::before { content: '!'; position: absolute; left: 0; color: var(--orange); opacity: 0.6; }

/* Tools in panel */
.tools-flex { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.tool-chip {
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 500;
  padding: 3px 9px;
  border-radius: 100px;
  color: var(--orange);
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.15);
}
.tool-chip.empty { color: var(--muted2); background: transparent; border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "messages":      [],   # {"role": "user"|"ai", "content": str, "meta": dict|None}
    "is_running":    False,
    "last_result":   None,
    "node_status":   {"planner": "idle", "worker": "idle", "reviewer": "idle"},
    "run_count":     0,
    "elapsed":       0,    # FIX: store elapsed as session_state key, not attribute
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def score_class(s: int) -> str:
    return "score-high" if s >= 8 else ("score-mid" if s >= 5 else "score-low")

def cx_class(c: str) -> str:
    return {"low": "cx-low", "medium": "cx-medium", "high": "cx-high"}.get(c, "cx-medium")

def render_workflow_card(statuses: dict, result: dict = None, elapsed: float = 0) -> str:
    nodes = [
        ("planner",  "🗂", "Planner",  "Plan"),
        ("worker",   "⚙", "Worker",   "Execute"),
        ("reviewer", "✦", "Reviewer", "Review"),
    ]
    state_labels = {"idle": "waiting", "active": "running…", "done": "done", "error": "error"}

    steps_html = ""
    for i, (key, icon, name, role) in enumerate(nodes):
        st_val = statuses.get(key, "idle")
        lbl    = state_labels.get(st_val, st_val)
        steps_html += f"""
<div class="wf-step {st_val}">
  <span class="wf-icon">{icon}</span>
  <span class="wf-name">{name}</span>
  <span class="wf-desc">{role} · {lbl}</span>
</div>"""
        if i < 2:
            steps_html += '<span class="wf-connector">→</span>'

    stats_html = ""
    plan_html  = ""
    if result:
        score    = result.get("quality_score", 0)
        plan     = result.get("plan", {})
        tools    = result.get("tools_used") or []
        revs     = result.get("revision_count", 0)
        steps    = len(plan.get("steps", []))
        complexity = plan.get("complexity", "medium")
        cx         = cx_class(complexity)

        tools_str = ", ".join(tools) if tools else "none"
        stats_html = f"""
<div class="stats-row">
  <span class="stat-chip highlight">⏱ {elapsed}s</span>
  <span class="stat-chip green">★ {score}/10</span>
  <span class="stat-chip">↻ {revs} rev</span>
  <span class="stat-chip">⚙ {tools_str}</span>
  <span class="stat-chip">◎ {steps} steps</span>
  <span class="complexity-badge {cx}">{complexity}</span>
</div>"""

        steps_list = plan.get("steps", [])
        if steps_list:
            step_rows = "".join(f"""
<div class="plan-step-inline">
  <span class="psi-num">0{s.get('id','?')}</span>
  <span class="psi-title">{s.get('title','')}</span>
</div>""" for s in steps_list)
            plan_html = f"""
<div class="plan-section">
  <div class="plan-heading">Execution Plan</div>
  <div class="plan-steps-inline">{step_rows}</div>
</div>"""

    return f"""
<div class="workflow-card">
  <div class="workflow-title">Agent Workflow</div>
  <div class="workflow-steps">{steps_html}</div>
  {stats_html}
  {plan_html}
</div>"""


def build_export(messages: list) -> str:
    lines = ["# NexusAI — Conversation Export\n"]
    for m in messages:
        role = "User" if m["role"] == "user" else "NexusAI"
        lines.append(f"## {role}\n\n{m['content']}\n")
        meta = m.get("meta")
        if meta:
            lines.append(
                f"*Score: {meta.get('quality_score',0)}/10 · "
                f"Time: {meta.get('elapsed',0)}s · "
                f"Revisions: {meta.get('revision_count',0)}*\n"
            )
    return "\n---\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
has_msgs = bool(st.session_state.messages)

st.markdown(f"""
<div class="nexus-nav">
  <div class="nav-brand">
    <div class="nav-logo-mark">◈</div>
    <span class="nav-name">NexusAI</span>
    <span class="nav-tag">Multi-Agent</span>
  </div>
  <div class="nav-right">
    <div class="nav-status">
      <span class="status-pulse"></span>
      <span>Llama 3 · 70B · Groq</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT: CHAT | DETAIL PANEL
# FIX: always use st.columns so the API is consistent — use [1, 0.001] trick
#      to get a "single column" layout without mixing containers and columns.
# ══════════════════════════════════════════════════════════════════════════════
last_result = st.session_state.last_result
show_panel  = last_result is not None

if show_panel:
    col_chat, col_panel = st.columns([2.4, 1], gap="small")
else:
    col_chat, col_panel = st.columns([1, 0.001], gap="small")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    # ── WELCOME SCREEN ────────────────────────────────────────────────────────
    if not has_msgs:
        examples = [
            ("🏗", "Design a scalable microservices architecture for an e-commerce platform"),
            ("🧠", "Explain transformers and implement self-attention in Python"),
            ("🚀", "Build a REST API with FastAPI, JWT auth, and Redis caching"),
            ("📊", "Create a go-to-market strategy for a B2B SaaS developer tool"),
        ]

        st.markdown("""
<div class="welcome-screen">
  <div class="welcome-orb">◈</div>
  <h1 class="welcome-title">What can I help you build?</h1>
  <p class="welcome-sub">
    A multi-agent AI system that plans, executes, and reviews your request
    through three specialized agents — automatically.
  </p>
  <div class="agent-badges">
    <span class="agent-badge badge-p"><span class="badge-dot"></span>Planner</span>
    <span style="color:#1e2d3d;font-size:0.8rem">→</span>
    <span class="agent-badge badge-w"><span class="badge-dot"></span>Worker</span>
    <span style="color:#1e2d3d;font-size:0.8rem">→</span>
    <span class="agent-badge badge-r"><span class="badge-dot"></span>Reviewer</span>
  </div>
</div>
""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        for i, (icon, text) in enumerate(examples):
            col = c1 if i % 2 == 0 else c2
            with col:
                short = text[:52] + "…" if len(text) > 52 else text
                if st.button(f"{icon}  {short}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["_prefill"] = text
                    st.rerun()

    # ── CHAT MESSAGES ─────────────────────────────────────────────────────────
    else:
        wf_placeholder = st.empty()
        if st.session_state.is_running or last_result:
            wf_placeholder.markdown(
                render_workflow_card(
                    st.session_state.node_status,
                    last_result,
                    st.session_state.elapsed,   # FIX: use session_state key
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
<div class="msg-user">
  <div class="msg-user-bubble">{msg['content']}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="msg-ai">
  <div class="msg-ai-avatar">◈</div>
  <div class="msg-ai-content">
    <div class="msg-ai-label">NexusAI</div>
    <div class="msg-ai-bubble">{msg['content']}</div>
  </div>
</div>""", unsafe_allow_html=True)

                meta = msg.get("meta")
                if meta:
                    col_dl, col_info = st.columns([1, 3])
                    with col_dl:
                        st.download_button(
                            "↓ Export",
                            data=build_export(st.session_state.messages),
                            file_name="nexusai_output.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key=f"dl_{id(msg)}",
                        )
                    with col_info:
                        with st.expander("{ } Raw JSON"):
                            st.json(meta.get("raw_result", {}))

        if st.session_state.is_running:
            st.markdown("""
<div class="thinking">
  <div class="thinking-avatar">◈</div>
  <div class="thinking-bubble">
    <span class="think-text">Agents working</span>
    <div class="think-dots">
      <span></span><span></span><span></span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── INPUT BAR ─────────────────────────────────────────────────────────────
    st.markdown('<div class="input-bar-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="input-bar">', unsafe_allow_html=True)

    # FIX: pop with default "" so text_area value is never None
    prefill = st.session_state.pop("_prefill", "") or ""
    inp_col, btn_col = st.columns([10, 1])

    with inp_col:
        user_query = st.text_area(
            "msg",
            value=prefill,
            height=100,          # FIX: was 44 — Streamlit minimum is 68px (100 is safe)
            placeholder="Ask anything — design, code, strategy, analysis…",
            label_visibility="collapsed",
            key="chat_input",
            disabled=st.session_state.is_running,
        )

    with btn_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        # FIX: guard against None from text_area before calling .strip()
        query_text = (user_query or "").strip()
        send = st.button(
            "↑",
            disabled=st.session_state.is_running or not query_text,
            key="send_btn",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # .input-bar

    clear_col, hint_col = st.columns([1, 5])
    with clear_col:
        if st.button("⟳ New chat", key="clear_btn"):
            # FIX: properly call memory_clear + reset_llm on new chat
            memory_clear()
            reset_llm()
            st.session_state.messages    = []
            st.session_state.last_result = None
            st.session_state.is_running  = False
            st.session_state.elapsed     = 0
            st.session_state.node_status = {"planner": "idle", "worker": "idle", "reviewer": "idle"}
            st.rerun()
    with hint_col:
        st.markdown(
            '<div class="input-hint">Planner · Worker · Reviewer — powered by Groq</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # .input-bar-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # .chat-panel


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT DETAIL PANEL
# ══════════════════════════════════════════════════════════════════════════════
if show_panel:
    r = last_result
    with col_panel:
        elapsed    = st.session_state.elapsed   # FIX: session_state key
        plan       = r.get("plan", {})
        steps      = plan.get("steps", [])
        score      = r.get("quality_score", 0)
        revisions  = r.get("revision_count", 0)
        tools      = r.get("tools_used") or []
        complexity = plan.get("complexity", "medium")
        # FIX: reviewer_json lives inside last_result dict returned by run_pipeline
        reviewer   = r.get("reviewer_json") or {}
        issues     = reviewer.get("issues", [])
        improvs    = reviewer.get("improvements", "")
        rcount     = st.session_state.run_count

        st.markdown('<div class="detail-panel" style="margin-top:56px">', unsafe_allow_html=True)

        sc = score_class(score)
        st.markdown(f"""
<div class="detail-section">
  <div class="detail-heading">Performance</div>
  <div class="metrics-grid">
    <div class="metric-box">
      <div class="metric-box-val">{score}/10</div>
      <div class="metric-box-label">★ Score</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-val">{elapsed}s</div>
      <div class="metric-box-label">⏱ Time</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-val">{revisions}</div>
      <div class="metric-box-label">↻ Revisions</div>
    </div>
    <div class="metric-box">
      <div class="metric-box-val">{len(steps)}</div>
      <div class="metric-box-label">◎ Steps</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        if steps:
            step_rows = "".join(f"""
<div class="panel-step">
  <span class="ps-num">0{s.get('id','?')}</span>
  <div>
    <div class="ps-title">{s.get('title','')}</div>
    <div class="ps-desc">{s.get('description','')}</div>
  </div>
</div>""" for s in steps)

            cx = cx_class(complexity)
            st.markdown(f"""
<div class="detail-section">
  <div class="detail-heading" style="display:flex;align-items:center;justify-content:space-between">
    <span>Execution Plan</span>
    <span class="complexity-badge {cx}">{complexity}</span>
  </div>
  {step_rows}
</div>
""", unsafe_allow_html=True)

        if tools:
            chips = "".join(f'<span class="tool-chip">{t}</span>' for t in tools)
        else:
            chips = '<span class="tool-chip empty">none invoked</span>'
        st.markdown(f"""
<div class="detail-section">
  <div class="detail-heading">Tools</div>
  <div class="tools-flex">{chips}</div>
</div>
""", unsafe_allow_html=True)

        if issues or improvs:
            issues_html = "".join(f'<div class="issue-item">{i}</div>' for i in (issues or []))
            improv_html  = f'<div style="font-size:0.72rem;color:#4a5568;margin-top:0.4rem;line-height:1.5">{improvs}</div>' if improvs else ""
            st.markdown(f"""
<div class="detail-section">
  <div class="detail-heading">Reviewer Notes</div>
  {issues_html}
  {improv_html}
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="detail-section">
  <div class="detail-heading">Session</div>
  <div style="font-family:var(--mono);font-size:0.65rem;color:#2d3748;line-height:2">
    <div>Runs this session: <span style="color:#4a5568">{rcount}</span></div>
    <div>Memory: <span style="color:#4a5568">{'FAISS active' if FAISS_AVAILABLE else 'text mode'}</span></div>
    <div>Model: <span style="color:#4a5568">{CONFIG.get('model','')}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # .detail-panel


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUN LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if send and query_text:
    if not os.environ.get("GROQ_API_KEY") and not CONFIG.get("groq_api_key"):
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query_text, "meta": None})
    st.session_state.is_running = True

    for n in ["planner", "worker", "reviewer"]:
        st.session_state.node_status[n] = "idle"
    st.session_state.node_status["planner"] = "active"

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND EXECUTION (triggered after rerun when is_running=True)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.is_running:
    last_user = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    if last_user:
        t0     = time.time()
        result = run_pipeline(last_user)
        elapsed = round(time.time() - t0, 2)

        trace = result.get("trace", [])
        err   = bool(result.get("error"))
        for n in ["planner", "worker", "reviewer"]:
            if n in trace:
                st.session_state.node_status[n] = "error" if err else "done"

        final = result.get("final_output") or result.get("error") or "No output generated."
        st.session_state.messages.append({
            "role":    "ai",
            "content": final,
            "meta": {
                "quality_score":  result.get("quality_score", 0),
                "revision_count": result.get("revision_count", 0),
                "tools_used":     result.get("tools_used", []),
                "elapsed":        elapsed,
                "raw_result": {
                    "plan":           result.get("plan"),
                    "quality_score":  result.get("quality_score"),
                    "revision_count": result.get("revision_count"),
                    "tools_used":     result.get("tools_used"),
                    "trace":          result.get("trace"),
                    # FIX: reviewer_json is in result dict from run_pipeline
                    "reviewer_json":  result.get("reviewer_json"),
                },
            },
        })

        st.session_state.last_result   = result
        st.session_state.elapsed       = elapsed   # FIX: proper session_state key
        st.session_state.is_running    = False
        st.session_state.run_count    += 1

        st.rerun()
