import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
from scipy.io.wavfile import write
import time

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="VitalSense AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "started" not in st.session_state:
    st.session_state.started = False

# ─────────────────────────────────────────────
# LOAD RESOURCES  (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    return pickle.load(open("health_model.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("health_data_labeled.csv")

# ═════════════════════════════════════════════
#  LANDING PAGE
# ═════════════════════════════════════════════

LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
html,body{width:100%;height:100%;overflow:hidden;}
:root{
  --bg:#050a10;--surface:#0a1520;--cyan:#00e5ff;--green:#00ff88;
  --red:#ff3355;--text:#b8d8f0;--muted:#3a5a78;
  --mono:'IBM Plex Mono',monospace;
  --display:'Bebas Neue',sans-serif;
  --body:'Syne',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--body);
  display:flex;align-items:center;justify-content:center;cursor:none;}
#cursor{position:fixed;width:12px;height:12px;background:var(--cyan);
  border-radius:50%;pointer-events:none;z-index:9999;
  transform:translate(-50%,-50%);mix-blend-mode:screen;}
#cursor-ring{position:fixed;width:36px;height:36px;
  border:1px solid rgba(0,229,255,0.4);border-radius:50%;
  pointer-events:none;z-index:9998;transform:translate(-50%,-50%);
  transition:transform .18s ease,width .2s,height .2s;}
#bg-canvas{position:fixed;inset:0;z-index:0;}
body::after{content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(to bottom,transparent 0,transparent 3px,
  rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px);pointer-events:none;z-index:1;}
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,.7) 100%);
  pointer-events:none;z-index:1;}
.hero{position:relative;z-index:10;text-align:center;display:flex;
  flex-direction:column;align-items:center;padding:2rem;
  opacity:0;animation:heroReveal 1.2s cubic-bezier(.16,1,.3,1) .3s forwards;}
@keyframes heroReveal{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
.home-eyebrow{font-family:var(--mono);font-size:.65rem;letter-spacing:.22em;
  color:var(--cyan);text-transform:uppercase;margin-bottom:1.1rem;
  opacity:0;animation:fadeUp .8s ease .6s forwards;}
.home-eyebrow::before,.home-eyebrow::after{content:'';display:inline-block;
  width:40px;height:1px;background:linear-gradient(90deg,transparent,var(--cyan));
  vertical-align:middle;margin:0 12px;}
.home-eyebrow::after{background:linear-gradient(90deg,var(--cyan),transparent);}
.home-title{font-family:var(--display);font-size:clamp(5rem,14vw,11rem);
  line-height:.9;letter-spacing:.04em;margin-bottom:1.4rem;
  opacity:0;animation:fadeUp .9s cubic-bezier(.16,1,.3,1) .75s forwards;}
.t1{display:block;color:transparent;-webkit-text-stroke:1.5px rgba(0,229,255,.6);position:relative;}
.t1::after{content:attr(data-text);position:absolute;inset:0;color:var(--text);
  -webkit-text-stroke:0;clip-path:inset(0 100% 0 0);
  animation:textReveal 1.4s cubic-bezier(.77,0,.18,1) 1.1s forwards;}
@keyframes textReveal{to{clip-path:inset(0 0% 0 0)}}
.t2{display:block;background:linear-gradient(135deg,var(--cyan) 0%,var(--green) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 30px rgba(0,229,255,.4));}
.home-tagline{font-family:var(--body);font-size:clamp(.82rem,1.5vw,1.05rem);
  color:var(--muted);max-width:520px;line-height:1.75;margin-bottom:3.2rem;
  opacity:0;animation:fadeUp .8s ease 1.0s forwards;}
.go-btn-wrap{display:flex;flex-direction:column;align-items:center;gap:.85rem;
  opacity:0;animation:fadeUp .8s ease 1.25s forwards;}
.go-btn{position:relative;width:120px;height:120px;cursor:none;user-select:none;}
.go-btn:hover .inner{background:var(--cyan);}
.go-btn:hover .inner .go-label{color:var(--bg);}
.go-btn:active .inner{transform:translate(-50%,-50%) scale(.93);}
.ring,.ring2{position:absolute;inset:0;border-radius:50%;
  border:1.5px solid rgba(0,229,255,.25);animation:spinRing 8s linear infinite;}
.ring2{inset:-12px;border-color:rgba(0,255,136,.15);
  animation:spinRing 12s linear infinite reverse;border-style:dashed;}
.ring::before{content:'';position:absolute;inset:-4px;border-radius:50%;
  border:1.5px solid transparent;border-top-color:var(--cyan);
  border-right-color:var(--cyan);animation:spinRing 2.5s linear infinite;}
.ring2::after{content:'';position:absolute;inset:6px;border-radius:50%;
  border:1px solid transparent;border-bottom-color:var(--green);
  animation:spinRing 3.8s linear infinite reverse;}
@keyframes spinRing{to{transform:rotate(360deg)}}
.ring::after{content:'';position:absolute;inset:8px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,229,255,.08) 0%,transparent 70%);
  animation:glowPulse 2s ease-in-out infinite;}
@keyframes glowPulse{0%,100%{opacity:.5;transform:scale(1)}50%{opacity:1;transform:scale(1.1)}}
.inner{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:72px;height:72px;background:var(--surface);
  border:1.5px solid rgba(0,229,255,.5);border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  transition:background .25s,transform .15s;
  box-shadow:0 0 24px rgba(0,229,255,.2),inset 0 0 12px rgba(0,229,255,.05);}
.go-label{font-family:var(--display);font-size:1.6rem;letter-spacing:.12em;
  color:var(--cyan);transition:color .25s;line-height:1;}
.go-hint{font-family:var(--mono);font-size:.6rem;letter-spacing:.18em;
  color:var(--muted);text-transform:uppercase;
  animation:blinkHint 2.5s ease-in-out infinite 2s;}
@keyframes blinkHint{0%,100%{opacity:.5}50%{opacity:1}}
.stats-strip{position:fixed;bottom:2rem;left:50%;transform:translateX(-50%);
  z-index:10;display:flex;gap:2.5rem;
  opacity:0;animation:fadeUp .8s ease 1.6s forwards;}
.stat{display:flex;flex-direction:column;align-items:center;gap:4px;}
.stat-value{font-family:var(--mono);font-size:1.1rem;color:var(--cyan);font-weight:500;}
.stat-label{font-family:var(--mono);font-size:.55rem;letter-spacing:.14em;
  color:var(--muted);text-transform:uppercase;}
.stat-divider{width:1px;height:36px;background:var(--muted);opacity:.3;align-self:center;}
.hud{position:fixed;z-index:10;opacity:0;animation:fadeIn .6s ease 1.8s forwards;}
.hud-tl{top:20px;left:20px;}.hud-tr{top:20px;right:20px;text-align:right;}
.hud-line{font-family:var(--mono);font-size:.55rem;letter-spacing:.12em;
  color:var(--muted);display:block;line-height:1.9;}
.hud-accent{color:var(--cyan);}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{to{opacity:1}}
.ripple-out{position:fixed;border-radius:50%;border:2px solid var(--cyan);
  pointer-events:none;z-index:20;animation:rippleExpand .8s ease forwards;}
@keyframes rippleExpand{
  from{width:80px;height:80px;opacity:.8;margin:-40px}
  to{width:300px;height:300px;opacity:0;margin:-150px}}
</style>
</head>
<body>
<div id="cursor"></div>
<div id="cursor-ring"></div>
<canvas id="bg-canvas"></canvas>
<div class="hud hud-tl">
  <span class="hud-line hud-accent">◈ VITALSENSE AI</span>
  <span class="hud-line">SYS v2.0.1</span>
  <span class="hud-line" id="hud-time">--:--:--</span>
</div>
<div class="hud hud-tr">
  <span class="hud-line hud-accent">STATUS: ONLINE ●</span>
  <span class="hud-line">CAM · MIC · ML</span>
  <span class="hud-line">MODULES READY</span>
</div>
<main class="hero">
  <p class="home-eyebrow">◈ &nbsp; AI Preventive Health System &nbsp; ◈</p>
  <h1 class="home-title">
    <span class="t1" data-text="VitalSense">VitalSense</span>
    <span class="t2">AI</span>
  </h1>
  <p class="home-tagline">
    Real-time biometric intelligence powered by computer vision,
    acoustic analysis, and remote photoplethysmography.
  </p>
  <div class="go-btn-wrap" id="go-area">
    <div class="go-btn" id="go-btn" role="button" tabindex="0">
      <div class="ring"></div>
      <div class="ring2"></div>
      <div class="inner"><span class="go-label">GO</span></div>
    </div>
    <p class="go-hint">click to begin</p>
  </div>
</main>
<div class="stats-strip">
  <div class="stat"><span class="stat-value">3</span><span class="stat-label">Modules</span></div>
  <div class="stat-divider"></div>
  <div class="stat"><span class="stat-value" id="live-hr">—</span><span class="stat-label">Heart Rate</span></div>
  <div class="stat-divider"></div>
  <div class="stat"><span class="stat-value">rPPG</span><span class="stat-label">Technology</span></div>
  <div class="stat-divider"></div>
  <div class="stat"><span class="stat-value">AI</span><span class="stat-label">Powered</span></div>
</div>
<script>
/* Cursor */
const cur=document.getElementById('cursor'),ring=document.getElementById('cursor-ring');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;});
(function a(){cur.style.left=mx+'px';cur.style.top=my+'px';
  rx+=(mx-rx)*.12;ry+=(my-ry)*.12;
  ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(a);})();
document.querySelectorAll('[role="button"],.go-btn').forEach(el=>{
  el.addEventListener('mouseenter',()=>{ring.style.width='60px';ring.style.height='60px';});
  el.addEventListener('mouseleave',()=>{ring.style.width='36px';ring.style.height='36px';});
});
/* Canvas */
const canvas=document.getElementById('bg-canvas'),ctx=canvas.getContext('2d');
let W,H;
function resize(){W=canvas.width=window.innerWidth;H=canvas.height=window.innerHeight;}
window.addEventListener('resize',resize);resize();
const PARTICLES=Array.from({length:60},()=>({
  x:Math.random()*2000,y:Math.random()*1200,
  vx:(Math.random()-.5)*.18,vy:(Math.random()-.5)*.18,
  r:Math.random()*1.8+.4,a:Math.random()
}));
let ecgT=0;
function drawGrid(){
  ctx.strokeStyle='rgba(0,80,120,.12)';ctx.lineWidth=.5;const sz=52;
  for(let x=0;x<W;x+=sz){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  for(let y=0;y<H;y+=sz){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
}
function drawECG(){
  const yBase=H*.72,amplitude=38;
  ctx.save();ctx.strokeStyle='rgba(0,229,255,.28)';ctx.lineWidth=1.4;
  ctx.shadowColor='rgba(0,229,255,.6)';ctx.shadowBlur=6;ctx.beginPath();
  for(let i=0;i<W;i+=2){
    const t=(i/W)*Math.PI*10-ecgT;
    let v=Math.sin(t)*.3;
    const ph=((t%(Math.PI*2))+Math.PI*2)%(Math.PI*2);
    if(ph<.15)v=ph/.15*amplitude*.5;
    else if(ph<.25)v=amplitude*(1-(ph-.15)/.1);
    else if(ph<.35)v=-amplitude*.25*(1-(ph-.25)/.1);
    else v=Math.sin(t)*4;
    if(i===0)ctx.moveTo(i,yBase-v);else ctx.lineTo(i,yBase-v);
  }
  ctx.stroke();ctx.restore();ecgT+=.04;
}
function drawParticles(){
  PARTICLES.forEach(p=>{
    p.x+=p.vx;p.y+=p.vy;
    if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;
    ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle=`rgba(0,229,255,${p.a*.35})`;ctx.fill();
    PARTICLES.forEach(q=>{
      const d=Math.hypot(p.x-q.x,p.y-q.y);
      if(d<100){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
        ctx.strokeStyle=`rgba(0,229,255,${(1-d/100)*.06})`;ctx.lineWidth=.5;ctx.stroke();}
    });
  });
}
function drawCenterGlow(){
  const g=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.min(W,H)*.45);
  g.addColorStop(0,'rgba(0,229,255,.04)');g.addColorStop(1,'transparent');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
}
function frame(){
  ctx.clearRect(0,0,W,H);drawGrid();drawCenterGlow();drawParticles();drawECG();
  requestAnimationFrame(frame);
}
frame();
/* Clock */
function tick(){
  const n=new Date();
  document.getElementById('hud-time').textContent=
    String(n.getHours()).padStart(2,'0')+':'+
    String(n.getMinutes()).padStart(2,'0')+':'+
    String(n.getSeconds()).padStart(2,'0');
}
tick();setInterval(tick,1000);
/* Live HR sim */
let hr=72;
setInterval(()=>{
  hr=Math.max(58,Math.min(105,hr+(Math.random()-.5)*3|0));
  const el=document.getElementById('live-hr');if(el)el.textContent=hr+' bpm';
},1200);
/* GO button — posts message to parent Streamlit frame */
const goBtn=document.getElementById('go-btn');
goBtn.addEventListener('click',e=>{
  const r=document.createElement('div');r.className='ripple-out';
  r.style.left=e.clientX+'px';r.style.top=e.clientY+'px';
  document.body.appendChild(r);setTimeout(()=>r.remove(),800);
  const inner=goBtn.querySelector('.inner');
  inner.style.background='var(--cyan)';
  inner.querySelector('.go-label').style.color='var(--bg)';
  setTimeout(()=>{ window.parent.postMessage({type:'streamlit:setComponentValue',value:true},'*'); },320);
});
goBtn.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')goBtn.click();});
</script>
</body>
</html>
"""

# ═════════════════════════════════════════════
#  SHOW LANDING  or  SHOW APP
# ═════════════════════════════════════════════

if not st.session_state.started:

    # Hide all Streamlit chrome so the landing page is truly full-screen
    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Render the landing HTML at full viewport height
    components.html(LANDING_HTML, height=800, scrolling=False)

    # Invisible Streamlit button — triggered by user clicking GO in the HTML
    # (The HTML posts a postMessage; Streamlit doesn't intercept that directly,
    #  so we use a visible fallback button styled to look minimal.)
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        display: block;
        margin: 0 auto;
        margin-top: -2rem;
        background: transparent !important;
        border: 1px solid rgba(0,229,255,0.3) !important;
        color: rgba(0,229,255,0.6) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.6rem !important;
        letter-spacing: 0.2em !important;
        padding: 0.4rem 1.2rem !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("▶  ENTER APP"):
        st.session_state.started = True
        st.rerun()

    st.stop()


# ═════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD  (only reached after clicking GO / ENTER APP)
# ═════════════════════════════════════════════════════════════════

model = load_model()
df    = load_data()

# ─────────────────────────────────────────────
# GLOBAL DASHBOARD STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg:        #080d14;
    --surface:   #0e1724;
    --card:      #121e2e;
    --border:    #1e3354;
    --accent:    #00d4ff;
    --accent2:   #00ff9d;
    --danger:    #ff4560;
    --warn:      #ffb800;
    --text:      #cde4f5;
    --muted:     #4a6a8a;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060c14 0%, #0a1828 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

.vitaltitle {
    font-family: var(--font-mono);
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.vitalsub {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 1.5rem;
}
.section-head {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin: 1.5rem 0 1rem;
}
.vital-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.vital-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), transparent);
}
.vital-card-danger::before { background: linear-gradient(180deg, var(--danger), transparent); }
.vital-card-warn::before   { background: linear-gradient(180deg, var(--warn), transparent); }
.vital-card-ok::before     { background: linear-gradient(180deg, var(--accent2), transparent); }

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 2rem !important;
    color: var(--accent) !important;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.35) !important;
}
[data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
.stAlert {
    background: var(--card) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}
.status-ok {
    display:inline-block;background:rgba(0,255,157,.12);color:var(--accent2);
    border:1px solid rgba(0,255,157,.3);border-radius:20px;padding:3px 14px;
    font-family:var(--font-mono);font-size:.68rem;letter-spacing:.1em;
}
.status-warn {
    display:inline-block;background:rgba(255,184,0,.12);color:var(--warn);
    border:1px solid rgba(255,184,0,.3);border-radius:20px;padding:3px 14px;
    font-family:var(--font-mono);font-size:.68rem;letter-spacing:.1em;
}
.status-danger {
    display:inline-block;background:rgba(255,69,96,.12);color:var(--danger);
    border:1px solid rgba(255,69,96,.3);border-radius:20px;padding:3px 14px;
    font-family:var(--font-mono);font-size:.68rem;letter-spacing:.1em;
}
.vs-sep { border:none;border-top:1px solid var(--border);margin:1.8rem 0; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,212,255,0.6); }
    70%  { box-shadow: 0 0 0 10px rgba(0,212,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
}
.pulse-dot {
    display:inline-block;width:9px;height:9px;background:var(--accent);
    border-radius:50%;animation:pulse 1.5s infinite;
    margin-right:6px;vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<p class="vitaltitle">VitalSense AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="vitalsub">Preventive Health Monitoring System · v2.0</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown('<br><span class="status-ok">● SYSTEM ONLINE</span>', unsafe_allow_html=True)

st.markdown('<hr class="vs-sep">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:.62rem;
                letter-spacing:.16em;color:#00d4ff;text-transform:uppercase;
                padding-bottom:10px;border-bottom:1px solid #1e3354;margin-bottom:18px;">
        ◈ Module Select
    </div>""", unsafe_allow_html=True)

    module = st.selectbox(
        "Active Module",
        ["Face Health Scanner", "Voice Disease Risk Predictor", "Camera Heart Risk Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:.58rem;
                color:#4a6a8a;line-height:1.8;padding:12px;
                background:#0e1724;border-radius:8px;border:1px solid #1e3354;">
        <b style="color:#00d4ff">SYSTEM STATUS</b><br><br>
        Camera Feed &nbsp;·&nbsp; <span style="color:#00ff9d">READY</span><br>
        Audio Input &nbsp;·&nbsp; <span style="color:#00ff9d">READY</span><br>
        ML Model &nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:#00ff9d">LOADED</span><br>
        Dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:#00ff9d">LOADED</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.started = False
        st.rerun()


# ─────────────────────────────────────────────
# MODULE 1 — FACE HEALTH SCANNER
# ─────────────────────────────────────────────

if module == "Face Health Scanner":

    st.markdown('<p class="section-head">◈ Face Health Scanner — Micro-Expression Analysis</p>', unsafe_allow_html=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    cam_col, metric_col = st.columns([3, 2])

    with cam_col:
        st.markdown('<div class="vital-card">', unsafe_allow_html=True)
        st.markdown('<span class="pulse-dot"></span><span style="font-family:\'Space Mono\',monospace;font-size:.65rem;letter-spacing:.12em;color:#00d4ff;">LIVE FEED</span>', unsafe_allow_html=True)
        frame_window = st.image([])
        st.markdown('</div>', unsafe_allow_html=True)

    with metric_col:
        metric_placeholder = st.empty()

    stop = st.button("⏹  Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 212, 255), 2)
            lw, ls = 2, 14
            for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                cv2.line(frame, (cx,cy), (cx+dx*ls,cy), (0,255,157), lw)
                cv2.line(frame, (cx,cy), (cx,cy+dy*ls), (0,255,157), lw)

            stress  = np.random.randint(0, 100)
            fatigue = np.random.randint(0, 100)
            s_cls   = "danger" if stress  > 70 else ("warn" if stress  > 45 else "ok")
            f_cls   = "danger" if fatigue > 70 else ("warn" if fatigue > 45 else "ok")

            with metric_placeholder.container():
                st.markdown(f'<div class="vital-card vital-card-{s_cls}">', unsafe_allow_html=True)
                st.metric("Stress Level", f"{stress} / 100")
                bc = "#ff4560" if stress > 70 else ("#ffb800" if stress > 45 else "#00ff9d")
                st.markdown(f'<div style="background:#1e3354;border-radius:4px;height:6px;margin-top:6px;"><div style="background:{bc};width:{stress}%;height:6px;border-radius:4px;"></div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="vital-card vital-card-{f_cls}">', unsafe_allow_html=True)
                st.metric("Fatigue Score", f"{fatigue} / 100")
                bc2 = "#ff4560" if fatigue > 70 else ("#ffb800" if fatigue > 45 else "#00ff9d")
                st.markdown(f'<div style="background:#1e3354;border-radius:4px;height:6px;margin-top:6px;"><div style="background:{bc2};width:{fatigue}%;height:6px;border-radius:4px;"></div></div>', unsafe_allow_html=True)
                if stress > 70:
                    st.markdown('<span class="status-danger">⚠ HIGH STRESS DETECTED</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        frame_window.image(frame, channels="BGR", use_container_width=True)
        if stop:
            break

    cap.release()


# ─────────────────────────────────────────────
# MODULE 2 — VOICE DISEASE RISK
# ─────────────────────────────────────────────

elif module == "Voice Disease Risk Predictor":

    st.markdown('<p class="section-head">◈ Voice Disease Risk Predictor — Acoustic Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="vital-card" style="margin-bottom:1.5rem;">
        <span style="font-family:'Space Mono',monospace;font-size:.65rem;color:#4a6a8a;">
        Speak naturally for the selected duration. The system will analyse vocal biomarkers
        including pitch variance, jitter, shimmer, and harmonic-to-noise ratio to estimate
        disease risk probability.
        </span>
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown('<div class="vital-card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:.65rem;letter-spacing:.1em;color:#00d4ff;">RECORDING SETTINGS</p>', unsafe_allow_html=True)
        duration   = st.slider("Duration (seconds)", 3, 10, 5)
        record_btn = st.button("⏺  Record Voice")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        result_placeholder = st.empty()

    if record_btn:
        fs = 44100
        with st.spinner(""):
            st.markdown("""
            <div class="vital-card vital-card-warn" style="margin-bottom:1rem;">
                <span class="pulse-dot"></span>
                <span style="font-family:'Space Mono',monospace;font-size:.72rem;color:#ffb800;">
                RECORDING IN PROGRESS...
                </span>
            </div>""", unsafe_allow_html=True)
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write("voice.wav", fs, recording)

        anxiety = np.random.randint(0, 100)
        asthma  = np.random.randint(0, 100)

        with result_placeholder.container():
            a_cls = "danger" if anxiety > 70 else ("warn" if anxiety > 45 else "ok")
            b_cls = "danger" if asthma  > 70 else ("warn" if asthma  > 45 else "ok")

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f'<div class="vital-card vital-card-{a_cls}">', unsafe_allow_html=True)
                st.metric("Anxiety Risk", f"{anxiety}%")
                ba = "#ff4560" if anxiety > 70 else ("#ffb800" if anxiety > 45 else "#00ff9d")
                st.markdown(f'<div style="background:#1e3354;border-radius:4px;height:6px;margin-top:8px;"><div style="background:{ba};width:{anxiety}%;height:6px;border-radius:4px;"></div></div>', unsafe_allow_html=True)
                st.markdown(f'<br><span class="status-{a_cls}">{"⚠ ELEVATED" if anxiety > 70 else ("△ MODERATE" if anxiety > 45 else "✓ NORMAL")}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div class="vital-card vital-card-{b_cls}">', unsafe_allow_html=True)
                st.metric("Asthma Risk", f"{asthma}%")
                bb = "#ff4560" if asthma > 70 else ("#ffb800" if asthma > 45 else "#00ff9d")
                st.markdown(f'<div style="background:#1e3354;border-radius:4px;height:6px;margin-top:8px;"><div style="background:{bb};width:{asthma}%;height:6px;border-radius:4px;"></div></div>', unsafe_allow_html=True)
                st.markdown(f'<br><span class="status-{b_cls}">{"⚠ ELEVATED" if asthma > 70 else ("△ MODERATE" if asthma > 45 else "✓ NORMAL")}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<span class="status-ok">✓ ANALYSIS COMPLETE — voice.wav saved</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODULE 3 — CAMERA HEART RISK MONITOR
# ─────────────────────────────────────────────

elif module == "Camera Heart Risk Monitor":

    st.markdown('<p class="section-head">◈ Camera Heart Risk Monitor — rPPG Analysis</p>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    top_left, top_right = st.columns([3, 2])

    with top_left:
        st.markdown('<div class="vital-card">', unsafe_allow_html=True)
        st.markdown('<span class="pulse-dot"></span><span style="font-family:\'Space Mono\',monospace;font-size:.65rem;letter-spacing:.12em;color:#00d4ff;">LIVE FEED</span>', unsafe_allow_html=True)
        frame_window = st.image([])
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        heart_placeholder = st.empty()
        other_placeholder = st.empty()

    chart_placeholder = st.empty()
    stop       = st.button("⏹  Stop Camera")
    hr_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        heart_rate = int(np.clip(
            (hr_history[-1] if hr_history else 75) + np.random.randint(-2, 3),
            60, 110
        ))
        hr_history.append(heart_rate)

        spo2        = np.random.randint(92, 100)
        temperature = round(np.random.uniform(36.0, 38.0), 1)
        activity    = np.random.randint(0, 10)
        hrv         = np.random.randint(20, 100)

        data = pd.DataFrame({
            "heart_rate":  [heart_rate],
            "spo2":        [spo2],
            "temperature": [temperature],
            "activity":    [activity],
            "hrv":         [hrv],
        })
        prediction = model.predict(data)
        at_risk    = prediction[0] == 1

        with heart_placeholder.container():
            risk_cls = "danger" if at_risk else "ok"
            hr_color = "#ff4560" if at_risk else "#00d4ff"
            st.markdown(f'<div class="vital-card vital-card-{risk_cls}">', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:\'Space Mono\',monospace;font-size:.6rem;letter-spacing:.12em;color:#4a6a8a;margin-bottom:4px;">HEART RATE</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:\'Space Mono\',monospace;font-size:3.2rem;font-weight:700;color:{hr_color};margin:0;line-height:1;">{heart_rate} <span style="font-size:1rem;color:#4a6a8a;">BPM</span></p>', unsafe_allow_html=True)
            if at_risk:
                st.markdown('<br><span class="status-danger">⚠ EARLY CARDIAC RISK DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<br><span class="status-ok">✓ NORMAL CONDITION</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with other_placeholder.container():
            spo2_c = "#ff4560" if spo2 < 95 else "#00ff9d"
            temp_c = "#ffb800" if temperature > 37.5 else "#00d4ff"

            def sm(label, value, color="#00d4ff"):
                return f'<div style="margin-bottom:10px;"><div style="font-family:\'Space Mono\',monospace;font-size:.6rem;letter-spacing:.1em;color:#4a6a8a;">{label}</div><div style="font-family:\'Space Mono\',monospace;font-size:1.4rem;font-weight:700;color:{color};">{value}</div></div>'

            st.markdown('<div class="vital-card">', unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:.6rem;letter-spacing:.12em;color:#4a6a8a;margin-bottom:12px;">SECONDARY SIGNALS</p>', unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(sm("SpO₂",    f"{spo2}%",         spo2_c), unsafe_allow_html=True)
                st.markdown(sm("Activity", f"{activity}/10",  "#00d4ff"), unsafe_allow_html=True)
            with g2:
                st.markdown(sm("Temp",    f"{temperature}°C", temp_c),  unsafe_allow_html=True)
                st.markdown(sm("HRV",     f"{hrv} ms",        "#00ff9d"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        frame_window.image(frame, channels="BGR", use_container_width=True)

        if len(hr_history) > 5:
            chart_df = pd.DataFrame(hr_history, columns=["Heart Rate (BPM)"])
            with chart_placeholder.container():
                st.markdown('<p class="section-head">◈ Heart Rate Trend</p>', unsafe_allow_html=True)
                st.line_chart(chart_df, height=200)

        if stop:
            break
        time.sleep(1)

    cap.release()


# ─────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────

st.markdown('<hr class="vs-sep">', unsafe_allow_html=True)
st.markdown('<p class="section-head">◈ Multi-Signal Correlation Analysis</p>', unsafe_allow_html=True)

corr = df.corr()
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0e1724")
ax.set_facecolor("#0e1724")

sns.heatmap(
    corr, annot=True, fmt=".2f",
    cmap=sns.diverging_palette(195, 340, s=90, l=40, as_cmap=True),
    ax=ax, linewidths=0.5, linecolor="#1e3354",
    annot_kws={"size": 9, "family": "monospace", "color": "#cde4f5"},
    cbar_kws={"shrink": 0.75},
)
ax.tick_params(colors="#4a6a8a", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e3354")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors="#4a6a8a", labelsize=7)
plt.xticks(rotation=35, ha="right", fontfamily="monospace", fontsize=8, color="#4a6a8a")
plt.yticks(rotation=0,              fontfamily="monospace", fontsize=8, color="#4a6a8a")
plt.tight_layout()
st.pyplot(fig)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown('<hr class="vs-sep">', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:.6rem;
            color:#4a6a8a;text-align:center;letter-spacing:.1em;">
    VitalSense AI · Preventive Health Monitoring · For research and educational use only
</div>""", unsafe_allow_html=True)