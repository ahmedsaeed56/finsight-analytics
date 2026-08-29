"""streamlit_app.py — JazzCash Analytics

A CLIENT, not the system.

Every call goes through src.api.client, which talks HTTP to the FastAPI server
or calls the graph in-process depending on JAZZCASH_USE_API. This file imports
nothing from src.graph, src.tools or src.pipeline.

THREE BUGS FIXED IN THIS VERSION
--------------------------------
1. THE RELOAD ON EVERY SECOND QUESTION. dataset_meta() hit the API on every
   Streamlit rerun. The @st.cache_data TTL closed the common case but a
   sub-second reload during the "dark flash" still slipped through, showing
   the "no dataset" warning mid-conversation. Now cached in session_state,
   which survives reruns cleanly, with an explicit clear after an upload.

2. SESSION STATE INITIALISED TOO LATE. thread and msgs were created inside
   page_ask(), AFTER the dataset check. When that check bailed, they were
   never created — so history vanished. They are now set up before anything
   can bail.

3. PAGE_ASK WARNS EVEN MID-CONVERSATION. If meta briefly returns None while
   a chat is in progress, the "no dataset loaded" warning replaced the whole
   thread. Now the warning only fires when the conversation is empty — an
   ongoing conversation is never interrupted by a transient meta miss.

AND TWO ANNOYANCES
------------------
The Gemini key is read from the environment and the input is not shown at all
when one is present. Same for LangSmith. Keys are typed once, in .env.

DEMO-MODE KEY ISOLATION
-----------------------
A second password (DEMO_PASSWORD) logs a visitor in as a demo user. Demo
users never inherit the deployer's Gemini key: their key env vars are cleared
at login and the credentials field is always shown, so they must paste their
own key to run a query. This keeps the public demo from spending the owner's
API quota.
"""

import base64
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# MUST run before anything reads a key.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="JazzCash Analytics", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

# Drop your image here. Missing file falls back to the gradient backdrop.
LOGIN_IMAGE = Path("assets/login.png")


def get_secret(name, default=""):
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LANGSMITH_API_KEY",
           "LANGSMITH_PROJECT", "LANGSMITH_TRACING", "LANGSMITH_ENDPOINT",
           "JAZZCASH_USE_API", "JAZZCASH_API_URL"):
    _v = get_secret(_k)
    if _v and not os.environ.get(_k):
        os.environ[_k] = _v

from src.api import client
from src.api.client import ApiError


@st.cache_data(show_spinner=False)
def _image_data_uri(path_str):
    """Read a local image once and return it as a CSS-ready data URI.

    Cached on the path string rather than the Path object — Streamlit hashes
    arguments, and a str hashes cleanly where a Path is fussier. Returns None
    when the file is absent, and every caller treats that as "use the
    gradient", so a missing asset is a different look rather than a crash.
    """
    path = Path(path_str)
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lstrip(".").lower()
    return f"data:image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix};base64,{encoded}"


# ==========================================================================
#  STYLE
# ==========================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg1:#0A0E1A; --bg2:#101725; --bg3:#070A12;
  --ink:#EEF1F7; --ink-2:#AAB3C5; --muted:#6B7488;

  --accent:#4C8DF6; --accent-hi:#7FB0FF; --accent-deep:#2C63C4;
  --teal:#39C0A8; --amber:#E0AE3D; --violet:#BE8FE8; --rose:#F0533F;

  --glass:rgba(255,255,255,.045);
  --glass-2:rgba(255,255,255,.07);
  --glass-border:rgba(255,255,255,.11);
  --glow:rgba(76,141,246,.26);

  --mono:'JetBrains Mono',ui-monospace,monospace;
  --display:'Sora',system-ui,sans-serif;
}

.stApp{
  background:
    radial-gradient(1100px 560px at 84% -12%, rgba(76,141,246,.15), transparent 58%),
    radial-gradient(900px 620px at -8% 108%, rgba(57,192,168,.07), transparent 60%),
    linear-gradient(155deg, var(--bg1), var(--bg2) 55%, var(--bg3));
  color:var(--ink);
  font-family:'Inter',system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.stApp p,.stApp li,.stApp label,.stMarkdown{ color:var(--ink); }

.block-container{
  max-width:1080px; padding-top:2.2rem; position:relative; z-index:1;
  animation:fadeUp .45s ease both;
}
@keyframes fadeUp{ from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:none;} }

h1,h2,h3{ color:var(--ink) !important; letter-spacing:-.02em; }
h1{ font-family:var(--display) !important; font-weight:600 !important; font-size:1.7rem !important; }
h3{ font-size:.98rem !important; font-weight:600 !important; }
code,pre,.jc-mono{ font-family:var(--mono) !important; }

hr{ border:none !important; height:1px !important;
    background:linear-gradient(90deg,transparent,rgba(76,141,246,.42),transparent) !important;
    margin:1.1rem 0 !important; }

/* ---------- gradient wordmark ---------- */
.jc-title{
  font-family:var(--display); font-weight:600; letter-spacing:-.01em;
  background:linear-gradient(92deg,#F2F6FF 0%,#9FC2FF 55%,#4C8DF6 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}

.jc-emblem{
  border-radius:18px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(150deg,rgba(76,141,246,.30),rgba(57,192,168,.12));
  border:1px solid rgba(127,176,255,.42);
  box-shadow:0 10px 30px rgba(76,141,246,.26);
  animation:glowPulse 4s ease-in-out infinite;
}
@keyframes glowPulse{
  0%,100%{ box-shadow:0 10px 26px rgba(76,141,246,.22); }
  50%{ box-shadow:0 14px 42px rgba(127,176,255,.40); }
}

/* ---------- login backdrop ---------- */
.jc-loginbg{
  position:fixed; inset:0; z-index:0;
  background-size:cover; background-position:center; background-repeat:no-repeat;
}

/* ---------- scrollbars ---------- */
*{ scrollbar-width:thin; scrollbar-color:#2E3648 transparent; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#2E3648; border-radius:8px;
  border:2px solid transparent; background-clip:padding-box; }
::-webkit-scrollbar-thumb:hover{ background:#454F68; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(16,23,37,.92),rgba(7,10,18,.96));
  border-right:1px solid var(--glass-border);
  backdrop-filter:blur(12px);
}
section[data-testid="stSidebar"] *{ color:var(--ink); }
section[data-testid="stSidebar"] h3{
  font-size:.66rem !important; font-weight:600 !important; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent-hi) !important;
  margin:.2rem 0 .5rem 0 !important;
}
section[data-testid="stSidebar"] hr{ margin:1rem 0 !important; opacity:.6; }
section[data-testid="stSidebar"] .stButton>button{
  background:var(--glass); color:var(--ink-2);
  border:1px solid var(--glass-border); border-radius:11px;
  text-align:left; padding:.52rem .8rem; font-size:.845rem; font-weight:500;
  width:100%; transition:all .16s ease;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  border-color:var(--accent-hi); background:rgba(76,141,246,.11);
  color:var(--ink); transform:translateX(2px);
  box-shadow:0 0 18px rgba(76,141,246,.20);
}

/* nav radio as pill list */
section[data-testid="stSidebar"] [role="radiogroup"]{ gap:.25rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label{
  background:var(--glass); border:1px solid var(--glass-border);
  border-radius:11px; padding:.5rem .75rem; margin:0;
  font-size:.86rem !important; font-weight:500;
  transition:all .16s ease; cursor:pointer;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  border-color:var(--accent-hi); background:rgba(76,141,246,.10);
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(135deg,rgba(76,141,246,.20),rgba(76,141,246,.07));
  border-color:var(--accent); box-shadow:inset 0 0 0 1px rgba(127,176,255,.22);
}

/* ---------- inputs ---------- */
.stTextInput input,.stNumberInput input,.stTextArea textarea{
  background:rgba(255,255,255,.055) !important;
  border:1px solid var(--glass-border) !important;
  color:var(--ink) !important; border-radius:12px !important;
  font-size:.88rem !important;
  transition:border-color .15s ease, box-shadow .15s ease;
}
.stTextInput input::placeholder{ color:#7A8399 !important; }
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color:var(--accent-hi) !important; box-shadow:0 0 0 3px var(--glow) !important; }

.stSelectbox div[data-baseweb="select"]>div:first-child{
  background:rgba(255,255,255,.055) !important;
  border:1px solid var(--glass-border) !important;
  color:var(--ink) !important; border-radius:12px !important; cursor:pointer !important; }
.stSelectbox svg{ fill:var(--accent-hi) !important; }
div[data-baseweb="popover"]{ z-index:9999 !important; }
ul[data-baseweb="menu"]{ background:#131A29 !important;
  border:1px solid var(--glass-border) !important; border-radius:12px !important; }
ul[data-baseweb="menu"] li{ color:var(--ink) !important; font-size:.86rem !important; }
ul[data-baseweb="menu"] li:hover{ background:rgba(76,141,246,.16) !important; }

[data-testid="stFileUploader"]{
  background:rgba(255,255,255,.03); border:1px dashed var(--glass-border);
  border-radius:13px; padding:.7rem; transition:all .16s ease; }
[data-testid="stFileUploader"]:hover{ border-color:var(--accent-hi); }
[data-testid="stFileUploader"] *{ color:var(--ink-2) !important; }

/* ---------- buttons, with the sweep ---------- */
.stButton>button[kind="primary"],.stFormSubmitButton>button{
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,var(--accent-hi),var(--accent-deep));
  color:#0A0E1A; border:none; font-weight:600; border-radius:12px;
  font-size:.88rem; letter-spacing:.2px;
  box-shadow:0 8px 22px rgba(76,141,246,.34);
  transition:transform .14s ease, box-shadow .14s ease, filter .14s ease;
}
.stButton>button[kind="primary"]::after,
.stFormSubmitButton>button::after{
  content:""; position:absolute; top:0; left:-130%; width:55%; height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,.65),transparent);
  transform:skewX(-20deg);
}
.stButton>button[kind="primary"]:hover::after,
.stFormSubmitButton>button:hover::after{ animation:shine .85s ease; }
@keyframes shine{ to{ left:140%; } }
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{
  transform:translateY(-1px); filter:brightness(1.06);
  box-shadow:0 14px 32px rgba(127,176,255,.44); }
.stButton>button[kind="primary"]:active,.stFormSubmitButton>button:active{
  transform:translateY(0) scale(.98);
  box-shadow:0 0 0 4px rgba(76,141,246,.28),0 6px 16px rgba(44,99,196,.5); }

/* ---------- login card ---------- */
[data-testid="stForm"]{
  background:rgba(10,14,26,.60); backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,.15); border-radius:20px;
  padding:1.5rem 1.5rem 1.2rem;
  box-shadow:0 30px 80px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.09);
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{
  background:var(--glass); backdrop-filter:blur(8px);
  border:1px solid var(--glass-border); border-radius:16px;
  padding:.7rem 1.05rem; margin-bottom:.55rem;
  box-shadow:0 10px 30px rgba(0,0,0,.28);
  animation:fadeUp .32s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  border-left:3px solid var(--accent);
  box-shadow:0 10px 30px rgba(0,0,0,.28),-8px 0 24px -12px rgba(76,141,246,.5); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:linear-gradient(180deg,rgba(76,141,246,.11),rgba(44,99,196,.06));
  border:1px solid rgba(127,176,255,.20); }
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"]{
  border:1px solid var(--accent-hi) !important; box-shadow:0 0 0 3px var(--glow); }

[data-testid="stChatInput"]{
  background:rgba(255,255,255,.055); border:1px solid var(--glass-border);
  border-radius:16px; backdrop-filter:blur(8px);
  transition:border-color .15s ease, box-shadow .15s ease; }
[data-testid="stChatInput"]:focus-within{
  border-color:var(--accent-hi); box-shadow:0 0 0 3px var(--glow); }
[data-testid="stChatInput"] textarea{ color:var(--ink) !important; }

/* ---------- metrics ---------- */
[data-testid="stMetric"]{
  background:var(--glass); backdrop-filter:blur(8px);
  border:1px solid var(--glass-border); border-radius:15px; padding:.9rem 1.1rem;
  box-shadow:0 10px 24px rgba(0,0,0,.26); }
[data-testid="stMetricValue"]{
  font-family:var(--mono) !important; color:var(--accent-hi) !important;
  font-size:1.4rem !important; font-weight:600 !important; }
[data-testid="stMetricLabel"]{
  color:var(--muted) !important; text-transform:uppercase;
  letter-spacing:.1em; font-size:.64rem !important; font-weight:600 !important; }

[data-testid="stExpander"]{
  background:rgba(255,255,255,.035); border:1px solid var(--glass-border);
  border-radius:13px; }
[data-testid="stExpander"] summary{ color:var(--accent-hi) !important; font-size:.83rem; }
[data-testid="stDataFrame"]{ border:1px solid var(--glass-border);
  border-radius:13px; overflow:hidden; }
[data-testid="stTabs"] button{ color:var(--muted) !important; font-size:.86rem !important; }
[data-testid="stTabs"] button[aria-selected="true"]{ color:var(--accent-hi) !important; }
[data-testid="stNotification"]{ border-radius:13px; }
#MainMenu,footer{ visibility:hidden; }
[data-testid="stHeader"]{ background:transparent; }

/* ---------- components ---------- */
.jc-badge{ display:inline-block; padding:.16rem .52rem; border-radius:6px;
  font-size:.665rem; font-weight:600; font-family:var(--mono); letter-spacing:.04em;
  margin-right:.35rem; border:1px solid var(--glass-border);
  background:var(--glass-2); color:var(--ink-2); }
.jc-badge.t1{ border-color:rgba(76,141,246,.45); background:rgba(76,141,246,.12); color:#9FC2FF; }
.jc-badge.t2{ border-color:rgba(224,174,61,.42); background:rgba(224,174,61,.11); color:#E8C468; }
.jc-badge.t3{ border-color:rgba(190,143,232,.42); background:rgba(190,143,232,.11); color:#D0AAF0; }
.jc-badge.refuse{ border-color:rgba(255,255,255,.16); color:var(--muted); }

.jc-meta{ color:var(--muted); font-size:.7rem; font-family:var(--mono);
  margin-top:.45rem; letter-spacing:.02em; }
.jc-expr{ background:rgba(0,0,0,.35); border:1px solid var(--glass-border);
  border-left:2px solid var(--violet); border-radius:9px; padding:.55rem .75rem;
  margin-top:.5rem; font-family:var(--mono); font-size:.76rem;
  color:#DCC6F2; overflow-x:auto; }
.jc-note{ color:var(--muted); font-size:.78rem; }

.jc-dot{ display:inline-block; width:7px; height:7px; border-radius:50%;
  margin-right:.4rem; vertical-align:middle; }
.jc-dot.ok{ background:var(--teal); box-shadow:0 0 8px rgba(57,192,168,.7); }
.jc-dot.warn{ background:var(--amber); box-shadow:0 0 8px rgba(224,174,61,.6); }
.jc-dot.bad{ background:var(--rose); box-shadow:0 0 8px rgba(240,83,63,.6); }
</style>
""", unsafe_allow_html=True)


def emblem(size=56, glyph="◆", font=1.5):
    return (f'<div class="jc-emblem" style="width:{size}px;height:{size}px;'
            f'margin:0 auto;font-size:{font}rem;color:#9FC2FF;">{glyph}</div>')


def page_header(title, subtitle=None):
    """Title, optional subtitle, rule.

    The subtitle is OPTIONAL and deliberately absent on the Ask page. A line
    of explanation above a chat box is read once and then occupies space
    forever — the interface should say what it is, not lecture about how it
    works.
    """
    sub = (f'<p style="color:var(--muted);margin:.15rem 0 0 0;font-size:.88rem;">'
           f'{subtitle}</p>') if subtitle else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.1rem;">'
        f'{emblem(40, "◆", 1.05)}'
        f'<h1 class="jc-title" style="margin:0;">{title}</h1></div>'
        f'{sub}<hr>', unsafe_allow_html=True)


TIER_OF = {
    "aggregate_metric": 1, "compare_groups": 1, "crosstab_rate": 1,
    "band_distribution": 1, "predict_default": 2, "predict_churn": 2,
    "score_population": 2, "simulate_loan": 2, "get_segment_profile": 2,
    "get_feature_importance": 2, "answer_freeform": 3,
}

PALETTE = ["#4C8DF6", "#39C0A8", "#E0AE3D", "#BE8FE8", "#F0533F", "#6E7681"]


# ==========================================================================
#  SESSION STATE  —  INITIALISED BEFORE ANYTHING CAN BAIL
# ==========================================================================
# These used to be created inside page_ask(), after the dataset check. When
# that check failed, the page returned early and they were never created — so
# the conversation disappeared. Identity has to exist before any branch runs.
#
# `meta` is also here now — see dataset_meta() below for why session_state
# beats @st.cache_data for this specific value.

if "thread" not in st.session_state:
    st.session_state.thread = str(uuid.uuid4())
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "auth" not in st.session_state:
    st.session_state.auth = False
if "is_demo" not in st.session_state:
    st.session_state.is_demo = False
if "meta" not in st.session_state:
    st.session_state.meta = None
    # On HF Spaces the container has no persistent disk on the free tier —
    # every restart wipes the pointer file and users would land on "Nothing
    # loaded" every visit. Auto-loading the reference makes the first
    # question work with zero clicks. On a paid persistent-volume plan the
    # pointer survives restarts and this call becomes a no-op (load_reference
    # just reloads the same parquets).
    try:
        client.load_reference()
        st.session_state.meta = client.get_dataset()
    except Exception:
        # First-visit auto-load is a convenience, not a requirement. If it
        # fails (missing API key on first boot, transient error) the sidebar
        # will show "Nothing loaded" with a manual button — same as before.
        pass 


# ==========================================================================
#  DATASET STATE
# ==========================================================================

def dataset_meta():
    """What is loaded, or None.

    HELD IN session_state, not in @st.cache_data. Streamlit reruns the whole
    script on every widget event — a chat submit, a sidebar click. The old
    cache_data with an 8-second TTL closed the common case, but a rerun that
    landed inside the "dark flash" between renders still slipped through and
    showed the "no dataset" warning mid-conversation. session_state survives
    reruns unconditionally and only re-fetches on an explicit clear.

    Returns None on the first call of a session, or after a refresh — that
    triggers exactly one API call. Every subsequent call in the session
    returns the cached value with no I/O.
    """
    if st.session_state.meta is None:
        try:
            st.session_state.meta = client.get_dataset()
        except ApiError:
            return None
    return st.session_state.meta


def refresh_dataset():
    """Drop the cached meta after something changed what is loaded.

    Called after an upload or a reference load. Without it, the sidebar would
    keep showing the previous dataset — a stale label is a worse lie than a
    slow one.
    """
    st.session_state.meta = None


# ==========================================================================
#  LOGIN
# ==========================================================================

def check_password():
    if st.session_state.get("auth"):
        return True

    image = _image_data_uri(str(LOGIN_IMAGE))
    if image:
        overlay = "linear-gradient(rgba(7,10,18,.72),rgba(7,10,18,.90))"
        backdrop = (f'<div class="jc-loginbg" style="background-image:'
                    f'{overlay},url(\'{image}\');"></div>')
    else:
        backdrop = ('<div class="jc-loginbg" style="background:'
                    'radial-gradient(900px 500px at 50% -8%,rgba(76,141,246,.16),'
                    'transparent 62%),#070A12;"></div>')

    st.markdown(
        f'<style>section[data-testid="stSidebar"]{{display:none !important;}}</style>'
        f'{backdrop}', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown("<div style='height:13vh'></div>", unsafe_allow_html=True)
        st.markdown(f"""
<div style="text-align:center;margin-bottom:1.3rem;position:relative;z-index:1;">
  {emblem(72, "◆", 2.0)}
  <h1 class="jc-title" style="font-size:2.2rem;margin:1rem 0 .15rem 0;">JazzCash Analytics</h1>
  <p style="color:#B9C3D6;font-size:.93rem;margin-top:.1rem;">Ask your loan book a question.</p>
</div>""", unsafe_allow_html=True)

        with st.form("login"):
            pw = st.text_input("Password", type="password",
                               label_visibility="collapsed", placeholder="Password")
            ok = st.form_submit_button("Sign in", use_container_width=True,
                                       type="primary")

        expected = get_secret("APP_PASSWORD")
        demo = get_secret("DEMO_PASSWORD")
        if ok:
            if not expected:
                st.error("APP_PASSWORD not found. Add `APP_PASSWORD=yourpassword` "
                         "to .env in the project root, then restart Streamlit.")
            elif pw and pw in (expected, demo):
                st.session_state.auth = True
                # A visitor who used the demo password is a demo user: they must
                # not spend the deployer's Gemini quota. Flag them, and strip the
                # pre-loaded key env vars so the graph can't fall back to them.
                st.session_state.is_demo = bool(demo) and (pw == demo)
                if st.session_state.is_demo:
                    for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                        os.environ.pop(_key, None)
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# ==========================================================================
#  SIDEBAR
# ==========================================================================

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1.1rem;">'
        f'{emblem(34, "◆", .95)}'
        f'<span class="jc-title" style="font-weight:600;font-size:1rem;">'
        f'JazzCash</span></div>', unsafe_allow_html=True)

    st.markdown("### Workspace")
    page = st.radio("nav", ["Ask", "Upload", "Explore", "Control room"],
                    label_visibility="collapsed")

    # ---- dataset -------------------------------------------------------
    st.markdown("---")
    st.markdown("### Dataset")
    _meta = dataset_meta()
    if _meta:
        st.markdown(
            f'<div style="background:var(--glass);border:1px solid var(--glass-border);'
            f'border-radius:12px;padding:.6rem .75rem;font-size:.78rem;line-height:1.75;">'
            f'<span class="jc-dot ok"></span>'
            f'<span style="color:var(--ink);font-weight:500;">{_meta["label"][:26]}</span><br>'
            f'<span class="jc-mono" style="color:var(--muted);font-size:.68rem;">'
            f'{_meta.get("fingerprint") or "no fingerprint"}</span><br>'
            f'<span style="color:var(--ink-2);font-size:.72rem;">'
            f'{sum(_meta["rows"].values()):,} rows · '
            f'{(_meta.get("as_of") or "—")[:10]}</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<span class="jc-dot warn"></span>'
                    '<span style="font-size:.8rem;color:var(--ink-2);">'
                    'Nothing loaded</span>', unsafe_allow_html=True)
        if st.button("Load reference extract"):
            with st.spinner("Loading…"):
                try:
                    client.load_reference()
                except ApiError as exc:
                    st.error(exc.detail)
                else:
                    refresh_dataset()
                    st.rerun()

    # ---- conversations -------------------------------------------------
    st.markdown("---")
    st.markdown("### Conversations")

    if st.button("＋  New thread"):
        st.session_state.thread = str(uuid.uuid4())
        st.session_state.msgs = []
        st.rerun()

    try:
        _threads = client.list_threads(limit=15)
    except ApiError:
        _threads = []

    for _t in _threads:
        _active = _t["thread_id"] == st.session_state.get("thread")
        _label = ("•  " if _active else "") + _t["title"][:28]
        if st.button(_label, key=f"thr-{_t['thread_id']}"):
            st.session_state.thread = _t["thread_id"]
            try:
                _turns = client.get_thread(_t["thread_id"], n=50)
            except ApiError:
                _turns = []
            st.session_state.msgs = [
                m
                for turn in _turns
                for m in (
                    {"role": "user", "content": turn["question"]},
                    {"role": "assistant", "content": turn["answer"],
                     "tool": turn.get("tool"), "replayed": True},
                )
            ]
            st.rerun()

    # ---- backend -------------------------------------------------------
    st.markdown("---")
    st.markdown("### Backend")

    _api_on = st.toggle("Use the API", value=client.use_api(),
                        help="On: this app is a client and the graph runs "
                             "behind FastAPI. Off: the graph runs in this "
                             "process.")
    os.environ["JAZZCASH_USE_API"] = "true" if _api_on else "false"

    if _api_on:
        if client.is_reachable():
            st.markdown('<span class="jc-dot ok"></span>'
                        '<span style="font-size:.79rem;color:var(--ink-2);">'
                        'Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="jc-dot bad"></span>'
                        '<span style="font-size:.79rem;color:var(--ink-2);">'
                        'Unreachable</span>', unsafe_allow_html=True)
            st.markdown('<p class="jc-note" style="margin-top:.4rem;">Start it with<br>'
                        '<code>python -m uvicorn src.api.main:app</code></p>',
                        unsafe_allow_html=True)
        with st.expander("API URL"):
            _url = st.text_input("URL", value=client.base_url(),
                                 key="api_url", label_visibility="collapsed")
            if _url:
                os.environ["JAZZCASH_API_URL"] = _url
    else:
        st.markdown('<span class="jc-dot warn"></span>'
                    '<span style="font-size:.79rem;color:var(--ink-2);">'
                    'In-process</span>', unsafe_allow_html=True)

    # ---- credentials ---------------------------------------------------
    # SHOWN WHEN MISSING, OR ALWAYS FOR DEMO USERS. A key already in .env needs
    # no input box — a password field pre-filled with a value you never change
    # is a prompt to do nothing. But a DEMO user's key env vars were cleared at
    # login, so the box always shows for them: they must paste their own key,
    # which is what keeps the deployer's Gemini quota safe on the public demo.
    #
    # Checks BOTH env var names because the Gemini client accepts either and
    # users may set only one.
    _is_demo = st.session_state.get("is_demo", False)
    _needs_key = (not _api_on) and (
        _is_demo or not (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
    )

    if _needs_key:
        st.markdown("---")
        st.markdown("### Credentials")
        if _is_demo:
            st.caption("🔐 Demo mode — paste your own Gemini key to run queries.")
        _gem = st.text_input("Gemini API key", type="password",
                             placeholder="Paste your Gemini key")
        if _gem:
            os.environ["GEMINI_API_KEY"] = _gem
            os.environ["GOOGLE_API_KEY"] = _gem
        elif _is_demo:
            # No key pasted yet this render — make sure nothing lingers.
            for _key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                os.environ.pop(_key, None)

    st.markdown("---")
    if st.button("Sign out"):
        st.session_state.auth = False
        st.session_state.is_demo = False
        st.rerun()


if not client.use_api() and not (
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
):
    st.warning("Add a Gemini API key in the sidebar, or switch on the API.")
    st.stop()


# ==========================================================================
#  MATPLOTLIB
# ==========================================================================

def _mpl():
    """matplotlib + seaborn themed to the page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "figure.facecolor": "#0E1420", "axes.facecolor": "#0E1420",
        "savefig.facecolor": "#0E1420", "text.color": "#AAB3C5",
        "axes.labelcolor": "#AAB3C5", "axes.edgecolor": "#212836",
        "xtick.color": "#6B7488", "ytick.color": "#6B7488",
        "grid.color": "#171E2C", "axes.grid": True, "grid.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "figure.dpi": 130,
    })
    return plt, sns


# ==========================================================================
#  UPLOAD
# ==========================================================================

def page_upload():
    page_header("Upload", "Three CSVs in, three feature tables out.")

    c1, c2, c3 = st.columns(3)
    f_cust = c1.file_uploader("customers.csv", type=["csv"])
    f_loan = c2.file_uploader("loans.csv", type=["csv"])
    f_txn = c3.file_uploader("transactions.csv", type=["csv"])

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("Run pipeline", type="primary",
                 disabled=not (f_cust and f_loan and f_txn)):
        with st.spinner("Cleaning, reconciling, building features…"):
            try:
                result = client.upload_dataset(
                    f_cust.getvalue(), f_loan.getvalue(), f_txn.getvalue()
                )
            except ApiError as exc:
                st.error(exc.detail)
                return

        st.session_state.pipeline_report = result
        refresh_dataset()          # the sidebar must not show the old file
        st.success("Loaded.")

    result = st.session_state.get("pipeline_report")
    if not result:
        st.markdown('<p class="jc-note">Or load the reference extract from the '
                    'sidebar.</p>', unsafe_allow_html=True)
        return

    st.markdown("---")
    report = result.get("report", {})
    dataset = result.get("dataset") or {}

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fingerprint", (result.get("fingerprint") or "—")[:10])
    k2.metric("As of", str(report.get("as_of", "—"))[:10])
    k3.metric("Loans", f"{dataset.get('rows', {}).get('default', 0):,}")
    k4.metric("Customers", f"{dataset.get('rows', {}).get('segment', 0):,}")

    t1, t2, t3 = st.tabs(["Reconciliation", "Drift", "Churn panel"])

    with t1:
        recon = report.get("reconciliation", {})
        st.markdown(f"**{'PASS' if recon.get('ok') else 'FLAGGED'}** — orphans "
                    f"and mismatches are kept, not dropped. The rows are real; "
                    f"what they cannot have is a trustworthy model score.")
        rows = [{"check": n, "status": "ok" if c.get("ok") else "flagged"}
                for n, c in recon.get("checks", {}).items()]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("**PSI against the training baseline.** Reports, never blocks.")
        for name, d in report.get("drift", {}).items():
            flagged = d.get("flagged", [])
            dot = "ok" if not flagged else "warn"
            st.markdown(f'<span class="jc-dot {dot}"></span><b>{name}</b> — '
                        f'{len(flagged) or "no"} column(s) flagged',
                        unsafe_allow_html=True)
            for col in flagged:
                e = d["columns"][col]
                st.markdown(f'<span class="jc-mono jc-note">&nbsp;&nbsp;&nbsp;{col} · '
                            f'PSI {e["psi"]:.4f} · {e.get("band","")}</span>',
                            unsafe_allow_html=True)

    with t3:
        st.json(report.get("churn_panel", {}))


# ==========================================================================
#  ASK
# ==========================================================================

def answer_chart(tool, result):
    """Draw the tool's own numbers. No model involved.

    Nothing is computed here — if a value is not in `result`, it is not on the
    chart, so the chart and the text cannot disagree.

    Thin groups are hatched rather than dropped. A rate on 300 loans and one on
    2,700 draw the same height, and the eye reads them as equally solid.
    """
    if not isinstance(result, dict):
        return None

    try:
        plt, sns = _mpl()
    except ImportError:
        return None

    thin = set(result.get("small_groups") or [])

    if tool == "aggregate_metric":
        values = result.get("result")
        if not isinstance(values, dict) or len(values) < 2:
            return None

        counts = result.get("n_per_group", {})
        labels = list(values.keys())

        fig, ax = plt.subplots(figsize=(8, 3.2))
        bars = ax.barh(labels, [values[k] for k in labels], color=PALETTE[0])
        for bar, label in zip(bars, labels):
            if label in thin:
                bar.set_hatch("///")
                bar.set_edgecolor("#070A12")
                bar.set_alpha(.65)

        overall = result.get("overall")
        if overall is not None:
            ax.axvline(overall, color=PALETTE[2], ls="--", lw=1)
            ax.text(overall, -.75, f" overall {overall:.1%}",
                    color=PALETTE[2], fontsize=7, va="top")

        for i, label in enumerate(labels):
            n = counts.get(label)
            ax.text(values[label], i, f"  {values[label]:.1%}"
                    + (f"  (n={n:,})" if n else ""),
                    va="center", fontsize=7.5, color="#AAB3C5")

        ax.invert_yaxis()
        ax.set_xlabel(f"{result.get('aggfunc','mean')} {result['metric']}")
        ax.margins(x=.22)
        return fig

    if tool == "compare_groups":
        rates = result.get("rates")
        if not isinstance(rates, dict) or len(rates) < 2:
            return None

        counts = result.get("n_per_group", {})
        labels = list(rates.keys())

        fig, ax = plt.subplots(figsize=(8, 2.6 + .3 * len(labels)))
        bars = ax.barh(labels, [rates[k] for k in labels], color=PALETTE[0])
        for bar, label in zip(bars, labels):
            if label in thin:
                bar.set_hatch("///")
                bar.set_edgecolor("#070A12")
                bar.set_alpha(.65)

        for i, label in enumerate(labels):
            n = counts.get(label)
            ax.text(rates[label], i, f"  {rates[label]:.1%}"
                    + (f"  (n={n:,})" if n else ""),
                    va="center", fontsize=7.5, color="#AAB3C5")

        ax.invert_yaxis()
        ax.set_xlabel(result["metric"])
        ax.margins(x=.22)

        p = result.get("p_value")
        if result.get("p_value_valid") and p is not None:
            ax.set_title(f"p = {p:.2g}" if p >= 1e-4 else "p < 0.0001",
                         fontsize=8, color="#6B7488", loc="left")
        elif result.get("p_value_note"):
            ax.set_title("not tested — a cell was too small to trust",
                         fontsize=8, color="#E0AE3D", loc="left")
        return fig

    if tool == "crosstab_rate":
        rates = result.get("rates")
        if not isinstance(rates, dict) or not rates:
            return None

        grid = pd.DataFrame(rates).T
        fig, ax = plt.subplots(figsize=(1.4 * len(grid.columns) + 3,
                                        .55 * len(grid) + 2))
        sns.heatmap(grid, cmap="mako", annot=True, fmt=".1%",
                    annot_kws={"size": 7.5}, linewidths=.5,
                    linecolor="#0E1420", cbar_kws={"shrink": .7}, ax=ax)
        ax.set_xlabel(result["col_by"]); ax.set_ylabel(result["row_by"])
        ax.tick_params(labelsize=7.5)
        ax.set_title("read across a row: does the effect hold in every column?",
                     fontsize=7.5, color="#6B7488", loc="left")
        return fig

    if tool == "band_distribution":
        counts = result.get("counts")
        if not isinstance(counts, dict) or len(counts) < 2:
            return None

        shares = result.get("shares", {})
        labels = list(counts.keys())

        fig, ax = plt.subplots(figsize=(8, 2.6 + .28 * len(labels)))
        ax.barh(labels, [counts[k] for k in labels], color=PALETTE[1])
        for i, label in enumerate(labels):
            s = shares.get(label)
            ax.text(counts[label], i,
                    f"  {counts[label]:,}" + (f"  ({s:.1%})" if s else ""),
                    va="center", fontsize=7.5, color="#AAB3C5")
        ax.invert_yaxis()
        ax.set_xlabel(result.get("unit", "rows"))
        ax.margins(x=.22)
        return fig

    return None


def render_answer(msg):
    tool, conf = msg.get("tool"), msg.get("confidence")
    tier = TIER_OF.get(tool)

    if tier:
        st.markdown(f'<span class="jc-badge t{tier}">TIER {tier}</span>'
                    f'<span class="jc-badge">{tool}</span>', unsafe_allow_html=True)
    elif tool == "out_of_scope":
        st.markdown('<span class="jc-badge refuse">OUT OF SCOPE</span>',
                    unsafe_allow_html=True)
    elif not msg.get("replayed"):
        st.markdown('<span class="jc-badge refuse">BLOCKED</span>',
                    unsafe_allow_html=True)

    st.markdown(msg["content"])

    fig = answer_chart(tool, msg.get("result"))
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(fig)
        if msg.get("thin"):
            st.markdown('<div class="jc-note">Hatched bars sit below 400 rows '
                        '— the rate is real but moves easily.</div>',
                        unsafe_allow_html=True)

    if msg.get("expression"):
        st.markdown('<div class="jc-note" style="margin-top:.55rem;">Generated at '
                    'runtime — nobody validated this before it ran:</div>'
                    f'<div class="jc-expr">{msg["expression"]}</div>',
                    unsafe_allow_html=True)

    meta = []
    if conf is not None:
        meta.append(f"confidence {conf:.2f}")
    if msg.get("retries"):
        meta.append(f"{msg['retries']} reroute(s)")
    if msg.get("latency"):
        meta.append(f"{msg['latency']:.1f}s")
    if meta:
        st.markdown(f'<div class="jc-meta">{" · ".join(meta)}</div>',
                    unsafe_allow_html=True)


def page_ask():
    """The chat.

    NO SUBTITLE AND NO STATUS BAR. The old version carried a line explaining
    that numbers are computed in Python, plus a strip listing file name,
    fingerprint, as-of date, row counts and transport. All of it is true and
    none of it belongs above a chat box — it is read once and then sits there
    forever, pushing the conversation down the page. The dataset it is talking
    to is named in the sidebar, permanently, which is where a status belongs.

    ONE RULE ABOUT THE "NO DATASET" WARNING. It fires only when the chat is
    EMPTY. If a conversation is already in progress and meta briefly returns
    None on a rerun (rare but possible), the warning would otherwise erase
    the whole thread from the screen and demand the user reload — which is
    the exact bug that showed up mid-conversation. An empty msgs list is the
    real "nothing to lose" signal.
    """
    meta = dataset_meta()
    page_header("Ask")

    if not meta:
        if not st.session_state.msgs:
            st.warning("No dataset loaded. Upload three files, or load the "
                       "reference extract from the sidebar.")
        return

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_answer(m)
            else:
                st.markdown(m["content"])

    q = st.chat_input("Enter your question…")
    if not q:
        return

    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        started = time.time()
        try:
            with st.spinner("Routing…"):
                out = client.ask(
                    q,
                    st.session_state.thread,
                    label=meta["label"],
                    fingerprint=meta.get("fingerprint"),
                )

            res = out.get("result")
            msg = {
                "role": "assistant",
                "content": out.get("answer") or "No answer was produced.",
                "tool": out.get("tool"),
                "confidence": out.get("confidence"),
                "retries": out.get("retries", 0),
                "expression": out.get("expression"),
                "result": res,
                "thin": bool(isinstance(res, dict) and res.get("small_groups")),
                "latency": out.get("latency_s", time.time() - started),
            }
        except ApiError as exc:
            msg = {"role": "assistant", "content": exc.detail, "tool": None,
                   "latency": time.time() - started}

        render_answer(msg)

    st.session_state.msgs.append(msg)


# ==========================================================================
#  EXPLORE
# ==========================================================================

def page_explore():
    if not dataset_meta():
        page_header("Explore")
        st.warning("No dataset loaded.")
        return

    page_header("Explore", "Direct pandas — no model involved.")

    try:
        plt, sns = _mpl()
    except ImportError:
        st.error("Charts need seaborn: `pip install seaborn matplotlib`")
        return

    c1, c2 = st.columns(2)
    table = c1.selectbox("Table", ["default", "churn", "segment"])
    kind = c2.selectbox("Chart", ["Distribution", "Rate by group", "Box by group",
                                  "Correlation heatmap", "Scatter", "Counts"])

    try:
        cols = client.get_columns(table)
    except ApiError as exc:
        st.error(exc.detail)
        return

    num_cols, cat_cols = cols["numeric"], cols["categorical"]
    fig = None
    spec = None

    if kind == "Distribution":
        col = st.selectbox("Column", num_cols)
        bins = st.slider("Bins", 10, 100, 40)
        spec = {"table": table, "chart_type": "distribution",
                "column": col, "bins": bins}

    elif kind == "Rate by group":
        a, b = st.columns(2)
        metric = a.selectbox("Metric", num_cols)
        group = b.selectbox("Group by", cat_cols)
        spec = {"table": table, "chart_type": "rate_by_group",
                "metric": metric, "group_by": group}

    elif kind == "Box by group":
        a, b = st.columns(2)
        metric = a.selectbox("Metric", num_cols)
        group = b.selectbox("Group by", cat_cols)
        spec = {"table": table, "chart_type": "box_by_group",
                "metric": metric, "group_by": group}

    elif kind == "Correlation heatmap":
        picked = st.multiselect("Columns", num_cols, default=num_cols[:8])
        if len(picked) >= 2:
            spec = {"table": table, "chart_type": "correlation",
                    "columns": picked}

    elif kind == "Scatter":
        a, b, c = st.columns(3)
        x = a.selectbox("X", num_cols)
        y = b.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1))
        hue = c.selectbox("Colour by", ["(none)"] + cat_cols)
        spec = {"table": table, "chart_type": "scatter", "x": x, "y": y,
                "hue": None if hue == "(none)" else hue}

    else:
        col = st.selectbox("Column", cat_cols)
        spec = {"table": table, "chart_type": "counts", "column": col}

    if spec is None:
        return

    try:
        data = client.explore(**spec)
    except ApiError as exc:
        st.error(exc.detail)
        return

    if hasattr(data, "model_dump"):
        data = data.model_dump()

    kind_key = data["chart_type"]

    if kind_key == "distribution":
        edges, counts = data["bin_edges"], data["counts"]
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(edges[:-1], counts, width=widths, align="edge",
               color=PALETTE[0], edgecolor="#0E1420", linewidth=.5)
        ax.set_xlabel(data.get("label", "")); ax.set_ylabel("count")

    elif kind_key == "rate_by_group":
        values = data["values"]
        labels = list(values.keys())
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.barh(labels, [values[k] for k in labels], color=PALETTE[0])
        ax.invert_yaxis()
        ax.set_xlabel(data.get("label", "")); ax.set_ylabel("")
        sizes = data.get("group_sizes") or {}
        if sizes:
            st.caption("Group sizes: " + " · ".join(
                f"{k}={v:,}" for k, v in sorted(sizes.items())))

    elif kind_key == "box_by_group":
        boxes = data["boxes"]
        stats = [
            {"label": b["group"], "whislo": b["min"], "q1": b["q1"],
             "med": b["median"], "q3": b["q3"], "whishi": b["max"],
             "fliers": b["outliers"]}
            for b in boxes
        ]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bxp(stats, showfliers=True, patch_artist=True,
               boxprops={"facecolor": PALETTE[0], "edgecolor": "#AAB3C5", "linewidth": .9},
               medianprops={"color": "#EEF1F7", "linewidth": 1.2},
               whiskerprops={"color": "#6B7488"}, capprops={"color": "#6B7488"},
               flierprops={"marker": ".", "markersize": 2,
                           "markerfacecolor": PALETTE[4], "markeredgecolor": "none"})
        ax.set_ylabel(data.get("label", ""))
        ax.tick_params(axis="x", rotation=25)

    elif kind_key == "correlation":
        order = data["matrix_columns"]
        grid = pd.DataFrame(data["matrix"]).reindex(index=order, columns=order)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(grid, cmap="mako", center=0, annot=True, fmt=".2f",
                    annot_kws={"size": 7}, linewidths=.4, linecolor="#0E1420",
                    cbar_kws={"shrink": .7}, ax=ax)
        ax.tick_params(labelsize=7)

    elif kind_key == "scatter":
        points = data["points"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        hues = {p.get("hue") for p in points if p.get("hue")}
        if hues:
            for i, h in enumerate(sorted(hues)):
                subset = [p for p in points if p.get("hue") == h]
                ax.scatter([p["x"] for p in subset], [p["y"] for p in subset],
                           s=9, alpha=.55, label=h,
                           color=PALETTE[i % len(PALETTE)], edgecolors="none")
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.scatter([p["x"] for p in points], [p["y"] for p in points],
                       s=9, alpha=.55, color=PALETTE[0], edgecolors="none")
        ax.set_xlabel(spec["x"]); ax.set_ylabel(spec["y"])

    else:
        values = data["values"]
        shares = data.get("shares") or {}
        labels = list(values.keys())
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.barh(labels, [values[k] for k in labels], color=PALETTE[1])
        for i, label in enumerate(labels):
            s = shares.get(label)
            ax.text(values[label], i,
                    f"  {int(values[label]):,}" + (f"  ({s:.1%})" if s else ""),
                    va="center", fontsize=7.5, color="#AAB3C5")
        ax.invert_yaxis()
        ax.set_xlabel("rows"); ax.set_ylabel("")
        ax.margins(x=.18)

    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if data.get("note"):
        st.caption(data["note"])


# ==========================================================================
#  CONTROL ROOM
# ==========================================================================

GUARDRAILS = {
    "validate": ("Parameter validator",
                 "Rejects a parameter the router proposed that the data does "
                 "not contain. Retryable — the error names the valid values."),
    "scope": ("Scope guardrail",
              "Blocks injection and empty questions before any model call."),
    "confidence": ("Confidence gate",
                   "Stops a route the router was not sure enough about."),
    "sandbox": ("Tier 3 sandbox",
                "Refuses generated code reaching outside pandas."),
    "iteration_cap": ("Iteration cap",
                      "Ends the Tier 3 retry loop after two attempts."),
}


def page_control():
    page_header("Control room",
                "Five control points, each with a counter. A count stuck at "
                "zero is how you find a guardrail that never worked.")

    window = st.selectbox("Window", ["All time", "Last 7 days", "Last 24 hours"])
    days = {"All time": None, "Last 7 days": 7, "Last 24 hours": 1}[window]

    try:
        payload = client.counters(days)
    except ApiError as exc:
        st.error(exc.detail)
        return

    counts = payload["counts"]

    st.metric("Guardrail events", f"{payload['total']:,}")
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    for key, (name, why) in GUARDRAILS.items():
        actions = counts.get(key, {})
        fired = sum(actions.values())
        colour = "var(--muted)" if fired == 0 else "var(--accent-hi)"
        breakdown = " · ".join(f"{a}: {n}" for a, n in actions.items()) or "never fired"
        st.markdown(
            f'<div style="background:var(--glass);backdrop-filter:blur(8px);'
            f'border:1px solid var(--glass-border);border-radius:14px;'
            f'padding:.8rem 1.05rem;margin-bottom:.55rem;'
            f'box-shadow:0 8px 22px rgba(0,0,0,.22);">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<b style="font-size:.9rem;">{name}</b>'
            f'<span class="jc-mono" style="color:{colour};font-size:1.15rem;'
            f'font-weight:600;">{fired}</span></div>'
            f'<div class="jc-note" style="margin-top:.3rem;">{why}</div>'
            f'<div class="jc-mono jc-note" style="margin-top:.35rem;color:var(--ink-2);">'
            f'{breakdown}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Evaluation set")
    path = Path("src/eval/eval_results.json")
    if path.exists():
        import json
        rows = json.loads(path.read_text(encoding="utf-8"))
        passed = sum(1 for r in rows if not r["failures"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Cases", len(rows))
        c2.metric("Passing", f"{passed}/{len(rows)}")
        c3.metric("Median latency",
                  f"{pd.Series([r['latency_s'] for r in rows]).median():.1f}s")
        st.dataframe(
            [{"case": r["id"], "tool": r["actual_tool"] or "—",
              "branch": r["actual_branch"], "conf": r["confidence"],
              "s": r["latency_s"], "pass": "" if r["failures"] else "ok"}
             for r in rows],
            use_container_width=True, hide_index=True, height=340)
        st.caption("Half the set expects a refusal, a clarification, or a block.")
    else:
        st.markdown('<p class="jc-note">Run <code>python -m src.eval.run_cases</code> '
                    'to populate this.</p>', unsafe_allow_html=True)


# ==========================================================================
{"Ask": page_ask, "Upload": page_upload,
 "Explore": page_explore, "Control room": page_control}[page]() 