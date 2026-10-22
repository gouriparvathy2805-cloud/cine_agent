"""app_v5.py - CineAgent v5.   Run:  streamlit run app_v5.py
Needs in the same folder: agent_v3.py, evaluate_v3.py, v4_helpers.py, tmdb.py, movies.csv
Optional: a free TMDB key (posters, trailers, streaming) in .streamlit/secrets.toml as TMDB_API_KEY = "..."
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import os
import random
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import evaluate_v3
import tmdb as TM
import v4_helpers as H
from agent_v3 import CRITERIA_LABEL, DEFAULT_W, MovieAgent, Profile

GOLD = "#f5c518"
st.set_page_config(page_title="CineAgent v5", page_icon="🎬", layout="wide")
ss = st.session_state

# ------------------------------------------------------------------ theme (dark / light switch)
THEMES = {
    "dark": dict(bg="#000000", bg2="#050506", card="#0a0a0b", card2="#0f0f11", line="#1f1f23", ink="#ececf1",
                 mute="#9a9aa5", accent="#f5c518", abg="#2b2208", abg2="#14110a", chip="#161618", chipline="#232327",
                 chat="#f4f4f6", ong="#000000", chatink="#000000", ok="#1d3b24", okink="#7be495", bad="#3b1d1d", badink="#ff8f8f"),
    "light": dict(bg="#f6f5f0", bg2="#ffffff", card="#ffffff", card2="#faf9f4", line="#e2dfd2", ink="#2b2d3a",
                  mute="#6a6d7a", accent="#a87a00", abg="#fff0bd", abg2="#fff8e0", chip="#f0eee4", chipline="#d9d6c8",
                  chat="#ffffff", ong="#3a2c00", chatink="#2b2d3a", ok="#dff5e3", okink="#1c6b33", bad="#fde2e2", badink="#a12828"),
}
LIGHT = bool(ss.get("light", False))
P = THEMES["light" if LIGHT else "dark"]

ROOT = ":root{" + ";".join(f"--{k}:{v}" for k, v in P.items()) + ";--gold:#f5c518}"
CSS = """
.stApp,[data-testid=stHeader],[data-testid=stBottom],[data-testid=stBottom]>div{background:var(--bg)!important}
.stApp,[data-testid=stMarkdownContainer],[data-testid=stWidgetLabel] *,h1,h2,h3,h4,h5,h6,.stApp label,.stApp li,
[data-testid=stMetricValue],[data-testid=stMetricLabel] *,[data-testid=stTable] *{color:var(--ink)!important}
[data-testid=stCaptionContainer],[data-testid=stCaptionContainer] *{color:var(--mute)!important}
[data-testid=stSidebar]{background:var(--bg2)!important;border-right:1px solid var(--line)}
.block-container{padding-top:2.2rem;max-width:1400px}
/* ---- sidebar navigation: big, readable ---- */
.brand{font-weight:800;letter-spacing:3px;font-size:24px;color:var(--ink);padding:4px 0 14px}
.brand span{background:var(--gold);color:var(--ong);border-radius:7px;padding:2px 9px;font-size:14px;letter-spacing:0;margin-left:8px}
[data-testid=stSidebar] [role=radiogroup]{gap:4px}
[data-testid=stSidebar] [role=radiogroup] label{display:flex!important;flex-direction:row!important;align-items:center!important;
 gap:12px;padding:10px 14px;border-radius:12px;width:100%;cursor:pointer;border:1px solid transparent;transition:background .15s,border-color .15s}
[data-testid=stSidebar] [role=radiogroup] label>div{margin:0!important}
[data-testid=stSidebar] [role=radiogroup] label>div:first-of-type{flex:none}
[data-testid=stSidebar] [role=radiogroup] label>div:last-of-type{flex:1;min-width:0}
[data-testid=stSidebar] [role=radiogroup] label p{font-size:16.5px!important;font-weight:700!important;line-height:1.25;margin:0!important;color:var(--ink)!important}
[data-testid=stSidebar] [role=radiogroup] label:hover{background:var(--card2);border-color:var(--line)}
[data-testid=stSidebar] [role=radiogroup] label:has(input:checked){background:var(--abg);border-color:var(--accent)}
[data-testid=stSidebar] [role=radiogroup] label:has(input:checked) p{color:var(--accent)!important;font-weight:800!important}
/* ---- hero / cards ---- */
.hero{border:1px solid var(--line);border-radius:20px;padding:34px 38px;margin-bottom:18px;
 background:radial-gradient(circle at 90% 0%,var(--abg) 0%,var(--bg) 62%)}
.hero h1{font-size:44px;margin:0 0 6px;letter-spacing:-1px}
.hero p{color:var(--mute)!important;font-size:16px;margin:0;max-width:660px}
.card{position:relative;display:flex;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:6px}
.card{height:262px}
.card .pz{width:150px;aspect-ratio:auto;align-self:stretch}
.card .bd{display:flex;flex-direction:column;overflow:hidden;padding:12px 14px}
.card .ttl{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card .meta{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:50px;flex:none}
.card .chips{white-space:nowrap;overflow:hidden;flex:none}
.card .why{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;flex:none}
.card .stream{flex-wrap:nowrap;overflow:hidden;white-space:nowrap;flex:none}
.card .svc{flex:none}
.card .svc.off{flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis}
.card .trail{margin-top:auto;align-self:flex-start;flex:none}
.card:hover{border-color:var(--accent)}
.pz{position:relative;width:112px;flex:none;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;font-size:34px;overflow:hidden}
.pz[data-big]{width:100%}
.pz img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.bd{padding:12px 14px 10px;min-width:0;flex:1}
.ttl{font-weight:700;font-size:16px;color:var(--ink);line-height:1.2;padding-right:50px}
.meta{color:var(--mute);font-size:12px;margin:4px 0 8px}
.chip{display:inline-block;background:var(--chip);border:1px solid var(--chipline);color:var(--ink);border-radius:20px;padding:2px 9px;font-size:11px;margin:0 4px 4px 0}
.chip.x{background:var(--abg);border-color:var(--accent);color:var(--accent)}
.chip.g{border-color:var(--accent);color:var(--accent)}
.chip.lk{background:var(--ok);border-color:var(--okink);color:var(--okink)}
.chip.dk{background:var(--bad);border-color:var(--badink);color:var(--badink)}
.card .ring{position:absolute;top:10px;right:10px}
.ring{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center}
.ring span{background:var(--card);color:var(--ink);width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
.why{color:var(--mute);font-size:12px;margin:4px 0 6px}
.stream{margin:4px 0 6px;display:flex;flex-wrap:wrap;align-items:center;gap:4px}
.stream .lbl{font-size:10px;letter-spacing:1px;color:var(--mute);margin-right:2px}
.svc{display:inline-flex;align-items:center;gap:5px;background:var(--card2);border:1px solid var(--chipline);color:var(--ink);border-radius:8px;padding:2px 8px 2px 3px;font-size:11px;font-weight:600}
.svc img{width:18px;height:18px;border-radius:5px}
.svc.off{padding:2px 8px;color:var(--mute);font-weight:500}
.trail{display:inline-block;margin-top:2px;background:var(--gold);color:var(--ong)!important;border-radius:8px;padding:3px 11px;font-size:12px;font-weight:700;text-decoration:none!important}
.trail:hover{filter:brightness(1.08)}
.under{margin:0 0 10px;color:var(--mute);font-size:12px;letter-spacing:.5px;text-transform:uppercase}
.tag{font-style:italic;color:var(--mute);margin:-4px 0 10px}
.sec{font-size:12px;letter-spacing:1px;text-transform:uppercase;color:var(--accent);margin:16px 0 4px;font-weight:700}
.lead{font-size:16px;line-height:1.65}
.strip{display:flex;gap:12px;overflow-x:auto;padding-bottom:6px}
.mini{width:110px;flex:none;font-size:12px}
.mini .pz{width:110px;border-radius:10px}
.mini b{display:block;margin-top:5px;line-height:1.2}
.mini span{color:var(--mute)}
.cpick{display:flex;gap:12px;align-items:center;background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:10px 12px;margin:8px 0}
.cpick:hover{border-color:var(--accent)}
.cpick .pz{width:54px;border-radius:9px;font-size:22px}
.cpick .num{position:absolute;top:2px;left:2px;z-index:2;background:var(--gold);color:var(--ong);border-radius:50%;width:18px;height:18px;font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center}
.cpick .ttl{font-size:15px;padding-right:0}
.note{color:var(--mute);font-size:12px;font-style:italic;margin-top:6px}
.follow{margin-top:10px;padding:8px 12px;border-left:3px solid var(--accent);background:var(--abg2);border-radius:6px;font-size:13px}
.cmphead{display:flex;gap:14px}.cmphead>div{flex:1;font-weight:700;font-size:14px}.cmphead .pz{width:100%;max-width:150px;border-radius:10px;margin-bottom:6px}
/* ---- native widgets (work in both themes) ---- */
.stButton>button,.stDownloadButton>button,[data-testid=stBaseButton-secondary],[data-testid=stBaseLinkButton-secondary]{background:var(--card)!important;border:1px solid var(--chipline)!important;border-radius:10px}
.stButton>button p,.stDownloadButton>button p,[data-testid=stBaseButton-secondary] p{color:var(--ink)!important}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent)!important}
.stButton>button:hover p{color:var(--accent)!important}
[data-testid=stMetric]{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 16px}
[data-testid=stExpander],[data-testid=stExpander] details{background:var(--card2)!important;border:1px solid var(--line)!important;border-radius:12px}
[data-testid=stExpander] summary,[data-testid=stExpander] summary *{color:var(--ink)!important}
[data-testid=stVerticalBlockBorderWrapper]{border-color:var(--line)!important}
[data-baseweb=select]>div,[data-baseweb=input]>div,[data-baseweb=textarea],[data-testid=stNumberInput] input,[data-testid=stTimeInput] input{background:var(--card)!important;border-color:var(--chipline)!important}
[data-baseweb=select] *,[data-baseweb=input] input,[data-testid=stTimeInput] input{color:var(--ink)!important}
[data-baseweb=popover] [data-baseweb=menu],[data-baseweb=popover] ul{background:var(--card)!important}
[data-baseweb=popover] li,[data-baseweb=popover] li *{color:var(--ink)!important}
[data-baseweb=tag]{background:var(--abg)!important}[data-baseweb=tag] *{color:var(--accent)!important}
hr{border-color:var(--line)!important}
[data-testid=stExpander] summary{background:var(--card2)!important}
[data-testid=stExpander] summary svg{fill:var(--ink)!important;color:var(--ink)!important}
[data-baseweb=select] div,[data-baseweb=input],[data-baseweb=input] div,[data-baseweb=base-input],[data-baseweb=base-input] input,[data-baseweb=textarea] textarea{background:var(--card)!important;color:var(--ink)!important}
[data-baseweb=select] svg{fill:var(--mute)!important}
[data-baseweb=select] input{-webkit-text-fill-color:var(--ink)!important}
[data-baseweb=tag],[data-baseweb=tag] div{background:var(--abg)!important}
[data-testid=stTooltipIcon] svg,[data-testid=stTooltipHoverTarget] svg{color:var(--mute)!important;fill:var(--mute)!important}
[data-testid=stTickBarMin],[data-testid=stTickBarMax]{color:var(--mute)!important}
[data-testid=stSliderThumbValue]{color:var(--accent)!important}
[data-testid=stCode] pre,[data-testid=stCode] code,[data-testid=stCode]{background:var(--card2)!important;color:var(--ink)!important}
[data-baseweb=checkbox] span:first-of-type{border-color:var(--mute)!important}
[data-testid=stToolbar] *,[data-testid=stHeader] button{color:var(--ink)!important}
[data-baseweb=tab-list]{background:transparent!important}
[data-baseweb=tab]{color:var(--mute)!important}
/* ---- gold primary buttons / added state ---- */
[data-testid=stBaseButton-primary],button[kind=primary]{background:var(--gold)!important;color:var(--ong)!important;border:1px solid var(--gold)!important;font-weight:700!important}
[data-testid=stBaseButton-primary]:hover{filter:brightness(1.08)}
[data-testid=stBaseButton-primary] p,button[kind=primary] p{color:var(--ong)!important;font-weight:700}
[data-testid=stBaseButton-secondary]:disabled{background:var(--abg2)!important;border:1px solid var(--accent)!important;opacity:1!important}
[data-testid=stBaseButton-secondary]:disabled p{color:var(--accent)!important;font-weight:700}
/* ---- chat ---- */
[data-testid=stChatInput]{background:var(--chat)!important;border:1px solid var(--chipline)!important;border-radius:18px!important}
[data-testid=stChatInput]:focus-within{border-color:var(--gold)!important;box-shadow:0 0 0 2px #f5c51866}
[data-testid=stChatInput] div,[data-testid=stChatInput] div[data-baseweb]{background:transparent!important}
[data-testid=stChatInput] textarea,[data-testid=stBottom] textarea{color:var(--chatink)!important;-webkit-text-fill-color:var(--chatink)!important;caret-color:var(--chatink)!important;background:transparent!important;font-size:15px}
[data-testid=stChatInput] textarea::placeholder{color:#6b6b75!important;-webkit-text-fill-color:#6b6b75!important;opacity:1}
[data-testid=stChatInput] button{background:var(--gold)!important;color:var(--ong)!important;border-radius:12px!important}
[data-testid=stChatInput] button svg{fill:var(--ong)!important;color:var(--ong)!important}
.chathead{display:flex;align-items:center;gap:14px;border:1px solid var(--line);border-radius:18px;padding:18px 22px;margin-bottom:14px;background:radial-gradient(circle at 100% 0%,var(--abg) 0%,var(--bg2) 60%)}
.chathead .av{width:46px;height:46px;border-radius:14px;background:var(--gold);color:var(--ong);display:flex;align-items:center;justify-content:center;font-size:24px}
.chathead h2{margin:0;font-size:22px}.chathead p{margin:2px 0 0;color:var(--mute)!important;font-size:13px}
[data-testid=stChatMessage]{background:var(--card)!important;border:1px solid var(--line);border-radius:18px;padding:14px 18px;margin-bottom:10px}
[data-testid=stChatMessage]:has([data-testid=stChatMessageAvatarUser]){background:var(--abg2)!important;border-color:var(--accent)}
"""
LIGHT_ONLY = """
[data-testid=stDataFrame]{filter:invert(1) hue-rotate(180deg)}   /* the grid is drawn on a canvas in the dark base theme */
[data-baseweb=tab][aria-selected=true]{color:var(--accent)!important}
"""
st.markdown(f"<style>{ROOT}{CSS}{LIGHT_ONLY if LIGHT else ''}</style>", unsafe_allow_html=True)

EMOJI = {"Action": "💥", "Comedy": "😂", "Drama": "🎭", "Horror": "👻", "Romance": "💕", "Sci-Fi": "🚀",
         "Thriller": "🔪", "Animation": "🎨", "Crime": "🕵️", "Fantasy": "🧙", "War": "🪖", "Adventure": "🧭",
         "Musical": "🎶", "Mystery": "🔍", "Family": "👨‍👩‍👧", "Sport": "🏁", "Biography": "📖", "Music": "🎵"}
PAGES = ["Discover", "Chat", "Marathon", "Watchlist", "My taste", "Explore", "Under the hood"]
ICONS = dict(zip(PAGES, ["🎯", "💬", "⏱️", "🔖", "❤️", "🧭", "🔧"]))
REGIONS = {"IN": "India", "US": "United States", "GB": "United Kingdom", "CA": "Canada", "AU": "Australia",
           "AE": "UAE", "SG": "Singapore", "DE": "Germany", "FR": "France", "JP": "Japan", "KR": "South Korea"}

HERE = Path(__file__).parent
WATCH_FILE = HERE / "watchlist_v5.csv"          # separate file, so the old app's watchlist is untouched
WATCH_COLS = ["Title", "Status", "My rating", "Notes", "Added"]
STATUSES = ["To watch", "Watching", "Watched"]


@st.cache_data
def load(mtime: float):
    return H.ensure_runtime(pd.read_csv(HERE / "movies.csv"))


try:
    df, RT_EST = load((HERE / "movies.csv").stat().st_mtime)
except FileNotFoundError:
    st.error("movies.csv was not found. Put it in the same folder as app_v5.py and rerun.")
    st.stop()
RUNTIME = dict(zip(df.title, df.runtime))
YEAR = dict(zip(df.title, df.year))


# ------------------------------------------------------------------ TMDB (posters / trailers / streaming)
def _secret_key() -> str:
    try:
        return str(st.secrets.get("TMDB_API_KEY", ""))
    except Exception:
        return ""


@st.cache_resource
def get_tmdb(key: str) -> TM.TMDB:
    return TM.TMDB(key, HERE / "tmdb_cache_v5.json")


T = get_tmdb((ss.get("tmdb_in") or _secret_key() or os.environ.get("TMDB_API_KEY", "")).strip())


@st.cache_resource
def get_wiki() -> TM.Wiki:
    return TM.Wiki(HERE / "wiki_cache_v5.json")


W = get_wiki()


def M(title: str) -> dict:
    return T.lookup(title, YEAR.get(title, ""))


def ensure_meta(titles):
    pairs = list(dict.fromkeys((t, YEAR.get(t, "")) for t in titles))
    if T.enabled and any(f"{t}|{y}" not in T.cache for t, y in pairs):
        with st.spinner("Fetching posters and streaming info…"):
            T.prefetch(pairs)


def pz(title, m, size="w342", extra=""):
    h = int(hashlib.md5(title.encode()).hexdigest()[:6], 16)
    g = df.genres[df.title == title].iloc[0].split("|")[0] if (df.title == title).any() else ""
    url = TM.poster_url(m, size)
    inner = f'<img src="{url}" alt="" loading="lazy">' if url else EMOJI.get(g, "🎬")
    bg = (f"linear-gradient(135deg,hsl({h % 360},60%,84%),hsl({(h + 40) % 360},50%,68%))" if LIGHT
          else f"linear-gradient(135deg,hsl({h % 360},45%,24%),#000)")
    return f'<div class="pz" style="background:{bg}">{extra}{inner}</div>'


def stream_html(title, m, detailed=False, limit=3):
    if not T.enabled:
        return ""
    region = ss.get("region", "IN")
    s = TM.streaming(m, region)
    svc = lambda x: (f'<span class="svc">' + (f'<img src="{TM.logo_url(x["logo"])}" alt="">' if x["logo"] else "") +
                     f'{html.escape(x["name"])}</span>')
    out = []
    if s["stream"]:
        out.append('<span class="lbl">STREAM</span>' + "".join(svc(x) for x in s["stream"][:None if detailed else limit]))
        if not detailed and len(s["stream"]) > limit:
            out.append(f'<span class="svc off">+{len(s["stream"]) - limit}</span>')
    elif s["rent"] or s["buy"]:
        names = ", ".join(dict.fromkeys(x["name"] for x in s["rent"] + s["buy"]))
        out.append(f'<span class="svc off">Rent / buy only: {html.escape(names[:46])}</span>')
    elif m.get("found"):
        out.append(f'<span class="svc off">Not streaming in {region} right now</span>')
    if detailed and (s["rent"] or s["buy"]) and s["stream"]:
        names = ", ".join(dict.fromkeys(x["name"] for x in s["rent"] + s["buy"]))
        out.append(f'<span class="svc off">Rent / buy: {html.escape(names)}</span>')
    return f'<div class="stream">{"".join(out)}</div>' if out else ""


def trailer_a(title, m):
    return (f'<a class="trail" href="{html.escape(TM.trailer_url(title, YEAR.get(title, ""), m))}" '
            f'target="_blank" rel="noopener">▶ Watch trailer</a>')


# ------------------------------------------------------------------ state
def fresh_agent():
    ss.agent = MovieAgent(df)
    ss.events = []                  # every like/dislike: title, label, profile used, learning state before/after


if "agent" not in ss:
    fresh_agent()
    ss.watch = None
    ss.chat, ss.chat_prof, ss.last_picks = [], Profile(), []
    ss.flags = dict(lsa=True, related=True, mmr=True, relax=True)
    ss.applied, ss.loaded = None, False


def A() -> MovieAgent:
    return ss.agent


agent: MovieAgent = A()
agent.flags.update(ss.flags)

FORM_KEYS = ["g", "m", "l", "av", "fav", "act", "r", "rt", "ctx", "era_on", "era", "persona"]
for _k in FORM_KEYS:           # keep form values alive while other pages are showing
    if _k in ss:
        ss[_k] = ss[_k]


def empty_watch():
    return pd.DataFrame({"Title": pd.Series(dtype=object), "Status": pd.Series(dtype=object),
                         "My rating": pd.Series(dtype=float), "Notes": pd.Series(dtype=object),
                         "Added": pd.Series(dtype=object)})


def load_watch():
    try:
        if WATCH_FILE.exists():
            w = pd.read_csv(WATCH_FILE)
            w = w[w["Title"].isin(set(df.title)) & w["Status"].isin(STATUSES)][WATCH_COLS].copy()
            w["My rating"] = pd.to_numeric(w["My rating"], errors="coerce").fillna(0.0)
            w["Notes"] = w["Notes"].fillna("").astype(str)
            return w.reset_index(drop=True)
    except Exception:
        pass
    return empty_watch()


def save_watch():
    try:
        ss.watch.to_csv(WATCH_FILE, index=False)
    except OSError:
        pass


if ss.watch is None:
    ss.watch, ss.wl_ver = load_watch(), 0


# ---- likes / dislikes with full undo --------------------------------------------------------
def liked():
    return [e["title"] for e in ss.events if e["label"] > 0]


def disliked():
    return [e["title"] for e in ss.events if e["label"] < 0]


def label_of(title):
    for e in reversed(ss.events):
        if e["title"] == title:
            return e["label"]
    return 0


def _snap(a):
    return dict(w=dict(a.weights), n=a.n_fb, gb={g: list(v) for g, v in a.genre_beta.items()})


def _restore(a, s):
    a.weights, a.n_fb, a.genre_beta = dict(s["w"]), s["n"], {g: list(v) for g, v in s["gb"].items()}


def drop_event(title):
    """Remove a title's feedback and rewind what the agent learned from it (later feedback is replayed)."""
    a = A()
    idx = max((i for i, e in enumerate(ss.events) if e["title"] == title), default=None)
    if idx is None:
        return False
    tail = ss.events[idx + 1:]
    _restore(a, ss.events[idx]["before"])
    ss.events = ss.events[:idx]
    for e in tail:
        e["before"] = _snap(a)
        a.feedback(e["prof"].copy(), e["title"], e["label"])
        e["after"] = dict(a.weights)
        ss.events.append(e)
    return True


def feedback(title, label):
    a = A()
    cur = label_of(title)
    if cur == label:
        st.toast(f"You already {'liked' if label > 0 else 'disliked'} {title}")
        return
    if cur:                                   # changed their mind: undo the old one first
        drop_event(title)
    p, _ = build_profile()
    p0, before = p.copy(), _snap(a)
    a.feedback(p, title, label)
    ss.events.append(dict(title=title, label=label, prof=p0, before=before, after=dict(a.weights)))
    st.toast(f"{'Liked' if label > 0 else 'Disliked'} {title}. You can undo it from My taste or the sidebar.")


def undo_title(title):
    cur = label_of(title)
    if drop_event(title):
        st.toast(f"Undid your {'like' if cur > 0 else 'dislike'} of {title}")


def undo_last():
    if ss.events:
        undo_title(ss.events[-1]["title"])


def reset_learning():
    fresh_agent()


# ------------------------------------------------------------------ profile + callbacks
def build_profile():
    a = ss.get("applied") or {}
    era = tuple(a.get("era", (1990, int(df.year.max())))) if a.get("era_on") else ()
    watched = list(ss.watch.loc[ss.watch["Status"] == "Watched", "Title"])
    p = Profile(genres=list(a.get("g", [])), moods=list(a.get("m", [])), languages=list(a.get("l", [])),
                min_rating=float(a.get("r", 0)), max_runtime=int(a.get("rt", 0)), era=era,
                avoid_genres=list(a.get("av", [])), favourites=list(a.get("fav", [])) + liked(),
                actors=list(a.get("act", [])), context="" if a.get("ctx", "Any") == "Any" else a["ctx"],
                seen=watched, disliked=disliked())
    return A().validate(p)


def apply_profile():
    ss.applied = {k: (list(ss[k]) if isinstance(ss[k], list) else ss[k]) for k in FORM_KEYS[:-1] if k in ss}
    ss.loaded = True


def add_watch(title):
    if title in set(ss.watch["Title"]):
        st.toast(f"{title} is already on your watchlist")
        return
    row = pd.DataFrame([{"Title": title, "Status": "To watch", "My rating": 0.0, "Notes": "",
                         "Added": dt.date.today().isoformat()}])
    ss.watch = row if ss.watch.empty else pd.concat([ss.watch, row], ignore_index=True)
    save_watch()
    st.toast(f"Added {title} to your watchlist")


def goto(p):
    ss.page = p


def clear_profile():
    for k, v in dict(g=[], m=[], l=[], av=[], fav=[], act=[], r=0.0, rt=0, ctx="Any", era_on=False).items():
        ss[k] = v
    ss.applied, ss.loaded = None, False


def load_persona():
    name = ss.get("persona")
    if name in evaluate_v3.PERSONAS:
        clear_profile()
        p, _ = A().validate(evaluate_v3.PERSONAS[name])
        ss.g, ss.m, ss.l, ss.av = p.genres, p.moods, p.languages, p.avoid_genres
        ss.r, ss.rt, ss.ctx = float(p.min_rating), int(p.max_runtime), p.context or "Any"
        if p.era:
            ss.era_on, ss.era = True, tuple(p.era)


def set_flag(k):
    ss.flags[k] = ss[f"fl_{k}"]


# ------------------------------------------------------------------ charts
def style(c, h=260):
    return (c.properties(height=h).configure(background="rgba(0,0,0,0)").configure_view(stroke=None)
            .configure_axis(labelColor=P["mute"], titleColor=P["mute"], gridColor=P["line"], domainColor=P["chipline"],
                            tickColor=P["chipline"])
            .configure_legend(labelColor=P["ink"], titleColor=P["mute"]))


def show(c, h=260):
    st.altair_chart(style(c, h), theme=None)


def heat(d, x, y, z, h=260, fmt=".0f"):
    base = alt.Chart(d).encode(x=alt.X(f"{x}:N", title=None, sort=None), y=alt.Y(f"{y}:N", title=None, sort=None))
    lo = P["bg"] if not LIGHT else "#ffffff"
    rect = base.mark_rect().encode(color=alt.Color(f"{z}:Q", scale=alt.Scale(range=[lo, GOLD]), legend=None),
                                   tooltip=[x, y, alt.Tooltip(f"{z}:Q", format=fmt)])
    mid = float(d[z].max()) * 0.6
    txt = base.mark_text(fontSize=11).encode(text=alt.Text(f"{z}:Q", format=fmt),
                                             color=alt.condition(f"datum.{z} > {mid}", alt.value(P["ong"]), alt.value(P["ink"])))
    show(rect + txt, h)


def bar(d, x, y, h=240, color=GOLD, sort="-x"):
    show(alt.Chart(d).mark_bar(color=color, cornerRadiusEnd=3).encode(
        x=alt.X(f"{x}:Q", title=None), y=alt.Y(f"{y}:N", sort=sort, title=None), tooltip=[y, x]), h)


# ------------------------------------------------------------------ shared widgets
def add_btn(title, key):
    if title in set(ss.watch["Title"]):
        st.button("✓ On your list", key=f"{key}_on", disabled=True, use_container_width=True)
    else:
        st.button("＋ Add to list", key=f"{key}_off", on_click=add_watch, args=(title,), type="primary",
                  use_container_width=True, help="Add to your watchlist")


def like_btns(title, key):
    cur = label_of(title)
    b1, b2 = st.columns(2)
    if cur > 0:
        b1.button("↩ Undo like", key=f"{key}_ul", on_click=undo_title, args=(title,), use_container_width=True)
    else:
        b1.button("👍 Like", key=f"{key}_l", on_click=feedback, args=(title, 1), use_container_width=True)
    if cur < 0:
        b2.button("↩ Undo dislike", key=f"{key}_ud", on_click=undo_title, args=(title,), use_container_width=True)
    else:
        b2.button("👎 Dislike", key=f"{key}_d", on_click=feedback, args=(title, -1), use_container_width=True)


def card(r, pfx):
    m = M(r.title)
    pct = int(min(100, r.utility))
    lab = label_of(r.title)
    chips = ('<span class="chip lk">👍 Liked</span>' if lab > 0 else '<span class="chip dk">👎 Disliked</span>' if lab < 0 else "")
    if r.explore:
        chips += '<span class="chip x">✨ exploration</span>'
    chips += "".join(f'<span class="chip">{html.escape(g)}</span>' for g in r.genres[:3])
    why = " · ".join(html.escape(x) for x in r.reasons[:2])
    st.markdown(
        f'<div class="card">{pz(r.title, m)}<div class="ring" style="background:conic-gradient({GOLD} {pct}%,{P["line"]} 0)">'
        f'<span>{pct}%</span></div><div class="bd"><div class="ttl">{html.escape(r.title)}</div>'
        f'<div class="meta">{r.year} · {html.escape(r.language)} · ⭐ {r.rating:.1f} · {r.runtime} min</div>'
        f'<div class="chips">{chips}</div><div class="why">{why}</div>{stream_html(r.title, m, limit=2)}{trailer_a(r.title, m)}</div></div>',
        unsafe_allow_html=True)
    like_btns(r.title, f"{pfx}{r.idx}")
    add_btn(r.title, f"{pfx}w{r.idx}")


def contrib_chart(r):
    d = pd.DataFrame({"Criterion": [CRITERIA_LABEL[k] for k in r.contrib], "Points": list(r.contrib.values())})
    bar(d, "Points", "Criterion", 170)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">CINEAGENT<span>v5</span></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed",
                    format_func=lambda p: f"{ICONS[p]}  {p}")
    st.divider()
    st.toggle("☀️ Light theme", key="light", help="Switch between the black theme and a light one.")
    st.selectbox("Streaming region", list(REGIONS), key="region", format_func=lambda c: f"{REGIONS[c]} ({c})",
                 help="Which country's streaming services to show.")
    st.button(f"↩ Undo last: {ss.events[-1]['title'][:20]}" if ss.events else "↩ Undo last like/dislike",
              on_click=undo_last, disabled=not ss.events, use_container_width=True)
    with st.expander("⚙️ Engine"):
        st.slider("Number of picks", 3, 10, 6, key="k")
        st.checkbox("Allow exploration pick", True, key="expl")
        st.slider("Diversity of picks", 0.0, 0.6, 0.3, 0.05, key="div",
                  help="0 = just the highest scores. Higher = avoid near-duplicate picks (MMR).")
        st.button("Reset learning", on_click=reset_learning)
    with st.expander("🎞️ Posters & streaming"):
        st.text_input("TMDB API key", type="password", key="tmdb_in",
                      help="Free at themoviedb.org → Settings → API. Either the v3 key or the v4 read token works.")
        st.caption(f"Status: {T.status}" if T.key else "No key yet: cards use colour tiles and the trailer button opens a "
                   "YouTube search.")
    st.caption(f"{len(liked())} likes · {len(disliked())} dislikes · {len(ss.watch)} on watchlist")
    if RT_EST:
        st.caption("Runtimes are estimates (no runtime column in movies.csv).")
    if T.key:
        st.caption("Posters and film data: TMDB. Streaming data: JustWatch. This product uses the TMDB API but is "
                   "not endorsed or certified by TMDB.")

profile, warns = build_profile()
if ss.get("loaded"):
    for w in warns:
        st.sidebar.warning(w)


def _state_key():
    return (repr(profile), ss.k, ss.expl, ss.div, tuple(sorted((k, round(v, 9)) for k, v in agent.weights.items())),
            tuple(sorted(agent.flags.items())), agent.n_fb)


def get_res():
    key = _state_key()
    c = ss.get("_res")
    if c and c[0] == key:
        return c[1]
    r = agent.recommend(profile, ss.k, ss.div, ss.expl)
    ss._res = (key, r)
    return r


def get_cf():
    key = _state_key()
    c = ss.get("_cf")
    if c and c[0] == key:
        return c[1]
    cf = agent.counterfactuals(profile, ss.k, ss.div)
    ss._cf = (key, cf)
    return cf


def mini_strip(idx_list, caption_fn):
    ensure_meta([df.title[j] for j in idx_list])
    items = "".join(f'<div class="mini">{pz(df.title[j], M(df.title[j]), "w185")}<b>{html.escape(df.title[j])}</b>'
                    f'<span>{caption_fn(j)}</span></div>' for j in idx_list)
    st.markdown(f'<div class="strip">{items}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ page: Discover (Home + Recommend)
def page_discover():
    res = get_res()
    st.markdown('<div class="hero"><h1>Tonight, sorted.</h1><p>A utility-based agent that learns your taste, '
                'explains every pick, and shows the poster, the trailer and where to stream it.</p></div>',
                unsafe_allow_html=True)
    filled = sum(bool(x) for x in [profile.genres, profile.moods, profile.languages, profile.min_rating,
                                   profile.max_runtime, profile.favourites, profile.actors, profile.context])
    k = st.columns(5)
    k[0].metric("Movies", len(df))
    k[1].metric("Languages", df.language.nunique())
    k[2].metric("Avg rating", f"{df.rating.mean():.2f}")
    k[3].metric("Feedback given", len(ss.events))
    k[4].metric("Profile completeness", f"{filled}/8")
    if not T.key:
        st.info("Add a free TMDB key in the sidebar (🎞️ Posters & streaming) to get real posters, direct trailers and "
                "streaming services on every card.")

    with st.expander("🎛️ Your taste profile", expanded=not ss.get("loaded")):
        left, right = st.columns(2)
        with left:
            st.multiselect("Genres", agent.all_genres, key="g")
            st.multiselect("Moods", agent.all_moods, key="m", format_func=str.title)
            st.multiselect("Languages", agent.all_langs, key="l")
            st.multiselect("Avoid genres", agent.all_genres, key="av")
        with right:
            st.selectbox("Watching with", ["Any", "Alone", "Partner", "Friends", "Family"], key="ctx")
            st.slider("Minimum rating", 0.0, 9.5, 0.0, 0.5, key="r")
            st.slider("Max runtime (min, 0 = any)", 0, 240, 0, 10, key="rt")
            if st.checkbox("Limit release years", key="era_on"):
                st.slider("Years", int(df.year.min()), int(df.year.max()), (1990, int(df.year.max())), key="era")
        st.multiselect("Favourite movies (optional)", sorted(df.title), key="fav")
        st.multiselect("Favourite actors (optional)", agent.all_actors, key="act")
        _go, _clr = st.columns([5, 1])
        _go.button("🎬 Load recommendations", type="primary", on_click=apply_profile, use_container_width=True)
        _clr.button("Clear", on_click=clear_profile, use_container_width=True, help="Reset the form")

    st.subheader("Top picks for you" if ss.get("loaded") else "Best all-round picks (set your taste above to personalise)")
    if ss.get("loaded"):
        for n in res.notices:
            st.info(n)
    if not res.picks:
        st.error("No movies qualify. Loosen a filter.")
        return
    c1, c2, _ = st.columns([2, 2, 3])
    sort = c1.selectbox("Sort by", ["Best match", "Rating", "Year (newest)", "Runtime (shortest)"])
    view = c2.radio("View", ["Cards", "Table"], horizontal=True)
    keyf = {"Best match": lambda r: 0, "Rating": lambda r: -r.rating, "Year (newest)": lambda r: -r.year,
            "Runtime (shortest)": lambda r: r.runtime}[sort]
    picks = sorted(res.picks, key=keyf)
    ensure_meta([r.title for r in picks])
    if view == "Cards":
        cols = st.columns(3)
        for n, r in enumerate(picks):
            with cols[n % 3]:
                card(r, "r")
                with st.expander("Why this pick?"):
                    contrib_chart(r)
                    for x in r.reasons:
                        st.write("• " + x)
    else:
        region = ss.get("region", "IN")
        t = pd.DataFrame([{"Title": r.title, "Match %": round(r.utility, 1), "Year": r.year, "Language": r.language,
                           "Rating": r.rating, "Runtime": r.runtime, "Genres": ", ".join(r.genres),
                           f"Streaming ({region})": ", ".join(x["name"] for x in TM.streaming(M(r.title), region)["stream"]) or "-",
                           "Exploration": r.explore} for r in picks])
        st.dataframe(t, hide_index=True)
        st.download_button("Download picks (CSV)", t.to_csv(index=False), "picks.csv", "text/csv")

    if ss.events and liked():
        last = liked()[-1]
        sim = H.similar_to(agent, agent.idx_of([last]), 6, exclude=agent.idx_of(liked() + disliked()))
        st.subheader(f"Because you liked {last}")
        mini_strip(sim, lambda j: f"{df.year[j]} · ⭐ {df.rating[j]:.1f}")
    cf = get_cf()
    if cf:
        with st.expander("🔀 What if…? (counterfactual explanations)"):
            for d, new in cf:
                st.write(f"If you **{d}**, you'd also get: " + ", ".join(new))
    with st.expander("📤 Share these picks"):
        lines = [f"{i + 1}. {r.title} ({r.year}, {r.language}) - match {r.utility:.0f}%" for i, r in enumerate(res.picks)]
        body = "My CineAgent picks:\n\n" + "\n".join(lines)
        st.code(body, language=None)
        st.link_button("✉️ Open in my email app", "mailto:?subject=" + quote("My movie picks") + "&body=" + quote(body))
    st.subheader("Jump to")
    q = st.columns(4)
    for c, p in zip(q, ["Chat", "Marathon", "Watchlist", "My taste"]):
        c.button(f"{ICONS[p]} {p}", key=f"go{p}", on_click=goto, args=(p,))


# ------------------------------------------------------------------ page: Chat
def queue(s):
    ss.pending = s


def _pick_dict(x):
    return dict(title=x.title, year=x.year, language=x.language, rating=x.rating, runtime=x.runtime,
                genres=list(x.genres), why=(x.reasons[0] if x.reasons else "solid all-round pick"))


def handle_chat(msg):
    ss.chat.append({"role": "user", "text": msg})
    p, chips, reset = H.parse_query(msg, agent, ss.chat_prof)
    ss.chat_prof = p
    q, _ = agent.validate(p)
    r = agent.recommend(q, 5, ss.div, False)
    ss.last_picks = [x.title for x in r.picks]
    if reset:
        ss.chat.append({"role": "assistant", "text": "Cleared. What are you in the mood for?"})
        return
    ensure_meta(ss.last_picks)
    m = {"role": "assistant", "chips": chips, "picks": [_pick_dict(x) for x in r.picks],
         "notices": list(r.notices), "follow": ""}
    nq = agent.next_question(q)
    if nq:
        m["follow"] = f"{nq['question']} Try: {', '.join(o.title() for o in nq['options'][:4])}"
    ss.chat.append(m)


def chat_bubble(m):
    if isinstance(m, tuple):
        m = {"role": m[0], "text": m[1]}
    role = m["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🎬"):
        if m.get("text"):
            st.markdown(m["text"])
        if "picks" in m:
            if m["chips"]:
                st.markdown('<div class="under">Here is what I understood</div>' +
                            "".join(f'<span class="chip g">{html.escape(c)}</span>' for c in m["chips"]),
                            unsafe_allow_html=True)
            else:
                st.markdown("No specific preference caught, so these are the all-round best matches.")
            rows = ""
            for n, x in enumerate(m["picks"], 1):
                meta = T.peek(x["title"], x["year"])
                gen = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in x["genres"][:3])
                rows += (f'<div class="cpick">{pz(x["title"], meta, "w185", f"<span class=num>{n}</span>")}<div>'
                         f'<div class="ttl">{html.escape(x["title"])}</div>'
                         f'<div class="meta" style="margin:2px 0 4px">{x["year"]} · {html.escape(x["language"])} · '
                         f'⭐ {x["rating"]:.1f} · {x["runtime"]} min</div>{gen}'
                         f'<div class="why">{html.escape(x["why"])}</div>{stream_html(x["title"], meta)}'
                         f'{trailer_a(x["title"], meta)}</div></div>')
            st.markdown(rows, unsafe_allow_html=True)
            for n in m["notices"]:
                st.markdown(f'<div class="note">{html.escape(n)}</div>', unsafe_allow_html=True)
            if m["follow"]:
                st.markdown(f'<div class="follow">💡 {html.escape(m["follow"])}</div>', unsafe_allow_html=True)


def page_chat():
    st.markdown('<div class="chathead"><div class="av">🎬</div><div><h2>Chat with the agent</h2>'
                '<p>Plain English in, ranked picks out. Preferences build up across messages. '
                'Say “reset” to start over.</p></div></div>', unsafe_allow_html=True)
    if not ss.chat:
        ss.chat.append({"role": "assistant", "text": "Hi! Tell me what you feel like watching, e.g. *a feel-good "
                        "Malayalam movie under 2.5 hours* or *dark thriller from the 2010s, no horror*."})
    top = st.columns([6, 1])
    with top[0].expander("What I currently understand"):
        known = {k: v for k, v in vars(ss.chat_prof).items() if v}
        if known:
            st.markdown("".join(
                f'<span class="chip g">{html.escape(k.replace("_", " "))}: '
                f'{html.escape(", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))}</span>'
                for k, v in known.items()), unsafe_allow_html=True)
        else:
            st.caption("Nothing yet.")
    if top[1].button("🗑 Clear", help="Clear the chat and what I understood"):
        ss.chat, ss.chat_prof, ss.last_picks = [], Profile(), []
        st.rerun()
    sugg = ["Scary movie from the 2010s", "Feel-good Malayalam under 2.5 hours", "Date night, highly rated",
            "Something like Inception", "Family movie, short"]
    st.markdown('<div class="under">Try asking</div>', unsafe_allow_html=True)
    for c, sg in zip(st.columns(len(sugg)), sugg):
        c.button(sg, key=f"sg{sg}", on_click=queue, args=(sg,), use_container_width=True)
    msg = st.chat_input("Describe what you want to watch…") or ss.pop("pending", None)
    if msg:
        with st.spinner("Thinking…"):
            handle_chat(msg)
    for m in ss.chat:
        chat_bubble(m)
    if ss.last_picks:
        st.divider()
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        t = c1.selectbox("Add one of the latest picks to your watchlist", ss.last_picks)
        with c2:
            add_btn(t, "chat_add")


# ------------------------------------------------------------------ page: Marathon
def page_marathon():
    st.title("Marathon planner")
    st.caption("Beam search for the best set of movies inside a time budget, rewarding utility and variety.")
    c = st.columns(5)
    hrs = c[0].slider("Time (hours)", 2.0, 12.0, 6.0, 0.5)
    films = c[1].slider("Max films", 2, 6, 4)
    gap = c[2].slider("Break (min)", 0, 30, 10, 5)
    start = c[3].time_input("Start at", dt.time(18, 0))
    order = c[4].selectbox("Order", ["Plan order", "Shortest first", "Best match first", "Oldest first"])
    if st.button("Plan my marathon"):
        ss.marathon = agent.marathon(profile, int(hrs * 60), films, gap)
        ss.marathon_gap, ss.marathon_start, ss.marathon_order = gap, start, order
    m = ss.get("marathon")
    if not m:
        return
    mv = list(m["movies"])
    if not mv:
        st.warning("Nothing fits that budget with your filters.")
        return
    o = ss.marathon_order
    if o == "Shortest first":
        mv.sort(key=lambda r: r.runtime)
    elif o == "Best match first":
        mv.sort(key=lambda r: -r.utility)
    elif o == "Oldest first":
        mv.sort(key=lambda r: r.year)
    ensure_meta([r.title for r in mv])
    t = dt.datetime.combine(dt.date.today(), ss.marathon_start)
    rows = []
    for r in mv:
        s, e = t, t + dt.timedelta(minutes=r.runtime)
        rows.append({"Title": r.title, "Start": s, "End": e, "Runtime": r.runtime, "Match": round(r.utility, 1),
                     "Genres": ", ".join(r.genres), "Year": r.year})
        t = e + dt.timedelta(minutes=ss.marathon_gap)
    d = pd.DataFrame(rows)
    k = st.columns(5)
    k[0].metric("Films", len(d))
    k[1].metric("Watch time", f"{m['total_minutes'] // 60}h {m['total_minutes'] % 60}m")
    k[2].metric("Budget used", f"{100 * m['total_minutes'] / m['budget']:.0f}%")
    k[3].metric("Avg rating", f"{np.mean([r.rating for r in mv]):.1f}")
    k[4].metric("Ends at", d.End.iloc[-1].strftime("%H:%M"))
    st.markdown('<div class="strip">' + "".join(
        f'<div class="mini">{pz(r.title, M(r.title), "w185")}<b>{html.escape(r.title)}</b><span>{r.runtime} min</span>'
        f'{stream_html(r.title, M(r.title))}</div>' for r in mv) + "</div>", unsafe_allow_html=True)
    show(alt.Chart(d).mark_bar(color=GOLD, cornerRadius=5, height=26).encode(
        x=alt.X("Start:T", title=None, axis=alt.Axis(format="%H:%M")), x2="End:T",
        y=alt.Y("Title:N", sort=None, title=None), tooltip=["Title", "Genres", "Runtime", "Match"]), 70 + 40 * len(d))
    out = d.assign(Start=d.Start.dt.strftime("%H:%M"), End=d.End.dt.strftime("%H:%M"))
    st.dataframe(out, hide_index=True)
    st.download_button("Download plan (CSV)", out.to_csv(index=False), "marathon.csv", "text/csv")


# ------------------------------------------------------------------ page: Watchlist
def add_many():
    for t in ss.get("wl_add", []):
        add_watch(t)
    ss.wl_add = []


def remove_watch():
    ss.watch = ss.watch[~ss.watch["Title"].isin(ss.get("wl_rm", []))].reset_index(drop=True)
    ss.wl_rm = []
    ss.wl_ver += 1
    save_watch()


def apply_wl_edits():
    ed = ss.get(f"wl_editor_{ss.wl_ver}") or {}
    for i, edits in ed.get("edited_rows", {}).items():
        for col, val in edits.items():
            ss.watch.loc[int(i), col] = val
    save_watch()


def page_watchlist():
    st.title("Watchlist")
    c1, c2 = st.columns([4, 1])
    on = set(ss.watch["Title"])
    c1.multiselect("Add movies", [t for t in sorted(df.title) if t not in on], key="wl_add")
    c2.write("")
    c2.button("Add", on_click=add_many)
    w = ss.watch
    k = st.columns(4)
    k[0].metric("To watch", int((w["Status"] == "To watch").sum()))
    k[1].metric("Watching", int((w["Status"] == "Watching").sum()))
    k[2].metric("Watched", int((w["Status"] == "Watched").sum()))
    hrs = sum(RUNTIME[t] for t in w.loc[w["Status"] == "To watch", "Title"]) / 60
    k[3].metric("Queue length", f"{hrs:.1f} h")
    if w.empty:
        st.info("Your watchlist is empty. Add movies above or press ＋ Add to list on any card.")
        return
    st.data_editor(
        w, hide_index=True, key=f"wl_editor_{ss.wl_ver}", disabled=["Title", "Added"], on_change=apply_wl_edits,
        column_config={"Status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
                       "My rating": st.column_config.NumberColumn(min_value=0, max_value=5, step=1, format="%.0f ⭐")})
    a, c = st.columns(2)
    with a:
        st.multiselect("Remove", list(ss.watch["Title"]), key="wl_rm")
        st.button("Remove selected", on_click=remove_watch)
    with c:
        st.caption("Titles marked Watched stop being recommended.")
        st.download_button("Export watchlist (CSV)", ss.watch.to_csv(index=False), "watchlist.csv", "text/csv")
    q = ss.watch[ss.watch["Status"] == "To watch"]
    if len(q):
        ensure_meta(list(q["Title"]))
        st.subheader(f"Where to watch your queue ({REGIONS[ss.get('region', 'IN')]})")
        if T.enabled:
            region = ss.get("region", "IN")
            st.dataframe(pd.DataFrame({"Title": list(q["Title"]),
                                       "Streaming": [", ".join(x["name"] for x in TM.streaming(M(t), region)["stream"]) or "-"
                                                     for t in q["Title"]]}), hide_index=True)
        else:
            st.caption("Add a TMDB key in the sidebar to see streaming services here.")
    if len(q) and st.button("🎲 Pick one for me tonight"):
        ids = agent.idx_of(list(q["Title"]))
        wts = [agent.score(i, profile)[0] + 1 for i in ids]
        i = random.choices(ids, weights=wts)[0]
        st.success(f"Tonight: **{df.title[i]}** ({df.year[i]}, {df.language[i]}, {df.runtime[i]} min)")
    loved = agent.idx_of(list(ss.watch.loc[(ss.watch["Status"] == "Watched") & (ss.watch["My rating"] >= 4), "Title"]))
    if loved:
        st.subheader("Because you loved your watched titles")
        sim = H.similar_to(agent, loved, 5, exclude=agent.idx_of(list(ss.watch["Title"])))
        mini_strip(sim, lambda j: f"{df.language[j]} · ⭐ {df.rating[j]:.1f}")


# ------------------------------------------------------------------ page: My taste (likes & dislikes)
def taste_row(t, key):
    m = M(t)
    i = agent.title_idx[t.lower()]
    st.markdown(
        f'<div class="cpick">{pz(t, m, "w185")}<div><div class="ttl">{html.escape(t)}</div>'
        f'<div class="meta" style="margin:2px 0 4px">{df.year[i]} · {html.escape(df.language[i])} · ⭐ {df.rating[i]:.1f}</div>'
        f'{"".join(f"<span class=chip>{html.escape(g)}</span>" for g in agent.df.genres_l[i][:3])}'
        f'{stream_html(t, m)}{trailer_a(t, m)}</div></div>', unsafe_allow_html=True)
    lab = label_of(t)
    b = st.columns(2)
    b[0].button("↩ Undo", key=f"{key}_u", on_click=undo_title, args=(t,), use_container_width=True,
                help="Remove this and rewind what the agent learned from it")
    b[1].button("⇄ Move to " + ("dislikes" if lab > 0 else "likes"), key=f"{key}_m", on_click=feedback,
                args=(t, -lab), use_container_width=True)


def page_taste():
    st.title("My taste")
    L, D = liked(), disliked()
    k = st.columns(4)
    k[0].metric("Liked", len(L))
    k[1].metric("Disliked", len(D))
    gl = Counter(g for t in L for g in agent.df.genres_l[agent.title_idx[t.lower()]])
    gd = Counter(g for t in D for g in agent.df.genres_l[agent.title_idx[t.lower()]])
    k[2].metric("Leaning towards", gl.most_common(1)[0][0] if gl else "-")
    k[3].metric("Leaning away from", gd.most_common(1)[0][0] if gd else "-")
    b = st.columns([2, 2, 4])
    b[0].button("↩ Undo last", on_click=undo_last, disabled=not ss.events, use_container_width=True, key="tk_undo")
    b[1].button("Clear everything", on_click=reset_learning, disabled=not ss.events, use_container_width=True,
                help="Removes all likes and dislikes and resets what the agent learned", key="tk_reset")
    if not ss.events:
        st.info("Nothing here yet. Press 👍 or 👎 on any recommendation and it will appear on this page. "
                "Every like or dislike can be undone.")
        return
    ensure_meta(L + D)
    left, right = st.columns(2)
    with left:
        st.subheader(f"👍 Liked ({len(L)})")
        if gl:
            st.markdown("".join(f'<span class="chip g">{html.escape(g)} · {n}</span>' for g, n in gl.most_common(5)),
                        unsafe_allow_html=True)
        for t in reversed(L):
            taste_row(t, f"tl{agent.title_idx[t.lower()]}")
        if not L:
            st.caption("No likes yet.")
    with right:
        st.subheader(f"👎 Disliked ({len(D)})")
        if gd:
            st.markdown("".join(f'<span class="chip">{html.escape(g)} · {n}</span>' for g, n in gd.most_common(5)),
                        unsafe_allow_html=True)
        for t in reversed(D):
            taste_row(t, f"td{agent.title_idx[t.lower()]}")
        if not D:
            st.caption("No dislikes yet.")
    with st.expander("History (oldest first)"):
        st.dataframe(pd.DataFrame([{"Step": n + 1, "Title": e["title"], "Feedback": "like" if e["label"] > 0 else "dislike"}
                                   for n, e in enumerate(ss.events)]), hide_index=True)


# ------------------------------------------------------------------ page: Explore (Inspect + Compare)
def long_text(i, m, plot=""):
    d = agent.df
    t, n = d.title[i], len(d)
    out = []
    story = m.get("overview") or d.description[i]
    out.append(("The story", story))
    if m.get("overview") and d.description[i] and d.description[i].lower().rstrip(".") not in story.lower():
        out.append(("In one line", d.description[i]))
    rank = int((d.rating > d.rating[i]).sum()) + 1
    same = d[d.language == d.language[i]]
    lrank = int((same.rating > d.rating[i]).sum()) + 1
    glance = (f"{t} is a {d.year[i]} {d.language[i]}-language {' / '.join(d.genres_l[i]).lower()} film running about "
              f"{m.get('runtime') or d.runtime[i]} minutes. Rated {d.rating[i]:.1f}, it sits at #{rank} of {n} titles in this "
              f"catalogue and #{lrank} among its {len(same)} {d.language[i]} films.")
    if d.moods_l[i]:
        glance += f" The mood is {', '.join(d.moods_l[i][:-1]) + (' and ' if len(d.moods_l[i]) > 1 else '') + d.moods_l[i][-1]}."
    out.append(("At a glance", glance + (("\n\n" + plot) if plot else "")))
    cast = m.get("cast") or d.cast_l[i]
    people = f"Starring {', '.join(cast)}." + (f" Directed by {m['director']}." if m.get("director") else "")
    out.append(("Cast & crew", people))
    if ss.get("loaded"):
        U, contrib, _ = agent.score(i, profile)
        top = sorted(contrib, key=contrib.get, reverse=True)[:2]
        out.append(("How it fits you", f"With your current taste profile it scores {U:.0f}%, mostly on "
                                       + " and ".join(CRITERIA_LABEL[k].lower() for k in top) + "."))
    return out


def page_explore():
    st.title("Explore")
    d = agent.df
    t = st.selectbox("Choose a movie (type to search)", sorted(d.title), key="xsel")
    i = agent.title_idx[t.lower()]
    sim = H.similar_to(agent, [i], 6)
    ensure_meta([t] + [d.title[j] for j in sim])
    m = M(t)
    plot = W.cache.get(f"{t}|{d.year[i]}", {}).get("plot") if f"{t}|{d.year[i]}" in W.cache else None
    if plot is None:
        with st.spinner("Loading the plot…"):
            plot = W.plot(t, int(d.year[i]), d.language[i])
    a, b = st.columns([1, 3])
    with a:
        st.markdown(f'<div style="border-radius:14px;overflow:hidden">{pz(t, m, "w500")}</div>'.replace(
            'class="pz"', 'class="pz" data-big', 1), unsafe_allow_html=True)
    with b:
        st.markdown(f"## {t}  ·  {d.year[i]}")
        if m.get("tagline"):
            st.markdown(f'<div class="tag">“{html.escape(m["tagline"])}”</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="chip g">{html.escape(x)}</span>' for x in d.genres_l[i])
                    + "".join(f'<span class="chip">{html.escape(x)}</span>' for x in d.moods_l[i]) +
                    f'<span class="chip">{html.escape(d.language[i])}</span><span class="chip">⭐ {d.rating[i]:.1f}</span>'
                    f'<span class="chip">{d.runtime[i]} min</span>', unsafe_allow_html=True)
        _sh = stream_html(t, m, detailed=True)
        if _sh:
            st.markdown(_sh, unsafe_allow_html=True)
        st.markdown(trailer_a(t, m), unsafe_allow_html=True)
        for head, text in long_text(i, m, plot):
            st.markdown(f'<div class="sec">{head}</div>' + "".join(
                f'<div class="lead" style="margin-bottom:10px">{html.escape(p)}</div>' for p in text.split("\n\n")),
                unsafe_allow_html=True)
        if plot:
            st.caption("Plot summary from Wikipedia (CC BY-SA). May contain spoilers.")
        else:
            st.caption("Couldn't load the plot paragraph (needs an internet connection).")
        st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
        links = st.columns(4)
        region = ss.get("region", "IN")
        links[0].link_button("Where to watch ↗", TM.streaming(m, region)["link"] or TM.justwatch_url(t, region))
        if m.get("url"):
            links[1].link_button("TMDB page ↗", m["url"])
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        bc = st.columns(3)
        with bc[0]:
            like_btns(t, "x")
        with bc[1]:
            add_btn(t, "x_w")
    st.subheader("Most similar")
    mini_strip(sim, lambda j: f"{round(100 * float(agent.S[i, j]))}% similar")

    st.divider()
    st.subheader("Compare")
    ss.cmp_with = [x for x in ss.get("cmp_with", []) if x != t]
    others = st.multiselect(f"Compare {t} with… (pick 1 or 2)", [x for x in sorted(d.title) if x != t],
                            max_selections=2, key="cmp_with")
    if not others:
        st.caption("Choose another movie to see them side by side.")
        return
    sel = [t] + others
    ids = agent.idx_of(sel)
    ensure_meta(sel)
    st.markdown('<div class="cmphead">' + "".join(
        f'<div>{pz(d.title[j], M(d.title[j]), "w342")}{html.escape(d.title[j])}</div>' for j in ids) + "</div>",
        unsafe_allow_html=True)
    region = ss.get("region", "IN")
    rows = {"Year": [d.year[j] for j in ids], "Language": [d.language[j] for j in ids],
            "Rating": [d.rating[j] for j in ids], "Runtime (min)": [d.runtime[j] for j in ids],
            "Genres": [", ".join(d.genres_l[j]) for j in ids], "Moods": [", ".join(d.moods_l[j]) for j in ids],
            "Cast": [", ".join(d.cast_l[j]) for j in ids],
            f"Streaming ({region})": [", ".join(x["name"] for x in TM.streaming(M(d.title[j]), region)["stream"]) or "-"
                                      for j in ids],
            "Match for you %": [round(agent.score(j, profile)[0], 1) for j in ids]}
    tbl = pd.DataFrame({d.title[j]: [str(rows[k][n]) for k in rows] for n, j in enumerate(ids)}, index=list(rows))
    st.dataframe(tbl)
    best = max(range(len(ids)), key=lambda n: rows["Match for you %"][n])
    if ss.get("loaded"):
        st.success(f"Best match for your current profile: **{d.title[ids[best]]}** ({rows['Match for you %'][best]}%)")
    st.markdown("**What they share**")
    for n, x in enumerate(ids):
        for y in ids[n + 1:]:
            sh = {"genres": set(d.genres_l[x]) & set(d.genres_l[y]), "moods": set(d.moods_l[x]) & set(d.moods_l[y]),
                  "cast": set(d.cast_l[x]) & set(d.cast_l[y])}
            common = "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sh.items() if v) or "nothing in common"
            st.write(f"**{d.title[x]}** & **{d.title[y]}** — {round(100 * float(agent.S[x, y]))}% similar content. Shared: {common}")


# ------------------------------------------------------------------ page: Under the hood (Analytics + Lab + About)
def section_analytics():
    d = agent.df
    t1, t2, t3 = st.tabs(["Catalogue", "Taste maps", "Your session"])
    with t1:
        a, b = st.columns(2)
        with a:
            st.subheader("Average rating by language")
            g = d.groupby("language").rating.agg(["mean", "count"]).reset_index()
            show(alt.Chart(g).mark_bar(color=GOLD, cornerRadiusEnd=3).encode(
                x=alt.X("mean:Q", title="Mean rating", scale=alt.Scale(domain=[6, 9.5])),
                y=alt.Y("language:N", sort="-x", title=None), tooltip=["language", "mean", "count"]), 260)
        with b:
            st.subheader("Movies per decade")
            dec = ((d.year // 10) * 10).astype(str) + "s"
            show(alt.Chart(dec.value_counts().rename_axis("Decade").reset_index(name="Movies").sort_values("Decade"))
                 .mark_bar(color=GOLD, cornerRadiusEnd=3).encode(x=alt.X("Decade:N", sort=None, title=None),
                                                                 y=alt.Y("Movies:Q", title=None), tooltip=["Decade", "Movies"]), 260)
        a, b = st.columns(2)
        with a:
            st.subheader("Genres")
            gc = pd.Series([g for x in d.genres_l for g in x]).value_counts().rename_axis("Genre").reset_index(name="Movies")
            bar(gc, "Movies", "Genre", 340)
        with b:
            st.subheader("Most-cast actors")
            ac = pd.Series([a for x in d.cast_l for a in x]).value_counts().head(10).rename_axis("Actor").reset_index(name="Movies")
            bar(ac, "Movies", "Actor", 340)
    with t2:
        st.caption("Learned relatedness between tags (co-occurrence cosine). Brighter = more related.")
        for name, mat in (("Genre relatedness", agent.rel_g), ("Mood relatedness", agent.rel_m)):
            st.subheader(name)
            long = mat.rename_axis("A").reset_index().melt(id_vars="A", var_name="B", value_name="Score")
            heat(long, "A", "B", "Score", 420, ".2f")
    with t3:
        st.subheader("What the agent values")
        wd = pd.DataFrame({"Criterion": [CRITERIA_LABEL[k] for k in agent.weights], "Weight": list(agent.weights.values())})
        bar(wd, "Weight", "Criterion", 260)
        if not ss.events:
            st.info("Like or dislike a few movies and the agent's learning will show up here.")
        else:
            st.subheader("How the criterion weights drifted")
            hist = pd.DataFrame([{"Step": n + 1, **e["after"]} for n, e in enumerate(ss.events)])
            long = hist.melt(id_vars="Step", var_name="Criterion", value_name="Weight")
            long["Criterion"] = long.Criterion.map(CRITERIA_LABEL)
            show(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("Step:Q", title="Feedback step"), y=alt.Y("Weight:Q"),
                color=alt.Color("Criterion:N", scale=alt.Scale(scheme="goldorange")), tooltip=["Criterion", "Weight"]), 300)
        st.subheader("Genre beliefs (Thompson sampling)")
        b = pd.DataFrame({g: {"Mean": a / (a + c), "Evidence": a + c - 2} for g, (a, c) in agent.genre_beta.items()}).T
        st.dataframe(b.sort_values("Mean", ascending=False).round(3))
        if len(ss.watch):
            st.subheader("Watchlist status")
            bar(ss.watch.Status.value_counts().rename_axis("Status").reset_index(name="Movies"), "Movies", "Status", 140)


def sync_weights():
    for k in DEFAULT_W:
        ss[f"wt_{k}"] = float(A().weights[k])


def apply_weights():
    tot = sum(ss[f"wt_{k}"] for k in DEFAULT_W) or 1.0
    A().weights = {k: ss[f"wt_{k}"] / tot for k in DEFAULT_W}
    sync_weights()


def default_weights():
    A().weights = dict(DEFAULT_W)
    sync_weights()


def section_lab():
    res = get_res()
    names = ["Agent trace", "Weight tuner", "Sensitivity", "Engine flags", "Personas", "Ablation", "Map"]
    sec = st.radio("Lab section", names, horizontal=True, key="lab_sec", label_visibility="collapsed")
    if sec == "Agent trace":
        st.caption("Perceive → Validate → Filter → Relax → Score → Rank → Explore → Explain")
        for step, detail in res.trace:
            st.markdown(f"**{step}** — {detail}")
        for w in warns:
            st.markdown(f"**Validate** — {w}")
        rows = []
        for i in range(len(df)):
            U, _, _ = agent.score(i, profile)
            rows.append({"Title": df.title[i], "Utility": round(U, 1),
                         "Status": "excluded (seen/avoided/liked)" if agent.excluded(i, profile) else "eligible"})
        st.dataframe(pd.DataFrame(rows).sort_values("Utility", ascending=False), hide_index=True, height=300)
    elif sec == "Weight tuner":
        st.caption("Override what the agent cares about. Weights are renormalised to sum to 1 when applied.")
        cols = st.columns(4)
        for n, k in enumerate(DEFAULT_W):
            with cols[n % 4]:
                st.slider(CRITERIA_LABEL[k], 0.0, 1.0, min(1.0, float(agent.weights[k])), 0.01, key=f"wt_{k}")
        b = st.columns(3)
        b[0].button("Apply weights", on_click=apply_weights)
        b[1].button("Sync from agent", on_click=sync_weights)
        b[2].button("Restore defaults", on_click=default_weights)
        dd = pd.DataFrame({"Default": DEFAULT_W, "Current": agent.weights})
        dd.index = [CRITERIA_LABEL[k] for k in dd.index]
        long = dd.rename_axis("Criterion").reset_index().melt(id_vars="Criterion", var_name="Set", value_name="Weight")
        show(alt.Chart(long).mark_bar().encode(
            y=alt.Y("Criterion:N", title=None), x=alt.X("Weight:Q", title=None), yOffset="Set:N",
            color=alt.Color("Set:N", scale=alt.Scale(range=["#777", GOLD]))), 320)
    elif sec == "Sensitivity":
        st.caption("Sweep one criterion's weight and see how much the top picks change.")
        key = st.selectbox("Criterion", list(DEFAULT_W), format_func=lambda k: CRITERIA_LABEL[k])
        if key not in agent.utilities(0, profile):
            st.warning("That criterion is not active for your current profile (you haven't filled it in), so changing its weight has no effect.")
        else:
            sd = H.sensitivity(agent, profile, key, np.round(np.linspace(0, 0.6, 13), 3), ss.k)
            show(alt.Chart(sd).mark_line(point=True, color=GOLD).encode(
                x=alt.X("Weight:Q"), y=alt.Y("Overlap:Q", title="Overlap with current top picks", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Weight", "Overlap", "Top pick"]), 260)
            st.dataframe(sd[["Weight", "Overlap", "Top pick", "Picks"]], hide_index=True)
    elif sec == "Engine flags":
        st.caption("Switch components off live and watch the picks change (the rest of the app uses these too).")
        c = st.columns(4)
        for col, (k, lab) in zip(c, [("lsa", "LSA similarity"), ("related", "Tag relatedness"),
                                     ("relax", "Constraint relaxation"), ("mmr", "MMR diversity")]):
            col.checkbox(lab, value=ss.flags[k], key=f"fl_{k}", on_change=set_flag, args=(k,))
        st.write("Current picks: " + ", ".join(r.title for r in res.picks))
    elif sec == "Personas":
        name = st.selectbox("Persona", list(evaluate_v3.PERSONAS))
        q, pw = agent.validate(evaluate_v3.PERSONAS[name])
        r = agent.recommend(q, 5, explore=False)
        for w in pw:
            st.caption(w)
        st.write(pd.DataFrame([{"Title": x.title, "Match %": round(x.utility, 1), "Language": x.language,
                                "Rating": x.rating} for x in r.picks]))
        st.dataframe(pd.DataFrame([evaluate_v3.metrics(agent, q, r.picks)]).round(3), hide_index=True)
    elif sec == "Ablation":
        st.caption("Turns one component off at a time over 8 personas.")
        if st.button("Run ablation"):
            ss.ablation = evaluate_v3.run(df)
        if "ablation" in ss:
            st.dataframe(ss.ablation, hide_index=True)
            met = st.selectbox("Metric", [c for c in ss.ablation.columns if c != "Variant"])
            show(alt.Chart(ss.ablation).mark_bar(color=GOLD).encode(
                x=alt.X(f"{met}:Q", title=None), y=alt.Y("Variant:N", sort=None, title=None)), 220)
    elif sec == "Map":
        st.caption("Every movie projected to 2D from the LSA vectors. Nearby = similar content.")
        by = st.radio("Colour by", ["Recommended", "Language", "Primary genre"], horizontal=True)
        pick = {r.idx for r in res.picks}
        mp = pd.DataFrame({"x": agent.coords[:, 0], "y": agent.coords[:, 1], "Title": df.title, "Rating": df.rating,
                           "Language": df.language, "Primary genre": [g[0] if g else "-" for g in agent.df.genres_l],
                           "Recommended": ["Recommended" if i in pick else "Other" for i in range(len(df))]})
        show(alt.Chart(mp).mark_circle(opacity=0.9).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), size=alt.Size("Rating:Q", legend=None, scale=alt.Scale(range=[40, 260])),
            color=alt.Color(f"{by}:N", scale=alt.Scale(scheme="goldorange") if by != "Recommended"
                            else alt.Scale(domain=["Recommended", "Other"], range=[GOLD, "#8a8a92"])),
            tooltip=["Title", "Language", "Primary genre", "Rating"]).interactive(), 480)


def section_about():
    st.subheader("PEAS")
    st.table(pd.DataFrame({
        "Performance": ["Relevance, constraint satisfaction, user satisfaction, explanation quality"],
        "Environment": ["Movie catalogue, TMDB metadata, interactive UI, the user's feedback and watch history"],
        "Actuators": ["Ranked poster cards, explanations, counterfactuals, chat replies, marathon plans, streaming links, CSV export"],
        "Sensors": ["Preference form, free-text chat, clarifying answers, like/dislike (with undo), watchlist ratings"]}).T.rename(columns={0: "v5"}))
    st.subheader("How it works")
    st.markdown("""
1. **Perceive & validate**: the profile is cleaned (fuzzy-matching names) and checked.
2. **Filter & relax**: hard constraints first; if too few movies remain they are relaxed one at a time and you are told which.
3. **Score**: weighted utility over genre, mood, language, similarity to favourites, actors, rating, runtime and era.
4. **Rank**: movies are ordered by their utility score.
5. **Explore**: Thompson sampling adds one pick from a genre you have not tried.
6. **Learn**: every like or dislike updates the criterion weights (Hedge) and the genre beliefs. Undo rewinds that update exactly and replays any later feedback.
7. **Explain**: per-criterion points, counterfactuals ("what if you lowered the rating floor?").
""")
    st.subheader("What's new in v5")
    st.markdown("Posters, trailers and streaming services from TMDB (JustWatch data) · Discover merges Home and Recommend · "
                "Explore merges the movie inspector and Compare, with long descriptions · Analytics, Lab and About live under "
                "“Under the hood” · bigger sidebar navigation · dark/light theme switch · My taste page with undo for every "
                "like and dislike · separate watchlist file so the old app is untouched.")
    st.subheader("Data credits")
    st.caption("Plot summaries come from Wikipedia (CC BY-SA). Posters, trailers, overviews and cast data come from TMDB. Streaming availability is provided by JustWatch "
               "through TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")
    if RT_EST:
        st.warning("movies.csv has no runtime column, so runtimes come from a built-in approximate table. "
                   "Add a `runtime` column (minutes) to use exact values.")


def page_under():
    st.title("Under the hood")
    st.caption("How the agent thinks, how it performs, and what it has learned about you.")
    sec = st.radio("Section", ["Analytics", "Lab", "About"], horizontal=True, key="uh_sec", label_visibility="collapsed")
    {"Analytics": section_analytics, "Lab": section_lab, "About": section_about}[sec]()


{"Discover": page_discover, "Chat": page_chat, "Marathon": page_marathon, "Watchlist": page_watchlist,
 "My taste": page_taste, "Explore": page_explore, "Under the hood": page_under}[page]()
