"""tmdb.py - tiny TMDB client for CineAgent v5 (no Streamlit import, standard library only).

One lookup per movie returns everything the UI needs: poster, tagline, long overview, director,
top cast, YouTube trailer key and per-country streaming providers (data supplied by JustWatch via TMDB).
Results are cached in a JSON file, so each movie is fetched only once, ever.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/"
EMPTY: dict = {"found": False}


def poster_url(meta: dict, size: str = "w342") -> str:
    return f"{IMG}{size}{meta['poster']}" if meta.get("poster") else ""


def logo_url(path: str) -> str:
    return f"{IMG}w45{path}" if path else ""


def trailer_url(title: str, year, meta: dict | None = None) -> str:
    """Direct YouTube trailer when TMDB knows one, otherwise a YouTube search (always works, no key)."""
    if meta and meta.get("trailer"):
        return f"https://www.youtube.com/watch?v={meta['trailer']}"
    return "https://www.youtube.com/results?search_query=" + quote(f"{title} {year} official trailer")


def justwatch_url(title: str, region: str = "IN") -> str:
    return f"https://www.justwatch.com/{region.lower()}/search?q=" + quote(title)


def streaming(meta: dict, region: str) -> dict:
    """Providers for one country as {'stream': [...], 'rent': [...], 'buy': [...], 'link': url}."""
    p = (meta.get("providers") or {}).get(region) or {}
    seen, stream = set(), []
    for kind in ("flatrate", "free", "ads"):
        for s in p.get(kind, []):
            if s["name"] not in seen:
                seen.add(s["name"])
                stream.append(s)
    return {"stream": stream, "rent": p.get("rent", []), "buy": p.get("buy", []), "link": p.get("link", "")}


class TMDB:
    def __init__(self, key: str | None = None, cache_path: str | Path | None = None, timeout: float = 6.0):
        self.key = (key or "").strip()
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict = {}
        self.lock = threading.Lock()
        self.cooldown_until = 0.0          # after a network error, stop hammering the API for a minute
        self.status = "no key" if not self.key else "ready"
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.cache = {}

    @property
    def enabled(self) -> bool:
        return bool(self.key) and self.status != "invalid key"

    # ------------------------------------------------------------------ http
    def _get(self, path: str, **params) -> dict:
        q = {k: v for k, v in params.items() if v not in (None, "")}
        headers = {"Accept": "application/json", "User-Agent": "CineAgent/5"}
        if len(self.key) > 40:                       # v4 "API Read Access Token" (JWT)
            headers["Authorization"] = "Bearer " + self.key
        else:                                        # classic v3 API key
            q["api_key"] = self.key
        req = Request(f"{API}{path}?{urlencode(q)}", headers=headers)
        with urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def _pick(results: list, title: str, year) -> dict | None:
        if not results:
            return None
        near = [r for r in results if str(r.get("release_date", ""))[:4].isdigit()
                and abs(int(r["release_date"][:4]) - int(year)) <= 1] or results
        t = title.lower()
        return max(near, key=lambda r: (r.get("title", "").lower() == t or r.get("original_title", "").lower() == t,
                                        r.get("popularity", 0)))

    @staticmethod
    def _shape(d: dict) -> dict:
        vids = [v for v in d.get("videos", {}).get("results", []) if v.get("site") == "YouTube"]
        vids.sort(key=lambda v: (v.get("type") == "Trailer", v.get("type") == "Teaser", bool(v.get("official"))),
                  reverse=True)
        credits = d.get("credits", {})
        prov = {}
        for cc, v in d.get("watch/providers", {}).get("results", {}).items():
            prov[cc] = {"link": v.get("link", "")}
            for kind in ("flatrate", "free", "ads", "rent", "buy"):
                if v.get(kind):
                    prov[cc][kind] = [{"name": s["provider_name"], "logo": s.get("logo_path") or ""} for s in v[kind]]
        return {"found": True, "id": d["id"], "poster": d.get("poster_path") or "", "overview": d.get("overview") or "",
                "tagline": d.get("tagline") or "", "runtime": d.get("runtime") or 0,
                "vote": round(d.get("vote_average") or 0, 1), "tmdb_genres": [g["name"] for g in d.get("genres", [])],
                "director": ", ".join(c["name"] for c in credits.get("crew", []) if c.get("job") == "Director")[:80],
                "cast": [c["name"] for c in credits.get("cast", [])[:6]],
                "trailer": vids[0]["key"] if vids and vids[0].get("type") in ("Trailer", "Teaser") else "",
                "providers": prov, "url": f"https://www.themoviedb.org/movie/{d['id']}"}

    def peek(self, title: str, year) -> dict:
        """Cache-only read (never touches the network)."""
        return self.cache.get(f"{title}|{year}", EMPTY)

    def lookup(self, title: str, year) -> dict:
        k = f"{title}|{year}"
        if k in self.cache:
            return self.cache[k]
        if not self.enabled or time.time() < self.cooldown_until:
            return EMPTY
        try:
            res = self._get("/search/movie", query=title, year=year, include_adult="false").get("results", [])
            if not res:                              # year can be off by a little in the source data
                res = self._get("/search/movie", query=title, include_adult="false").get("results", [])
            hit = self._pick(res, title, year)
            meta = EMPTY
            if hit:
                meta = self._shape(self._get(f"/movie/{hit['id']}", language="en-US",
                                             append_to_response="videos,watch/providers,credits"))
            with self.lock:
                self.cache[k] = meta
            self.status = "ready"
            return meta
        except HTTPError as e:
            if e.code in (401, 403):
                self.status = "invalid key"
            else:
                self.cooldown_until = time.time() + 60
                self.status = f"TMDB error {e.code}"
        except (URLError, TimeoutError, OSError, ValueError):
            self.cooldown_until = time.time() + 60
            self.status = "offline"
        return EMPTY

    def prefetch(self, items) -> None:
        """items: iterable of (title, year). Fetches the missing ones in parallel, then saves the cache once."""
        todo = [(t, y) for t, y in dict.fromkeys(items) if f"{t}|{y}" not in self.cache]
        if not todo or not self.enabled:
            return
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda a: self.lookup(*a), todo))
        self.save()

    def save(self) -> None:
        if not self.cache_path:
            return
        try:
            with self.lock:
                tmp = self.cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.cache_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------
# Wikipedia plot summaries: free, no account and no key needed
import re  # noqa: E402


class Wiki:
    API = "https://en.wikipedia.org/w/api.php"
    HEAD = re.compile(r"^==\s*(Plot|Plot summary|Synopsis|Premise|Storyline|Story)\s*==\s*$", re.M | re.I)

    def __init__(self, cache_path: str | Path | None = None, timeout: float = 8.0):
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict = {}
        self.cooldown_until = 0.0
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.cache = {}

    def _get(self, **params) -> dict:
        q = dict(params, format="json", formatversion=2)
        req = Request(f"{self.API}?{urlencode(q)}", headers={"User-Agent": "CineAgent/5 (student project)"})
        with urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @classmethod
    def plot_from(cls, extract: str, lo: int = 600, hi: int = 1300) -> str:
        m = cls.HEAD.search(extract)
        if not m:
            return ""
        rest = extract[m.end():]
        nxt = re.search(r"^==[^=].*$", rest, re.M)          # next level-2 heading ends the section
        body = re.sub(r"^=+.*?=+\s*$", "", rest[:nxt.start()] if nxt else rest, flags=re.M)
        text = ""
        for p in (x.strip() for x in re.split(r"\n+", body)):
            if len(p) < 40:
                continue
            text += (" " if text else "") + p
            if len(text) >= lo:
                break
        if len(text) > hi:                                   # cut at a sentence end so it reads as one paragraph
            cut = text[:hi].rfind(". ")
            text = text[:cut + 1] if cut > lo // 2 else text[:hi].rstrip() + "…"
        return text

    def plot(self, title: str, year: int, language: str = "") -> str:
        k = f"{title}|{year}"
        if k in self.cache:
            return self.cache[k].get("plot", "")
        if time.time() < self.cooldown_until:
            return ""
        cands = [f"{title} ({year} film)"]
        if language:
            cands += [f"{title} ({year} {language} film)", f"{title} ({language} film)"]
        cands += [f"{title} (film)", title]
        try:
            for c in cands:
                pages = self._get(action="query", prop="extracts", explaintext=1, exsectionformat="wiki",
                                  redirects=1, titles=c).get("query", {}).get("pages", [])
                if not pages or pages[0].get("missing") or not pages[0].get("extract"):
                    continue
                ex = pages[0]["extract"]
                head = ex[:1500]
                if "film" in head[:600].lower() and any(str(y) in head for y in (year - 1, year, year + 1)):
                    txt = self.plot_from(ex)
                    if txt:
                        self.cache[k] = {"plot": txt, "page": pages[0].get("title", c)}
                        self.save()
                        return txt
            self.cache[k] = {"plot": ""}                     # looked properly, nothing usable: don't retry every time
            self.save()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            self.cooldown_until = time.time() + 60
        return ""

    def save(self) -> None:
        if self.cache_path:
            try:
                tmp = self.cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.cache_path)
            except OSError:
                pass
