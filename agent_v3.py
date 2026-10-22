"""
agent_v3.py - Movie Recommendation Agent v3
Utility-based agent + learning + exploration + group / marathon planning.

Reads the same movies.csv schema as v2:
title, year, language, genres(|), rating, cast(|), moods(|), description, runtime
"""
from __future__ import annotations

import copy
import difflib
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_W = dict(genre=.22, mood=.20, language=.08, favourites=.15,
                 actors=.08, rating=.15, runtime=.06, era=.06)
FAMILY_BAN = {"Horror", "Crime", "Thriller", "War"}
# must use the lowercase mood names that exist in movies.csv, otherwise the "Watching with" context scores 0
CONTEXT_MOODS = {"Alone": ["thoughtful", "dark"], "Partner": ["romantic", "heartwarming"],
                 "Friends": ["funny", "thrilling"], "Family": ["heartwarming", "funny"]}
CRITERIA_LABEL = dict(genre="Genre", mood="Mood", language="Language", favourites="Similar to favourites",
                      actors="Actors", rating="Rating", runtime="Runtime", era="Era")


def split(s):
    return [x.strip() for x in s.split("|") if x.strip()] if isinstance(s, str) else []


# --------------------------------------------------------------------------- data classes
@dataclass
class Profile:
    genres: list = field(default_factory=list)
    moods: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    min_rating: float = 0.0
    max_runtime: int = 0            # 0 = no limit
    era: tuple = ()                 # (from_year, to_year) or ()
    avoid_genres: list = field(default_factory=list)
    avoid_moods: list = field(default_factory=list)
    favourites: list = field(default_factory=list)   # movie titles
    actors: list = field(default_factory=list)
    context: str = ""               # Alone / Partner / Friends / Family
    seen: list = field(default_factory=list)
    disliked: list = field(default_factory=list)

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Rec:
    idx: int
    title: str
    year: int
    language: str
    genres: list
    rating: float
    runtime: int
    utility: float
    contrib: dict
    reasons: list
    explore: bool = False


@dataclass
class Result:
    picks: list
    notices: list
    trace: list          # list of (step, detail)
    pool_size: int


# --------------------------------------------------------------------------- agent
class MovieAgent:
    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.df = df.reset_index(drop=True).copy()
        d = self.df
        for c in ("genres", "cast", "moods"):
            d[c + "_l"] = d[c].map(split)
        d["description"] = d["description"].fillna("")
        self.seed = seed
        self.n_fb = 0          # number of feedback events; part of the exploration seed
        self.title_idx = {t.lower(): i for i, t in enumerate(d.title)}

        # --- text similarity: TF-IDF + LSA hybrid
        text = (d.title + " " + d.description + " " + (d.genres.str.replace("|", " ") + " ") * 2
                + d.moods.str.replace("|", " ") + " " + d.cast.str.replace("|", " "))
        X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(text)
        self.sim_tfidf = cosine_similarity(X)
        k = max(2, min(24, X.shape[1] - 1, X.shape[0] - 1))
        Z = TruncatedSVD(k, random_state=0).fit_transform(X)
        self.sim_full = np.clip(0.5 * self.sim_tfidf + 0.5 * cosine_similarity(Z), 0, 1)
        self.coords = PCA(2, random_state=0).fit_transform(Z)

        # --- learned tag relatedness (co-occurrence cosine)
        self.rel_g = self._relatedness("genres_l")
        self.rel_m = self._relatedness("moods_l")
        self.all_genres = sorted(self.rel_g.index)
        self.all_moods = sorted(self.rel_m.index)
        self.all_langs = sorted(d.language.unique())
        self.all_actors = sorted({a for c in d.cast_l for a in c})

        # --- learning state
        self.weights = dict(DEFAULT_W)
        self.genre_beta = {g: [1.0, 1.0] for g in self.all_genres}   # Thompson sampling
        self.flags = dict(lsa=True, related=True, mmr=True, relax=True)

    # ------------------------------------------------------------------ helpers
    def _relatedness(self, col):
        tags = sorted({t for ts in self.df[col] for t in ts})
        M = pd.DataFrame([[1.0 if t in ts else 0.0 for t in tags] for ts in self.df[col]], columns=tags)
        C = cosine_similarity(M.T.values)
        np.fill_diagonal(C, 0.0)
        return pd.DataFrame(C, index=tags, columns=tags)

    @property
    def S(self):
        return self.sim_full if self.flags["lsa"] else self.sim_tfidf

    def idx_of(self, titles):
        """Titles -> row indices (unknown titles are skipped)."""
        return [self.title_idx[t.lower()] for t in titles if t.lower() in self.title_idx]

    _idx = idx_of      # backwards-compatible alias

    def _tag_u(self, tags, wanted, rel):
        best = 0.0
        for w in wanted:
            if w in tags:
                return 1.0
            if self.flags["related"] and w in rel.index:
                best = max(best, 0.6 * max((rel.at[w, t] for t in tags if t in rel.columns), default=0.0))
        return best

    # ------------------------------------------------------------------ validation
    def validate(self, p: Profile):
        q, warns = p.copy(), []

        def fix(vals, vocab, label):
            low = {v.lower(): v for v in vocab}
            out = []
            for v in vals:
                key = v.lower() if v.lower() in low else next(
                    iter(difflib.get_close_matches(v.lower(), list(low), 1, 0.7)), None)
                if key is None:
                    warns.append(f"Unknown {label} '{v}' ignored")
                    continue
                out.append(low[key])
                if key != v.lower():
                    warns.append(f"{label} '{v}' interpreted as '{low[key]}'")
            return out
        q.genres = fix(q.genres, self.all_genres, "genre")
        q.avoid_genres = fix(q.avoid_genres, self.all_genres, "genre")
        q.moods = fix(q.moods, self.all_moods, "mood")
        q.avoid_moods = fix(q.avoid_moods, self.all_moods, "mood")
        q.languages = fix(q.languages, self.all_langs, "language")
        q.favourites = fix(q.favourites, list(self.df.title), "movie")
        q.actors = fix(q.actors, self.all_actors, "actor")
        q.min_rating = float(np.clip(q.min_rating, 0, 10))
        q.max_runtime = max(0, int(q.max_runtime))
        if q.era:
            q.era = tuple(sorted(q.era))
        return q, warns

    # ------------------------------------------------------------------ utility
    def utilities(self, i, p):
        r, u, S = self.df.iloc[i], {}, self.S
        if p.genres:
            u["genre"] = self._tag_u(r.genres_l, p.genres, self.rel_g)
        ctx = CONTEXT_MOODS.get(p.context, [])
        if p.moods or ctx:
            a = self._tag_u(r.moods_l, p.moods, self.rel_m) if p.moods else None
            b = self._tag_u(r.moods_l, ctx, self.rel_m) if ctx else None
            u["mood"] = a if b is None else b if a is None else .75 * a + .25 * b
        if p.languages:
            u["language"] = 1.0 if r.language in p.languages else 0.0
        favs = self.idx_of(p.favourites)
        if favs:
            sims = [S[i, f] for f in favs if f != i]
            u["favourites"] = float(np.clip(max(sims) / 0.55, 0, 1)) if sims else 0.0
        if p.actors or favs:
            cast, v = set(r.cast_l), 0.0
            if any(a in cast for a in p.actors):
                v = 1.0
            elif favs and cast & {a for f in favs for a in self.df.cast_l[f]}:
                v = 0.5
            u["actors"] = v
        if p.min_rating > 0:
            u["rating"] = 1 / (1 + math.exp(-1.5 * (r.rating - 7.5)))
        if p.max_runtime:
            u["runtime"] = max(0.0, 1 - max(0, r.runtime - p.max_runtime) / 45)
        if p.era:
            lo, hi = p.era
            dist = 0 if lo <= r.year <= hi else min(abs(r.year - lo), abs(r.year - hi))
            u["era"] = max(0.0, 1 - dist / 15)
        return u

    def score(self, i, p):
        u = self.utilities(i, p)
        if not u:                               # nothing filled in -> fall back to rating
            u = {"rating": 1 / (1 + math.exp(-1.5 * (self.df.rating[i] - 7.5)))}
        w = {k: self.weights[k] for k in u}
        tot = sum(w.values())
        dis = self.idx_of(p.disliked)
        factor = 1 - 0.5 * max((self.S[i, j] for j in dis if j != i), default=0.0)
        contrib = {k: 100 * w[k] / tot * u[k] * factor for k in u}
        return sum(contrib.values()), contrib, u

    # ------------------------------------------------------------------ selection (hard filters + relaxation)
    def excluded(self, i, p):
        r = self.df.iloc[i]
        if r.title in p.seen or r.title in p.disliked or r.title in p.favourites:
            return True
        if set(r.genres_l) & set(p.avoid_genres) or set(r.moods_l) & set(p.avoid_moods):
            return True
        return p.context == "Family" and bool(set(r.genres_l) & FAMILY_BAN)

    def select(self, p, k, relax=True):
        base = [i for i in range(len(self.df)) if not self.excluded(i, p)]
        stages = [("all constraints", p.min_rating, True, True)]
        if relax and self.flags["relax"]:
            if p.min_rating > 0:
                for s in range(1, 5):
                    if p.min_rating - .5 * s >= 0:
                        stages.append((f"minimum rating relaxed to {p.min_rating - .5 * s:.1f}",
                                       p.min_rating - .5 * s, True, True))
            floor = max(0, p.min_rating - 2)
            if p.genres:
                stages.append(("genre-overlap requirement dropped", floor, False, True))
            if p.languages:
                stages.append(("language requirement dropped", floor, False, False))
        pool, notices = [], []
        for n, (name, mr, need_g, need_l) in enumerate(stages):
            pool = [i for i in base
                    if self.df.rating[i] >= mr
                    and (not need_l or not p.languages or self.df.language[i] in p.languages)
                    and (not need_g or not p.genres or set(self.df.genres_l[i]) & set(p.genres))]
            if len(pool) >= k:
                if n > 0:
                    notices.append(f"Not enough movies matched everything, so I relaxed: {name}.")
                return pool, notices, name
        if len(stages) > 1:
            notices.append(f"Even after relaxing ({stages[-1][0]}), fewer than {k} movies qualified.")
        return pool, notices, stages[-1][0]

    # ------------------------------------------------------------------ recommend
    def _reasons(self, contrib, u, r, p):
        out = []
        for k, v in sorted(contrib.items(), key=lambda kv: -kv[1])[:3]:
            if v <= 0.5:
                continue
            extra = ""
            if k == "genre":
                hit = [g for g in r.genres_l if g in p.genres]
                extra = f" ({', '.join(hit)})" if hit else " (related genre)"
            elif k == "mood":
                hit = [m for m in r.moods_l if m in p.moods]
                extra = f" ({', '.join(hit)})" if hit else ""
            out.append(f"{CRITERIA_LABEL[k]}{extra}: +{v:.1f} pts")
        return out

    def _rec(self, i, p, explore=False, note=""):
        U, contrib, u = self.score(i, p)
        r = self.df.iloc[i]
        reasons = self._reasons(contrib, u, r, p)
        if explore:
            reasons.insert(0, note)
        return Rec(i, r.title, int(r.year), r.language, r.genres_l, float(r.rating), int(r.runtime),
                   U, contrib, reasons, explore)

    def recommend(self, p, k=5, diversity=0.3, explore=True, relax=True):
        trace = [("Perceive", f"{len(p.genres)} genres, {len(p.moods)} moods, {len(p.languages)} languages, "
                              f"{len(p.favourites)} favourites, context={p.context or 'none'}")]
        pool, notices, stage = self.select(p, k, relax)
        trace.append(("Filter + relax", f"{len(pool)} candidates after: {stage}"))
        scored = {i: self.score(i, p)[0] for i in pool}
        trace.append(("Score", "utility computed for every candidate (weights renormalised to the criteria you filled in)"))
        d = diversity if self.flags["mmr"] else 0.0
        n_main = k - 1 if (explore and k >= 3) else k
        picked, left = [], set(pool)
        while left and len(picked) < n_main:
            def mmr(c):
                red = max((self.S[c, s] for s in picked), default=0.0)
                return (1 - d) * scored[c] / 100 - d * red, self.df.rating[c]
            best = max(left, key=mmr)
            picked.append(best)
            left.discard(best)
        trace.append(("Rank (MMR)", f"diversity weight {d:.2f}; picked {len(picked)} movies"))
        picks = [self._rec(i, p) for i in picked]
        if explore and k >= 3 and left:
            tried = set(p.genres)
            # same inputs + same learned state -> same exploration pick (no flicker between reruns)
            rng = random.Random(f"{self.seed}|{self.n_fb}|{','.join(sorted(tried))}|{k}")
            order = sorted((g for g in self.all_genres if g not in tried),
                           key=lambda g: -rng.betavariate(*self.genre_beta[g]))
            for g in order:
                cand = [c for c in left if g in self.df.genres_l[c]]
                if cand:
                    c = max(cand, key=lambda c: scored[c])
                    picks.append(self._rec(c, p, True, f"Exploration pick: tests a genre you haven't tried ({g})"))
                    trace.append(("Explore (Thompson)", f"sampled genre '{g}' from Beta posteriors"))
                    break
        picks.sort(key=lambda r: (r.explore, -r.utility))
        trace.append(("Explain", "per-criterion points stored for each pick"))
        return Result(picks, notices, trace, len(pool))

    # ------------------------------------------------------------------ learning
    def feedback(self, p, title, label):
        """Hedge update: w_k *= exp(lr*label*(u_k - mean u)); renormalise active mass. Mutates p."""
        old = dict(self.weights)
        self.n_fb += 1
        if title.lower() in self.title_idx:
            i = self.title_idx[title.lower()]
            u = self.utilities(i, p)
            if len(u) > 1:
                mean = sum(u.values()) / len(u)
                mass = sum(self.weights[k] for k in u)
                new = {k: self.weights[k] * math.exp(label * (u[k] - mean)) for k in u}
                s = sum(new.values())
                for k in u:
                    self.weights[k] = new[k] / s * mass
            for g in self.df.genres_l[i]:
                self.genre_beta[g][0 if label > 0 else 1] += 1
        if label > 0 and title not in p.favourites:
            p.favourites.append(title)
        if label < 0 and title not in p.disliked:
            p.disliked.append(title)
        return old, dict(self.weights)

    # ------------------------------------------------------------------ extras
    def counterfactuals(self, p, k=5, diversity=0.3):
        base = {r.title for r in self.recommend(p, k, diversity, explore=False, relax=False).picks}
        tests = []
        if p.min_rating > 0:
            q = p.copy(); q.min_rating = max(0, p.min_rating - 1)
            tests.append((f"lower the minimum rating to {q.min_rating:g}", q))
        if p.max_runtime:
            q = p.copy(); q.max_runtime += 20
            tests.append((f"allow 20 more minutes of runtime (to {q.max_runtime})", q))
        if p.languages:
            q = p.copy(); q.languages = []
            tests.append(("accept any language", q))
        for g in p.avoid_genres:
            q = p.copy(); q.avoid_genres = [x for x in p.avoid_genres if x != g]
            tests.append((f"stop avoiding {g}", q))
        out = []
        for desc, q in tests:
            new = [r.title for r in self.recommend(q, k, diversity, explore=False, relax=False).picks
                   if r.title not in base]
            if new:
                out.append((desc, new))
        return out

    def next_question(self, p):
        """Conversational mode: ask about the unfilled attribute with the highest entropy over candidates."""
        pool, _, _ = self.select(p, 5, relax=False)
        pool = pool or list(range(len(self.df)))
        specs = []
        if not p.genres:
            specs.append(("genres", "Which genre are you in the mood for?", lambda i: self.df.genres_l[i]))
        if not p.moods:
            specs.append(("moods", "What mood are you in?", lambda i: self.df.moods_l[i]))
        if not p.languages:
            specs.append(("languages", "Any preferred language?", lambda i: [self.df.language[i]]))
        best = None
        for attr, q, fn in specs:
            cnt = pd.Series([v for i in pool for v in fn(i)]).value_counts()
            if cnt.empty:
                continue
            pr = cnt / cnt.sum()
            ent = float(-(pr * np.log2(pr)).sum())
            if best is None or ent > best[0]:
                best = (ent, attr, q, list(cnt.index[:6]))
        return None if best is None else dict(attr=best[1], question=best[2], options=best[3], entropy=best[0])

    def group(self, members: dict, rule="least_misery", k=5):
        ids = [i for i in range(len(self.df)) if not any(self.excluded(i, p) for p in members.values())]
        M = np.vstack([[self.score(i, p)[0] for i in ids] for p in members.values()])
        if rule == "average":
            agg = M.mean(0)
        elif rule == "borda":
            agg = M.argsort(1).argsort(1).sum(0).astype(float)
        else:
            agg = M.min(0)
        rows = []
        for j in np.argsort(-agg)[:k]:
            row = {"Title": self.df.title[ids[j]], "Group score": round(float(agg[j]), 1)}
            row.update({n: round(float(M[m, j]), 1) for m, n in enumerate(members)})
            rows.append(row)
        return pd.DataFrame(rows)

    def marathon(self, p, budget=360, max_films=5, gap=10, pool_size=25):
        pool, _, _ = self.select(p, pool_size, relax=True)
        U = {i: self.score(i, p)[0] for i in pool}
        pool = sorted(pool, key=lambda i: -U[i])[:pool_size]
        rt = self.df.runtime

        def value(ch):
            gset = {g for i in ch for g in self.df.genres_l[i]}
            pairs = [self.S[a, b] for a in ch for b in ch if a < b]
            return sum(U[i] for i in ch) + 5 * len(gset) - (15 * np.mean(pairs) if pairs else 0)
        beam, best = [((), 0, 0.0)], ((), 0, 0.0)
        for _ in range(max_films):
            nxt = []
            for ch, t, _v in beam:
                for j in pool:
                    if j in ch or (ch and j < ch[-1]):
                        continue
                    t2 = t + int(rt[j]) + (gap if ch else 0)
                    if t2 <= budget:
                        c2 = ch + (j,)
                        nxt.append((c2, t2, value(c2)))
            if not nxt:
                break
            nxt.sort(key=lambda x: -x[2])
            beam = nxt[:300]
            if beam[0][2] > best[2]:
                best = beam[0]
        ch, t, v = best
        return dict(movies=[self._rec(i, p) for i in ch], total_minutes=t, value=v, budget=budget)
