"""
evaluate_v3.py - ablation study. Run:  python evaluate_v3.py
Turns one component off at a time and measures the effect over a set of personas.
(Persona genres/moods that don't exist in your movies.csv are dropped automatically,
 so use the exact lowercase mood names from the catalogue.)
"""
import numpy as np
import pandas as pd

from agent_v3 import MovieAgent, Profile
from v4_helpers import ensure_runtime

PERSONAS = {
    "Thriller fan":   Profile(genres=["Thriller", "Crime"], moods=["dark", "thrilling"], min_rating=7.5),
    "Family night":   Profile(genres=["Animation", "Comedy"], moods=["heartwarming", "funny"], context="Family", min_rating=7.0),
    "Date night":     Profile(genres=["Romance", "Drama"], moods=["romantic"], context="Partner", min_rating=7.0),
    "Short sci-fi":   Profile(genres=["Sci-Fi"], moods=["thoughtful"], max_runtime=120, min_rating=7.5),
    "Hindi drama":    Profile(genres=["Drama"], languages=["Hindi"], min_rating=7.0),
    "Strict corner":  Profile(genres=["Horror"], languages=["Malayalam"], min_rating=9.0),   # forces relaxation
    "Korean mood":    Profile(genres=["Thriller"], languages=["Korean"], moods=["dark"], min_rating=7.5),
    "Classic lover":  Profile(genres=["Drama"], era=(1990, 2005), min_rating=8.0),
}

VARIANTS = {
    "Full agent v3":        {},
    "- no MMR diversity":   {"mmr": False},
    "- no LSA (TF-IDF)":    {"lsa": False},
    "- no tag relatedness": {"related": False},
    "- no relaxation":      {"relax": False},
}


def metrics(agent, p, picks, k=5):
    if not picks:
        return dict(sat=0.0, cov=0.0, div=0.0, rating=0.0, filled=0.0)
    df = agent.df
    sat = np.mean([(df.rating[r.idx] >= p.min_rating)
                   and (not p.languages or df.language[r.idx] in p.languages)
                   and (not p.genres or bool(set(r.genres) & set(p.genres))) for r in picks])
    cov = np.mean([np.mean([bool(set(r.genres) & set(p.genres)) if p.genres else 1,
                            bool(set(df.moods_l[r.idx]) & set(p.moods)) if p.moods else 1]) for r in picks])
    ids = [r.idx for r in picks]
    pairs = [agent.S[a, b] for a in ids for b in ids if a < b]
    return dict(sat=sat, cov=cov, div=1 - (np.mean(pairs) if pairs else 0),
                rating=np.mean([r.rating for r in picks]), filled=len(picks) / k)


def run(df, k=5):
    if "runtime" not in df.columns:          # the agent needs a runtime column
        df, _ = ensure_runtime(df)
    rows = []
    for name, flags in VARIANTS.items():
        agent = MovieAgent(df)
        agent.flags.update(flags)
        ms = []
        for p in PERSONAS.values():
            q, _ = agent.validate(p)
            ms.append(metrics(agent, q, agent.recommend(q, k, explore=False).picks, k))
        m = pd.DataFrame(ms).mean()
        # use m["..."], not m.cov / m.div: those names are pandas Series *methods*, not your columns
        rows.append({"Variant": name, "Constraint satisfaction": m["sat"], "Preference coverage": m["cov"],
                     "Diversity": m["div"], "Avg rating": m["rating"], "Slots filled": m["filled"]})
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    print(run(pd.read_csv("movies.csv")).to_string(index=False))
