"""v4_helpers.py - logic used by app_v4.py that does not need Streamlit (so it is easy to test)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from agent_v3 import Profile

# Approximate running times (minutes). Only used when movies.csv has no 'runtime' column.
RUNTIMES = {
    "The Shawshank Redemption": 142, "The Dark Knight": 152, "Inception": 148, "Interstellar": 169,
    "The Grand Budapest Hotel": 99, "Forrest Gump": 142, "Superbad": 113, "The Conjuring": 112, "Get Out": 104,
    "La La Land": 128, "Mad Max: Fury Road": 120, "The Martian": 144, "Paddington 2": 103, "Knives Out": 130,
    "Rush": 123, "Whiplash": 106, "Spider-Man: Into the Spider-Verse": 117, "Coco": 105, "The Notebook": 123,
    "Se7en": 127, "Jurassic Park": 127, "3 Idiots": 170, "Dangal": 161, "Zindagi Na Milegi Dobara": 155,
    "Andhadhun": 139, "Queen": 146, "Gully Boy": 153, "Dil Chahta Hai": 183, "Tumbbad": 104,
    "Taare Zameen Par": 165, "Kahaani": 122, "Drishyam": 160, "Premam": 156, "Manjummel Boys": 135,
    "Kumbalangi Nights": 135, "Bangalore Days": 170, "Maheshinte Prathikaaram": 120, "Jallikattu": 91,
    "Angamaly Diaries": 128, "Thanneer Mathan Dinangal": 143, "Ustad Hotel": 150, "Bramayugam": 139,
    "Minnal Murali": 158, "Vikram": 174, "Soorarai Pottru": 153, "Super Deluxe": 176, "96": 158,
    "Vikram Vedha": 147, "Kaithi": 145, "Parasite": 132, "Train to Busan": 118, "Oldboy": 120,
    "Miracle in Cell No. 7": 127, "Spirited Away": 125, "Your Name": 107, "Departures": 130,
    "Seven Samurai": 207, "Amelie": 122, "The Intouchables": 112, "Pan's Labyrinth": 118,
    "The Secret in Their Eyes": 129,
    "The Godfather": 175,
    "The Godfather Part II": 202,
    "Pulp Fiction": 154,
    "Fight Club": 139,
    "The Matrix": 136,
    "Goodfellas": 145,
    "The Lord of the Rings: The Fellowship of the Ring": 178,
    "The Lord of the Rings: The Return of the King": 201,
    "Star Wars: The Empire Strikes Back": 124,
    "Back to the Future": 116,
    "Gladiator": 155,
    "The Silence of the Lambs": 118,
    "Saving Private Ryan": 169,
    "Schindler's List": 195,
    "The Green Mile": 189,
    "Terminator 2: Judgment Day": 137,
    "Alien": 117,
    "Aliens": 137,
    "Blade Runner 2049": 164,
    "Dune": 155,
    "Dune: Part Two": 166,
    "Oppenheimer": 180,
    "The Prestige": 130,
    "Memento": 113,
    "Gone Girl": 149,
    "Shutter Island": 138,
    "The Wolf of Wall Street": 180,
    "The Departed": 151,
    "Joker": 122,
    "Avengers: Endgame": 181,
    "Iron Man": 126,
    "Guardians of the Galaxy": 121,
    "Black Panther": 134,
    "Spider-Man: No Way Home": 148,
    "The Incredibles": 115,
    "Toy Story": 81,
    "Finding Nemo": 100,
    "Inside Out": 95,
    "WALL-E": 98,
    "The Lion King": 88,
    "Shrek": 90,
    "How to Train Your Dragon": 98,
    "Ratatouille": 111,
    "Soul": 100,
    "The Truman Show": 103,
    "Eternal Sunshine of the Spotless Mind": 108,
    "Good Will Hunting": 126,
    "Dead Poets Society": 128,
    "The Pursuit of Happyness": 117,
    "A Beautiful Mind": 135,
    "The Social Network": 120,
    "Titanic": 194,
    "Casablanca": 102,
    "12 Angry Men": 96,
    "Psycho": 109,
    "Jaws": 124,
    "E.T. the Extra-Terrestrial": 115,
    "Raiders of the Lost Ark": 115,
    "Ferris Bueller's Day Off": 103,
    "The Hangover": 100,
    "Bridesmaids": 125,
    "Mamma Mia!": 108,
    "The Greatest Showman": 105,
    "Bohemian Rhapsody": 134,
    "Pride & Prejudice": 129,
    "Before Sunrise": 101,
    "The Princess Bride": 98,
    "Harry Potter and the Prisoner of Azkaban": 142,
    "Pirates of the Caribbean: The Curse of the Black Pearl": 143,
    "Top Gun: Maverick": 130,
    "Mission: Impossible - Fallout": 147,
    "John Wick": 101,
    "Skyfall": 143,
    "Casino Royale": 144,
    "Edge of Tomorrow": 113,
    "Arrival": 116,
    "Ex Machina": 108,
    "Gravity": 91,
    "A Quiet Place": 90,
    "Drishyam 2": 154,
    "Lucifer": 174,
    "Pulimurugan": 161,
    "Manichitrathazhu": 165,
    "Kireedam": 146,
    "Spadikam": 160,
    "Thanmathra": 160,
    "Kilukkam": 155,
    "Nadodikkattu": 140,
    "Sandesham": 140,
    "In Harihar Nagar": 144,
    "Ramji Rao Speaking": 143,
    "Thenmavin Kombath": 160,
    "Oru Vadakkan Veeragatha": 166,
    "Amaram": 130,
    "Pranchiyettan & the Saint": 150,
    "Kannur Squad": 160,
    "Bheeshma Parvam": 157,
    "Nanpakal Nerathu Mayakkam": 104,
    "Kaathal - The Core": 114,
    "Premalu": 156,
    "Aavesham": 158,
    "Aadujeevitham": 173,
    "2018: Everyone Is a Hero": 148,
    "Romancham": 146,
    "Nna Thaan Case Kodu": 138,
    "Jaya Jaya Jaya Jaya Hey": 138,
    "Joji": 113,
    "Malik": 160,
    "Kappela": 109,
    "Ayyappanum Koshiyum": 177,
    "Trance": 170,
    "Virus": 153,
    "Uyare": 130,
    "Ishq": 127,
    "Android Kunjappan Version 5.25": 135,
    "Charlie": 156,
    "Ennu Ninte Moideen": 172,
    "Ohm Shanthi Oshana": 140,
    "Traffic": 120,
    "Salt N' Pepper": 140,
    "Annayum Rasoolum": 130,
    "Diamond Necklace": 137,
    "22 Female Kottayam": 130,
    "Mumbai Police": 120,
    "Memories": 127,
    "Ayalum Njanum Thammil": 135,
    "Njan Prakashan": 140,
    "Sudani from Nigeria": 123,
    "Thondimuthalum Driksakshiyum": 130,
    "Kammatti Paadam": 177,
    "Take Off": 138,
    "Varathan": 135,
    "Parava": 140,
    "Joseph": 123,
    "Mayaanadhi": 135,
    "Anjaam Pathira": 143,
    "Home": 153,
    "Nayattu": 120,
    "The Great Indian Kitchen": 100,
    "Kuruthi": 117,
    "Hridayam": 172,
    "Rorschach": 141,
    "Thallumaala": 162,
    "Jan.E.Man": 110,
    "Spirit": 165,
    "Celluloid": 120,
    "Oppam": 160,
    "Munthirivallikal Thalirkkumbol": 152,
    "Marakkar: Arabikadalinte Simham": 178,
    "Aarkkariyam": 127,
    "Devasuram": 165,
    "Aaram Thampuran": 170,
    "Narasimham": 170,
    "Pavithram": 140,
    "Manassinakkare": 135,
    "Meesha Madhavan": 160,
    "CID Moosa": 150,
    "Punjabi House": 165,
    "Gandhinagar 2nd Street": 140,
    "Neelakasham Pachakadal Chuvanna Bhoomi": 130,
    "Bhaskar the Rascal": 150,
    "Munnariyippu": 105,
    "Shutter": 125,
    "Neram": 121,
    "Thattathin Marayathu": 138,
    "Vikramadithyan": 155,
    "Aadu": 152,
    "Sholay": 204,
    "Dilwale Dulhania Le Jayenge": 189,
    "Lagaan": 224,
    "Rang De Basanti": 167,
    "Swades": 210,
    "Chak De! India": 153,
    "Munna Bhai M.B.B.S.": 156,
    "Lage Raho Munna Bhai": 144,
    "PK": 153,
    "Bajrangi Bhaijaan": 163,
    "Kabhi Khushi Kabhie Gham": 210,
    "Kuch Kuch Hota Hai": 185,
    "Devdas": 185,
    "Jab We Met": 138,
    "Barfi!": 151,
    "Yeh Jawaani Hai Deewani": 160,
    "Rockstar": 159,
    "Wake Up Sid": 138,
    "Tamasha": 139,
    "Dev.D": 144,
    "Gangs of Wasseypur": 160,
    "Gangs of Wasseypur Part 2": 160,
    "Black Friday": 143,
    "A Wednesday!": 104,
    "Special 26": 144,
    "Badla": 118,
    "Pink": 136,
    "Article 15": 130,
    "Talvar": 132,
    "Haider": 160,
    "Udta Punjab": 148,
    "Raazi": 138,
    "Uri: The Surgical Strike": 138,
    "Sardar Udham": 162,
    "Shershaah": 135,
    "Bhaag Milkha Bhaag": 186,
    "Mary Kom": 122,
    "M.S. Dhoni: The Untold Story": 184,
    "Sanju": 161,
    "Super 30": 154,
    "Hichki": 116,
    "English Vinglish": 134,
    "Piku": 123,
    "Vicky Donor": 126,
    "Badhaai Ho": 124,
    "Stree": 128,
    "Bhool Bhulaiyaa": 159,
    "Newton": 106,
    "Masaan": 109,
    "The Lunchbox": 104,
    "Om Shanti Om": 162,
    "Dhoom 2": 152,
    "War": 156,
    "Pathaan": 146,
    "Jawan": 169,
    "Animal": 201,
    "12th Fail": 147,
    "Laapataa Ladies": 122,
    "Kai Po Che!": 130,
    "Kapoor & Sons": 132,
    "Rock On!!": 145,
    "Mughal-E-Azam": 197,
    "Mother India": 172,
    "Pyaasa": 146,
    "Anand": 122,
    "Deewaar": 174,
    "Mr. India": 179,
    "Hera Pheri": 156,
    "Andaz Apna Apna": 160,
    "Chhichhore": 143,
    "Baby": 159,
    "Neerja": 122,
    "The Sky Is Pink": 150,
    "Tanu Weds Manu": 105,
    "Dear Zindagi": 151,
    "Highway": 133,
    "2 States": 149,
    "Veer-Zaara": 192,
    "Kal Ho Naa Ho": 186,
    "Border": 178,
    "Dil Se..": 158,
    "Sarfarosh": 174,
    "Rangeela": 150,
    "Khosla Ka Ghosla": 120,
    "Oye Lucky! Lucky Oye!": 140,
    "Johnny Gaddaar": 130,
    "Delhi Belly": 103,
    "Shahid": 129,
    "Hum Dil De Chuke Sanam": 188,
    "Ludo": 149,
    "Asuran": 141,
    "Vada Chennai": 164,
    "Jai Bhim": 164,
    "Ratsasan": 170,
    "Pariyerum Perumal": 154,
    "Anbe Sivam": 160,
    "Nayakan": 145,
    "Jigarthanda": 171,
    "Ponniyin Selvan: I": 167,
    "Master": 179,
    "Memories of Murder": 131,
    "The Handmaiden": 145,
    "Burning": 148,
    "The Wailing": 156,
    "A Taxi Driver": 137,
    "The Man from Nowhere": 119,
    "I Saw the Devil": 144,
    "The Host": 120,
    "Decision to Leave": 138,
    "Extreme Job": 111,
    "Princess Mononoke": 134,
    "My Neighbor Totoro": 86,
    "Grave of the Fireflies": 89,
    "Howl's Moving Castle": 119,
    "Akira": 124,
    "Ikiru": 143,
    "Rashomon": 88,
    "Shoplifters": 121,
    "A Silent Voice": 130,
    "Drive My Car": 179,
    "Les Miserables": 104,
    "The 400 Blows": 99,
    "La Haine": 98,
    "Portrait of a Lady on Fire": 122,
    "The Diving Bell and the Butterfly": 112,
    "A Prophet": 155,
    "Amour": 127,
    "Persepolis": 96,
    "Les Diaboliques": 117,
    "Delicatessen": 99,
    "Y Tu Mama Tambien": 106,
    "The Orphanage": 105,
    "Talk to Her": 112,
    "Amores Perros": 154,
    "All About My Mother": 101,
    "Roma": 135,
    "The Platform": 94,
    "Wild Tales": 122,
    "Pain and Glory": 113,
    "The Sea Inside": 125,
}


def ensure_runtime(df: pd.DataFrame):
    """Return (df with an int 'runtime' column, True if the runtimes are estimates)."""
    d = df.copy()
    estimated = "runtime" not in d.columns
    if estimated:
        d["runtime"] = d["title"].map(RUNTIMES).fillna(120).astype(int)
    else:
        d["runtime"] = pd.to_numeric(d["runtime"], errors="coerce").fillna(120).astype(int)
    return d, estimated


# ----------------------------------------------------------------------------- chat parsing
GENRE_WORDS = {"scary": "Horror", "spooky": "Horror", "creepy": "Horror", "funny": "Comedy", "hilarious": "Comedy",
               "laugh": "Comedy", "romcom": "Romance", "love story": "Romance", "space": "Sci-Fi", "scifi": "Sci-Fi",
               "sci fi": "Sci-Fi", "futuristic": "Sci-Fi", "cartoon": "Animation", "animated": "Animation",
               "anime": "Animation", "suspense": "Thriller", "detective": "Mystery", "whodunit": "Mystery",
               "superhero": "Action", "sports": "Sport", "biopic": "Biography", "kids": "Family"}
MOOD_WORDS = {"sad": "emotional", "cry": "emotional", "tearjerker": "emotional", "feel good": "heartwarming",
              "feel-good": "heartwarming", "wholesome": "heartwarming", "cute": "heartwarming",
              "cheerful": "uplifting", "twisty": "mind-bending", "trippy": "mind-bending", "gritty": "dark",
              "motivational": "inspiring", "easy": "light", "chill": "light", "tense": "intense",
              "edge of my seat": "thrilling", "weird": "quirky", "nostalgia": "nostalgic"}
LANG_WORDS = {"bollywood": "Hindi", "mollywood": "Malayalam", "kollywood": "Tamil", "k-drama": "Korean",
              "kdrama": "Korean", "ghibli": "Japanese"}
NEG = r"\b(?:no|avoid|without|skip|not|nothing)\s+(?:any\s+|too\s+|much\s+)?$"


def _find(word, text):
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text)


def _add(lst, v):
    if v not in lst:
        lst.append(v)


def parse_query(text: str, agent, prof: Profile):
    """Turn a sentence into profile updates. Returns (profile, chips, was_reset)."""
    t = text.lower().strip()
    if re.search(r"\b(reset|start over|clear everything|forget it)\b", t):
        return Profile(), ["profile cleared"], True
    p, chips = prof.copy(), []
    top_year = int(agent.df.year.max())

    def tag(vocab, mapping, pos, neg, label):
        cands = [(w, v) for w, v in mapping.items() if v in vocab] + [(v.lower(), v) for v in vocab]
        for w, v in cands:
            m = _find(w, t)
            if not m:
                continue
            if re.search(NEG, t[:m.start()]):
                if v not in neg:
                    _add(neg, v)
                    chips.append(f"avoid {v}")
            elif v not in pos:
                _add(pos, v)
                chips.append(f"{label}: {v}")

    avoid_g = p.avoid_genres
    tag(agent.all_genres, GENRE_WORDS, p.genres, avoid_g, "genre")
    p.genres = [g for g in p.genres if g not in avoid_g]
    avoid_m = p.avoid_moods
    tag(agent.all_moods, MOOD_WORDS, p.moods, avoid_m, "mood")
    p.moods = [m for m in p.moods if m not in avoid_m]
    for w, v in {**LANG_WORDS, **{x.lower(): x for x in agent.all_langs}}.items():
        if v in agent.all_langs and _find(w, t) and v not in p.languages:
            p.languages.append(v)
            chips.append(f"language: {v}")

    m = re.search(r"\b(19|20)?(\d)0s\b", t)
    if m:
        pre = m.group(1) or ("19" if int(m.group(2)) >= 3 else "20")
        y = int(pre) * 100 + int(m.group(2)) * 10
        p.era = (y, y + 9)
        chips.append(f"era: {y}s")
    else:
        a = re.search(r"(?:after|since|from)\s+((?:19|20)\d{2})", t)
        b = re.search(r"(?:before|until|prior to)\s+((?:19|20)\d{2})", t)
        if a or b:
            p.era = (int(a.group(1)) if a else 1900, int(b.group(1)) if b else top_year)
            chips.append(f"era: {p.era[0]}-{p.era[1]}")
        elif re.search(r"\b(recent|new|latest|modern)\b", t):
            p.era = (2015, top_year)
            chips.append("era: 2015+")
        elif re.search(r"\b(classic|classics|old|vintage)\b", t):
            p.era = (1900, 2000)
            chips.append("era: classics (<=2000)")

    m = re.search(r"(?:above|over|at least|rated|rating|min(?:imum)?)\s*(\d(?:\.\d)?)"
                  r"(?![\d.]|\s*(?:hours?|hrs?|h|minutes?|mins?|m)\b)", t)
    if m and 0 < float(m.group(1)) <= 10:
        p.min_rating = float(m.group(1))
        chips.append(f"rating >= {p.min_rating:g}")
    elif re.search(r"\b(highly rated|top rated|best|acclaimed)\b", t):
        p.min_rating = 8.0
        chips.append("rating >= 8")

    m = re.search(r"(?:under|below|less than|within|max(?:imum)?|shorter than)\s*(\d+(?:\.\d+)?)\s*"
                  r"(hours?|hrs?|h|minutes?|mins?|m)\b", t)
    if m:
        v = float(m.group(1))
        p.max_runtime = int(v * 60) if m.group(2).startswith("h") else int(v)
        chips.append(f"runtime <= {p.max_runtime} min")
    elif re.search(r"\bshort\b", t):
        p.max_runtime = 110
        chips.append("runtime <= 110 min")

    ctx = [("Family", r"\b(family|kids|children)\b"), ("Partner", r"\b(date|partner|girlfriend|boyfriend|wife|husband)\b"),
           ("Friends", r"\bfriends?\b"), ("Alone", r"\b(alone|solo|myself)\b")]
    for name, pat in ctx:
        if re.search(pat, t):
            p.context = name
            chips.append(f"watching: {name.lower()}")
            break

    cue = re.search(r"\b(like|similar to|love|loved|liked|favou?rite|enjoyed|watched|seen)\b", t)
    for title in agent.df.title:
        # one-word titles ("War", "Home", "Animal"...) are ordinary words, so they only count after a cue like "like X"
        if " " not in title.strip() and not cue:
            continue
        if _find(title.lower(), t):
            lst = p.seen if re.search(r"\b(seen|watched)\b", t) else p.favourites
            if title not in lst:
                lst.append(title)
                chips.append(("seen: " if lst is p.seen else "like: ") + title)
    for a in agent.all_actors:
        if _find(a.lower(), t) and a not in p.actors:
            p.actors.append(a)
            chips.append(f"actor: {a}")
    return p, chips, False


# ----------------------------------------------------------------------------- lab
def sensitivity(agent, p: Profile, key: str, grid, k: int = 5) -> pd.DataFrame:
    """Sweep one criterion's weight; report overlap of the top-k with the top-k at the current weight."""
    keep = agent.weights[key]
    try:
        base = {r.idx for r in agent.recommend(p, k, explore=False).picks}
        rows = []
        for g in grid:
            agent.weights[key] = float(g)
            picks = agent.recommend(p, k, explore=False).picks
            ids = {r.idx for r in picks}
            rows.append({"Weight": float(g), "Overlap": len(ids & base) / max(1, len(base)),
                         "Top pick": picks[0].title if picks else "-", "Picks": ", ".join(r.title for r in picks)})
    finally:
        agent.weights[key] = keep
    return pd.DataFrame(rows)


def similar_to(agent, idx_list, k=5, exclude=()):
    """Top-k movie indices most similar (mean cosine) to a set of movies."""
    if not idx_list:
        return []
    s = np.mean([agent.S[i] for i in idx_list], axis=0)
    skip = set(idx_list) | set(exclude)
    order = [int(j) for j in np.argsort(-s) if j not in skip]
    return order[:k]
