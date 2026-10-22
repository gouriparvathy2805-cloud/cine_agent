# CineAgent v5
New files: app_v5.py, tmdb.py, .streamlit/config.toml (+ secrets example).
Keep the existing: agent_v3.py, evaluate_v3.py, v4_helpers.py, movies.csv, requirements.txt (no new packages).

    streamlit run app_v5.py

Free TMDB key: themoviedb.org > Settings > API. Put it in .streamlit/secrets.toml as TMDB_API_KEY,
or paste it in the app sidebar (Posters & streaming). Without a key the app still works (colour tiles,
trailer button opens a YouTube search).

Separate link: deploy a NEW app (Streamlit Community Cloud > New app) with main file app_v5.py.
Your v4 link keeps running untouched. v5 saves to watchlist_v5.csv and tmdb_cache_v5.json.
