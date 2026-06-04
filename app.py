from __future__ import annotations

import os
import glob
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
import time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

try:
    import requests
except Exception:  # requests is optional; Spotify preview gracefully falls back without it
    requests = None

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mood_model.pkl"
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo"
]
MOODS = ["Happy", "Energetic", "Calm", "Romantic", "Angry", "Low Energy", "Sad"]

# Songs that are NOT taken from dataset.csv. These are only used as an "outside dataset"
# mixed recommendation pool, because this project does not use Spotify API / external API.
OUTSIDE_DATASET: Dict[str, List[Dict[str, Any]]] = {
    "Happy": [
        {"title": "Happy", "artist": "Pharrell Williams", "genre": "pop", "language": "English"},
        {"title": "Good as Hell", "artist": "Lizzo", "genre": "pop", "language": "English"},
        {"title": "Can’t Stop the Feeling!", "artist": "Justin Timberlake", "genre": "dance", "language": "English"},
        {"title": "Sugar", "artist": "Maroon 5", "genre": "pop", "language": "English"},
        {"title": "Dynamite", "artist": "BTS", "genre": "k-pop", "language": "Korean"},
        {"title": "Cupid", "artist": "FIFTY FIFTY", "genre": "k-pop", "language": "Korean"},
        {"title": "Hati-Hati di Jalan", "artist": "Tulus", "genre": "pop", "language": "Indonesian"},
        {"title": "Tak Kan Hilang", "artist": "Budi Doremi", "genre": "pop", "language": "Indonesian"},
        {"title": "Shake It Off", "artist": "Taylor Swift", "genre": "pop", "language": "English"},
        {"title": "Treasure", "artist": "Bruno Mars", "genre": "funk", "language": "English"},
    ],
    "Energetic": [
        {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "synth-pop", "language": "English"},
        {"title": "Levels", "artist": "Avicii", "genre": "edm", "language": "English"},
        {"title": "Stronger", "artist": "Kanye West", "genre": "hip-hop", "language": "English"},
        {"title": "Titanium", "artist": "David Guetta;Sia", "genre": "edm", "language": "English"},
        {"title": "Bang Bang Bang", "artist": "BIGBANG", "genre": "k-pop", "language": "Korean"},
        {"title": "Super Shy", "artist": "NewJeans", "genre": "k-pop", "language": "Korean"},
        {"title": "Ddu-Du Ddu-Du", "artist": "BLACKPINK", "genre": "k-pop", "language": "Korean"},
        {"title": "Satu-Satu", "artist": "Idgitaf", "genre": "pop", "language": "Indonesian"},
        {"title": "Uptown Funk", "artist": "Mark Ronson;Bruno Mars", "genre": "funk", "language": "English"},
        {"title": "Don't Stop Me Now", "artist": "Queen", "genre": "rock", "language": "English"},
    ],
    "Calm": [
        {"title": "Weightless", "artist": "Marconi Union", "genre": "ambient", "language": "Instrumental"},
        {"title": "Bloom", "artist": "The Paper Kites", "genre": "indie", "language": "English"},
        {"title": "Holocene", "artist": "Bon Iver", "genre": "indie", "language": "English"},
        {"title": "River Flows in You", "artist": "Yiruma", "genre": "piano", "language": "Instrumental"},
        {"title": "Rehat", "artist": "Kunto Aji", "genre": "pop", "language": "Indonesian"},
        {"title": "Secukupnya", "artist": "Hindia", "genre": "indie", "language": "Indonesian"},
        {"title": "Spring Day", "artist": "BTS", "genre": "k-pop", "language": "Korean"},
        {"title": "Ditto", "artist": "NewJeans", "genre": "k-pop", "language": "Korean"},
        {"title": "Sunset Lover", "artist": "Petit Biscuit", "genre": "electronic", "language": "Instrumental"},
        {"title": "Banana Pancakes", "artist": "Jack Johnson", "genre": "acoustic", "language": "English"},
    ],
    "Romantic": [
        {"title": "Perfect", "artist": "Ed Sheeran", "genre": "pop", "language": "English"},
        {"title": "All of Me", "artist": "John Legend", "genre": "r-n-b", "language": "English"},
        {"title": "Until I Found You", "artist": "Stephen Sanchez", "genre": "romance", "language": "English"},
        {"title": "Komang", "artist": "Raim Laode", "genre": "pop", "language": "Indonesian"},
        {"title": "Adu Rayu", "artist": "Yovie Widianto;Tulus;Glenn Fredly", "genre": "pop", "language": "Indonesian"},
        {"title": "Melukis Senja", "artist": "Budi Doremi", "genre": "pop", "language": "Indonesian"},
        {"title": "Love Scenario", "artist": "iKON", "genre": "k-pop", "language": "Korean"},
        {"title": "Only", "artist": "LeeHi", "genre": "k-pop", "language": "Korean"},
        {"title": "La Vie En Rose", "artist": "Édith Piaf", "genre": "french", "language": "French"},
        {"title": "Just the Way You Are", "artist": "Bruno Mars", "genre": "pop", "language": "English"},
    ],
    "Angry": [
        {"title": "Numb", "artist": "Linkin Park", "genre": "alt-rock", "language": "English"},
        {"title": "Killing in the Name", "artist": "Rage Against The Machine", "genre": "rock", "language": "English"},
        {"title": "Break Stuff", "artist": "Limp Bizkit", "genre": "metal", "language": "English"},
        {"title": "Duality", "artist": "Slipknot", "genre": "metal", "language": "English"},
        {"title": "God's Menu", "artist": "Stray Kids", "genre": "k-pop", "language": "Korean"},
        {"title": "MIROH", "artist": "Stray Kids", "genre": "k-pop", "language": "Korean"},
        {"title": "Bohemian Rhapsody", "artist": "Reality Club", "genre": "rock", "language": "Indonesian"},
        {"title": "C.H.R.I.S.Y.E.", "artist": "Diskoria;Laleilmanino;Eva Celia", "genre": "disco", "language": "Indonesian"},
        {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "grunge", "language": "English"},
        {"title": "Bodies", "artist": "Drowning Pool", "genre": "metal", "language": "English"},
    ],
    "Low Energy": [
        {"title": "Space Song", "artist": "Beach House", "genre": "dream-pop", "language": "English"},
        {"title": "Night Changes", "artist": "One Direction", "genre": "pop", "language": "English"},
        {"title": "Slow Dancing in the Dark", "artist": "Joji", "genre": "r-n-b", "language": "English"},
        {"title": "Sparks", "artist": "Coldplay", "genre": "alternative", "language": "English"},
        {"title": "Evaluasi", "artist": "Hindia", "genre": "indie", "language": "Indonesian"},
        {"title": "Jakarta Hari Ini", "artist": "For Revenge;Stereo Wall", "genre": "emo", "language": "Indonesian"},
        {"title": "Through the Night", "artist": "IU", "genre": "k-pop", "language": "Korean"},
        {"title": "Still With You", "artist": "Jungkook", "genre": "k-pop", "language": "Korean"},
        {"title": "The Night We Met", "artist": "Lord Huron", "genre": "folk", "language": "English"},
        {"title": "Asleep", "artist": "The Smiths", "genre": "indie", "language": "English"},
    ],
    "Sad": [
        {"title": "Someone Like You", "artist": "Adele", "genre": "pop", "language": "English"},
        {"title": "Fix You", "artist": "Coldplay", "genre": "alternative", "language": "English"},
        {"title": "drivers license", "artist": "Olivia Rodrigo", "genre": "pop", "language": "English"},
        {"title": "Happier", "artist": "Olivia Rodrigo", "genre": "pop", "language": "English"},
        {"title": "Bertaut", "artist": "Nadin Amizah", "genre": "indie", "language": "Indonesian"},
        {"title": "Runtuh", "artist": "Feby Putri;Fiersa Besari", "genre": "pop", "language": "Indonesian"},
        {"title": "To My Youth", "artist": "BOL4", "genre": "k-pop", "language": "Korean"},
        {"title": "Eight", "artist": "IU;SUGA", "genre": "k-pop", "language": "Korean"},
        {"title": "Let Her Go", "artist": "Passenger", "genre": "folk", "language": "English"},
        {"title": "When I Was Your Man", "artist": "Bruno Mars", "genre": "pop", "language": "English"},
    ],
}

LANGUAGE_BY_GENRE = {
    "cantopop": "Chinese", "mandopop": "Chinese", "j-pop": "Japanese", "j-rock": "Japanese",
    "j-dance": "Japanese", "j-idol": "Japanese", "anime": "Japanese", "k-pop": "Korean",
    "latin": "Spanish", "latino": "Spanish", "reggaeton": "Spanish", "salsa": "Spanish",
    "spanish": "Spanish", "tango": "Spanish", "brazil": "Portuguese", "forro": "Portuguese",
    "mpb": "Portuguese", "pagode": "Portuguese", "samba": "Portuguese", "sertanejo": "Portuguese",
    "french": "French", "german": "German", "swedish": "Swedish", "turkish": "Turkish",
    "iranian": "Persian", "indian": "Hindi", "malay": "Malay", "piano": "Instrumental",
    "ambient": "Instrumental", "classical": "Instrumental", "study": "Instrumental", "sleep": "Instrumental",
}


def infer_language(genre: str) -> str:
    return LANGUAGE_BY_GENRE.get(str(genre).lower(), "English")


def spotify_search_url(title: Any, artist: Any) -> str:
    from urllib.parse import quote_plus
    query = quote_plus(f"{title} {artist}")
    return f"https://open.spotify.com/search/{query}"

_SPOTIFY_TOKEN_CACHE: Dict[str, Any] = {"token": None, "expires_at": 0}


def spotify_track_url(track_id: str, title: str, artist: str) -> str:
    track_id = str(track_id or "").strip()
    if track_id:
        return f"https://open.spotify.com/track/{track_id}"
    return spotify_search_url(title, artist)


def spotify_embed_url(track_id: str) -> str:
    track_id = str(track_id or "").strip()
    if not track_id:
        return ""
    return f"https://open.spotify.com/embed/track/{track_id}"


def get_spotify_token() -> Optional[str]:
    """Return Spotify client-credentials token when env vars are available."""
    if requests is None:
        return None
    if _SPOTIFY_TOKEN_CACHE.get("token") and time.time() < float(_SPOTIFY_TOKEN_CACHE.get("expires_at", 0)):
        return str(_SPOTIFY_TOKEN_CACHE["token"])
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        _SPOTIFY_TOKEN_CACHE["token"] = data.get("access_token")
        _SPOTIFY_TOKEN_CACHE["expires_at"] = time.time() + int(data.get("expires_in", 3600)) - 60
        return _SPOTIFY_TOKEN_CACHE["token"]
    except Exception:
        return None


@lru_cache(maxsize=2000)
def fetch_spotify_metadata(title: str, artist: str, track_id: str = "") -> Dict[str, str]:
    metadata = {
        "spotify_url": spotify_track_url(track_id, title, artist),
        "spotify_embed_url": spotify_embed_url(track_id),
        "preview_url": "",
        "track_id": str(track_id or "").strip(),
    }
    token = get_spotify_token()
    if not token or requests is None:
        return metadata
    headers = {"Authorization": f"Bearer {token}"}
    try:
        if metadata["track_id"]:
            response = requests.get(f"https://api.spotify.com/v1/tracks/{metadata['track_id']}", headers=headers, timeout=8)
        else:
            query = f'track:"{title}" artist:"{artist}"'
            response = requests.get(
                "https://api.spotify.com/v1/search",
                headers=headers,
                params={"q": query, "type": "track", "limit": 1},
                timeout=8,
            )
        response.raise_for_status()
        data = response.json()
        item = data if metadata["track_id"] else (data.get("tracks", {}).get("items", []) or [{}])[0]
        if item:
            found_id = item.get("id") or metadata["track_id"]
            metadata["track_id"] = found_id or ""
            metadata["spotify_url"] = item.get("external_urls", {}).get("spotify") or spotify_track_url(found_id, title, artist)
            metadata["spotify_embed_url"] = spotify_embed_url(found_id)
            metadata["preview_url"] = item.get("preview_url") or ""
    except Exception:
        pass
    return metadata



def ensure_model() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return

    mlruns_zip = BASE_DIR / "mlruns.zip"
    if mlruns_zip.exists():
        extract_dir = BASE_DIR / ".mlruns_extracted"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(mlruns_zip, "r") as zf:
            zf.extractall(extract_dir)
        candidates = glob.glob(str(extract_dir / "**" / "artifacts" / "model.pkl"), recursive=True)
        if candidates:
            latest = max(candidates, key=lambda p: os.path.getmtime(p))
            MODEL_PATH.write_bytes(Path(latest).read_bytes())
            return

    direct_model = BASE_DIR / "model.pkl"
    if direct_model.exists():
        MODEL_PATH.write_bytes(direct_model.read_bytes())


ensure_model()
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def assign_mood(row: pd.Series) -> str:
    v, e, d, a = row["valence"], row["energy"], row["danceability"], row["acousticness"]
    if v >= 0.5 and e >= 0.5:
        return "Energetic" if d >= 0.7 else "Happy"
    if v < 0.5 and e >= 0.5:
        return "Angry"
    if v >= 0.5 and e < 0.5:
        return "Romantic" if a >= 0.5 else "Calm"
    return "Low Energy" if a >= 0.6 else "Sad"


@lru_cache(maxsize=1)
def load_song_database() -> Optional[pd.DataFrame]:
    songs_pkl = BASE_DIR / "songs.pkl"
    dataset_csv = BASE_DIR / "dataset.csv"

    if songs_pkl.exists():
        df = pd.DataFrame(joblib.load(songs_pkl))
    elif dataset_csv.exists():
        df = pd.read_csv(dataset_csv)
    else:
        return None

    required = set(FEATURES + ["track_name", "artists"])
    if not required.issubset(df.columns):
        return None

    df = df.dropna(subset=FEATURES + ["track_name", "artists"]).copy()
    df = df.drop_duplicates(subset=["track_name", "artists"])
    if "mood" not in df.columns:
        df["mood"] = df.apply(assign_mood, axis=1)
    if "track_genre" not in df.columns:
        df["track_genre"] = "dataset"
    df["language"] = df["track_genre"].apply(infer_language)
    if "popularity" in df.columns:
        df = df[df["popularity"].fillna(0) >= 10].copy()
    else:
        df["popularity"] = 50
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def get_scaler() -> Optional[StandardScaler]:
    df = load_song_database()
    if df is None:
        return None
    scaler_path = BASE_DIR / "scaler.pkl"
    if scaler_path.exists():
        try:
            return joblib.load(scaler_path)
        except Exception:
            pass
    scaler = StandardScaler()
    scaler.fit(df[FEATURES])
    return scaler


def clean_feature_payload(data: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for feature in FEATURES:
        value = data.get(feature)
        if value is None or value == "":
            raise ValueError(f"Missing feature: {feature}")
        row[feature] = float(value)
    return pd.DataFrame([row], columns=FEATURES)


def probability_for(predicted: str, frame: pd.DataFrame) -> float:
    if model is None or not hasattr(model, "predict_proba"):
        return 0.92
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    if predicted in classes:
        return float(probabilities[classes.index(predicted)])
    return float(np.max(probabilities))


def mood_centroid(mood: str, df: pd.DataFrame) -> pd.DataFrame:
    mood_df = df[df["mood"].astype(str).str.lower() == mood.lower()]
    if mood_df.empty:
        mood_df = df
    return pd.DataFrame([mood_df[FEATURES].mean(numeric_only=True).to_dict()], columns=FEATURES)


def normalize_filter(value: Any) -> str:
    value = str(value or "all").strip()
    return "all" if value == "" else value


def apply_filters(df: pd.DataFrame, genre: str = "all", language: str = "all") -> pd.DataFrame:
    filtered = df.copy()
    genre = normalize_filter(genre).lower()
    language = normalize_filter(language).lower()
    if genre != "all":
        filtered = filtered[filtered["track_genre"].astype(str).str.lower() == genre]
    if language != "all":
        filtered = filtered[filtered["language"].astype(str).str.lower() == language]
    return filtered


def recommend_from_dataset(
    mood: str,
    feature_frame: Optional[pd.DataFrame] = None,
    limit: int = 6,
    genre: str = "all",
    language: str = "all",
    avoid_keys: Optional[List[str]] = None,
    repeat_chance: float = 0.33,
) -> List[Dict[str, Any]]:
    df = load_song_database()
    scaler = get_scaler()
    if df is None or scaler is None or limit <= 0:
        return []

    filtered_df = apply_filters(df, genre, language)
    if filtered_df.empty:
        filtered_df = df.copy()

    mood_df = filtered_df[filtered_df["mood"].astype(str).str.lower() == mood.lower()].copy()
    if mood_df.empty:
        mood_df = filtered_df.copy()

    avoid_set = {str(key).lower() for key in (avoid_keys or [])}
    allow_repeat = random.random() < repeat_chance
    if avoid_set and not allow_repeat:
        fresh_df = mood_df[~mood_df.apply(lambda row: song_key(row.get("track_name", ""), row.get("artists", "")) in avoid_set, axis=1)].copy()
        # Keep the page full even when the mood/filter has too few unseen songs left.
        if len(fresh_df) >= max(1, min(limit, 10)):
            mood_df = fresh_df

    query = feature_frame if feature_frame is not None else mood_centroid(mood, df)
    X = scaler.transform(mood_df[FEATURES])
    q = scaler.transform(query[FEATURES])[0]
    distances = np.linalg.norm(X - q, axis=1)
    mood_df["distance"] = distances
    mood_df["match_score"] = 100 - np.clip(distances * 10, 0, 30)

    # Take a wider nearest-neighbor candidate set, then sample randomly.
    # This keeps it KNN-like but prevents the same songs from appearing every click.
    candidate_count = min(len(mood_df), max(limit * 12, 120))
    candidates = mood_df.sort_values(["distance", "popularity"], ascending=[True, False]).head(candidate_count)
    if len(candidates) > limit:
        candidates = candidates.sample(n=limit, weights=(candidates["popularity"].fillna(0) + 1), replace=False)
    candidates = candidates.sort_values("match_score", ascending=False)

    output = []
    for row in candidates.to_dict("records"):
        title = row.get("track_name", "Unknown Track")
        artist = row.get("artists", "Unknown Artist")
        spotify_meta = fetch_spotify_metadata(title, artist, row.get("track_id", ""))
        output.append({
            "title": title,
            "artist": artist,
            "score": round(float(row.get("match_score", 88)), 1),
            "mood": mood,
            "source": "dataset",
            "genre": row.get("track_genre", "dataset"),
            "language": row.get("language", "English"),
            **spotify_meta,
        })
    return output


def outside_matches(song: Dict[str, Any], genre: str, language: str) -> bool:
    genre = normalize_filter(genre).lower()
    language = normalize_filter(language).lower()
    if genre != "all" and str(song.get("genre", "")).lower() != genre:
        return False
    if language != "all" and str(song.get("language", "")).lower() != language:
        return False
    return True


def song_key(title: Any, artist: Any) -> str:
    return f"{str(title).strip().lower()}::{str(artist).strip().lower()}"


def recommend_outside_dataset(
    mood: str,
    limit: int = 10,
    genre: str = "all",
    language: str = "all",
    avoid_keys: Optional[List[str]] = None,
    repeat_chance: float = 0.33,
) -> List[Dict[str, Any]]:
    pool = [song for song in OUTSIDE_DATASET.get(mood, []) if outside_matches(song, genre, language)]
    avoid_set = {str(key).lower() for key in (avoid_keys or [])}
    if avoid_set and random.random() >= repeat_chance:
        fresh_pool = [song for song in pool if song_key(song.get("title", ""), song.get("artist", "")) not in avoid_set]
        if fresh_pool:
            pool = fresh_pool
    if len(pool) < limit and (normalize_filter(genre) != "all" or normalize_filter(language) != "all"):
        # Keep results useful even when the manual outside pool has few exact matches.
        pool.extend([song for song in OUTSIDE_DATASET.get(mood, []) if song not in pool])
    if not pool:
        return []
    picked = random.sample(pool, k=min(limit, len(pool)))
    return [
        {
            "title": song["title"],
            "artist": song["artist"],
            "score": max(72, 96 - idx * random.uniform(1.2, 3.4)),
            "mood": mood,
            "source": "outside dataset",
            "genre": song.get("genre", "external pool"),
            "language": song.get("language", "English"),
            **fetch_spotify_metadata(song["title"], song["artist"], ""),
        }
        for idx, song in enumerate(picked)
    ]


def mixed_recommendations(
    mood: str,
    feature_frame: Optional[pd.DataFrame] = None,
    total_limit: int = 30,
    outside_ratio: float = 0.625,
    genre: str = "all",
    language: str = "all",
    avoid_keys: Optional[List[str]] = None,
    repeat_chance: float = 0.33,
) -> List[Dict[str, Any]]:
    # User can select 20-40 songs. Outside pool is used first, then dataset fills the rest.
    # Repeat control: old songs only have about 33% chance to be allowed again after refresh.
    total_limit = max(20, min(40, int(total_limit or 30)))
    outside_limit = round(total_limit * outside_ratio)
    outside_songs = recommend_outside_dataset(mood, outside_limit, genre, language, avoid_keys, repeat_chance)
    dataset_limit = max(total_limit - len(outside_songs), total_limit - outside_limit)
    dataset_songs = recommend_from_dataset(mood, feature_frame, dataset_limit, genre, language, avoid_keys, repeat_chance)
    songs = outside_songs + dataset_songs

    if len(songs) < total_limit:
        existing = {song_key(song.get("title", ""), song.get("artist", "")) for song in songs}
        fillers = recommend_from_dataset(mood, feature_frame, total_limit - len(songs), genre, language, [], 1.0)
        songs.extend([song for song in fillers if song_key(song.get("title", ""), song.get("artist", "")) not in existing])

    random.shuffle(songs)
    return songs[:total_limit]


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    df = load_song_database()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": list(getattr(model, "classes_", MOODS)) if model is not None else MOODS,
        "uses_real_song_dataset": df is not None,
        "dataset_song_count": int(len(df)) if df is not None else 0,
        "outside_dataset_enabled": True,
        "spotify_preview_enabled": bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")),
        "spotify_embed_enabled": True,
    })


@app.get("/api/filters")
def filters():
    df = load_song_database()
    dataset_genres = []
    languages = sorted({"English", "Indonesian", "Korean", "Japanese", "Chinese", "Spanish", "Portuguese", "French", "Instrumental"})
    outside_genres = sorted({song.get("genre", "external pool") for songs in OUTSIDE_DATASET.values() for song in songs})
    if df is not None:
        dataset_genres = sorted(df["track_genre"].dropna().astype(str).unique().tolist())
        languages = sorted(set(languages) | set(df["language"].dropna().astype(str).unique().tolist()))
    return jsonify({
        "genres": sorted(set(dataset_genres) | set(outside_genres)),
        "languages": languages,
    })


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model file not found. Put model.pkl or mlruns.zip in the project root."}), 500
    try:
        payload = request.get_json(force=True) or {}
        frame = clean_feature_payload(payload)
        predicted = str(model.predict(frame)[0])
        confidence = probability_for(predicted, frame)
        limit = max(20, min(40, int(payload.get("limit", 30))))
        recommendations = mixed_recommendations(predicted, frame, total_limit=limit, genre=payload.get("genre", "all"), language=payload.get("language", "all"), avoid_keys=payload.get("avoid_keys", []), repeat_chance=float(payload.get("repeat_chance", 0.33)))
        return jsonify({
            "mood": predicted,
            "confidence": round(confidence, 4),
            "recommendations": recommendations,
            "source": "dataset + outside dataset",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/recommend")
def recommend():
    payload = request.get_json(force=True) or {}
    mood = str(payload.get("mood", "Happy"))
    if mood not in MOODS:
        return jsonify({"error": f"Unknown mood: {mood}"}), 400
    limit = max(20, min(40, int(payload.get("limit", 30))))
    recommendations = mixed_recommendations(mood, None, total_limit=limit, genre=payload.get("genre", "all"), language=payload.get("language", "all"), avoid_keys=payload.get("avoid_keys", []), repeat_chance=float(payload.get("repeat_chance", 0.33)))
    return jsonify({
        "mood": mood,
        "confidence": 0.94,
        "recommendations": recommendations,
        "source": "dataset + outside dataset",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
