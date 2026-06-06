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

SPOTIFY_MOOD_QUERIES: Dict[str, Dict[str, List[str]]] = {
    # Outside-dataset Spotify results are intentionally weighted around 70% English.
    # Spotify does not expose reliable track language, so Moodify uses query buckets as safer labels.
    "Happy": {
        "English": [
            "happy upbeat english pop", "feel good english hits", "sunny english indie pop",
            "english summer pop", "english dance pop happy", "english good vibes songs",
            "english cheerful pop"
        ],
        "Korean": ["happy k-pop", "bright k-pop songs"],
        "Indonesian": ["happy indonesian pop", "lagu pop indonesia bahagia"],
    },
    "Energetic": {
        "English": [
            "english workout hits", "high energy english edm", "english party pop",
            "english pump up songs", "english gym playlist", "english energetic rock",
            "english dance hits"
        ],
        "Korean": ["energetic k-pop", "k-pop workout songs"],
        "Indonesian": ["lagu indonesia semangat", "indonesian energetic pop"],
    },
    "Calm": {
        "English": [
            "english calm acoustic", "english chill indie", "english soft pop calm",
            "english peaceful songs", "english relaxing acoustic", "english mellow indie",
            "english chill folk"
        ],
        "Instrumental": ["ambient piano instrumental", "lofi chill instrumental"],
        "Indonesian": ["calm indonesian indie", "lagu indonesia santai"],
        "Korean": ["calm k-pop songs"],
    },
    "Romantic": {
        "English": [
            "english romantic love songs", "modern english love pop", "english rnb love songs",
            "english wedding love songs", "english acoustic love songs", "english romantic ballads",
            "english soft love songs"
        ],
        "Korean": ["romantic k-pop", "korean love songs"],
        "Indonesian": ["indonesian love songs", "lagu cinta indonesia"],
        "French": ["french love songs"],
    },
    "Angry": {
        "English": [
            "english angry rock", "english metal workout", "english rage rock",
            "english hard rock", "english aggressive hip hop", "english punk rock anger",
            "english heavy metal songs"
        ],
        "Korean": ["aggressive k-pop", "k-pop hard songs"],
        "Japanese": ["j-rock angry songs"],
    },
    "Low Energy": {
        "English": [
            "english slow chill songs", "english dream pop", "english late night songs",
            "english soft indie", "english sleepy songs", "english mellow bedroom pop",
            "english slow acoustic"
        ],
        "Korean": ["slow k-pop", "korean night songs"],
        "Indonesian": ["lagu indonesia mellow", "indonesian slow songs"],
        "Instrumental": ["lofi sleep instrumental"],
    },
    "Sad": {
        "English": [
            "english sad songs", "english heartbreak pop", "english sad indie",
            "english breakup songs", "english emotional ballads", "english sad acoustic",
            "english melancholic pop"
        ],
        "Korean": ["sad k-pop", "korean sad songs"],
        "Indonesian": ["indonesian sad songs", "lagu galau indonesia"],
    },
}


# Legacy manual outside-song pool kept only as a backup reference.
# Runtime outside-dataset recommendations now come from Spotify Web API when
# SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are set in environment variables.
OUTSIDE_DATASET: Dict[str, List[Dict[str, Any]]] = {
    'Happy': [
        {"title": 'Happy', "artist": 'Pharrell Williams', "genre": 'pop', "language": 'English'},
        {"title": 'Good as Hell', "artist": 'Lizzo', "genre": 'pop', "language": 'English'},
        {"title": 'Can’t Stop the Feeling!', "artist": 'Justin Timberlake', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Sugar', "artist": 'Maroon 5', "genre": 'pop', "language": 'English'},
        {"title": 'Dynamite', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Cupid', "artist": 'FIFTY FIFTY', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Hati-Hati di Jalan', "artist": 'Tulus', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Tak Kan Hilang', "artist": 'Budi Doremi', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Shake It Off', "artist": 'Taylor Swift', "genre": 'pop', "language": 'English'},
        {"title": 'Treasure', "artist": 'Bruno Mars', "genre": 'funk', "language": 'English'},
        {"title": 'Walking on Sunshine', "artist": 'Katrina and the Waves', "genre": 'pop-rock', "language": 'English'},
        {"title": 'Best Day Of My Life', "artist": 'American Authors', "genre": 'pop-rock', "language": 'English'},
        {"title": 'Shut Up and Dance', "artist": 'WALK THE MOON', "genre": 'pop-rock', "language": 'English'},
        {"title": 'I Gotta Feeling', "artist": 'Black Eyed Peas', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Firework', "artist": 'Katy Perry', "genre": 'pop', "language": 'English'},
        {"title": 'Levitating', "artist": 'Dua Lipa', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Adventure of a Lifetime', "artist": 'Coldplay', "genre": 'pop-rock', "language": 'English'},
        {"title": 'Count on Me', "artist": 'Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": 'Permission to Dance', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Love Myself', "artist": 'Hailee Steinfeld', "genre": 'pop', "language": 'English'},
        {"title": 'On Top Of The World', "artist": 'Imagine Dragons', "genre": 'pop-rock', "language": 'English'},
        {"title": 'Pocketful of Sunshine', "artist": 'Natasha Bedingfield', "genre": 'pop', "language": 'English'},
        {"title": 'Unwritten', "artist": 'Natasha Bedingfield', "genre": 'pop', "language": 'English'},
        {"title": 'Lovely Day', "artist": 'Bill Withers', "genre": 'soul', "language": 'English'},
        {"title": 'September', "artist": 'Earth, Wind & Fire', "genre": 'funk', "language": 'English'},
        {"title": 'Good Time', "artist": 'Owl City;Carly Rae Jepsen', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Call Me Maybe', "artist": 'Carly Rae Jepsen', "genre": 'pop', "language": 'English'},
        {"title": 'Roar', "artist": 'Katy Perry', "genre": 'pop', "language": 'English'},
        {"title": "Can't Stop", "artist": 'Red Hot Chili Peppers', "genre": 'rock', "language": 'English'},
        {"title": 'As It Was', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Watermelon Sugar', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Kiss Me More', "artist": 'Doja Cat;SZA', "genre": 'pop', "language": 'English'},
        {"title": 'About Damn Time', "artist": 'Lizzo', "genre": 'pop', "language": 'English'},
        {"title": 'Cake By The Ocean', "artist": 'DNCE', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Sunday Best', "artist": 'Surfaces', "genre": 'pop', "language": 'English'},
        {"title": 'Sunflower', "artist": 'Post Malone;Swae Lee', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Sweet Caroline', "artist": 'Neil Diamond', "genre": 'pop', "language": 'English'},
        {"title": 'Hey Ya!', "artist": 'Outkast', "genre": 'funk', "language": 'English'},
        {"title": 'Valerie', "artist": 'Mark Ronson;Amy Winehouse', "genre": 'soul', "language": 'English'},
        {"title": "I'm Yours", "artist": 'Jason Mraz', "genre": 'acoustic', "language": 'English'},
        {"title": 'Banana Pancakes', "artist": 'Jack Johnson', "genre": 'acoustic', "language": 'English'},
        {"title": 'Riptide', "artist": 'Vance Joy', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Put Your Records On', "artist": 'Corinne Bailey Rae', "genre": 'soul', "language": 'English'},
        {"title": 'A-O-K', "artist": 'Tai Verdes', "genre": 'pop', "language": 'English'},
        {"title": 'Heat Waves', "artist": 'Glass Animals', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Sweet Disposition', "artist": 'The Temper Trap', "genre": 'indie-rock', "language": 'English'},
        {"title": 'The Lazy Song', "artist": 'Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": '24K Magic', "artist": 'Bruno Mars', "genre": 'funk', "language": 'English'},
        {"title": 'Geronimo', "artist": 'Sheppard', "genre": 'pop', "language": 'English'},
        {"title": 'Rather Be', "artist": 'Clean Bandit;Jess Glynne', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Symphony', "artist": 'Clean Bandit;Zara Larsson', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Love On Top', "artist": 'Beyoncé', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Crazy In Love', "artist": 'Beyoncé;Jay-Z', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Gerua', "artist": 'Pritam;Arijit Singh;Antara Mitra', "genre": 'bollywood', "language": 'Hindi'},
        {"title": 'Kesempurnaan Cinta', "artist": 'Rizky Febian', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Zona Nyaman', "artist": 'Fourtwnty', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Monokrom', "artist": 'Tulus', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Manusia Kuat', "artist": 'Tulus', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Cinta Luar Biasa', "artist": 'Andmesh', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Laskar Pelangi', "artist": 'Nidji', "genre": 'pop-rock', "language": 'Indonesian'},
        {"title": 'Kangen', "artist": 'Dewa 19', "genre": 'pop-rock', "language": 'Indonesian'},
        {"title": "It's You", "artist": 'Sezairi', "genre": 'pop', "language": 'English'},
        {"title": 'Every Summertime', "artist": 'NIKI', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Lowkey', "artist": 'NIKI', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Left and Right', "artist": 'Charlie Puth;Jung Kook', "genre": 'pop', "language": 'English'},
        {"title": 'Seven', "artist": 'Jung Kook;Latto', "genre": 'pop', "language": 'English'},
        {"title": 'OMG', "artist": 'NewJeans', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Hype Boy', "artist": 'NewJeans', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Cheer Up', "artist": 'TWICE', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Alcohol-Free', "artist": 'TWICE', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Feel My Rhythm', "artist": 'Red Velvet', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Polaroid Love', "artist": 'ENHYPEN', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'After LIKE', "artist": 'IVE', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Queencard', "artist": '(G)I-DLE', "genre": 'k-pop', "language": 'Korean'},
    ],
    'Energetic': [
        {"title": 'Blinding Lights', "artist": 'The Weeknd', "genre": 'synth-pop', "language": 'English'},
        {"title": 'Levels', "artist": 'Avicii', "genre": 'edm', "language": 'English'},
        {"title": 'Stronger', "artist": 'Kanye West', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Titanium', "artist": 'David Guetta;Sia', "genre": 'edm', "language": 'English'},
        {"title": 'Bang Bang Bang', "artist": 'BIGBANG', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Super Shy', "artist": 'NewJeans', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Ddu-Du Ddu-Du', "artist": 'BLACKPINK', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Satu-Satu', "artist": 'Idgitaf', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Uptown Funk', "artist": 'Mark Ronson;Bruno Mars', "genre": 'funk', "language": 'English'},
        {"title": "Don't Stop Me Now", "artist": 'Queen', "genre": 'rock', "language": 'English'},
        {"title": 'Animals', "artist": 'Martin Garrix', "genre": 'edm', "language": 'English'},
        {"title": 'Wake Me Up', "artist": 'Avicii', "genre": 'edm', "language": 'English'},
        {"title": "Can't Hold Us", "artist": 'Macklemore & Ryan Lewis', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Remember the Name', "artist": 'Fort Minor', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Believer', "artist": 'Imagine Dragons', "genre": 'rock', "language": 'English'},
        {"title": 'Radioactive', "artist": 'Imagine Dragons', "genre": 'rock', "language": 'English'},
        {"title": 'POWER', "artist": 'Kanye West', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Run BTS', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'ANTIFRAGILE', "artist": 'LE SSERAFIM', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Kick It', "artist": 'NCT 127', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Sandstorm', "artist": 'Darude', "genre": 'trance', "language": 'Instrumental'},
        {"title": 'The Nights', "artist": 'Avicii', "genre": 'edm', "language": 'English'},
        {"title": "Don't You Worry Child", "artist": 'Swedish House Mafia;John Martin', "genre": 'edm', "language": 'English'},
        {"title": 'Turn Down for What', "artist": 'DJ Snake;Lil Jon', "genre": 'edm', "language": 'English'},
        {"title": 'Where Them Girls At', "artist": 'David Guetta;Flo Rida;Nicki Minaj', "genre": 'dance', "language": 'English'},
        {"title": 'Hey Brother', "artist": 'Avicii', "genre": 'edm', "language": 'English'},
        {"title": 'Without You', "artist": 'David Guetta;Usher', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Clarity', "artist": 'Zedd;Foxes', "genre": 'edm', "language": 'English'},
        {"title": 'Stay The Night', "artist": 'Zedd;Hayley Williams', "genre": 'edm', "language": 'English'},
        {"title": 'The Middle', "artist": 'Zedd;Maren Morris;Grey', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Starships', "artist": 'Nicki Minaj', "genre": 'pop-rap', "language": 'English'},
        {"title": 'Till I Collapse', "artist": 'Eminem;Nate Dogg', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Lose Yourself', "artist": 'Eminem', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Eye of the Tiger', "artist": 'Survivor', "genre": 'rock', "language": 'English'},
        {"title": 'Thunderstruck', "artist": 'AC/DC', "genre": 'rock', "language": 'English'},
        {"title": 'Back In Black', "artist": 'AC/DC', "genre": 'rock', "language": 'English'},
        {"title": 'Enter Sandman', "artist": 'Metallica', "genre": 'metal', "language": 'English'},
        {"title": 'Seven Nation Army', "artist": 'The White Stripes', "genre": 'rock', "language": 'English'},
        {"title": 'Mr. Brightside', "artist": 'The Killers', "genre": 'indie-rock', "language": 'English'},
        {"title": 'Sex on Fire', "artist": 'Kings of Leon', "genre": 'rock', "language": 'English'},
        {"title": 'Pump It', "artist": 'Black Eyed Peas', "genre": 'hip-hop', "language": 'English'},
        {"title": "Let's Get It Started", "artist": 'Black Eyed Peas', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Party Rock Anthem', "artist": 'LMFAO', "genre": 'dance-pop', "language": 'English'},
        {"title": 'Moves Like Jagger', "artist": 'Maroon 5;Christina Aguilera', "genre": 'pop', "language": 'English'},
        {"title": 'Danza Kuduro', "artist": 'Don Omar;Lucenzo', "genre": 'latin', "language": 'Spanish'},
        {"title": 'Gasolina', "artist": 'Daddy Yankee', "genre": 'reggaeton', "language": 'Spanish'},
        {"title": 'Pepas', "artist": 'Farruko', "genre": 'reggaeton', "language": 'Spanish'},
        {"title": 'Mi Gente', "artist": 'J Balvin;Willy William', "genre": 'reggaeton', "language": 'Spanish'},
        {"title": 'Taki Taki', "artist": 'DJ Snake;Selena Gomez;Ozuna;Cardi B', "genre": 'reggaeton', "language": 'Spanish'},
        {"title": 'Despacito', "artist": 'Luis Fonsi;Daddy Yankee', "genre": 'latin', "language": 'Spanish'},
        {"title": 'How You Like That', "artist": 'BLACKPINK', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Kill This Love', "artist": 'BLACKPINK', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'I AM', "artist": 'IVE', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Bouncy', "artist": 'ATEEZ', "genre": 'k-pop', "language": 'Korean'},
        {"title": "God's Menu", "artist": 'Stray Kids', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Thunderous', "artist": 'Stray Kids', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Fire', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Not Today', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'MIC Drop', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Tempo', "artist": 'EXO', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Growl', "artist": 'EXO', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Boom', "artist": 'NCT DREAM', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Cherry Bomb', "artist": 'NCT 127', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Jopping', "artist": 'SuperM', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Wonderland', "artist": 'ATEEZ', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Rungkad', "artist": 'Happy Asmara', "genre": 'dangdut', "language": 'Indonesian'},
        {"title": 'Ojo Dibandingke', "artist": 'Farel Prayoga', "genre": 'dangdut', "language": 'Indonesian'},
        {"title": 'Kopi Dangdut', "artist": 'Fahmi Shahab', "genre": 'dangdut', "language": 'Indonesian'},
        {"title": 'Bento', "artist": 'Iwan Fals', "genre": 'rock', "language": 'Indonesian'},
        {"title": 'Separuh Aku', "artist": 'NOAH', "genre": 'pop-rock', "language": 'Indonesian'},
        {"title": 'Bendera', "artist": 'Cokelat', "genre": 'rock', "language": 'Indonesian'},
        {"title": 'Gebyar-Gebyar', "artist": 'Gombloh', "genre": 'rock', "language": 'Indonesian'},
    ],
    'Calm': [
        {"title": 'Weightless', "artist": 'Marconi Union', "genre": 'ambient', "language": 'Instrumental'},
        {"title": 'Bloom', "artist": 'The Paper Kites', "genre": 'indie', "language": 'English'},
        {"title": 'Holocene', "artist": 'Bon Iver', "genre": 'indie', "language": 'English'},
        {"title": 'River Flows in You', "artist": 'Yiruma', "genre": 'piano', "language": 'Instrumental'},
        {"title": 'Rehat', "artist": 'Kunto Aji', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Secukupnya', "artist": 'Hindia', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Spring Day', "artist": 'BTS', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Ditto', "artist": 'NewJeans', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Sunset Lover', "artist": 'Petit Biscuit', "genre": 'electronic', "language": 'Instrumental'},
        {"title": 'Banana Pancakes', "artist": 'Jack Johnson', "genre": 'acoustic', "language": 'English'},
        {"title": 'Yellow', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'The Scientist', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Cherry Wine', "artist": 'Hozier', "genre": 'folk', "language": 'English'},
        {"title": 'Mystery of Love', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Photograph', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'To Build a Home', "artist": 'The Cinematic Orchestra', "genre": 'classical', "language": 'English'},
        {"title": 'Through the Night', "artist": 'IU', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Blue Side', "artist": 'j-hope', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Skinny Love', "artist": 'Bon Iver', "genre": 'indie', "language": 'English'},
        {"title": 'Flightless Bird, American Mouth', "artist": 'Iron & Wine', "genre": 'folk', "language": 'English'},
        {"title": 'First Day Of My Life', "artist": 'Bright Eyes', "genre": 'folk', "language": 'English'},
        {"title": 'Ophelia', "artist": 'The Lumineers', "genre": 'folk', "language": 'English'},
        {"title": 'Ho Hey', "artist": 'The Lumineers', "genre": 'folk', "language": 'English'},
        {"title": 'Heartbeats', "artist": 'José González', "genre": 'acoustic', "language": 'English'},
        {"title": 'Stay Alive', "artist": 'José González', "genre": 'acoustic', "language": 'English'},
        {"title": 'Slow Burn', "artist": 'Kacey Musgraves', "genre": 'country', "language": 'English'},
        {"title": 'Golden Hour', "artist": 'Kacey Musgraves', "genre": 'country', "language": 'English'},
        {"title": 'Sweet Creature', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Matilda', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Vienna', "artist": 'Billy Joel', "genre": 'pop', "language": 'English'},
        {"title": 'Your Song', "artist": 'Elton John', "genre": 'pop', "language": 'English'},
        {"title": 'Dreams', "artist": 'Fleetwood Mac', "genre": 'rock', "language": 'English'},
        {"title": 'Landslide', "artist": 'Fleetwood Mac', "genre": 'rock', "language": 'English'},
        {"title": 'New Light', "artist": 'John Mayer', "genre": 'pop', "language": 'English'},
        {"title": 'Gravity', "artist": 'John Mayer', "genre": 'blues', "language": 'English'},
        {"title": 'Stop This Train', "artist": 'John Mayer', "genre": 'acoustic', "language": 'English'},
        {"title": 'Better Together', "artist": 'Jack Johnson', "genre": 'acoustic', "language": 'English'},
        {"title": 'Flake', "artist": 'Jack Johnson', "genre": 'acoustic', "language": 'English'},
        {"title": 'Amsterdam', "artist": 'Gregory Alan Isakov', "genre": 'folk', "language": 'English'},
        {"title": 'Big Black Car', "artist": 'Gregory Alan Isakov', "genre": 'folk', "language": 'English'},
        {"title": 'Naked As We Came', "artist": 'Iron & Wine', "genre": 'folk', "language": 'English'},
        {"title": 'Roslyn', "artist": 'Bon Iver;St. Vincent', "genre": 'indie', "language": 'English'},
        {"title": 'Anchor', "artist": 'Novo Amor', "genre": 'indie', "language": 'English'},
        {"title": 'State Lines', "artist": 'Novo Amor', "genre": 'indie', "language": 'English'},
        {"title": 'Saturn', "artist": 'Sleeping At Last', "genre": 'ambient', "language": 'English'},
        {"title": 'Turning Page', "artist": 'Sleeping At Last', "genre": 'piano', "language": 'English'},
        {"title": "Comptine d'un autre été", "artist": 'Yann Tiersen', "genre": 'piano', "language": 'Instrumental'},
        {"title": 'Nuvole Bianche', "artist": 'Ludovico Einaudi', "genre": 'piano', "language": 'Instrumental'},
        {"title": 'Experience', "artist": 'Ludovico Einaudi', "genre": 'classical', "language": 'Instrumental'},
        {"title": 'Kiss the Rain', "artist": 'Yiruma', "genre": 'piano', "language": 'Instrumental'},
        {"title": 'Merry-Go-Round of Life', "artist": 'Joe Hisaishi', "genre": 'soundtrack', "language": 'Instrumental'},
        {"title": "One Summer's Day", "artist": 'Joe Hisaishi', "genre": 'soundtrack', "language": 'Instrumental'},
        {"title": 'Gymnopédie No.1', "artist": 'Erik Satie', "genre": 'classical', "language": 'Instrumental'},
        {"title": 'Clair de Lune', "artist": 'Claude Debussy', "genre": 'classical', "language": 'Instrumental'},
        {"title": 'Gymnopédie No.3', "artist": 'Erik Satie', "genre": 'classical', "language": 'Instrumental'},
        {"title": 'Sampai Jadi Debu', "artist": 'Banda Neira', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Yang Patah Tumbuh, Yang Hilang Berganti', "artist": 'Banda Neira', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Besok Mungkin Kita Sampai', "artist": 'Hindia', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Untuk Perempuan Yang Sedang Dalam Pelukan', "artist": 'Payung Teduh', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Akad', "artist": 'Payung Teduh', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Resah', "artist": 'Payung Teduh', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Biru', "artist": 'Kunto Aji', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Pilu Membiru', "artist": 'Kunto Aji', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Amin Paling Serius', "artist": 'Sal Priadi;Nadin Amizah', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Sorai', "artist": 'Nadin Amizah', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Best Part', "artist": 'Daniel Caesar;H.E.R.', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Japanese Denim', "artist": 'Daniel Caesar', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Get You', "artist": 'Daniel Caesar;Kali Uchis', "genre": 'r-n-b', "language": 'English'},
    ],
    'Romantic': [
        {"title": 'Perfect', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'All of Me', "artist": 'John Legend', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Until I Found You', "artist": 'Stephen Sanchez', "genre": 'romance', "language": 'English'},
        {"title": 'Komang', "artist": 'Raim Laode', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Adu Rayu', "artist": 'Yovie Widianto;Tulus;Glenn Fredly', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Melukis Senja', "artist": 'Budi Doremi', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Love Scenario', "artist": 'iKON', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Only', "artist": 'LeeHi', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'La Vie En Rose', "artist": 'Édith Piaf', "genre": 'french', "language": 'French'},
        {"title": 'Just the Way You Are', "artist": 'Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": 'Thinking Out Loud', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'A Thousand Years', "artist": 'Christina Perri', "genre": 'pop', "language": 'English'},
        {"title": 'Lover', "artist": 'Taylor Swift', "genre": 'pop', "language": 'English'},
        {"title": 'Beautiful In White', "artist": 'Shane Filan', "genre": 'pop', "language": 'English'},
        {"title": 'Sempurna', "artist": 'Andra and The Backbone', "genre": 'pop-rock', "language": 'Indonesian'},
        {"title": 'A Whole New World', "artist": 'ZAYN;Zhavia Ward', "genre": 'soundtrack', "language": 'English'},
        {"title": 'Die With A Smile', "artist": 'Lady Gaga;Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": 'Die For You', "artist": 'The Weeknd', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Call Out My Name', "artist": 'The Weeknd', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Earned It', "artist": 'The Weeknd', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Adore You', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Golden', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Late Night Talking', "artist": 'Harry Styles', "genre": 'pop', "language": 'English'},
        {"title": 'Kiss Me', "artist": 'Sixpence None The Richer', "genre": 'pop', "language": 'English'},
        {"title": 'Lucky', "artist": 'Jason Mraz;Colbie Caillat', "genre": 'acoustic', "language": 'English'},
        {"title": 'Bubbly', "artist": 'Colbie Caillat', "genre": 'pop', "language": 'English'},
        {"title": 'Make You Feel My Love', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Love Me Like You Do', "artist": 'Ellie Goulding', "genre": 'pop', "language": 'English'},
        {"title": "Say You Won't Let Go", "artist": 'James Arthur', "genre": 'pop', "language": 'English'},
        {"title": 'You Are The Reason', "artist": 'Calum Scott', "genre": 'pop', "language": 'English'},
        {"title": 'Dancing On My Own', "artist": 'Calum Scott', "genre": 'pop', "language": 'English'},
        {"title": "I Won't Give Up", "artist": 'Jason Mraz', "genre": 'acoustic', "language": 'English'},
        {"title": 'Marry You', "artist": 'Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": 'Versace on the Floor', "artist": 'Bruno Mars', "genre": 'r-n-b', "language": 'English'},
        {"title": 'At My Worst', "artist": 'Pink Sweat$', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Nothing', "artist": 'Bruno Major', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'Easily', "artist": 'Bruno Major', "genre": 'jazz-pop', "language": 'English'},
        {"title": "Wouldn't Mean A Thing", "artist": 'Bruno Major', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'Like Someone In Love', "artist": 'Bruno Major', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'Always', "artist": 'Daniel Caesar', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Valentine', "artist": 'Laufey', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'From The Start', "artist": 'Laufey', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'Let You Break My Heart Again', "artist": 'Laufey', "genre": 'jazz-pop', "language": 'English'},
        {"title": 'Until I Met You', "artist": 'Stephen Sanchez', "genre": 'pop', "language": 'English'},
        {"title": 'Those Eyes', "artist": 'New West', "genre": 'pop', "language": 'English'},
        {"title": 'Line Without a Hook', "artist": 'Ricky Montgomery', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Sweet', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Apocalypse', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": "Nothing's Gonna Hurt You Baby", "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Angel Baby', "artist": 'Troye Sivan', "genre": 'pop', "language": 'English'},
        {"title": 'I Like Me Better', "artist": 'Lauv', "genre": 'pop', "language": 'English'},
        {"title": 'Paris in the Rain', "artist": 'Lauv', "genre": 'pop', "language": 'English'},
        {"title": 'Mean It', "artist": 'Lauv;LANY', "genre": 'pop', "language": 'English'},
        {"title": 'ILYSB', "artist": 'LANY', "genre": 'pop', "language": 'English'},
        {"title": 'Malibu Nights', "artist": 'LANY', "genre": 'pop', "language": 'English'},
        {"title": 'Location Unknown', "artist": 'HONNE;BEKA', "genre": 'electronic', "language": 'English'},
        {"title": 'No Song Without You', "artist": 'HONNE', "genre": 'electronic', "language": 'English'},
        {"title": 'Day 1', "artist": 'HONNE', "genre": 'electronic', "language": 'English'},
        {"title": 'Tak Ingin Usai', "artist": 'Keisya Levronka', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Janji Suci', "artist": 'Yovie & Nuno', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Teman Hidup', "artist": 'Tulus', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Jatuh Suka', "artist": 'Tulus', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Cinta Luar Biasa', "artist": 'Andmesh', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Sampai Tua Nanti', "artist": 'Andmesh', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Akhirnya Ku Menemukanmu', "artist": 'NaFF', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Cantik', "artist": 'Kahitna', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Soulmate', "artist": 'Kahitna', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Menikahimu', "artist": 'Kahitna', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'Beautiful', "artist": 'Crush', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Everytime', "artist": 'CHEN;Punch', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Some', "artist": 'BOL4', "genre": 'k-pop', "language": 'Korean'},
    ],
    'Angry': [
        {"title": 'Numb', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Killing in the Name', "artist": 'Rage Against The Machine', "genre": 'rock', "language": 'English'},
        {"title": 'Break Stuff', "artist": 'Limp Bizkit', "genre": 'metal', "language": 'English'},
        {"title": 'Duality', "artist": 'Slipknot', "genre": 'metal', "language": 'English'},
        {"title": "God's Menu", "artist": 'Stray Kids', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'MIROH', "artist": 'Stray Kids', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Bohemian Rhapsody', "artist": 'Queen', "genre": 'rock', "language": 'English'},
        {"title": 'C.H.R.I.S.Y.E.', "artist": 'Diskoria;Laleilmanino;Eva Celia', "genre": 'disco', "language": 'Indonesian'},
        {"title": 'Smells Like Teen Spirit', "artist": 'Nirvana', "genre": 'grunge', "language": 'English'},
        {"title": 'Bodies', "artist": 'Drowning Pool', "genre": 'metal', "language": 'English'},
        {"title": 'In The End', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Faint', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'One Step Closer', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Papercut', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Crawling', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Given Up', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Bleed It Out', "artist": 'Linkin Park', "genre": 'alt-rock', "language": 'English'},
        {"title": 'The Pretender', "artist": 'Foo Fighters', "genre": 'rock', "language": 'English'},
        {"title": 'Everlong', "artist": 'Foo Fighters', "genre": 'rock', "language": 'English'},
        {"title": 'All My Life', "artist": 'Foo Fighters', "genre": 'rock', "language": 'English'},
        {"title": 'No One Knows', "artist": 'Queens of the Stone Age', "genre": 'rock', "language": 'English'},
        {"title": 'Chop Suey!', "artist": 'System Of A Down', "genre": 'metal', "language": 'English'},
        {"title": 'Toxicity', "artist": 'System Of A Down', "genre": 'metal', "language": 'English'},
        {"title": 'B.Y.O.B.', "artist": 'System Of A Down', "genre": 'metal', "language": 'English'},
        {"title": 'Down With The Sickness', "artist": 'Disturbed', "genre": 'metal', "language": 'English'},
        {"title": 'Last Resort', "artist": 'Papa Roach', "genre": 'nu-metal', "language": 'English'},
        {"title": 'Before I Forget', "artist": 'Slipknot', "genre": 'metal', "language": 'English'},
        {"title": 'Psychosocial', "artist": 'Slipknot', "genre": 'metal', "language": 'English'},
        {"title": 'Wait and Bleed', "artist": 'Slipknot', "genre": 'metal', "language": 'English'},
        {"title": 'Master of Puppets', "artist": 'Metallica', "genre": 'metal', "language": 'English'},
        {"title": 'Nothing Else Matters', "artist": 'Metallica', "genre": 'metal', "language": 'English'},
        {"title": 'Seek & Destroy', "artist": 'Metallica', "genre": 'metal', "language": 'English'},
        {"title": 'Paranoid', "artist": 'Black Sabbath', "genre": 'metal', "language": 'English'},
        {"title": 'Iron Man', "artist": 'Black Sabbath', "genre": 'metal', "language": 'English'},
        {"title": 'Ace of Spades', "artist": 'Motörhead', "genre": 'metal', "language": 'English'},
        {"title": 'Raining Blood', "artist": 'Slayer', "genre": 'metal', "language": 'English'},
        {"title": 'War Pigs', "artist": 'Black Sabbath', "genre": 'metal', "language": 'English'},
        {"title": 'Basket Case', "artist": 'Green Day', "genre": 'punk-rock', "language": 'English'},
        {"title": 'American Idiot', "artist": 'Green Day', "genre": 'punk-rock', "language": 'English'},
        {"title": 'Holiday', "artist": 'Green Day', "genre": 'punk-rock', "language": 'English'},
        {"title": 'Misery Business', "artist": 'Paramore', "genre": 'pop-punk', "language": 'English'},
        {"title": 'Ignorance', "artist": 'Paramore', "genre": 'pop-punk', "language": 'English'},
        {"title": 'Decode', "artist": 'Paramore', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Bring Me To Life', "artist": 'Evanescence', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Going Under', "artist": 'Evanescence', "genre": 'alt-rock', "language": 'English'},
        {"title": 'My Immortal', "artist": 'Evanescence', "genre": 'alt-rock', "language": 'English'},
        {"title": 'Animal I Have Become', "artist": 'Three Days Grace', "genre": 'rock', "language": 'English'},
        {"title": 'I Hate Everything About You', "artist": 'Three Days Grace', "genre": 'rock', "language": 'English'},
        {"title": 'Riot', "artist": 'Three Days Grace', "genre": 'rock', "language": 'English'},
        {"title": 'Monster', "artist": 'Skillet', "genre": 'rock', "language": 'English'},
        {"title": 'Hero', "artist": 'Skillet', "genre": 'rock', "language": 'English'},
        {"title": 'Feel Invincible', "artist": 'Skillet', "genre": 'rock', "language": 'English'},
        {"title": "You're Gonna Go Far, Kid", "artist": 'The Offspring', "genre": 'punk-rock', "language": 'English'},
        {"title": "The Kids Aren't Alright", "artist": 'The Offspring', "genre": 'punk-rock', "language": 'English'},
        {"title": 'In Bloom', "artist": 'Nirvana', "genre": 'grunge', "language": 'English'},
        {"title": 'Come As You Are', "artist": 'Nirvana', "genre": 'grunge', "language": 'English'},
        {"title": 'Lithium', "artist": 'Nirvana', "genre": 'grunge', "language": 'English'},
        {"title": 'Sabotage', "artist": 'Beastie Boys', "genre": 'rap-rock', "language": 'English'},
        {"title": 'Bulls On Parade', "artist": 'Rage Against The Machine', "genre": 'rap-rock', "language": 'English'},
        {"title": 'Guerrilla Radio', "artist": 'Rage Against The Machine', "genre": 'rap-rock', "language": 'English'},
        {"title": 'HUMBLE.', "artist": 'Kendrick Lamar', "genre": 'hip-hop', "language": 'English'},
        {"title": 'DNA.', "artist": 'Kendrick Lamar', "genre": 'hip-hop', "language": 'English'},
        {"title": 'm.A.A.d city', "artist": 'Kendrick Lamar', "genre": 'hip-hop', "language": 'English'},
        {"title": 'SICKO MODE', "artist": 'Travis Scott', "genre": 'hip-hop', "language": 'English'},
        {"title": 'goosebumps', "artist": 'Travis Scott', "genre": 'hip-hop', "language": 'English'},
        {"title": 'POWER', "artist": 'Kanye West', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Black Skinhead', "artist": 'Kanye West', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Lose Yourself', "artist": 'Eminem', "genre": 'hip-hop', "language": 'English'},
        {"title": 'Till I Collapse', "artist": 'Eminem;Nate Dogg', "genre": 'hip-hop', "language": 'English'},
    ],
    'Low Energy': [
        {"title": 'Space Song', "artist": 'Beach House', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Night Changes', "artist": 'One Direction', "genre": 'pop', "language": 'English'},
        {"title": 'Slow Dancing in the Dark', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Sparks', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Evaluasi', "artist": 'Hindia', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Jakarta Hari Ini', "artist": 'For Revenge;Stereo Wall', "genre": 'emo', "language": 'Indonesian'},
        {"title": 'Through the Night', "artist": 'IU', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Still With You', "artist": 'Jung Kook', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'The Night We Met', "artist": 'Lord Huron', "genre": 'folk', "language": 'English'},
        {"title": 'Asleep', "artist": 'The Smiths', "genre": 'indie', "language": 'English'},
        {"title": 'Glimpse of Us', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Sanctuary', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'YEAH RIGHT', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Like You Do', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Ew', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Will He', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Attention', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Die For You', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Motion Picture Soundtrack', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'No Surprises', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'Exit Music', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'High and Dry', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'Fake Plastic Trees', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'Creep', "artist": 'Radiohead', "genre": 'alternative', "language": 'English'},
        {"title": 'Fix You', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Trouble', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'O', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Gravity', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Magic', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Yellow', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Apocalypse', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Sweet', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'K.', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Cry', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Heavenly', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Sunsetz', "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": "Nothing's Gonna Hurt You Baby", "artist": 'Cigarettes After Sex', "genre": 'dream-pop', "language": 'English'},
        {"title": 'Fourth of July', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Visions of Gideon', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Mystery of Love', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Should Have Known Better', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Casimir Pulaski Day', "artist": 'Sufjan Stevens', "genre": 'folk', "language": 'English'},
        {"title": 'Blue Jeans', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": 'Video Games', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": 'Young and Beautiful', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": 'Summertime Sadness', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": 'Let The Light In', "artist": 'Lana Del Rey;Father John Misty', "genre": 'alternative', "language": 'English'},
        {"title": 'Mariners Apartment Complex', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": 'Cherry', "artist": 'Lana Del Rey', "genre": 'alternative', "language": 'English'},
        {"title": '505', "artist": 'Arctic Monkeys', "genre": 'indie-rock', "language": 'English'},
        {"title": 'Do I Wanna Know?', "artist": 'Arctic Monkeys', "genre": 'indie-rock', "language": 'English'},
        {"title": 'I Wanna Be Yours', "artist": 'Arctic Monkeys', "genre": 'indie-rock', "language": 'English'},
        {"title": 'The Less I Know The Better', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'Eventually', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'New Person, Same Old Mistakes', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'Let It Happen', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'Borderline', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'Breathe Deeper', "artist": 'Tame Impala', "genre": 'psychedelic', "language": 'English'},
        {"title": 'Ribs', "artist": 'Lorde', "genre": 'pop', "language": 'English'},
        {"title": 'Liability', "artist": 'Lorde', "genre": 'pop', "language": 'English'},
        {"title": 'Supercut', "artist": 'Lorde', "genre": 'pop', "language": 'English'},
        {"title": 'Ocean Eyes', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": "when the party's over", "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": 'everything i wanted', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": 'idontwannabeyouanymore', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": 'TV', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": 'What Was I Made For?', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
        {"title": 'Bored', "artist": 'Billie Eilish', "genre": 'pop', "language": 'English'},
    ],
    'Sad': [
        {"title": 'Someone Like You', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Fix You', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'drivers license', "artist": 'Olivia Rodrigo', "genre": 'pop', "language": 'English'},
        {"title": 'Happier', "artist": 'Olivia Rodrigo', "genre": 'pop', "language": 'English'},
        {"title": 'Bertaut', "artist": 'Nadin Amizah', "genre": 'indie', "language": 'Indonesian'},
        {"title": 'Runtuh', "artist": 'Feby Putri;Fiersa Besari', "genre": 'pop', "language": 'Indonesian'},
        {"title": 'To My Youth', "artist": 'BOL4', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Eight', "artist": 'IU;SUGA', "genre": 'k-pop', "language": 'Korean'},
        {"title": 'Let Her Go', "artist": 'Passenger', "genre": 'folk', "language": 'English'},
        {"title": 'When I Was Your Man', "artist": 'Bruno Mars', "genre": 'pop', "language": 'English'},
        {"title": 'All I Want', "artist": 'Kodaline', "genre": 'alternative', "language": 'English'},
        {"title": 'Before You Go', "artist": 'Lewis Capaldi', "genre": 'pop', "language": 'English'},
        {"title": 'Skinny Love', "artist": 'Bon Iver', "genre": 'indie', "language": 'English'},
        {"title": 'Arcade', "artist": 'Duncan Laurence', "genre": 'pop', "language": 'English'},
        {"title": 'traitor', "artist": 'Olivia Rodrigo', "genre": 'pop', "language": 'English'},
        {"title": 'Glimpse of Us', "artist": 'Joji', "genre": 'r-n-b', "language": 'English'},
        {"title": 'Aku Milikmu', "artist": 'Dewa 19', "genre": 'pop-rock', "language": 'Indonesian'},
        {"title": 'Someone You Loved', "artist": 'Lewis Capaldi', "genre": 'pop', "language": 'English'},
        {"title": 'Bruises', "artist": 'Lewis Capaldi', "genre": 'pop', "language": 'English'},
        {"title": 'Hold Me While You Wait', "artist": 'Lewis Capaldi', "genre": 'pop', "language": 'English'},
        {"title": 'Another Love', "artist": 'Tom Odell', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Heal', "artist": 'Tom Odell', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Black Friday', "artist": 'Tom Odell', "genre": 'indie-pop', "language": 'English'},
        {"title": 'Jealous', "artist": 'Labrinth', "genre": 'soul', "language": 'English'},
        {"title": 'Train Wreck', "artist": 'James Arthur', "genre": 'pop', "language": 'English'},
        {"title": 'Say Something', "artist": 'A Great Big World;Christina Aguilera', "genre": 'pop', "language": 'English'},
        {"title": 'The A Team', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'Supermarket Flowers', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'Photograph', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'I See Fire', "artist": 'Ed Sheeran', "genre": 'pop', "language": 'English'},
        {"title": 'Tears Dry On Their Own', "artist": 'Amy Winehouse', "genre": 'soul', "language": 'English'},
        {"title": 'Back To Black', "artist": 'Amy Winehouse', "genre": 'soul', "language": 'English'},
        {"title": 'Love Is A Losing Game', "artist": 'Amy Winehouse', "genre": 'soul', "language": 'English'},
        {"title": 'All I Ask', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Easy On Me', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Hello', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Set Fire to the Rain', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Turning Tables', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Chasing Pavements', "artist": 'Adele', "genre": 'pop', "language": 'English'},
        {"title": 'Liability', "artist": 'Lorde', "genre": 'pop', "language": 'English'},
        {"title": 'The Scientist', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Viva La Vida', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Everglow', "artist": 'Coldplay', "genre": 'alternative', "language": 'English'},
        {"title": 'Let Somebody Go', "artist": 'Coldplay;Selena Gomez', "genre": 'pop', "language": 'English'},
        {"title": 'Oceans', "artist": 'Seafret', "genre": 'folk', "language": 'English'},
        {"title": 'Atlantis', "artist": 'Seafret', "genre": 'folk', "language": 'English'},
        {"title": 'Youth', "artist": 'Daughter', "genre": 'indie', "language": 'English'},
        {"title": 'Medicine', "artist": 'Daughter', "genre": 'indie', "language": 'English'},
        {"title": 'Smother', "artist": 'Daughter', "genre": 'indie', "language": 'English'},
        {"title": 'Funeral', "artist": 'Phoebe Bridgers', "genre": 'indie', "language": 'English'},
        {"title": 'Motion Sickness', "artist": 'Phoebe Bridgers', "genre": 'indie', "language": 'English'},
        {"title": 'Scott Street', "artist": 'Phoebe Bridgers', "genre": 'indie', "language": 'English'},
        {"title": 'I Know The End', "artist": 'Phoebe Bridgers', "genre": 'indie', "language": 'English'},
        {"title": 'Heather', "artist": 'Conan Gray', "genre": 'pop', "language": 'English'},
        {"title": 'The Cut That Always Bleeds', "artist": 'Conan Gray', "genre": 'pop', "language": 'English'},
        {"title": 'Maniac', "artist": 'Conan Gray', "genre": 'pop', "language": 'English'},
        {"title": 'Memories', "artist": 'Conan Gray', "genre": 'pop', "language": 'English'},
        {"title": 'July', "artist": 'Noah Cyrus', "genre": 'pop', "language": 'English'},
        {"title": 'Lonely', "artist": 'Noah Cyrus', "genre": 'pop', "language": 'English'},
        {"title": 'In My Blood', "artist": 'Shawn Mendes', "genre": 'pop', "language": 'English'},
        {"title": 'Stitches', "artist": 'Shawn Mendes', "genre": 'pop', "language": 'English'},
        {"title": "It'll Be Okay", "artist": 'Shawn Mendes', "genre": 'pop', "language": 'English'},
        {"title": 'Ghost', "artist": 'Justin Bieber', "genre": 'pop', "language": 'English'},
        {"title": 'Lonely', "artist": 'Justin Bieber;Benny Blanco', "genre": 'pop', "language": 'English'},
        {"title": 'Stay', "artist": 'Rihanna;Mikky Ekko', "genre": 'pop', "language": 'English'},
        {"title": 'Take A Bow', "artist": 'Rihanna', "genre": 'pop', "language": 'English'},
        {"title": 'Unfaithful', "artist": 'Rihanna', "genre": 'pop', "language": 'English'},
        {"title": 'Lose You To Love Me', "artist": 'Selena Gomez', "genre": 'pop', "language": 'English'},
        {"title": 'The Heart Wants What It Wants', "artist": 'Selena Gomez', "genre": 'pop', "language": 'English'},
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


def infer_spotify_language_from_query(query: str, fallback: str = "English") -> str:
    """Best-effort label only. Spotify Search API does not return track language."""
    q = query.lower()
    if "indonesian" in q or "indonesia" in q or "lagu" in q:
        return "Indonesian"
    if "k-pop" in q or "korean" in q:
        return "Korean"
    if "j-rock" in q or "japanese" in q:
        return "Japanese"
    if "french" in q:
        return "French"
    if "piano" in q or "ambient" in q or "lofi" in q or "instrumental" in q:
        return "Instrumental"
    return fallback or "English"


def infer_spotify_genre_from_query(query: str) -> str:
    q = query.lower()
    if "k-pop" in q:
        return "k-pop"
    if "edm" in q:
        return "edm"
    if "rock" in q:
        return "rock"
    if "metal" in q:
        return "metal"
    if "hip hop" in q:
        return "hip-hop"
    if "rnb" in q or "r&b" in q:
        return "r-n-b"
    if "indie" in q:
        return "indie"
    if "acoustic" in q:
        return "acoustic"
    if "piano" in q:
        return "piano"
    if "ambient" in q:
        return "ambient"
    if "lofi" in q:
        return "lofi"
    return "pop"


def search_spotify_tracks(
    query: str,
    limit: int = 30,
    offset: Optional[int] = None,
    language_label: str = "English",
) -> List[Dict[str, Any]]:
    """Search Spotify for real outside-dataset tracks. Requires env vars."""
    token = get_spotify_token()
    if not token or requests is None:
        return []

    # Random offset keeps outside-dataset results fresh without hardcoded song lists.
    if offset is None:
        offset = random.randint(0, 120)

    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": query,
                "type": "track",
                "market": "US",
                "limit": max(1, min(50, int(limit))),
                "offset": max(0, min(950, int(offset))),
            },
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", []) or []
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for item in items:
        track_id = item.get("id") or ""
        artists = ";".join(artist.get("name", "") for artist in item.get("artists", []) if artist.get("name"))
        title = item.get("name") or "Unknown Track"
        if not title or not artists:
            continue
        results.append({
            "title": title,
            "artist": artists,
            "track_id": track_id,
            "spotify_url": item.get("external_urls", {}).get("spotify") or spotify_track_url(track_id, title, artists),
            "spotify_embed_url": spotify_embed_url(track_id),
            "preview_url": item.get("preview_url") or "",
            "album_cover": ((item.get("album", {}).get("images") or [{}])[0].get("url") or ""),
            "genre": infer_spotify_genre_from_query(query),
            "language": infer_spotify_language_from_query(query, language_label),
        })
    return results


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


def flatten_spotify_queries(mood: str, only_language: Optional[str] = None) -> List[Dict[str, str]]:
    buckets = SPOTIFY_MOOD_QUERIES.get(mood) or SPOTIFY_MOOD_QUERIES.get("Happy", {})
    entries: List[Dict[str, str]] = []
    for language_label, queries in buckets.items():
        if only_language and language_label != only_language:
            continue
        for query in queries:
            entries.append({"query": query, "language": language_label})
    random.shuffle(entries)
    return entries


def collect_spotify_recommendations(
    mood: str,
    entries: List[Dict[str, str]],
    target_count: int,
    avoid_set: set,
    seen: set,
    allow_repeat: bool,
    score_min: float,
    score_max: float,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if target_count <= 0:
        return output

    # Multiple passes with random offsets gives fresher results and avoids returning the same top Spotify results.
    for entry in entries:
        if len(output) >= target_count:
            break
        query = entry["query"]
        language_label = entry.get("language", "English")
        needed = max(20, min(50, (target_count - len(output)) * 4))
        candidates = search_spotify_tracks(query, limit=needed, language_label=language_label)
        random.shuffle(candidates)
        for song in candidates:
            key = song_key(song.get("title", ""), song.get("artist", ""))
            if not key or key in seen:
                continue
            if avoid_set and key in avoid_set and not allow_repeat:
                continue
            seen.add(key)
            output.append({
                "title": song.get("title", "Unknown Track"),
                "artist": song.get("artist", "Unknown Artist"),
                "score": round(random.uniform(score_min, score_max), 1),
                "mood": mood,
                "source": "outside dataset",
                "genre": song.get("genre", "spotify"),
                "language": song.get("language", language_label),
                "spotify_url": song.get("spotify_url", ""),
                "spotify_embed_url": song.get("spotify_embed_url", ""),
                "preview_url": song.get("preview_url", ""),
                "track_id": song.get("track_id", ""),
                "album_cover": song.get("album_cover", ""),
            })
            if len(output) >= target_count:
                break
    return output


def recommend_outside_dataset(
    mood: str,
    limit: int = 10,
    genre: str = "all",
    language: str = "all",
    avoid_keys: Optional[List[str]] = None,
    repeat_chance: float = 0.33,
) -> List[Dict[str, Any]]:
    """Return outside-dataset recommendations from Spotify API with ~70% English targeting."""
    if limit <= 0:
        return []

    avoid_set = {str(key).lower() for key in (avoid_keys or [])}
    allow_repeat = random.random() < repeat_chance
    seen = set()

    # Spotify does not provide reliable language per track. We target English by query mix instead:
    # around 70% English queries and 30% international/instrumental queries.
    english_target = round(limit * 0.70)
    international_target = max(0, limit - english_target)

    english_entries = flatten_spotify_queries(mood, "English")
    international_entries = flatten_spotify_queries(mood)
    international_entries = [entry for entry in international_entries if entry.get("language") != "English"]
    if not international_entries:
        international_entries = flatten_spotify_queries(mood, "English")

    output: List[Dict[str, Any]] = []
    output.extend(collect_spotify_recommendations(
        mood, english_entries, english_target, avoid_set, seen, allow_repeat, 78, 97
    ))
    output.extend(collect_spotify_recommendations(
        mood, international_entries, international_target, avoid_set, seen, allow_repeat, 74, 95
    ))

    # If Spotify returns fewer songs than needed, fill the rest with any mood query but still prefer English first.
    if len(output) < limit:
        output.extend(collect_spotify_recommendations(
            mood, flatten_spotify_queries(mood, "English"), limit - len(output), avoid_set, seen, True, 72, 94
        ))
    if len(output) < limit:
        output.extend(collect_spotify_recommendations(
            mood, flatten_spotify_queries(mood), limit - len(output), avoid_set, seen, True, 70, 92
        ))

    random.shuffle(output)
    return output[:limit]


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
        "outside_dataset_enabled": bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")),
        "outside_dataset_source": "Spotify API" if os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET") else "dataset fallback only",
        "spotify_preview_enabled": bool(os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET")),
        "spotify_embed_enabled": True,
    })


@app.get("/api/filters")
def filters():
    df = load_song_database()
    dataset_genres = []
    languages = sorted({"English", "Indonesian", "Korean", "Japanese", "Chinese", "Spanish", "Portuguese", "French", "Instrumental"})
    outside_genres = sorted({"pop", "edm", "rock", "metal", "hip-hop", "r-n-b", "indie", "acoustic", "piano", "ambient", "lofi", "k-pop"})
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
