# Moodify

Moodify is a mood-based music recommendation web application that predicts a user's mood using audio features and recommends songs based on the predicted mood.

The application uses a Random Forest Classifier for mood prediction and K-Nearest Neighbors (KNN) for song recommendations. In addition to dataset-based recommendations, Moodify also integrates with the Spotify API to provide real-world song suggestions.

## Features

* Mood prediction using audio features
* Random Forest mood classification
* KNN-based song recommendations
* Spotify API integration
* Song preview and Spotify links
* Adjustable recommendation count (20–40 songs)
* Prioritize New Songs option
* Fully responsive web interface
* Public deployment via Vercel

## Technology Stack

* Python
* Flask
* Scikit-learn
* Random Forest
* K-Nearest Neighbors (KNN)
* HTML, CSS, JavaScript
* Spotify Web API
* Vercel

## Live Demo

https://your-vercel-url.vercel.app

## Project Structure

```text
Moodify
│
├── app.py
├── requirements.txt
├── dataset.csv
├── train_model.py
├── vercel.json
│
├── models/
│   └── mood_model.pkl
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── README.md
```

## Running Locally

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/Moodify.git
cd Moodify
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## API Endpoints

### Health Check

```http
GET /health
```

### Mood Prediction & Recommendation

```http
POST /api/predict
```

Example request:

```json
{
  "danceability": 7,
  "energy": 8,
  "acousticness": 3,
  "valence": 8,
  "tempo": 130
}
```

## Performance

The application includes inference latency measurement for evaluating machine learning performance.

Example metrics:

* Random Forest Inference Latency
* Recommendation Generation Latency
* Total Server Processing Latency

## Authors

Developed as part of a Machine Learning course project.
