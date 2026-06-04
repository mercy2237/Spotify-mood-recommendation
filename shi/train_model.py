import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print(f"LOAD AND CLEAN DATA")
print("="*50)
#load data
df=pd.read_csv("dataset.csv")
df=df.dropna()
df= df.drop(columns=['Unnamed: 0','key','mode','time_signature','duration_ms','liveness'])

#remove duplicates
duplicates= df.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")
df = df.drop_duplicates()
print(f"\nTotal songs for training: {len(df)}")

print(f"ASSIGN MOOD LABLES")
print("="*50)
# Assign mood labels
def assign_mood(row):
    v=row['valence']
    e=row['energy']
    d=row['danceability']
    a=row['acousticness']

    # High valence + High energy
    if v >= 0.5 and e >= 0.5:
        if d >= 0.7:
            return 'Energetic'
        else:
            return 'Happy'

    # Low valence + High energy
    elif v < 0.5 and e >= 0.5:
        return 'Angry'

    # High valence + Low energy
    elif v >= 0.5 and e < 0.5:
        if a >= 0.5:
            return 'Romantic'
        else:
            return 'Calm'

    # Low valence + Low energy
    else:
        if a >= 0.6:
            return 'Low Energy'
        else:
            return 'Sad'

df['mood'] = df.apply(assign_mood, axis=1)
print("Mood distribution:")
print(df['mood'].value_counts())

print(f"\nPREPROCESSING")
print("="*50)

features= ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','valence','tempo']
X = df[features]
y = df['mood']
#train test split 70/30
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set : {len(x_train)} songs")
print(f"Test set     : {len(x_test)} songs")

#scale x for KNN
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

#train rfc
print("\nTRAIN RANDOM FOREST FOR MOOD CLASSIFICATION")
print("="*50)
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
print("Random Forest Results")
print("-" * 42)
print(classification_report(y_test, y_pred_rfc))

#train KNN
print("TRAIN KNN")
print("="*50)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)
print("KNN Results")
print("-" * 42)
print(classification_report(y_test, y_pred_knn))

#save model
print("\nSAVE MODELS AND PKL FILES")
print("="*50)
joblib.dump(rfc,'model.pkl')
joblib.dump(knn,'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
#songs for app filter so less popular songs are excluded
songs_for_app = df[df['popularity'] >= 10].copy()
joblib.dump(songs_for_app[['track_id', 'track_name', 'artists', 'mood'] + features],'songs.pkl')

print("model.pkl     — Random Forest classifier")
print("knn_model.pkl — KNN classifier")
print("scaler.pkl    — StandardScaler for KNN")
print("songs.pkl     — Popular songs with track_id and mood labels")
print(f"\nSongs saved for app: {len(songs_for_app)}")