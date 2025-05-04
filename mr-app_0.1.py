import streamlit as st
import pandas as pd
import requests
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('train.csv')

# Select audio features
features = ['danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

# Fill missing values
data[features] = data[features].fillna(data[features].mean())

# Standardize features
scaler = StandardScaler()
feature_data = scaler.fit_transform(data[features])

# Compute cosine similarity
similarity = cosine_similarity(feature_data)

# Function to recommend songs
def recommend(track_name, num_recommendations=5):
    track_index = data[data['Track Name'].str.lower() == track_name.lower()].index
    if len(track_index) == 0:
        return None
    track_index = track_index[0]

    similarity_scores = list(enumerate(similarity[track_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, (index, score) in enumerate(similarity_scores[1:num_recommendations + 1]):
        track = data.iloc[index]
        recommendations.append({
            'Track Name': track['Track Name'],
            'Artist': track['Artist Name'],
            'Preview URL': get_deezer_preview(track['Track Name'], track['Artist Name']),
            'Album Cover': get_album_cover(track['Track Name'], track['Artist Name'])
        })
    return recommendations

# Function to get Deezer preview URL
def get_deezer_preview(track_name, artist_name):
    query = urllib.parse.quote(f"{track_name} {artist_name}")
    url = f"https://api.deezer.com/search?q={query}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for item in data['data']:
                title = item.get('title', '').lower()
                artist = item.get('artist', {}).get('name', '').lower()
                # Basic fuzzy matching logic
                if track_name.lower() in title and artist_name.lower() in artist:
                    return item.get('preview')
            # If nothing closely matches, just return the first preview
            if data['data']:
                return data['data'][0].get('preview')
    except Exception as e:
        print("Deezer API error:", e)
    return None

# Function to get Deezer album cover URL
def get_album_cover(track_name, artist_name):
    query = urllib.parse.quote(f"{track_name} {artist_name}")
    url = f"https://api.deezer.com/search?q={query}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for item in data['data']:
                title = item.get('title', '').lower()
                artist = item.get('artist', {}).get('name', '').lower()
                # Basic fuzzy matching logic
                if track_name.lower() in title and artist_name.lower() in artist:
                    return item.get('album', {}).get('cover_big')
            # If nothing closely matches, just return the first album cover
            if data['data']:
                return data['data'][0].get('album', {}).get('cover_big')
    except Exception as e:
        print("Deezer API error:", e)
    return None

# Streamlit UI
st.title('🎵 Music Recommender System')

st.write("Select a song you like, and we'll recommend similar songs!")

all_tracks = data['Track Name'].dropna().unique()
selected_song = st.selectbox('Select a Song:', sorted(all_tracks))

if st.button('Recommend'):
    results = recommend(selected_song)
    if results:
        st.subheader('You might also like:')
        for idx, rec in enumerate(results, start=1):
            st.write(f"**{idx}. {rec['Track Name']}** by {rec['Artist']}")
            if rec['Preview URL']:
                st.audio(rec['Preview URL'], format='audio/mp3')
            else:
                st.caption("Preview not available.")
            if rec['Album Cover']:
                st.image(rec['Album Cover'], width=100)
            else:
                st.caption("Album cover not available.")
    else:
        st.error('Sorry, we could not find recommendations for this song.')
