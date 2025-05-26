import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommender System")

# ===== Load dữ liệu =====
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies_df, ratings_df = load_data()

# ===== Xử lý genre thành vector =====
movies_df['genres'] = movies_df['genres'].fillna('')
all_genres = sorted(set(g for genres in movies_df['genres'] for g in genres.split('|')))

def genres_to_vector(genres_str):
    genres = set(genres_str.split('|'))
    return [1 if genre in genres else 0 for genre in all_genres]

movies_df['genre_vec'] = movies_df['genres'].apply(genres_to_vector)

# ===== Giao diện người dùng =====
user_id = st.number_input("🔢 Nhập User ID:", min_value=1, value=1, step=1)
method = st.selectbox("📊 Chọn phương pháp gợi ý:", ["Content-Based", "Collaborative Filtering", "Hybrid"])

# ===== Ratings đã xem =====
liked_movie_ids = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4)]['movieId']
watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId']

# ===== Content-Based Filtering =====
liked_vectors = movies_df[movies_df['movieId'].isin(liked_movie_ids)]['genre_vec'].tolist()
user_profile = np.mean(liked_vectors, axis=0).reshape(1, -1) if liked_vectors else np.zeros((1, len(all_genres)))

unwatched_movies = movies_df[~movies_df['movieId'].isin(watched_movie_ids)].copy()
movie_vectors = np.vstack(unwatched_movies['genre_vec'].values)
similarities = cosine_similarity(user_profile, movie_vectors).flatten()
unwatched_movies['similarity'] = similarities

# ===== Collaborative Filtering (simple mean score) =====
movie_avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
movie_avg_ratings.columns = ['movieId', 'avg_rating']
unwatched_movies = pd.merge(unwatched_movies, movie_avg_ratings, on='movieId', how='left').fillna(0)

# ===== Hybrid Score =====
sim_norm = (unwatched_movies['similarity'] - unwatched_movies['similarity'].min()) / (
    unwatched_movies['similarity'].max() - unwatched_movies['similarity'].min() + 1e-8)
pred_norm = (unwatched_movies['avg_rating'] - unwatched_movies['avg_rating'].min()) / (
    unwatched_movies['avg_rating'].max() - unwatched_movies['avg_rating'].min() + 1e-8)

unwatched_movies['hybrid_score'] = 0.5 * sim_norm + 0.5 * pred_norm

# ===== Hiển thị =====
st.subheader("🎯 Gợi ý phim:")

if method == "Content-Based":
    results = unwatched_movies.sort_values(by='similarity', ascending=False).head(10)
    for _, row in results.iterrows():
        st.markdown(f"🎬 **{row['title']}** - Similarity: `{row['similarity']:.3f}`")

elif method == "Collaborative Filtering":
    results = unwatched_movies.sort_values(by='avg_rating', ascending=False).head(10)
    for _, row in results.iterrows():
        st.markdown(f"🎬 **{row['title']}** - Avg Rating: `{row['avg_rating']:.2f}`")

elif method == "Hybrid":
    results = unwatched_movies.sort_values(by='hybrid_score', ascending=False).head(10)
    for _, row in results.iterrows():
        st.markdown(f"🎬 **{row['title']}** - Hybrid Score: `{row['hybrid_score']:.3f}`")
