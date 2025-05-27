import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from collections import defaultdict

st.title("Movie Recommender System (with SVD)")

# ===== Load dữ liệu =====
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies_df, ratings_df = load_data()

# ===== Train SVD Model =====
@st.cache_resource
def train_svd_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(random_state=42)  # Add random_state for reproducibility
    model.fit(trainset)
    return model

model = train_svd_model(ratings_df)

# ===== Xử lý genres thành vector =====
movies_df['genres'] = movies_df['genres'].fillna('')
all_genres = sorted(set(g for genres in movies_df['genres'] for g in genres.split('|')))

def genres_to_vector(genres_str):
    genres = set(genres_str.split('|'))
    return [1 if genre in genres else 0 for genre in all_genres]

movies_df['genre_vec'] = movies_df['genres'].apply(genres_to_vector)

# ===== Giao diện người dùng =====
user_id = st.number_input("Nhập User ID:", min_value=1, value=10, step=1)  # Default to user_id=10
method = st.selectbox("Chọn phương pháp gợi ý:", ["Content-Based", "Collaborative Filtering", "Hybrid"])

# ===== Ratings đã xem và vector người dùng =====
liked_movie_ids = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4)]['movieId']
watched_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId']

liked_vectors = movies_df[movies_df['movieId'].isin(liked_movie_ids)]['genre_vec'].tolist()
user_profile = np.mean(liked_vectors, axis=0).reshape(1, -1) if liked_vectors else np.zeros((1, len(all_genres)))

# ===== Collaborative Filtering: Align with first snippet =====
if method == "Collaborative Filtering":
    all_movie_ids = movies_df['movieId'].astype(str).unique()
    seen_movies = ratings_df[ratings_df['userId'] == int(user_id)]['movieId'].astype(str).unique()
    unseen_movies = [m for m in all_movie_ids if m not in seen_movies]
    
    predictions_unseen = [model.predict(str(user_id), movie_id) for movie_id in unseen_movies]
    predictions_unseen.sort(key=lambda x: x.est, reverse=True)
    
    top_5_unseen = predictions_unseen[:5]
    movie_id_to_title = dict(zip(movies_df['movieId'].astype(str), movies_df['title']))
    
    st.subheader("Gợi ý phim:")
    for pred in top_5_unseen:
        title = movie_id_to_title.get(str(pred.iid), "Unknown Title")
        st.markdown(f"**{title}** - Predicted Rating: `{pred.est:.2f}`")

# ===== Content-Based Filtering =====
else:
    unwatched_movies = movies_df[~movies_df['movieId'].isin(watched_movie_ids)].copy()
    movie_vectors = np.vstack(unwatched_movies['genre_vec'].values)
    similarities = cosine_similarity(user_profile, movie_vectors).flatten()
    unwatched_movies['similarity'] = similarities
    
    # ===== Collaborative Filtering: SVD Predict for Hybrid =====
    predicted_ratings = [model.predict(str(user_id), str(mid)).est for mid in unwatched_movies['movieId']]
    unwatched_movies['predicted_rating'] = predicted_ratings
    
    # ===== Hybrid Score =====
    sim_norm = (unwatched_movies['similarity'] - unwatched_movies['similarity'].min()) / (
        unwatched_movies['similarity'].max() - unwatched_movies['similarity'].min() + 1e-8)
    pred_norm = (unwatched_movies['predicted_rating'] - unwatched_movies['predicted_rating'].min()) / (
        unwatched_movies['predicted_rating'].max() - unwatched_movies['predicted_rating'].min() + 1e-8)
    unwatched_movies['hybrid_score'] = 0.5 * sim_norm + 0.5 * pred_norm
    
    # ===== Hiển thị =====
    st.subheader("Gợi ý phim:")
    if method == "Content-Based":
        results = unwatched_movies.sort_values(by='similarity', ascending=False).head(5)
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** - Similarity: `{row['similarity']:.3f}`")
    
    elif method == "Hybrid":
        results = unwatched_movies.sort_values(by='hybrid_score', ascending=False).head(5)
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** - Hybrid Score: `{row['hybrid_score']:.3f}`")