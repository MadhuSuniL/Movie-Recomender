import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data
posters = joblib.load("data/movie_posters.joblib")
vectors = joblib.load("data/movie_vectors.joblib")
titles = joblib.load("data/movie_title.joblib")

# Recommendation Logic
def recommend_movies(movie_title, num_recommendations=30):
    movie_index = np.where(titles == movie_title)[0][0]
    movie_vector = vectors[movie_index].reshape(1, -1)
    movie_poster = posters[movie_index]
    similarities = cosine_similarity(movie_vector, vectors).flatten()
    similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]
    similar_movies = [(titles[i], posters[i]) for i in similar_indices]
    return similar_movies, movie_poster

# Page Config
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎥", layout="wide")

# Main Title
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommender System</h1>
    <p style='text-align: center; font-size: 18px;'>Find movies similar to your favorite ones in a click!</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Select Movie")
selected_movie = st.sidebar.selectbox("🎞️ Choose a movie:", options=titles, key="movie_selector")
num_recommendations = st.sidebar.slider("🎯 Number of recommendations", min_value=1, max_value=100, value=10, step=1)
st.sidebar.markdown("---")

# Recommendation Button
if st.sidebar.button("🚀 Get Recommendations", use_container_width=True):
    recommended_movies, selected_movie_poster = recommend_movies(selected_movie, num_recommendations)

    # Centered selected movie display
    st.markdown("## 🎞️ Selected Movie")
    center_col1, center_col2, center_col3 = st.columns([3, 3, 3])
    with center_col2:
        st.image(selected_movie_poster, caption=f"🎬 {selected_movie}", use_container_width=True)
        st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

    # Recommended movies
    st.markdown(f"<h2 style='color: #00BFFF;'>📽️ Top {num_recommendations} Recommended Movies</h2>", unsafe_allow_html=True)
    num_cols = 5
    rows = (num_recommendations + num_cols - 1) // num_cols
    for row in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            movie_idx = row * num_cols + col_idx
            if movie_idx < len(recommended_movies):
                title, poster = recommended_movies[movie_idx]
                with cols[col_idx]:
                    st.image(poster, use_container_width=True, caption=f"🎬 {title}")
                    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    
else:
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)                
    center_col1, center_col2, center_col3 = st.columns([2, 1, 2])
    with center_col2:
        st.text("No movie selected yet.")
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)                
