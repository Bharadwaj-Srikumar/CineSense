import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
# from urllib.parse import quote_plus  # Uncomment if needed for encoding passwords
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# Load MongoDB URI securely from Streamlit secrets
uri = st.secrets["MONGODB_URI"]

# Load movies from MongoDB and cache the result to avoid repeated DB queries
@st.cache_data
def load_movielens_from_mongo():
    client = MongoClient(uri)
    db = client["MovieLens"]  # Access the 'MovieLens' database
    collection1 = db["movies"]  # Access the 'movies' collection
    collection2 = db["ratings"]  # Access the 'ratings' collection
    movies = pd.DataFrame(list(collection1.find({}, {"_id": 0})))  # Fetch all documents, exclude MongoDB's _id & Convert to a Pandas DataFrame
    ratings = pd.DataFrame(list(collection2.find({}, {"_id": 0})))  # Fetch all documents, exclude MongoDB's _id & Convert to a Pandas DataFrame
    # Normalize movie titles to create a consistent internal ID (used in ratings matrix)
    movies['movieID'] = movies['title'].str.replace(" ", "_").str.lower()
    return ratings, movies

# Generate movie recommendations for a selected user using K-Nearest Neighbors
def recommend_movies_for_user(user_id, df_ratings, df_movies, n_neighbors=5, n_recommendations=5):
    # Create a user-item rating matrix with users as rows, movies as columns
    user_item_matrix = df_ratings.pivot_table(index='userId', columns='movieID', values='rating').fillna(0)

    # Fit a KNN model using cosine similarity to identify similar users
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_matrix)

    # If user is not in matrix (edge case), return an empty list
    if user_id not in user_item_matrix.index:
        return []

    # Find the k nearest neighbors (excluding the user themselves)
    distances, indices = knn_model.kneighbors([user_item_matrix.loc[user_id]], n_neighbors=n_neighbors+1)
    neighbor_indices = indices.flatten()[1:]  # Skip the first one (it's the user)
    similar_users = user_item_matrix.index[neighbor_indices]

    # Get the list of movies this user has already rated
    user_rated_movies = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    # Collect ratings from similar users for movies the current user hasn't rated
    movie_scores = defaultdict(list)
    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.loc[sim_user]
        for movie_id, rating in sim_user_ratings.items():
            if movie_id not in user_rated_movies and rating > 0:
                movie_scores[movie_id].append(rating)

    # Calculate average score for each movie
    avg_scores = [(movie_id, np.mean(scores)) for movie_id, scores in movie_scores.items()]
    sorted_recs = sorted(avg_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Map internal movieIDs back to actual titles for display
    recommendations = []
    for movie_id, score in sorted_recs:
        title_row = df_movies[df_movies['movieID'] == movie_id]
        if not title_row.empty:
            title = title_row['title'].values[0]
            recommendations.append((title, score))

    return recommendations

# Load data once per user session
df_ratings, df_movies = load_movielens_from_mongo()

# Streamlit App UI
st.title("🎬 CineSense: Movie Recommender")

# User selection dropdown
user_ids = sorted(df_ratings["userId"].unique())
selected_user = st.selectbox("Select a user:", user_ids)

# When user clicks the button, show recommendations
if st.button("Get Recommendations"):
    recs = recommend_movies_for_user(selected_user, df_ratings, df_movies)
    st.subheader("Top 5 Recommended Movies:")
    if recs:
        for title, rating in recs:
            st.write(f"- {title} (Estimated Preference: {round(rating, 2)})")
    else:
        st.write("No recommendations found.")