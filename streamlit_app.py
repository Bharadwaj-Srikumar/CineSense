import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

# Load MongoDB URI securely from Streamlit secrets
uri = st.secrets["MONGODB_URI"]

# Load movies from MongoDB and cache the result
@st.cache_data
def load_movies_from_mongo():
    client = MongoClient(uri)
    db = client["imdb"]
    collection = db["movies"]
    movies = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(movies)
    # Create a normalized movieID for internal use
    df['movieID'] = df['Series_Title'].str.replace(" ", "_").str.lower()
    return df

# Generate random ratings and cache the result
@st.cache_data
def generate_ratings(df_movies):
    np.random.seed(42)
    n_users = 100
    user_ids = [f"user_{i+1}" for i in range(n_users)]
    ratings_data = []

    # Each user randomly rates 20 to 50 movies
    for user in user_ids:
        rated_movies = np.random.choice(df_movies['movieID'], size=np.random.randint(20, 50), replace=False)
        for movie in rated_movies:
            rating = np.random.randint(1, 6)  # Ratings from 1 to 5
            ratings_data.append((user, movie, rating))

    df_ratings = pd.DataFrame(ratings_data, columns=["userID", "movieID", "rating"])
    return df_ratings

# Recommend movies for a given user using collaborative filtering with KNN
def recommend_movies_for_user(user_id, df_ratings, df_movies, n_neighbors=5, n_recommendations=5):
    # Create a user-item matrix (users as rows, movies as columns)
    user_item_matrix = df_ratings.pivot_table(index='userID', columns='movieID', values='rating').fillna(0)

    # Fit KNN model on this matrix
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_item_matrix)

    if user_id not in user_item_matrix.index:
        return []

    # Get nearest neighbors for the selected user
    distances, indices = knn_model.kneighbors([user_item_matrix.loc[user_id]], n_neighbors=n_neighbors+1)
    neighbor_indices = indices.flatten()[1:]  # Exclude the user itself
    similar_users = user_item_matrix.index[neighbor_indices]

    # Find movies this user hasn't rated yet
    user_rated_movies = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    # Collect scores from neighbors for unrated movies
    movie_scores = defaultdict(list)
    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.loc[sim_user]
        for movie_id, rating in sim_user_ratings.items():
            if movie_id not in user_rated_movies and rating > 0:
                movie_scores[movie_id].append(rating)

    # Average the scores for each movie and pick top N
    avg_scores = [(movie_id, np.mean(scores)) for movie_id, scores in movie_scores.items()]
    sorted_recs = sorted(avg_scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Map movie IDs to movie titles
    recommendations = []
    for movie_id, score in sorted_recs:
        title_row = df_movies[df_movies['movieID'] == movie_id]
        if not title_row.empty:
            title = title_row['Series_Title'].values[0]
            recommendations.append((title, score))

    return recommendations

# Load data once per session
df_movies = load_movies_from_mongo()
df_ratings = generate_ratings(df_movies)

# Streamlit UI
st.title("🎬 IMDB Movie Recommender (KNN-Based)")

st.write("Sample Movies from MongoDB:")
st.dataframe(df_movies[['Series_Title', 'Genre', 'IMDB_Rating']].head())

user_ids = sorted(df_ratings["userID"].unique())
selected_user = st.selectbox("Select your user:", user_ids)

if st.button("Get Recommendations"):
    recs = recommend_movies_for_user(selected_user, df_ratings, df_movies)
    st.subheader("Top 5 Recommended Movies:")
    if recs:
        for title, rating in recs:
            st.write(f"- {title} (Estimated Preference: {round(rating, 2)})")
    else:
        st.write("No recommendations found.")
