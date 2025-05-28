import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from pymongo import MongoClient
from urllib.parse import quote_plus
from collections import defaultdict

# MongoDB Atlas credentials (use environment variables in production)
username = quote_plus("Bharad")
password = quote_plus("Brad@CineSense1.")
uri = f"mongodb+srv://{username}:{password}@cluster0.xr2tvbu.mongodb.net/?retryWrites=true&w=majority"

# Cache MongoDB data (loaded once per session)
@st.cache_data
def load_movies_from_mongo():
    client = MongoClient(uri)
    db = client["imdb"]
    collection = db["movies"]
    movies = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(movies)
    df['movieID'] = df['Series_Title'].str.replace(" ", "_").str.lower()
    return df

# Cache random ratings and trainset to avoid recomputation
@st.cache_data
def generate_ratings_and_train(df_movies):
    np.random.seed(42)
    n_users = 100
    user_ids = [f"user_{i+1}" for i in range(n_users)]
    ratings_data = []

    for user in user_ids:
        rated_movies = np.random.choice(df_movies['movieID'], size=np.random.randint(20, 50), replace=False)
        for movie in rated_movies:
            rating = np.random.randint(1, 6)
            ratings_data.append((user, movie, rating))

    df_ratings = pd.DataFrame(ratings_data, columns=["userID", "movieID", "rating"])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[["userID", "movieID", "rating"]], reader)
    trainset = data.build_full_trainset()
    return df_ratings, trainset

# Cache model training (resource intensive)
@st.cache_resource
def train_model(_trainset):
    model = SVD()
    model.fit(_trainset)
    return model

# Recommendation logic
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def recommend_movies_for_user(user_id, model, trainset, df_movies):
    anti_testset = trainset.build_anti_testset()
    predictions = model.test(anti_testset)
    top_n = get_top_n(predictions)
    if user_id not in top_n:
        return []
    recommendations = []
    for movie_id, rating in top_n[user_id]:
        title = df_movies[df_movies['movieID'] == movie_id]['Series_Title'].values[0]
        recommendations.append((title, rating))
    return recommendations

# Load everything efficiently
df_movies = load_movies_from_mongo()
df_ratings, trainset = generate_ratings_and_train(df_movies)
model = train_model(trainset)

# Streamlit UI
st.title("🎬 IMDB Movie Recommender")

st.write("Sample Movies from MongoDB:")
st.dataframe(df_movies[['Series_Title', 'Genre', 'IMDB_Rating']].head())

user_ids = sorted(df_ratings["userID"].unique())
selected_user = st.selectbox("Select your user:", user_ids)

if st.button("Get Recommendations"):
    recs = recommend_movies_for_user(selected_user, model, trainset, df_movies)
    st.subheader("Top 5 Recommended Movies:")
    for title, rating in recs:
        st.write(f"- {title} (Predicted Rating: {round(rating, 2)})")
