# IMDB Movie Recommender System

An interactive movie recommender system using collaborative filtering (K-Nearest Neighbors). It is built with Streamlit and fetches movie data from a MongoDB Atlas database. The application simulates user ratings and generates personalized movie recommendations based on user similarity.

## Features

- Loads IMDB-style movie data from MongoDB
- Simulates ratings from 100 random users
- Recommends movies using collaborative filtering (KNN with cosine similarity)
- Built with Streamlit and deployable to Streamlit Cloud
- Secure MongoDB URI management using Streamlit secrets

## Technologies Used

- Python
- Pandas, NumPy, scikit-learn
- Streamlit
- MongoDB (with pymongo)
- K-Nearest Neighbors algorithm

## How It Works

1. **Data Loading**: The application fetches movie data from a MongoDB collection.
2. **User Simulation**: Randomized ratings are generated for 100 synthetic users, each rating between 20 to 50 movies.
3. **Collaborative Filtering**: A user-item matrix is built, and KNN is used to find similar users based on cosine similarity.
4. **Recommendations**: Unrated movies are scored based on neighbors' ratings and recommended accordingly.

## Streamlit Cloud Deployment

This app is designed to run directly on [Streamlit Cloud](https://streamlit.io/cloud).


### Secrets Configuration

On **Streamlit Cloud**, you should define the following secret in the app's dashboard under **App settings > Secrets**:

MONGODB_URI = "mongodb-connection-uri"

### How to run it on your own machine

1. Clone the repository

   ```
   git clone https://github.com/Bharadwaj-Srikumar/CineSense.git
   
   cd imdb-recommender
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
## Future Enhancements

- Integrate real user login and actual rating collection
- Add support for content-based filtering
- Replace synthetic ratings with real-world datasets (e.g., MovieLens)
- Include recommendation explanations or similarity visualizations
- Add persistent recommendation logs using MongoDB