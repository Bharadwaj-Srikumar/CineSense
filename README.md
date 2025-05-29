# CineSense: Movie Recommender System

An interactive movie recommender system using collaborative filtering (K-Nearest Neighbors). Built with Streamlit, the app now uses real user ratings from the MovieLens dataset to generate personalized movie recommendations based on user similarity.

## Features
- Loads movie metadata from the MovieLens dataset
- Uses real user ratings from MovieLens (ratings.csv)
- Recommends movies using collaborative filtering (KNN with cosine similarity)
- Interactive frontend built with Streamlit
- Deployable to Streamlit Cloud

## Technologies Used
- Python
- Pandas, NumPy, scikit-learn
- Streamlit
- MovieLens dataset (CSV)
- K-Nearest Neighbors algorithm (collaborative filtering)

## How It Works
- Data Loading: The application loads movie metadata (movies.csv) and user ratings (ratings.csv) from the MovieLens "latest small" dataset.
- User-Item Matrix Creation: A sparse user-item rating matrix is constructed using real user ratings.
- Collaborative Filtering: KNN (cosine similarity) is applied to find similar users and predict unseen movie ratings.
- Recommendations: Top N movies are recommended based on nearest neighbors’ ratings and unseen titles.

## Dataset Used
This app uses the official MovieLens "latest small" dataset (100,000+ real ratings):

🔗 https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

- Extract ratings.csv and movies.csv into the project root.

## Streamlit Cloud Deployment
This app is ready to run on Streamlit Cloud.

### Secrets Configuration
On Streamlit Cloud, define secrets under App settings > Secrets (optional, if using MongoDB):

MONGODB_URI = "your-mongodb-connection-uri"

## Run Locally

### Clone the repository:
git clone https://github.com/Bharadwaj-Srikumar/CineSense.git


cd imdb-recommender

### Run the Streamlit app:
streamlit run streamlit_app.py

## Future Enhancements
- Integrate real user login and personal rating history
- Add content-based filtering (genres, descriptions, tags)
- Visualize user similarity and recommendation reasons
- Store favorite movies or watchlists using MongoDB
- Add hybrid recommender (collaborative + content)