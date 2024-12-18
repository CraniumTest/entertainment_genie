import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Step 1: Data Collection
# Load a sample dataset
movies = pd.read_csv('../data/movies.csv')  # Contains movieId, title, genres
ratings = pd.read_csv('../data/ratings.csv')  # Contains userId, movieId, rating, timestamp

# Step 2: Data Preprocessing
# Handling missing values, if any
movies.fillna('', inplace=True)
ratings.dropna(inplace=True)

# Step 3: Feature Engineering
# Content-Based Filtering - TF-IDF Vectorization on movie genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Collaborative Filtering - Preparing data
X_train, X_val = train_test_split(ratings, test_size=0.2, random_state=42)

# Step 4: Building the Recommendation Engine
# Collaborative Filtering - Using NearestNeighbors as a simplistic approach
user_movie_matrix = X_train.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.values)

# Content-Based Filtering - Similarity between movies
cosine_sim = tfidf_matrix * tfidf_matrix.T

# Function to recommend movies based on content
def recommend_movie_content(movie_title, num_recommendations=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Function to recommend movies to a user based on collaborative filtering
def recommend_movie_users(user_id, num_recommendations=5):
    distances, indices = model_knn.kneighbors(user_movie_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=num_recommendations + 1)
    for i in range(1, len(distances.flatten())):
        idx = user_movie_matrix.index[indices.flatten()[i]]
        print(f'Movie_ID: {idx} with Distance: {distances.flatten()[i]}')

# Step 5: Evaluation
def evaluate_recommendations(user_predictions, user_actual):
    # Simply using RMSE for quick evaluation
    mse = mean_squared_error(user_predictions, user_actual)
    rmse = np.sqrt(mse)
    print(f"RMSE Error: {rmse:.2f}")

# Usage:
# Example usage for movie content recommendation
print("Content-Based Recommendations for 'Toy Story':")
print(recommend_movie_content('Toy Story'))

# Example usage for collaborative filtering user recommendations
print("\nCollaborative Filtering Recommendations for User 1:")
recommend_movie_users(1)
