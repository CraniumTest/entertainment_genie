# README

## Overview

This document provides an overview of the prototype implementation for the AI-Driven Personalized Entertainment Platform, also known as "Entertainment Genie." The platform is designed to offer personalized entertainment content recommendations to users, leveraging advanced machine learning techniques. The core of this implementation focuses on building a recommendation engine using collaborative and content-based filtering techniques.

## Directory Structure

The project is structured as follows:

- `entertainment_genie/`
  - `data/`: This directory is intended for storing datasets like movies and ratings, which are essential for generating recommendations.
  - `models/`: This directory is reserved for saving machine learning models.
  - `scripts/`: This directory contains the Python scripts required for data preprocessing, feature engineering, model training, and recommendation generation.

## Main Components

### 1. Data Collection

The prototype utilizes sample datasets, such as MovieLens, to simulate content recommendation. The primary datasets involve:
- Movies: Information includes movie IDs, titles, and genres.
- Ratings: User ratings of movies, which include user IDs, movie IDs, and rating scores.

### 2. Data Preprocessing

Key preprocessing steps include:
- Handling missing values in the movie and ratings datasets.
- Transforming textual data, such as genres, into a suitable format for machine learning models.

### 3. Feature Engineering

The recommendation system integrates two primary approaches:
- **Content-Based Filtering**: By using a TF-IDF vectorizer, the system analyzes movie genres to determine content similarity.
- **Collaborative Filtering**: A user-movie interaction matrix is created to capture user preferences and utilized to train a Nearest Neighbors model.

### 4. Recommendation Engine

The system provides recommendation functionality through two main methods:
- **Content-Based Recommendations**: Offers movie recommendations based on content similarity calculated from genres.
- **Collaborative Filtering Recommendations**: Recommends movies to a user by identifying items favored by similar users.

### 5. Evaluation

To assess the effectiveness of recommendations, the prototype includes a basic evaluation metric, RMSE (Root Mean Square Error), which provides a measure of discrepancy between predicted and actual user ratings.

## How to Use

1. **Setup**: 
   - Ensure that the necessary datasets are placed in the `data` directory.
   - Install required Python packages listed in `requirements.txt` using a package manager like pip.

2. **Running the Recommendation Engine**:
   - Execute the `recommendation_engine.py` script in the `scripts` directory to generate and evaluate recommendations.

3. **Example Outputs**:
   - The script provides sample outputs that display content-based and collaborative filtering recommendations for specific inputs, such as a movie title ("Toy Story") and a user ID (User 1).

## Dependencies

The system is built using Python and requires the following packages, which are specified in the `requirements.txt` file:
- pandas
- numpy
- scikit-learn

This initial prototype serves as a foundation for further enhancements to deliver personalized entertainment experiences. Future developments could include integrating more sophisticated models, mood detection, and interactive content delivery features.
