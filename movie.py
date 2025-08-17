import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the movie data
movies = pd.read_csv('movies.csv')

# Step 2: Prepare movie tags by combining genres, keywords, and overview
movies['genres'] = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['overview']
movies['tags'] = movies['tags'].str.lower()

# Step 3: Vectorize tags using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# Step 4: Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Function to get recommendations based on a single movie
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.lower()
    if title not in movies['title'].str.lower().values:
        return "Movie not found in database."
    idx = movies[movies['title'].str.lower() == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Step 7: Mood-based filtering function
def mood_filter(mood):
    mood_keywords = {
        'happy': ['comedy', 'funny', 'feelgood', 'joy'],
        'sad': ['drama', 'tragic', 'sad', 'tearjerker'],
        'thrill': ['thriller', 'suspense', 'action', 'mystery'],
        'chill': ['romance', 'light', 'relax', 'family'],
        'motivated': ['inspiration', 'biography', 'true story', 'hero']
    }
    keywords = mood_keywords.get(mood.lower(), [])
    filtered = movies[movies['tags'].apply(lambda x: any(k in x for k in keywords))]
    return filtered

# Step 8: Group recommendation by averaging favorite movies
def group_recommendations(favorite_titles):
    favorite_titles = [title.lower() for title in favorite_titles]
    indices = []
    for title in favorite_titles:
        if title in movies['title'].str.lower().values:
            idx = movies[movies['title'].str.lower() == title].index[0]
            indices.append(idx)
    if not indices:
        return "None of the favorite movies found in database."
    avg_vector = np.mean(tfidf_matrix[indices].toarray(), axis=0).reshape(1, -1)
    sim_scores = cosine_similarity(avg_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-15:][::-1]
    top_indices = [i for i in top_indices if i not in indices][:10]
    return movies['title'].iloc[top_indices]

# Example usage:
if __name__ == "__main__":
    print("Single movie recommendations for 'Inception':")
    print(get_recommendations('Inception'))

    print("\nMood-based recommendations for mood 'happy':")
    happy_movies = mood_filter('happy')
    print(happy_movies['title'].head(10))

    print("\nGroup recommendations for favorites ['Inception', 'Titanic', 'The Dark Knight']:")
    print(group_recommendations(['Inception', 'Titanic', 'The Dark Knight']))
