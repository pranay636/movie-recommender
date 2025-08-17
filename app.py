import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Set page config ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# --- Background setup ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: 0;
        }}
        .stApp > div {{
            position: relative;
            z-index: 1;
        }}
        h1, h2, h3, h4, h5, h6, label, p, div, span {{
            color: white !important;
            font-weight: 700 !important;
        }}
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div {{
            background-color: rgba(0,0,0,0.7);
            color: white !important;
            font-weight: 600;
        }}
        .stButton>button {{
            background-color: #e50914;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #b20710;
            color: white !important;
        }}
        .recommend-card {{
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.png")

# --- Load Movies Dataset ---
@st.cache_data
def load_movies():
    df = pd.read_csv("movies_metadata.csv", low_memory=False)
    df = df.dropna(subset=["title", "overview"])  # remove missing
    df = df.reset_index(drop=True)
    return df

df = load_movies()
movie_list = sorted(df["title"].unique().tolist())

# --- Build TF-IDF (no cosine_sim precompute) ---
@st.cache_resource
def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_tfidf(df)

# --- Recommendation Function ---
def get_recommendations(title, mood, df, tfidf_matrix, top_n=10):
    if title not in df["title"].values:
        return []

    idx = df.index[df["title"] == title][0]

    # Compute similarity only for this movie
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-50:][::-1]  # top 50
    recs = df.iloc[sim_indices]

    # Remove the movie itself
    recs = recs[recs["title"] != title]

    # Mood filter → map moods to genres
    mood_map = {
        "Happy": "Comedy",
        "Sad": "Drama",
        "Adventurous": "Adventure",
        "Romantic": "Romance",
        "Thriller": "Thriller"
    }

    genre = mood_map.get(mood, None)
    if genre:
        filtered = recs[recs["genres"].astype(str).str.contains(genre, case=False, na=False)]
        if not filtered.empty:
            recs = filtered

    return recs["title"].head(top_n).tolist()

# --- Main app ---
st.title("🎬 Movie Recommender")

st.subheader("Step 1 · Choose your mood")
mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Adventurous", "Romantic", "Thriller"])

st.subheader("Step 2 · Enter a movie you like")
movie_name = st.selectbox(
    "Search or select your favorite movie",
    options=movie_list,
    index=None,
    placeholder="Start typing to search..."
)

if st.button("Get Recommendations"):
    if not movie_name:
        st.warning("⚠ Please enter a movie name before continuing.")
    else:
        recs = get_recommendations(movie_name, mood, df, tfidf_matrix, top_n=10)

        st.subheader("Step 3 · Your 10 recommendations")
        if recs:
            st.markdown(
                f"""
                <div class="recommend-card">
                    <h4>Mood: {mood}</h4>
                    <h4>Based on: {movie_name}</h4>
                    <h3>🍿 Recommended Movies</h3>
                    <ol>
                        {''.join([f"<li>{m}</li>" for m in recs])}
                    </ol>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("❌ Sorry, no recommendations found for this combination.")

        st.button("Start Over 🔄")
