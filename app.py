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

# --- Load Movies ---
@st.cache_data
def load_movies():
    file_id = "1sdposT1B7g1HDU5_LTPehQ3VzGsORfbc"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url, low_memory=False)
    df = df.dropna(subset=["title", "overview"])  
    df = df.reset_index(drop=True)
    return df

df = load_movies()
movie_list = sorted(df["title"].unique().tolist())

# --- Build TF-IDF ---
@st.cache_resource
def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_tfidf(df)

# --- Recommendation Functions ---
def get_recommendations(title, mood, df, tfidf_matrix, top_n=10):
    if title not in df["title"].values:
        return []

    idx = df.index[df["title"] == title][0]

    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-50:][::-1]  
    recs = df.iloc[sim_indices]

    recs = recs[recs["title"] != title]

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

def recommend_by_mood(mood, df, top_n=10):
    mood_map = {
        "Happy": "Comedy",
        "Sad": "Drama",
        "Adventurous": "Adventure",
        "Romantic": "Romance",
        "Thriller": "Thriller"
    }
    genre = mood_map.get(mood, None)
    if genre:
        recs = df[df["genres"].astype(str).str.contains(genre, case=False, na=False)]
        return recs["title"].drop_duplicates().head(top_n).tolist()
    return []

# --- Main App ---
st.title("🎬 Movie Recommender")

st.subheader("Step 1 · Choose your mood")
mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Adventurous", "Romantic", "Thriller"])

st.subheader("Step 2 · Enter a movie you like (optional)")
movie_name = st.selectbox(
    "Search or select your favorite movie (optional)",
    options=["-- No specific movie --"] + movie_list,
    index=0
)

if st.button("Get Recommendations"):
    if movie_name == "-- No specific movie --":
        recs = recommend_by_mood(mood, df, top_n=10)
        based_on_text = "Mood only"
    else:
        recs = get_recommendations(movie_name, mood, df, tfidf_matrix, top_n=10)
        based_on_text = movie_name

    st.subheader("Step 3 · Your 10 recommendations")
    if recs:
        st.markdown(
            f"""
            <div class="recommend-card">
                <h4>Mood: {mood}</h4>
                <h4>Based on: {based_on_text}</h4>
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

    st.button("Start Over 🔄")
