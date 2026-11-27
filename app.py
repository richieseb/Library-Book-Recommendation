import pandas as pd
import streamlit as st

from src.analytics import compute_resource_analytics
from src.data_loader import load_library_data
from src.recommender import HybridBookRecommender
from src.search import search_books

st.set_page_config(page_title="Library Recommender & Analytics", layout="wide")


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_library_data("data/books_clean.csv", "data/ratings_synthetic.csv")


@st.cache_resource
def get_recommender(books: pd.DataFrame, ratings: pd.DataFrame) -> HybridBookRecommender:
    return HybridBookRecommender(books, ratings)


books, ratings = get_data()
recommender = get_recommender(books, ratings)

st.title("Smart Library: Search, Recommendations & Analytics")

tab_search, tab_rec, tab_analytics = st.tabs(
    ["Search Catalog", "Personalized Recommendations", "Resource Analytics"]
)

with tab_search:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("Search by title, author, genre, or topic", value="")
    with col2:
        genres = ["All"] + sorted(books["genre"].dropna().unique().tolist())
        selected_genre = st.selectbox("Genre", genres)
    with col3:
        topics = ["All"] + sorted(books["topic"].dropna().unique().tolist())
        selected_topic = st.selectbox("Topic", topics)

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
    with col5:
        sort_by = st.selectbox(
            "Sort By", ["rating", "reviews", "price", "years_on_list", "last_year", "title"]
        )
    with col6:
        ascending = st.checkbox("Ascending", value=False)

    results = search_books(
        books=books,
        query=query,
        genre=selected_genre,
        topic=selected_topic,
        min_rating=min_rating,
        sort_by=sort_by,
        ascending=ascending,
        limit=30,
    )
    st.caption(f"Showing {len(results)} result(s)")
    st.dataframe(results, use_container_width=True)

with tab_rec:
    user_ids = sorted(ratings["user_id"].unique().tolist())
    col_u, col_a, col_n = st.columns([1, 1, 1])
    with col_u:
        user_id = st.selectbox("Select User ID", user_ids)
    with col_a:
        alpha = st.slider("Content vs. Collaborative Weight (alpha)", 0.0, 1.0, 0.5, 0.05)
    with col_n:
        top_n = st.slider("Number of Recommendations", 5, 25, 10, 1)

    exclude_rated = st.checkbox("Exclude already-rated books", value=True)

    user_history = (
        ratings[ratings["user_id"] == user_id]
        .merge(books[["book_id", "title", "author", "genre", "topic"]], on="book_id", how="left")
        .sort_values(by="rating", ascending=False)
    )
    with st.expander(f"User {user_id} Rating History ({len(user_history)} books)"):
        st.dataframe(user_history, use_container_width=True)

    recs = recommender.recommend(
        user_id=int(user_id),
        top_n=int(top_n),
        alpha=float(alpha),
        exclude_rated=exclude_rated,
    )
    st.dataframe(recs, use_container_width=True)

with tab_analytics:
    reports = compute_resource_analytics(books, ratings)

    st.subheader("Topic & Genre Demand")
    st.dataframe(reports["topic_demand"], use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Lowest-Interaction Books")
        st.dataframe(reports["underutilized_books"], use_container_width=True)
    with col_right:
        st.subheader("Discovery Gaps (High Rating, Low Interaction)")
        st.dataframe(reports["discovery_gaps"], use_container_width=True)
