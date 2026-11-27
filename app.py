import pandas as pd
import streamlit as st

from src.auth import AuthManager
from src.catalogue import add_book, delete_book, update_book
from src.data_loader import load_library_data, save_library_data
from src.ratings import submit_rating
from src.recommender import HybridBookRecommender
from src.search import search_books

st.set_page_config(page_title="Library Recommendation System", layout="wide")

BOOKS_PATH = "data/books_clean.csv"
RATINGS_PATH = "data/ratings_synthetic.csv"
USERS_PATH = "data/users.json"


def _init_state() -> None:
    if "books" not in st.session_state or "ratings" not in st.session_state:
        books_df, ratings_df = load_library_data(BOOKS_PATH, RATINGS_PATH)
        st.session_state.books = books_df
        st.session_state.ratings = ratings_df
        st.session_state.recommender_dirty = True

    if "auth" not in st.session_state:
        st.session_state.auth = AuthManager(USERS_PATH)

    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if "recommender" not in st.session_state or st.session_state.get("recommender_dirty", False):
        st.session_state.recommender = HybridBookRecommender(
            st.session_state.books, st.session_state.ratings
        )
        st.session_state.recommender_dirty = False


def _persist_and_refresh(refresh_recommender: bool = False) -> None:
    save_library_data(st.session_state.books, st.session_state.ratings, BOOKS_PATH, RATINGS_PATH)
    if refresh_recommender:
        st.session_state.recommender_dirty = True
        st.session_state.recommender = HybridBookRecommender(
            st.session_state.books, st.session_state.ratings
        )
        st.session_state.recommender_dirty = False


_init_state()
books: pd.DataFrame = st.session_state.books
ratings: pd.DataFrame = st.session_state.ratings
recommender: HybridBookRecommender = st.session_state.recommender
auth: AuthManager = st.session_state.auth

# --- Sidebar: User Registration & Login ---
with st.sidebar:
    st.header("Account")
    if st.session_state.current_user:
        user = st.session_state.current_user
        st.success(f"Signed in as **{user['username']}** (ID: {user['user_id']})")
        if st.button("Log Out", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    else:
        auth_mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)
        uname = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if auth_mode == "Login":
            if st.button("Log In", type="primary", use_container_width=True):
                ok, msg, info = auth.login_user(uname, pwd)
                if ok:
                    st.session_state.current_user = info
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            if st.button("Create Account", type="primary", use_container_width=True):
                ok, msg, info = auth.register_user(uname, pwd)
                if ok:
                    st.session_state.current_user = info
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

st.title("Smart Library Book Discovery & Recommendation System")

tab_search, tab_rec, tab_rate, tab_catalogue = st.tabs(
    ["Search Books", "Recommendations", "Rate Books (1-5)", "Catalogue Management"]
)

# --- Tab 1: Book Searching ---
with tab_search:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search by title, author, category, or ISBN")
    with c2:
        cat_options = ["All"] + sorted(books["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", cat_options)
    with c3:
        avail_options = ["All", "Available", "Checked Out"]
        selected_avail = st.selectbox("Availability", avail_options)

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        min_r = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
    with c5:
        sort_col = st.selectbox("Sort By", ["rating", "reviews", "title", "author", "category"])
    with c6:
        asc = st.checkbox("Ascending", value=False)

    search_df = search_books(
        books=books,
        query=q,
        category=selected_cat,
        availability_status=selected_avail,
        min_rating=min_r,
        sort_by=sort_col,
        ascending=asc,
        limit=30,
    )
    st.caption(f"Found {len(search_df)} matching book(s)")
    st.dataframe(search_df, use_container_width=True, hide_index=True)

# --- Tab 2: Content-Based, Collaborative & Hybrid Recommendations (Top 5 or 10) ---
with tab_rec:
    if not st.session_state.current_user:
        st.info("Please log in or register from the sidebar to view personalized recommendations.")
    else:
        active_user_id = int(st.session_state.current_user["user_id"])
        rc1, rc2, rc3 = st.columns([1.2, 1, 1])
        with rc1:
            mode = st.selectbox(
                "Recommendation Strategy",
                ["Hybrid", "Content-Based", "Collaborative Filtering"],
            )
        with rc2:
            top_n = st.radio("Display Count", [5, 10], horizontal=True)
        with rc3:
            alpha = st.slider(
                "Hybrid Content Weight (alpha)",
                0.0,
                1.0,
                0.5,
                0.05,
                disabled=(mode != "Hybrid"),
            )

        exclude_rated = st.checkbox("Exclude books already rated by this user", value=True)

        if mode == "Content-Based":
            rec_df = recommender.recommend_content_based(
                user_id=active_user_id, top_n=int(top_n), exclude_rated=exclude_rated
            )
        elif mode == "Collaborative Filtering":
            rec_df = recommender.recommend_collaborative(
                user_id=active_user_id, top_n=int(top_n), exclude_rated=exclude_rated
            )
        else:
            rec_df = recommender.recommend_hybrid(
                user_id=active_user_id,
                top_n=int(top_n),
                alpha=float(alpha),
                exclude_rated=exclude_rated,
            )

        st.subheader(f"Top {top_n} {mode} Recommendations")
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

# --- Tab 3: Book Rating (1 to 5) ---
with tab_rate:
    if not st.session_state.current_user:
        st.info("Please log in or register from the sidebar to submit a 1 to 5 book rating.")
    else:
        active_user_id = int(st.session_state.current_user["user_id"])
        book_choices = {
            f"{row['title']} — {row['author']} (ID: {row['book_id']})": int(row["book_id"])
            for _, row in books.iterrows()
        }
        selected_label = st.selectbox("Choose a Book", list(book_choices.keys()))
        chosen_book_id = book_choices[selected_label]

        existing_user_rating = ratings[
            (ratings["user_id"] == active_user_id) & (ratings["book_id"] == chosen_book_id)
        ]
        default_stars = int(existing_user_rating["rating"].iloc[0]) if not existing_user_rating.empty else 4

        star_rating = st.radio("Your Rating (1 to 5)", [1, 2, 3, 4, 5], index=default_stars - 1, horizontal=True)

        if st.button("Submit Rating", type="primary"):
            new_books, new_ratings, status_msg = submit_rating(
                books=st.session_state.books,
                ratings=st.session_state.ratings,
                user_id=active_user_id,
                book_id=chosen_book_id,
                rating=int(star_rating),
            )
            st.session_state.books = new_books
            st.session_state.ratings = new_ratings
            _persist_and_refresh(refresh_recommender=True)
            st.success(status_msg)
            st.rerun()

        user_rated = (
            st.session_state.ratings[st.session_state.ratings["user_id"] == active_user_id]
            .merge(
                st.session_state.books[["book_id", "title", "author", "category", "availability_status"]],
                on="book_id",
                how="left",
            )
            .sort_values(by="rating", ascending=False)
        )
        st.subheader(f"Your Rated Books ({len(user_rated)})")
        st.dataframe(user_rated, use_container_width=True, hide_index=True)

# --- Tab 4: Book Catalogue Management ---
with tab_catalogue:
    cat_sub1, cat_sub2, cat_sub3 = st.tabs(["Add Book", "Edit Book", "Delete Book"])

    with cat_sub1:
        with st.form("add_book_form", clear_on_submit=True):
            f_title = st.text_input("Title")
            f_author = st.text_input("Author")
            f_category = st.text_input("Category", placeholder="e.g., Fiction - Thriller Mystery")
            f_isbn = st.text_input("ISBN", placeholder="e.g., 978-0-12345-678-9")
            f_avail = st.selectbox("Availability Status", ["Available", "Checked Out"])
            f_desc = st.text_area("Description")
            submitted_add = st.form_submit_button("Add Book to Catalogue")

        if submitted_add:
            upd_books, msg = add_book(
                books=st.session_state.books,
                title=f_title,
                author=f_author,
                category=f_category,
                description=f_desc,
                isbn=f_isbn,
                availability_status=f_avail,
            )
            if msg.startswith("Added"):
                st.session_state.books = upd_books
                _persist_and_refresh(refresh_recommender=True)
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

    with cat_sub2:
        edit_map = {
            f"[{row['book_id']}] {row['title']}": int(row["book_id"])
            for _, row in books.iterrows()
        }
        selected_edit_key = st.selectbox("Select Book to Edit", list(edit_map.keys()))
        edit_id = edit_map[selected_edit_key]
        current_row = books[books["book_id"] == edit_id].iloc[0]

        with st.form("edit_book_form"):
            e_title = st.text_input("Title", value=str(current_row["title"]))
            e_author = st.text_input("Author", value=str(current_row["author"]))
            e_category = st.text_input("Category", value=str(current_row["category"]))
            e_isbn = st.text_input("ISBN", value=str(current_row["isbn"]))
            e_avail = st.selectbox(
                "Availability Status",
                ["Available", "Checked Out"],
                index=0 if current_row["availability_status"] == "Available" else 1,
            )
            e_desc = st.text_area("Description", value=str(current_row["description"]))
            submitted_edit = st.form_submit_button("Save Changes")

        if submitted_edit:
            upd_books, msg = update_book(
                books=st.session_state.books,
                book_id=edit_id,
                title=e_title,
                author=e_author,
                category=e_category,
                description=e_desc,
                isbn=e_isbn,
                availability_status=e_avail,
            )
            if msg.startswith("Updated"):
                st.session_state.books = upd_books
                _persist_and_refresh(refresh_recommender=True)
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

    with cat_sub3:
        del_map = {
            f"[{row['book_id']}] {row['title']} — {row['author']}": int(row["book_id"])
            for _, row in books.iterrows()
        }
        selected_del_key = st.selectbox("Select Book to Delete", list(del_map.keys()))
        del_id = del_map[selected_del_key]
        confirm_del = st.checkbox("Confirm permanent deletion of this book")

        if st.button("Delete Book", type="secondary", disabled=not confirm_del):
            upd_books, upd_ratings, msg = delete_book(
                books=st.session_state.books,
                ratings=st.session_state.ratings,
                book_id=del_id,
            )
            if msg.startswith("Deleted"):
                st.session_state.books = upd_books
                st.session_state.ratings = upd_ratings
                _persist_and_refresh(refresh_recommender=True)
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)
