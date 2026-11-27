from pathlib import Path
import pandas as pd


def load_library_data(
    books_path: str | Path = "data/books_clean.csv",
    ratings_path: str | Path = "data/ratings_synthetic.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    required_book_cols = {
        "book_id", "title", "author", "genre", "rating",
        "reviews", "price", "years_on_list", "first_year", "last_year", "topic",
    }
    required_rating_cols = {"user_id", "book_id", "rating"}

    if not required_book_cols.issubset(books.columns):
        missing = required_book_cols - set(books.columns)
        raise ValueError(f"books_clean.csv missing columns: {sorted(missing)}")

    if not required_rating_cols.issubset(ratings.columns):
        missing = required_rating_cols - set(ratings.columns)
        raise ValueError(f"ratings_synthetic.csv missing columns: {sorted(missing)}")

    books = books.drop_duplicates(subset=["book_id"]).copy()
    books["book_id"] = books["book_id"].astype(int)
    books["title"] = books["title"].fillna("").astype(str).str.strip()
    books["author"] = books["author"].fillna("Unknown").astype(str).str.strip()
    books["genre"] = books["genre"].fillna("Unknown").astype(str).str.strip()
    books["topic"] = books["topic"].fillna("general").astype(str).str.strip()

    for col in ["rating", "price"]:
        books[col] = pd.to_numeric(books[col], errors="coerce")
    for col in ["reviews", "years_on_list", "first_year", "last_year"]:
        books[col] = pd.to_numeric(books[col], errors="coerce").fillna(0).astype(int)

    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])

    valid_book_ids = set(books["book_id"])
    orphan_count = (~ratings["book_id"].isin(valid_book_ids)).sum()
    if orphan_count > 0:
        ratings = ratings[ratings["book_id"].isin(valid_book_ids)].copy()

    # Keep latest rating if a user rated the same book multiple times
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last").reset_index(drop=True)
    books = books.reset_index(drop=True)

    return books, ratings


if __name__ == "__main__":
    books_df, ratings_df = load_library_data()
    print(f"Books: {len(books_df)} | Users: {ratings_df['user_id'].nunique()} | Ratings: {len(ratings_df)}")
