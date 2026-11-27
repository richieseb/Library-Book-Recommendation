from pathlib import Path
import pandas as pd


def load_library_data(
    books_path: str | Path = "data/books_clean.csv",
    ratings_path: str | Path = "data/ratings_synthetic.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    required_book_cols = {
        "book_id",
        "isbn",
        "title",
        "author",
        "category",
        "description",
        "availability_status",
        "rating",
        "reviews",
        "years_on_list",
        "first_year",
        "last_year",
    }
    required_rating_cols = {"user_id", "book_id", "rating"}

    if not required_book_cols.issubset(books.columns):
        missing = sorted(required_book_cols - set(books.columns))
        raise ValueError(f"books_clean.csv missing required columns: {missing}")

    if not required_rating_cols.issubset(ratings.columns):
        missing = sorted(required_rating_cols - set(ratings.columns))
        raise ValueError(f"ratings_synthetic.csv missing required columns: {missing}")

    books = books.drop_duplicates(subset=["book_id"]).copy()
    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce").dropna().astype(int)
    books["isbn"] = books["isbn"].fillna("").astype(str).str.strip()
    books["title"] = books["title"].fillna("").astype(str).str.strip()
    books["author"] = books["author"].fillna("Unknown").astype(str).str.strip()
    books["category"] = books["category"].fillna("General").astype(str).str.strip()
    books["description"] = books["description"].fillna("").astype(str).str.strip()

    books["availability_status"] = (
        books["availability_status"]
        .fillna("Available")
        .astype(str)
        .str.strip()
        .replace({"": "Available"})
    )
    books.loc[
        ~books["availability_status"].isin(["Available", "Checked Out"]),
        "availability_status",
    ] = "Available"

    books["rating"] = pd.to_numeric(books["rating"], errors="coerce").fillna(0.0)
    for col in ["reviews", "years_on_list", "first_year", "last_year"]:
        books[col] = pd.to_numeric(books[col], errors="coerce").fillna(0).astype(int)

    if "genre" in books.columns:
        books["genre"] = books["genre"].fillna("").astype(str).str.strip()
    if "topic" in books.columns:
        books["topic"] = books["topic"].fillna("").astype(str).str.strip()

    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["book_id"] = pd.to_numeric(ratings["book_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)

    valid_book_ids = set(books["book_id"])
    ratings = ratings[
        ratings["book_id"].isin(valid_book_ids) & ratings["rating"].between(1, 5)
    ].copy()
    ratings["rating"] = ratings["rating"].astype(int)
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last").reset_index(
        drop=True
    )

    books = books.sort_values("book_id").reset_index(drop=True)
    return books, ratings


def save_library_data(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    books_path: str | Path = "data/books_clean.csv",
    ratings_path: str | Path = "data/ratings_synthetic.csv",
) -> None:
    books_out = Path(books_path)
    ratings_out = Path(ratings_path)
    books_out.parent.mkdir(parents=True, exist_ok=True)
    ratings_out.parent.mkdir(parents=True, exist_ok=True)

    books.to_csv(books_out, index=False)
    ratings.to_csv(ratings_out, index=False)


if __name__ == "__main__":
    books_df, ratings_df = load_library_data()
    print(
        f"Loaded {len(books_df)} books | {ratings_df['user_id'].nunique()} users | "
        f"{len(ratings_df)} ratings"
    )
