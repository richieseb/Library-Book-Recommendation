from pathlib import Path
import pandas as pd


def load_library_data(
    books_path: str | Path = "data/books_clean.csv",
    ratings_path: str | Path = "data/ratings_synthetic.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    required_base_cols = {"book_id", "title", "author", "rating", "reviews"}
    required_rating_cols = {"user_id", "book_id", "rating"}

    if not required_base_cols.issubset(books.columns):
        missing = sorted(required_base_cols - set(books.columns))
        raise ValueError(f"books_clean.csv missing base columns: {missing}")

    if not required_rating_cols.issubset(ratings.columns):
        missing = sorted(required_rating_cols - set(ratings.columns))
        raise ValueError(f"ratings_synthetic.csv missing required columns: {missing}")

    books = books.drop_duplicates(subset=["book_id"]).copy()
    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce").dropna().astype(int)
    books["title"] = books["title"].fillna("").astype(str).str.strip()
    books["author"] = books["author"].fillna("Unknown").astype(str).str.strip()

    for col in ["years_on_list", "first_year", "last_year"]:
        if col not in books.columns:
            books[col] = 0
        books[col] = pd.to_numeric(books[col], errors="coerce").fillna(0).astype(int)

    if "category" not in books.columns:
        genre = books["genre"].fillna("General").astype(str).str.strip() if "genre" in books.columns else "General"
        topic = (
            books["topic"].fillna("general").astype(str).str.replace("_", " ", regex=False).str.title()
            if "topic" in books.columns
            else "General"
        )
        books["category"] = genre + " - " + topic
    else:
        books["category"] = books["category"].fillna("General").astype(str).str.strip()

    if "isbn" not in books.columns:
        books["isbn"] = [f"978-0-{10000 + int(bid):05d}-0" for bid in books["book_id"]]
    else:
        books["isbn"] = books["isbn"].fillna("").astype(str).str.strip()

    if "description" not in books.columns:
        books["description"] = [
            f"{row['title']} by {row['author']} in {row['category']}."
            for _, row in books.iterrows()
        ]
    else:
        books["description"] = books["description"].fillna("").astype(str).str.strip()

    if "availability_status" not in books.columns:
        books["availability_status"] = [
            "Available" if int(bid) % 5 != 0 else "Checked Out" for bid in books["book_id"]
        ]
    else:
        books["availability_status"] = (
            books["availability_status"].fillna("Available").astype(str).str.strip()
        )
        books.loc[
            ~books["availability_status"].isin(["Available", "Checked Out"]),
            "availability_status",
        ] = "Available"

    books["rating"] = pd.to_numeric(books["rating"], errors="coerce").fillna(0.0)
    books["reviews"] = pd.to_numeric(books["reviews"], errors="coerce").fillna(0).astype(int)

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
