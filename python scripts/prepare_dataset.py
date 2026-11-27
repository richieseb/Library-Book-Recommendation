from pathlib import Path
import pandas as pd


def prepare_datasets(
    raw_books_path: str | Path = "books_clean.csv",
    raw_ratings_path: str | Path = "ratings_synthetic.csv",
    out_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    books = pd.read_csv(raw_books_path)
    ratings = pd.read_csv(raw_ratings_path)

    books = books.drop_duplicates(subset=["book_id"]).copy()
    books["book_id"] = books["book_id"].astype(int)

    # Derive category from genre + topic
    topic_clean = books["topic"].fillna("general").astype(str).str.replace("_", " ", regex=False).str.title()
    genre_clean = books["genre"].fillna("General").astype(str).str.strip()
    books["category"] = genre_clean + " - " + topic_clean

    # Populate ISBN and description if absent
    if "isbn" not in books.columns:
        books["isbn"] = [f"978-0-{10000 + int(bid):05d}-0" for bid in books["book_id"]]

    if "description" not in books.columns:
        books["description"] = [
            f"{row['title']} by {row['author']} in {row['category']} (listed {row['first_year']}-{row['last_year']})."
            for _, row in books.iterrows()
        ]

    # Populate availability_status if absent
    if "availability_status" not in books.columns:
        books["availability_status"] = [
            "Available" if int(bid) % 5 != 0 else "Checked Out" for bid in books["book_id"]
        ]

    # Remove out-of-scope price field if present
    books = books.drop(columns=["price"], errors="ignore")

    ordered_cols = [
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
        "genre",
        "topic",
    ]
    books = books[[c for c in ordered_cols if c in books.columns]].reset_index(drop=True)

    # Validate ratings (1 to 5) and keep valid book_ids
    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings[
        ratings["book_id"].isin(set(books["book_id"])) & ratings["rating"].between(1, 5)
    ].copy()
    ratings["rating"] = ratings["rating"].astype(int)
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last").reset_index(drop=True)

    books.to_csv(out_path / "books_clean.csv", index=False)
    ratings.to_csv(out_path / "ratings_synthetic.csv", index=False)
    return books, ratings


if __name__ == "__main__":
    b_df, r_df = prepare_datasets()
    print(f"Saved {len(b_df)} books and {len(r_df)} ratings to data/")
