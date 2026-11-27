import pandas as pd


def search_books(
    books: pd.DataFrame,
    query: str = "",
    category: str | None = None,
    availability_status: str | None = None,
    min_rating: float = 0.0,
    sort_by: str = "rating",
    ascending: bool = False,
    limit: int = 25,
) -> pd.DataFrame:
    df = books.copy()

    if query and query.strip():
        q = query.strip().lower()
        mask = (
            df["title"].str.lower().str.contains(q, na=False)
            | df["author"].str.lower().str.contains(q, na=False)
            | df["category"].str.lower().str.contains(q, na=False)
            | df["isbn"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    if category and category.strip().lower() != "all":
        df = df[df["category"].str.lower() == category.strip().lower()]

    if availability_status and availability_status.strip().lower() != "all":
        df = df[
            df["availability_status"].str.lower()
            == availability_status.strip().lower()
        ]

    if min_rating > 0:
        df = df[df["rating"] >= float(min_rating)]

    allowed_sort = {"rating", "reviews", "title", "author", "category", "last_year"}
    if sort_by in allowed_sort and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    display_cols = [
        c
        for c in [
            "book_id",
            "isbn",
            "title",
            "author",
            "category",
            "description",
            "availability_status",
            "rating",
            "reviews",
        ]
        if c in df.columns
    ]
    return df[display_cols].head(int(limit)).reset_index(drop=True)
