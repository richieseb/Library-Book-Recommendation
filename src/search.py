import pandas as pd


def search_books(
    books: pd.DataFrame,
    query: str = "",
    genre: str | None = None,
    topic: str | None = None,
    min_rating: float = 0.0,
    max_price: float | None = None,
    sort_by: str = "rating",
    ascending: bool = False,
    limit: int = 20,
) -> pd.DataFrame:
    df = books.copy()

    if query and query.strip():
        q = query.strip().lower()
        mask = (
            df["title"].str.lower().str.contains(q, na=False)
            | df["author"].str.lower().str.contains(q, na=False)
            | df["genre"].str.lower().str.contains(q, na=False)
            | df["topic"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    if genre and genre.lower() != "all":
        df = df[df["genre"].str.lower() == genre.strip().lower()]

    if topic and topic.lower() != "all":
        df = df[df["topic"].str.lower() == topic.strip().lower()]

    if min_rating > 0:
        df = df[df["rating"] >= min_rating]

    if max_price is not None:
        df = df[df["price"] <= max_price]

    allowed_sort = {"rating", "reviews", "price", "years_on_list", "last_year", "title"}
    if sort_by in allowed_sort:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    return df.head(limit).reset_index(drop=True)
