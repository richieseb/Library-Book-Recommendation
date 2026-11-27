import pandas as pd


def compute_resource_analytics(
    books: pd.DataFrame, ratings: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    rating_stats = ratings.groupby("book_id", as_index=False).agg(
        interaction_count=("rating", "count"),
        avg_user_rating=("rating", "mean"),
    )

    catalog_usage = books.merge(rating_stats, on="book_id", how="left")
    catalog_usage["interaction_count"] = (
        catalog_usage["interaction_count"].fillna(0).astype(int)
    )
    catalog_usage["avg_user_rating"] = catalog_usage["avg_user_rating"].round(2)

    # 1. Lowest-engagement books (under-utilized catalog items)
    underutilized_books = (
        catalog_usage.sort_values(
            by=["interaction_count", "reviews", "rating"],
            ascending=[True, True, False],
        )[
            [
                "book_id",
                "title",
                "author",
                "genre",
                "topic",
                "interaction_count",
                "reviews",
                "rating",
            ]
        ]
        .head(15)
        .reset_index(drop=True)
    )

    # 2. Topic & genre demand concentration
    topic_demand = (
        catalog_usage.groupby(["genre", "topic"], as_index=False)
        .agg(
            book_count=("book_id", "count"),
            total_interactions=("interaction_count", "sum"),
            total_reviews=("reviews", "sum"),
            mean_catalog_rating=("rating", "mean"),
        )
        .sort_values(by="total_interactions", ascending=False)
        .reset_index(drop=True)
    )
    topic_demand["mean_catalog_rating"] = topic_demand["mean_catalog_rating"].round(2)

    # 3. High catalog rating vs. low user interaction (discovery gap)
    rating_cutoff = catalog_usage["rating"].median()
    interaction_cutoff = catalog_usage["interaction_count"].quantile(0.25)

    discovery_gaps = (
        catalog_usage[
            (catalog_usage["rating"] >= rating_cutoff)
            & (catalog_usage["interaction_count"] <= interaction_cutoff)
        ]
        .sort_values(by=["rating", "interaction_count"], ascending=[False, True])[
            [
                "book_id",
                "title",
                "author",
                "genre",
                "topic",
                "rating",
                "interaction_count",
            ]
        ]
        .head(15)
        .reset_index(drop=True)
    )

    return {
        "underutilized_books": underutilized_books,
        "topic_demand": topic_demand,
        "discovery_gaps": discovery_gaps,
    }


if __name__ == "__main__":
    from src.data_loader import load_library_data

    books_df, ratings_df = load_library_data()
    reports = compute_resource_analytics(books_df, ratings_df)
    for name, df in reports.items():
        print(f"\n=== {name} ===")
        print(df.head(5).to_string(index=False))
