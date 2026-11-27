import pandas as pd


def submit_rating(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    user_id: int,
    book_id: int,
    rating: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    uid = int(user_id)
    bid = int(book_id)
    score = int(rating)

    if not (1 <= score <= 5):
        return books, ratings, "Rating must be between 1 and 5."

    if bid not in set(books["book_id"]):
        return books, ratings, f"Book ID {bid} does not exist in the catalogue."

    updated_ratings = ratings.copy()
    mask = (updated_ratings["user_id"] == uid) & (updated_ratings["book_id"] == bid)

    if mask.any():
        updated_ratings.loc[mask, "rating"] = score
        action = "Updated"
    else:
        new_row = pd.DataFrame([{"user_id": uid, "book_id": bid, "rating": score}])
        updated_ratings = pd.concat([updated_ratings, new_row], ignore_index=True)
        action = "Submitted"

    updated_books = refresh_book_rating_stats(books, updated_ratings, bid)
    return updated_books, updated_ratings, f"{action} rating {score}/5 for Book ID {bid}."


def refresh_book_rating_stats(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    book_id: int,
) -> pd.DataFrame:
    bid = int(book_id)
    updated_books = books.copy()
    book_ratings = ratings[ratings["book_id"] == bid]["rating"]

    if book_ratings.empty:
        return updated_books

    # Blend existing catalog summary with user ratings so synthetic dataset averages stay stable
    existing_rating = float(updated_books.loc[updated_books["book_id"] == bid, "rating"].iloc[0])
    existing_reviews = int(updated_books.loc[updated_books["book_id"] == bid, "reviews"].iloc[0])
    user_mean = float(book_ratings.mean())
    user_count = int(len(book_ratings))

    total_weight = max(existing_reviews + user_count, 1)
    blended_rating = (existing_rating * existing_reviews + user_mean * user_count) / total_weight

    updated_books.loc[updated_books["book_id"] == bid, "rating"] = round(blended_rating, 2)
    updated_books.loc[updated_books["book_id"] == bid, "reviews"] = max(existing_reviews, user_count)
    return updated_books
