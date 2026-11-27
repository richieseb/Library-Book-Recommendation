import pandas as pd


def add_book(
    books: pd.DataFrame,
    title: str,
    author: str,
    category: str,
    description: str,
    isbn: str,
    availability_status: str = "Available",
    rating: float = 0.0,
    reviews: int = 0,
    years_on_list: int = 0,
    first_year: int = 2026,
    last_year: int = 2026,
) -> tuple[pd.DataFrame, str]:
    clean_title = title.strip()
    clean_author = author.strip()
    clean_category = category.strip()
    clean_desc = description.strip()
    clean_isbn = isbn.strip()
    clean_status = availability_status.strip() if availability_status.strip() in {"Available", "Checked Out"} else "Available"

    if not clean_title or not clean_author or not clean_category or not clean_isbn:
        return books, "Title, author, category, and ISBN are required."

    if "isbn" in books.columns and clean_isbn in set(books["isbn"].astype(str).str.strip()):
        return books, f"A book with ISBN '{clean_isbn}' already exists."

    next_id = int(books["book_id"].max()) + 1 if not books.empty else 0
    new_row = {
        "book_id": next_id,
        "isbn": clean_isbn,
        "title": clean_title,
        "author": clean_author,
        "category": clean_category,
        "description": clean_desc,
        "availability_status": clean_status,
        "rating": float(rating),
        "reviews": int(reviews),
        "years_on_list": int(years_on_list),
        "first_year": int(first_year),
        "last_year": int(last_year),
    }

    updated = pd.concat([books, pd.DataFrame([new_row])], ignore_index=True)
    return updated, f"Added '{clean_title}' (ID: {next_id})."


def update_book(
    books: pd.DataFrame,
    book_id: int,
    title: str,
    author: str,
    category: str,
    description: str,
    isbn: str,
    availability_status: str,
) -> tuple[pd.DataFrame, str]:
    mask = books["book_id"] == int(book_id)
    if not mask.any():
        return books, f"Book ID {book_id} not found."

    clean_isbn = isbn.strip()
    if "isbn" in books.columns and clean_isbn:
        duplicate_isbn = books[(books["book_id"] != int(book_id)) & (books["isbn"].astype(str).str.strip() == clean_isbn)]
        if not duplicate_isbn.empty:
            return books, f"ISBN '{clean_isbn}' is already assigned to another book."

    updated = books.copy()
    updated.loc[mask, "title"] = title.strip()
    updated.loc[mask, "author"] = author.strip()
    updated.loc[mask, "category"] = category.strip()
    updated.loc[mask, "description"] = description.strip()
    updated.loc[mask, "isbn"] = clean_isbn
    if availability_status in {"Available", "Checked Out"}:
        updated.loc[mask, "availability_status"] = availability_status
    return updated, f"Updated Book ID {book_id}."


def delete_book(books: pd.DataFrame, ratings: pd.DataFrame, book_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    target_id = int(book_id)
    if target_id not in set(books["book_id"]):
        return books, ratings, f"Book ID {target_id} not found."

    updated_books = books[books["book_id"] != target_id].reset_index(drop=True)
    updated_ratings = ratings[ratings["book_id"] != target_id].reset_index(drop=True)
    return updated_books, updated_ratings, f"Deleted Book ID {target_id} and its associated ratings."
