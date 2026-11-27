# Smart Library: Book Discovery, Hybrid Recommender & Resource Analytics

A modular Python mini project for book discovery, personalized recommendations, and catalog engagement analytics built over `books_clean.csv` and `ratings_synthetic.csv` with a Streamlit web interface.

---

## Problem Statement & Objectives

### Problem Statement
* Manual catalog searching slows down book discovery.
* Readers frequently miss relevant titles across genres and topics.
* Static catalog listings lack personalized recommendations.
* Catalog engagement is uneven across topics and individual titles.
* Handling diverse reader preferences requires combining content metadata with interaction signals.

### Objectives
* **Objective 1:** Develop a recommendation engine that suggests books tailored to user preferences using content similarity (`title`, `author`, `genre`, `topic`) and user-item collaborative filtering, with a popularity fallback for cold-start users.
* **Objective 2:** Improve book discovery and reader satisfaction through multi-field search, filtering, sorting, and personalized recommendations.
* **Objective 3:** Surface catalog engagement analytics—highlighting lowest-interaction books, genre/topic demand concentration, and high-rated/low-interaction discovery gaps.

---

## Dataset Summary

| Dataset | Rows | Key Columns & Stats |
| :--- | :--- | :--- |
| `data/books_clean.csv` | 347 books | `book_id`, `title`, `author`, `genre` (`Non Fiction`: 190, `Fiction`: 157), `rating` (3.3–4.9, mean 4.61, median 4.60), `reviews` (37–87,841, mean 9,773.30), `price` ($0.00–$105.00, mean $13.02), `years_on_list`, `first_year`–`last_year` (2009–2019), `topic` (14 categories) |
| `data/ratings_synthetic.csv` | 12,845 ratings | `user_id` (500 unique users), `book_id` (all 347 books covered), `rating` |

---

## Project Architecture & Module Guide

```text
library-recommender/
├── data/
│   ├── books_clean.csv
│   └── ratings_synthetic.csv
├── src/
│   ├── __init__.py       # Exposes load_library_data at package level
│   ├── data_loader.py    # Loads CSVs, enforces types, removes orphan/duplicate rows
│   ├── search.py         # Keyword + genre/topic/rating/price filtering & sorting
│   ├── recommender.py    # HybridBookRecommender (TF-IDF + item similarity + popularity)
│   └── analytics.py      # compute_resource_analytics (low engagement, topic demand, gaps)
├── app.py                # Streamlit UI (Search, Recommendations, Resource Analytics)
├── requirements.txt
├── .gitignore
└── README.md
