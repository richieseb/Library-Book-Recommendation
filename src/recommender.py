import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridBookRecommender:
    def __init__(self, books: pd.DataFrame, ratings: pd.DataFrame):
        self.books = books.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.book_id_to_idx = {
            int(bid): idx for idx, bid in enumerate(self.books["book_id"])
        }
        self._build_content_matrix()
        self._build_user_item_matrix()

    def _build_content_matrix(self) -> None:
        corpus = (
            self.books["title"].fillna("")
            + " "
            + self.books["author"].fillna("")
            + " "
            + self.books["genre"].fillna("")
            + " "
            + self.books["topic"].fillna("")
        )
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        self.content_sim = cosine_similarity(self.tfidf_matrix)

    def _build_user_item_matrix(self) -> None:
        if self.ratings.empty:
            self.user_item = pd.DataFrame()
            self.item_sim = np.zeros((len(self.books), len(self.books)))
            return

        self.user_item = self.ratings.pivot_table(
            index="user_id", columns="book_id", values="rating", fill_value=0.0
        )
        item_matrix = self.user_item.T.reindex(index=self.books["book_id"], fill_value=0.0)
        self.item_sim = cosine_similarity(item_matrix.values)

    def _min_max_norm(self, scores: np.ndarray) -> np.ndarray:
        span = scores.max() - scores.min()
        if span == 0:
            return np.zeros_like(scores, dtype=float)
        return (scores - scores.min()) / span

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        alpha: float = 0.5,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        n_books = len(self.books)
        content_scores = np.zeros(n_books, dtype=float)
        collab_scores = np.zeros(n_books, dtype=float)

        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        rated_book_ids = set(user_ratings["book_id"].astype(int))

        if not user_ratings.empty:
            # Weighted profile across books rated by the user
            weights = np.zeros(n_books, dtype=float)
            for _, row in user_ratings.iterrows():
                bid = int(row["book_id"])
                if bid in self.book_id_to_idx:
                    idx = self.book_id_to_idx[bid]
                    weights[idx] = float(row["rating"])

            if weights.sum() > 0:
                content_scores = self.content_sim.dot(weights) / weights.sum()

            if user_id in self.user_item.index:
                u_vec = self.user_item.loc[user_id].reindex(
                    self.books["book_id"], fill_value=0.0
                ).values
                denom = np.abs(self.item_sim).sum(axis=1)
                raw_collab = self.item_sim.dot(u_vec)
                collab_scores = np.divide(
                    raw_collab, denom, out=np.zeros_like(raw_collab), where=denom > 0
                )

        # Popularity fallback when user has no ratings
        pop_scores = (
            self._min_max_norm(self.books["rating"].fillna(0).values) * 0.6
            + self._min_max_norm(np.log1p(self.books["reviews"].clip(lower=0).values)) * 0.4
        )

        if user_ratings.empty:
            final_scores = pop_scores
        else:
            final_scores = (
                alpha * self._min_max_norm(content_scores)
                + (1.0 - alpha) * self._min_max_norm(collab_scores)
                + 0.15 * pop_scores
            )

        recs = self.books.copy()
        recs["score"] = final_scores

        if exclude_rated and rated_book_ids:
            recs = recs[~recs["book_id"].isin(rated_book_ids)]

        recs = recs.sort_values(by="score", ascending=False).head(top_n).reset_index(drop=True)
        return recs[["book_id", "title", "author", "genre", "topic", "rating", "reviews", "score"]]
