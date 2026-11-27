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
        # Weight category and author slightly higher alongside description text
        corpus = (
            self.books["description"].fillna("").astype(str)
            + " "
            + (self.books["author"].fillna("").astype(str) + " ") * 2
            + (self.books["category"].fillna("").astype(str) + " ") * 2
        )
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        self.content_sim = cosine_similarity(self.tfidf_matrix)

    def _build_user_item_matrix(self) -> None:
        if self.ratings.empty:
            self.user_item = pd.DataFrame()
            self.item_sim = np.zeros((len(self.books), len(self.books)), dtype=float)
            return

        self.user_item = self.ratings.pivot_table(
            index="user_id",
            columns="book_id",
            values="rating",
            aggfunc="mean",
            fill_value=0.0,
        )
        aligned_items = self.user_item.T.reindex(index=self.books["book_id"], fill_value=0.0)
        self.item_sim = cosine_similarity(aligned_items.values)

    @staticmethod
    def _min_max_norm(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        span = arr.max() - arr.min()
        if span <= 0.0:
            return np.zeros_like(arr, dtype=float)
        return (arr - arr.min()) / span

    def recommend_content_based(
        self,
        user_id: int,
        top_n: int = 5,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        return self._format_results(
            user_id=user_id,
            top_n=top_n,
            alpha=1.0,
            use_collab=False,
            exclude_rated=exclude_rated,
        )

    def recommend_collaborative(
        self,
        user_id: int,
        top_n: int = 5,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        return self._format_results(
            user_id=user_id,
            top_n=top_n,
            alpha=0.0,
            use_collab=True,
            exclude_rated=exclude_rated,
        )

    def recommend_hybrid(
        self,
        user_id: int,
        top_n: int = 10,
        alpha: float = 0.5,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        return self._format_results(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            use_collab=True,
            exclude_rated=exclude_rated,
        )

    def _compute_scores(self, user_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
        n_books = len(self.books)
        content_scores = np.zeros(n_books, dtype=float)
        collab_scores = np.zeros(n_books, dtype=float)

        user_rows = self.ratings[self.ratings["user_id"] == int(user_id)]
        rated_book_ids = set(user_rows["book_id"].astype(int))

        if not user_rows.empty:
            weights = np.zeros(n_books, dtype=float)
            for _, row in user_rows.iterrows():
                bid = int(row["book_id"])
                if bid in self.book_id_to_idx:
                    weights[self.book_id_to_idx[bid]] = float(row["rating"])

            if weights.sum() > 0.0:
                content_scores = self.content_sim.dot(weights) / weights.sum()

            if not self.user_item.empty and int(user_id) in self.user_item.index:
                user_vec = (
                    self.user_item.loc[int(user_id)]
                    .reindex(self.books["book_id"], fill_value=0.0)
                    .values.astype(float)
                )
                denom = np.abs(self.item_sim).sum(axis=1)
                raw_collab = self.item_sim.dot(user_vec)
                collab_scores = np.divide(
                    raw_collab,
                    denom,
                    out=np.zeros_like(raw_collab, dtype=float),
                    where=denom > 0.0,
                )

        fallback_scores = (
            0.7 * self._min_max_norm(self.books["rating"].fillna(0.0).values)
            + 0.3 * self._min_max_norm(np.log1p(self.books["reviews"].clip(lower=0).values))
        )
        return content_scores, collab_scores, fallback_scores, rated_book_ids

    def _format_results(
        self,
        user_id: int,
        top_n: int,
        alpha: float,
        use_collab: bool,
        exclude_rated: bool,
    ) -> pd.DataFrame:
        allowed_top_n = top_n if top_n in (5, 10) else (5 if top_n < 8 else 10)
        clamped_alpha = float(np.clip(alpha, 0.0, 1.0))

        content_scores, collab_scores, fallback_scores, rated_book_ids = self._compute_scores(
            user_id
        )
        user_rows = self.ratings[self.ratings["user_id"] == int(user_id)]

        if user_rows.empty:
            final_scores = fallback_scores
        elif not use_collab:
            norm_c = self._min_max_norm(content_scores)
            final_scores = norm_c if norm_c.max() > 0 else fallback_scores
        else:
            norm_c = self._min_max_norm(content_scores)
            norm_f = self._min_max_norm(collab_scores)
            if norm_c.max() == 0 and norm_f.max() == 0:
                final_scores = fallback_scores
            else:
                final_scores = clamped_alpha * norm_c + (1.0 - clamped_alpha) * norm_f

        out = self.books.copy()
        out["recommendation_score"] = np.round(final_scores, 4)

        if exclude_rated and rated_book_ids:
            out = out[~out["book_id"].isin(rated_book_ids)]

        out = out.sort_values(
            by=["recommendation_score", "rating", "reviews"],
            ascending=[False, False, False],
        ).head(allowed_top_n).reset_index(drop=True)

        cols = [
            "book_id",
            "isbn",
            "title",
            "author",
            "category",
            "description",
            "availability_status",
            "rating",
            "recommendation_score",
        ]
        return out[[c for c in cols if c in out.columns]]
