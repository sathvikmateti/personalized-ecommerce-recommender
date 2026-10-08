from pathlib import Path
import pickle
import numpy as np


class ItemCFRecommender:

    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)

        print("Loading recommendation artifacts...")

        self.knn_model = self._load("item_cf_model.pkl")
        self.user_item_matrix = self._load("user_item_matrix.pkl")
        self.user_to_idx = self._load("user_to_idx.pkl")
        self.item_to_idx = self._load("item_to_idx.pkl")
        self.idx_to_item = self._load("idx_to_item.pkl")

        print("Recommendation artifacts loaded.")

    def _load(self, filename):
        path = self.model_dir / filename

        with open(path, "rb") as f:
            return pickle.load(f)

    def recommend(self, visitor_id, n=10):

        if visitor_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[visitor_id]

        user_vector = self.user_item_matrix[user_idx]

        seen_indices = user_vector.indices
        seen_scores = user_vector.data

        if len(seen_indices) == 0:
            return []

        recommendation_scores = {}

        # Limit history used during online inference.
        # Highest interaction scores are most informative.
        max_history = 20

        history_order = np.argsort(seen_scores)[::-1][:max_history]

        selected_items = seen_indices[history_order]
        selected_scores = seen_scores[history_order]

        for item_idx, interaction_score in zip(
            selected_items,
            selected_scores
        ):

            # IMPORTANT:
            # Model was fitted on item_user_matrix,
            # so query using the corresponding item row.
            item_vector = self.user_item_matrix[:, item_idx].T

            distances, neighbor_indices = self.knn_model.kneighbors(
                item_vector,
                n_neighbors=11
            )

            for distance, neighbor_idx in zip(
                distances[0][1:],
                neighbor_indices[0][1:]
            ):

                if neighbor_idx in seen_indices:
                    continue

                similarity = 1.0 - distance

                recommendation_scores[neighbor_idx] = (
                    recommendation_scores.get(neighbor_idx, 0.0)
                    + similarity * float(interaction_score)
                )

        ranked_items = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        return [
            {
                "itemid": int(self.idx_to_item[item_idx]),
                "score": round(float(score), 4)
            }
            for item_idx, score in ranked_items
        ]