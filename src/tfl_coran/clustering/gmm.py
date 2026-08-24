from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture


class SoftClusterer:
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.model = GaussianMixture(
            n_components=int(config["clusters"]),
            covariance_type=str(config["covariance_type"]),
            max_iter=int(config["max_iter"]),
            reg_covar=float(config["reg_covar"]),
            n_init=int(config.get("n_init", 3)),
            random_state=seed,
            warm_start=True,
        )
        self._aligned_means: np.ndarray | None = None
        self._order: np.ndarray | None = None

    @staticmethod
    def _normalize(responsibilities: np.ndarray) -> np.ndarray:
        responsibilities = np.nan_to_num(responsibilities, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = responsibilities.sum(axis=1, keepdims=True)
        invalid = row_sum[:, 0] <= 0.0
        if np.any(invalid):
            responsibilities[invalid] = 1.0 / responsibilities.shape[1]
            row_sum = responsibilities.sum(axis=1, keepdims=True)
        return responsibilities / row_sum

    def fit_predict(self, embeddings: np.ndarray, hard: bool = False) -> np.ndarray:
        data = np.asarray(embeddings, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < self.model.n_components:
            raise ValueError("GMM needs a 2-D array with at least K samples")
        self.model.fit(data)
        responsibilities = self.model.predict_proba(data)
        means = self.model.means_
        if self._aligned_means is None:
            # Canonical first-fit order: lexicographic latent means.
            order = np.lexsort(
                tuple(means[:, column] for column in reversed(range(means.shape[1])))
            )
        else:
            cost = np.linalg.norm(self._aligned_means[:, None, :] - means[None, :, :], axis=2)
            old_indices, new_indices = linear_sum_assignment(cost)
            order = np.empty(self.model.n_components, dtype=np.int64)
            order[old_indices] = new_indices
        responsibilities = responsibilities[:, order]
        self._aligned_means = means[order].copy()
        self._order = order.copy()
        responsibilities = self._normalize(responsibilities)
        if hard:
            labels = responsibilities.argmax(axis=1)
            responsibilities = np.eye(self.model.n_components, dtype=np.float64)[labels]
        return responsibilities

    def predict(self, embeddings: np.ndarray, hard: bool = False) -> np.ndarray:
        if self._order is None:
            raise RuntimeError("fit_predict must be called before predict")
        data = np.asarray(embeddings, dtype=np.float64)
        responsibilities = self.model.predict_proba(data)[:, self._order]
        responsibilities = self._normalize(responsibilities)
        if hard:
            labels = responsibilities.argmax(axis=1)
            responsibilities = np.eye(self.model.n_components, dtype=np.float64)[labels]
        return responsibilities
