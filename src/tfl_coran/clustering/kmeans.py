from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans


class HardKMeansClusterer:
    """Deterministic hard-cluster fallback for the VGAE-on/GMM-off ablation."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.clusters = int(config["clusters"])
        self.seed = int(seed)
        self.model: KMeans | None = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        data = np.asarray(embeddings, dtype=np.float64)
        if self.model is None:
            model = KMeans(n_clusters=self.clusters, n_init=10, random_state=self.seed)
        else:
            model = KMeans(
                n_clusters=self.clusters,
                init=self.model.cluster_centers_,
                n_init=1,
                random_state=self.seed,
            )
        labels = model.fit_predict(data)
        self.model = model
        return np.eye(self.clusters, dtype=np.float64)[labels]

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit_predict must be called before predict")
        labels = self.model.predict(np.asarray(embeddings, dtype=np.float64))
        return np.eye(self.clusters, dtype=np.float64)[labels]
