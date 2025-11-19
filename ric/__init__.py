# ric/__init__.py
from .topology import build_features_matrix, build_similarity_adjacency, normalize_adj
from .vgae import VGAEEncoder, infer_embeddings
from .gmm import SoftGMM
from .fl_server import aggregate_membership_weighted, personalize_by_membership
from .transfer import warm_start_from_neighbors
