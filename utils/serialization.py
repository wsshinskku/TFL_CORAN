# utils/serialization.py
# Checkpointing helpers for UE agent weights and RIC cluster models.
from __future__ import annotations
import os, io, zlib, json, time
from typing import Dict, List, Any, Optional

import torch

Weights = Dict[str, torch.Tensor]

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# ------------------------------ agents ------------------------------ #

def save_agent_weights(state: Weights, out_path: str, compress: bool = True):
    """
    Save a single agent's state_dict to disk (CPU tensors). Supports zlib compression.
    """
    _ensure_dir(os.path.dirname(out_path) or ".")
    data = {k: v.detach().cpu() for k, v in state.items()}
    buf = io.BytesIO()
    torch.save(data, buf)
    payload = zlib.compress(buf.getvalue(), 9) if compress else buf.getvalue()
    with open(out_path, "wb") as f: f.write(payload)

def load_agent_weights(path: str, device: str = "cpu", compressed: bool = True) -> Weights:
    with open(path, "rb") as f: raw = f.read()
    if compressed:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
    buf = io.BytesIO(raw)
    state: Weights = torch.load(buf, map_location=device)
    return state

# ------------------------------ cluster models ------------------------------ #

def save_cluster_models(cluster_models: List[Weights], out_path: str, compress: bool = True):
    """
    Save list of cluster model weights [w̄_k] (k=0..K-1).
    """
    _ensure_dir(os.path.dirname(out_path) or ".")
    payload = []
    for k, w in enumerate(cluster_models):
        payload.append({kk: vv.detach().cpu() for kk, vv in w.items()})
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = zlib.compress(buf.getvalue(), 9) if compress else buf.getvalue()
    with open(out_path, "wb") as f: f.write(data)

def load_cluster_models(path: str, device: str = "cpu", compressed: bool = True) -> List[Weights]:
    with open(path, "rb") as f: raw = f.read()
    if compressed:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
    buf = io.BytesIO(raw)
    lst: List[Weights] = torch.load(buf, map_location=device)
    return lst

# ------------------------------ full checkpoint ------------------------------ #

def save_full_checkpoint(
    out_dir: str,
    round_idx: int,
    agents: Dict[str, Any],                # expects .get_weights()
    cluster_models: List[Weights],
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a full experiment checkpoint (per FL round).
    Layout:
      out_dir/
        ckpt_round_{r:05d}/
          agent_{ue_id}.pt
          clusters.pt
          meta.json
    """
    ckpt_dir = os.path.join(out_dir, f"ckpt_round_{round_idx:05d}")
    _ensure_dir(ckpt_dir)
    # agents
    for ue_id, agent in agents.items():
        save_agent_weights(agent.get_weights(), os.path.join(ckpt_dir, f"agent_{ue_id}.pt"))
    # clusters
    save_cluster_models(cluster_models, os.path.join(ckpt_dir, "clusters.pt"))
    # meta
    meta = {"round": round_idx, "time": time.time()}
    if extra: meta.update(extra)
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return ckpt_dir

# --- Compatibility wrappers for tests ---
def save_state_dict(state: Dict[str, torch.Tensor], path: str):
    """Compatibility wrapper for tests."""
    save_agent_weights(state, path)

def load_state_dict(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Compatibility wrapper for tests."""
    return load_agent_weights(path, device=map_location)

