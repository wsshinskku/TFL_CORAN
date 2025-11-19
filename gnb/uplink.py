# gnb/uplink.py
# Δw, SITM 등을 전송 친화적 포맷으로 패킹/언패킹합니다.
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import torch
import zlib
import pickle

def _tensor_to_bytes(t: torch.Tensor, qbits: int = 0) -> bytes:
    if qbits in (0, 32):
        arr = t.detach().cpu().numpy().astype(np.float32)
    elif qbits == 16:
        arr = t.detach().cpu().numpy().astype(np.float16)
    elif qbits == 8:
        # symmetric per-tensor quantization
        x = t.detach().cpu().numpy().astype(np.float32)
        s = np.max(np.abs(x)) + 1e-12
        q = np.clip(np.round(x / s * 127.0), -128, 127).astype(np.int8)
        arr = np.stack([q, np.array([s], dtype=np.float32)], axis=0)   # store scale
    else:
        raise ValueError("qbits must be one of {0,8,16,32}")
    return pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)

def _bytes_to_tensor(b: bytes, qbits: int = 0, device: str = "cpu") -> torch.Tensor:
    arr = pickle.loads(b)
    if qbits in (0, 32):
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    elif qbits == 16:
        return torch.from_numpy(arr).to(device=device, dtype=torch.float16).float()
    elif qbits == 8:
        q = arr[0].astype(np.int8)
        s = float(arr[1][0])
        x = (q.astype(np.float32) / 127.0) * s
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)
    else:
        raise ValueError("qbits must be one of {0,8,16,32}")

def pack_state_dict(sd: Dict[str, torch.Tensor], qbits: int = 0, compress: bool = True) -> bytes:
    blob = {k: _tensor_to_bytes(v, qbits=qbits) for k, v in sd.items()}
    raw = pickle.dumps({"qbits": qbits, "blob": blob}, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, 9) if compress else raw

def unpack_state_dict(data: bytes, device: str = "cpu") -> Dict[str, torch.Tensor]:
    raw = zlib.decompress(data)
    pay = pickle.loads(raw)
    qbits = int(pay["qbits"])
    blob = {k: _bytes_to_tensor(v, qbits=qbits, device=device) for k, v in pay["blob"].items()}
    return blob

def pack_sitm(sitm: Dict[str, np.ndarray], compress: bool = True) -> bytes:
    raw = pickle.dumps(sitm, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, 9) if compress else raw

def unpack_sitm(data: bytes) -> Dict[str, np.ndarray]:
    raw = zlib.decompress(data)
    return pickle.loads(raw)
