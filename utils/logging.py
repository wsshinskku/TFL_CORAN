# utils/logging.py
# Unified logging helpers: Python logging + TensorBoard + CSV/JSONL
# Designed for multi-timescale runs (episode/FL round/embedding refresh). 
from __future__ import annotations
import os, sys, csv, json, logging, time
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter  # optional
except Exception:
    SummaryWriter = None  # type: ignore

# ------------------------------ base logger ------------------------------ #

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def setup_logging(
    run_dir: str,
    run_name: str = "tfl_coran",
    level: str = "INFO",
    stdout: bool = True,
    logfile: Optional[str] = None
) -> logging.Logger:
    """
    Create a named logger writing to stdout and/or file.
    """
    _ensure_dir(run_dir)
    logger = logging.getLogger(run_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # clear duplicate handlers (useful on notebooks/reloads)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    if stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt); logger.addHandler(sh)
    if logfile:
        _ensure_dir(os.path.dirname(logfile))
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

# ------------------------------ tensorboard ------------------------------ #

@dataclass
class TBLogger:
    log_dir: str
    enabled: bool = True

    def __post_init__(self):
        self.writer = None
        if self.enabled and SummaryWriter is not None:
            _ensure_dir(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalars(self, tag: str, scalars: Dict[str, float], step: int):
        if self.writer is None: return
        for k, v in scalars.items():
            try:
                self.writer.add_scalar(f"{tag}/{k}", float(v), step)
            except Exception:
                pass

    def add_text(self, tag: str, text: str, step: int):
        if self.writer is None: return
        try:
            self.writer.add_text(tag, text, step)
        except Exception:
            pass

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

# ------------------------------ CSV / JSONL ------------------------------ #

class CSVLogger:
    """
    Append-only CSV metric logger.
    Header is inferred from first row; later rows may contain a superset of keys (missing -> blank).
    """
    def __init__(self, path: str):
        _ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self._fieldnames = None
        self._fh = open(self.path, "a", newline="")
        self._writer = None

    def log(self, row: Dict[str, Any]):
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            if os.stat(self.path).st_size == 0:
                self._writer.writeheader()
        else:
            # union of previous + new keys
            new_keys = [k for k in row.keys() if k not in self._fieldnames]
            if new_keys:
                self._fieldnames.extend(new_keys)
                # rewrite header? keep simple: write only new row with missing filled
                self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})  # type: ignore
        self._fh.flush()

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

class JSONLLogger:
    """Append-only JSON Lines logger."""
    def __init__(self, path: str):
        _ensure_dir(os.path.dirname(path) or ".")
        self.path = path
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, obj: Dict[str, Any]):
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

# ------------------------------ factory ------------------------------ #

def build_loggers(output_dir: str, enable_tb: bool = True, enable_csv: bool = True):
    """
    Create default (tb, csv, jsonl) loggers under:
      output_dir/tb, output_dir/metrics.csv, output_dir/metrics.jsonl
    """
    tb = TBLogger(os.path.join(output_dir, "tb"), enabled=enable_tb)
    csvlog = CSVLogger(os.path.join(output_dir, "metrics.csv")) if enable_csv else None
    jsonl = JSONLLogger(os.path.join(output_dir, "metrics.jsonl"))
    return tb, csvlog, jsonl
