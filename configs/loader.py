"""
Config loader & validator for TFL-CORAN.
- Supports stacking multiple YAMLs (base -> overrides -> ablations)
- CLI-style overrides: key1.key2=value (optional)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import copy
import os
import re
import yaml

# ------------------------- dataclass schema ------------------------- #

@dataclass
class PathsCfg:
    vgae_checkpoint: str
    cluster_models_dir: str
    logs_dir: str

@dataclass
class RunCfg:
    seed: int = 42
    device: str = "cuda"
    run_name: str = "tfl_coran_default"
    output_dir: str = "runs/tfl_coran_default"
    deterministic: bool = False

@dataclass
class LoggingCfg:
    level: str = "INFO"
    tensorboard: bool = True
    csv: bool = True
    save_every_round: int = 1

@dataclass
class TopologyCfg:
    cells: int = 3
    isd_m: int = 500
    ue_counts: Dict[str, int] = field(default_factory=lambda: {"urban":150,"commercial":120,"school":105})

@dataclass
class CarrierCfg:
    center_freq_ghz: float = 3.5
    bandwidth_mhz: int = 100
    n_subbands: int = 6

@dataclass
class TimeScalesCfg:
    episode_len: int = 200
    tau_E: int = 5
    tau_F: int = 10
    warmup_episodes: int = 0

@dataclass
class VGAECfg:
    in_dim: int = 4
    gcn_dims: List[int] = field(default_factory=lambda: [64,32])
    latent_dim: int = 32

@dataclass
class GMMCfg:
    K: int = 3
    covariance_type: str = "full"
    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-6

@dataclass
class EpsilonCfg:
    start: float = 1.0
    end: float = 0.01
    decay_episodes: int = 100

@dataclass
class DDQNCfg:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    target_update_episodes: int = 10
    replay_capacity: int = 100000
    epsilon: EpsilonCfg = field(default_factory=EpsilonCfg)
    optimizer: str = "adam"
    grad_clip_norm: float = 10.0

@dataclass
class ModelsCfg:
    vgae: VGAECfg = field(default_factory=VGAECfg)
    gmm: GMMCfg = field(default_factory=GMMCfg)
    ddqn: DDQNCfg = field(default_factory=DDQNCfg)

@dataclass
class FLCfg:
    aggregator: str = "membership_weighted"  # membership_weighted | fedavg | hard_cluster
    weight_dtype: str = "fp32"
    fuse_personalized: bool = True
    clip_delta_global_norm: float = 100.0
    keep_last_cluster_models: int = 5

@dataclass
class TransferCfg:
    enabled: bool = True
    delta: float = 0.5
    neighbor_pool_size: int = 32
    similarity: str = "cosine"

@dataclass
class PrivacyCfg:
    send_embeddings_only: bool = False
    compress_updates: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "bits": 8})

@dataclass
class MobilityCfg:
    speed_mps: Dict[str, float] = field(default_factory=lambda: {"min": 0.0, "max": 5.0})
    delta_v: float = 0.2
    delta_dir_deg: float = 15.0
    handover_on_boundary: bool = True

@dataclass
class TrafficTargetsCfg:
    rate_bps: int
    latency_s: float
    reliability: float

@dataclass
class TrafficCfg:
    classes: List[str] = field(default_factory=lambda: ["eMBB","URLLC","mMTC"])
    targets: Dict[str, TrafficTargetsCfg] = field(default_factory=dict)
    new_activation_ratio_per_round: float = 0.03
    reassignment_ratio_per_round: float = 0.10

@dataclass
class SchedulerCfg:
    reuse_allowed: bool = True
    utility_weights: Dict[str, float] = field(default_factory=lambda: {"priority":1.0,"predicted_rate":1.0,"qos_deficit":1.0})
    harq_like_retx: bool = True
    sinr_to_mcs_table: Optional[str] = "data/tables/sinr_mcs.csv"

@dataclass
class EnvCfg:
    mode: str = "sim"
    mobility: MobilityCfg = field(default_factory=MobilityCfg)
    traffic: TrafficCfg = field(default_factory=TrafficCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    transport: Dict[str, Any] = field(default_factory=dict)  # only used in o_ran mode

@dataclass
class EvaluationCfg:
    metrics: List[str] = field(default_factory=lambda: ["qos_satisfaction","avg_throughput","avg_latency","adaptation_time"])
    num_seeds: int = 5
    report_dir: str = "runs/tfl_coran_default/reports"
    baselines: List[str] = field(default_factory=lambda: ["heuristic","drl","fdrl","cfdrl"])

@dataclass
class Config:
    cfg_version: int = 1
    run: RunCfg = field(default_factory=RunCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    paths: PathsCfg = field(default_factory=lambda: PathsCfg(
        vgae_checkpoint="data/checkpoints/vgae_encoder.pt",
        cluster_models_dir="data/checkpoints/cluster_models",
        logs_dir="runs/tfl_coran_default/logs"
    ))
    topology: TopologyCfg = field(default_factory=TopologyCfg)
    carrier: CarrierCfg = field(default_factory=CarrierCfg)
    time_scales: TimeScalesCfg = field(default_factory=TimeScalesCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    fl: FLCfg = field(default_factory=FLCfg)
    transfer: TransferCfg = field(default_factory=TransferCfg)
    privacy: PrivacyCfg = field(default_factory=PrivacyCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)

# --------------------------- load & merge --------------------------- #

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge b into a (modifies and returns a)."""
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _interpolate_env_vars(d: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ${VAR} and ${a.b} style placeholders if present."""
    pat = re.compile(r"\$\{([^}]+)\}")
    def _expand(val: Any, ctx: Dict[str, Any]) -> Any:
        if isinstance(val, str):
            def repl(m):
                key = m.group(1)
                # support ${ENV} or ${a.b.c} from ctx
                if key in os.environ:
                    return os.environ[key]
                cur = ctx
                for part in key.split("."):
                    cur = cur.get(part, "")
                return str(cur)
            return pat.sub(repl, val)
        if isinstance(val, dict):
            return {k: _expand(v, ctx) for k, v in val.items()}
        if isinstance(val, list):
            return [_expand(v, ctx) for v in val]
        return val
    return _expand(d, d)

def load_config(paths: List[str], overrides: Optional[List[str]]=None) -> Config:
    """Load one or more YAML files and apply CLI-style overrides."""
    merged: Dict[str, Any] = {}
    for p in paths:
        y = _load_yaml(p)
        _deep_merge(merged, y)
    if overrides:
        for ov in overrides:
            # example: "time_scales.tau_F=3"
            key, val = ov.split("=", 1)
            _apply_override(merged, key.strip(), _parse_val(val.strip()))
    merged = _interpolate_env_vars(merged)
    # materialize dataclasses
    cfg = _to_dataclass(Config, merged)
    validate_config(cfg)
    return cfg

# --------------------------- overrides ----------------------------- #

def _parse_val(v: str) -> Any:
    if v.lower() in ("true","false"):
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v  # string

def _apply_override(d: Dict[str, Any], dotted: str, value: Any):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

# ---------------------- dict -> dataclass -------------------------- #

def _to_dataclass(cls, data: Dict[str, Any]):
    """Recursively instantiate dataclasses from dict"""
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    kwargs = {}
    for name, fieldinfo in cls.__dataclass_fields__.items():
        ftype = fieldinfo.type
        if name not in data:
            kwargs[name] = getattr(cls(), name)
            continue
        val = data[name]
        if hasattr(ftype, "__dataclass_fields__"):
            kwargs[name] = _to_dataclass(ftype, val)
        elif getattr(ftype, "__origin__", None) is list:
            kwargs[name] = val
        elif getattr(ftype, "__origin__", None) is dict:
            kwargs[name] = val
        else:
            kwargs[name] = val
    return cls(**kwargs)

# ----------------------------- validate --------------------------- #

def validate_config(cfg: Config):
    assert cfg.time_scales.episode_len > 0, "episode_len must be > 0"
    assert cfg.time_scales.tau_E >= 1, "tau_E must be >= 1"
    assert cfg.time_scales.tau_F >= 1, "tau_F must be >= 1"
    assert cfg.models.gmm.K >= 1, "GMM K must be >= 1"
    assert 0.0 < cfg.models.ddqn.gamma < 1.0, "gamma must be in (0,1)"
    eps = cfg.models.ddqn.epsilon
    assert 0.0 <= eps.end <= eps.start <= 1.0, "epsilon range invalid"
    if cfg.fl.aggregator not in ("membership_weighted","fedavg","hard_cluster"):
        raise ValueError(f"Unknown aggregator '{cfg.fl.aggregator}'")
    if cfg.env.mode not in ("sim","o_ran"):
        raise ValueError(f"env.mode must be 'sim' or 'o_ran'")
    # paths sanity
    if not cfg.paths.vgae_checkpoint:
        raise ValueError("paths.vgae_checkpoint must be set")

def save_resolved_config(cfg: Config, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)

# ----------------------------- example ---------------------------- #
if __name__ == "__main__":
    # 예) python configs/loader.py configs/default.yaml configs/env_sim.yaml time_scales.tau_F=3
    import sys
    files = [p for p in sys.argv[1:] if "=" not in p]
    ovs = [p for p in sys.argv[1:] if "=" in p]
    cfg = load_config(files, ovs)
    print(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True))
