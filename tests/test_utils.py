# tests/test_utils.py
import os, tempfile, json
import torch
from utils.serialization import save_state_dict, load_state_dict
from utils.logger import CSVLogger, JSONLLogger

def test_serialization_roundtrip(tmp_path):
    path = tmp_path / "w.pt"
    sd = {"a": torch.randn(2), "b": torch.randn(3)}
    save_state_dict(sd, str(path))
    sd2 = load_state_dict(str(path), map_location="cpu")
    assert set(sd2.keys()) == set(sd.keys())
    assert torch.allclose(sd2["a"], sd["a"], atol=0, rtol=0)

def test_csv_jsonl_logging(tmp_path):
    csvp = tmp_path / "m.csv"
    jlp = tmp_path / "m.jsonl"
    csvlog = CSVLogger(str(csvp))
    jsonllog = JSONLLogger(str(jlp))
    csvlog.log(1, {"QoS_%": 87.7, "AvgDelay_ms": 10.2})
    jsonllog.log(1, {"QoS_%": 87.7, "AvgDelay_ms": 10.2})
    csvlog.close(); jsonllog.close()
    assert csvp.exists() and jlp.exists()
    # JSONL 첫 줄 파싱
    with open(jlp, "r") as f:
        line = f.readline()
    rec = json.loads(line)
    assert rec["step"] == 1 and "QoS_%" in rec
