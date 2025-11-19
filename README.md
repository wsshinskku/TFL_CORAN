# TFL-CORAN: Transfer-Enhanced Federated Learning with Dynamic Clustering for 5G Open RAN

> UE–gNB–RIC 3계층, **VGAE 임베딩** + **GMM 소프트 클러스터링** + **멤버십 가중 FL** + **전이 초기화**로
> UE-레벨 트래픽 제어를 학습하는 프레임워크입니다. 식 (1)–(6), (9)–(10)과 표 I–III의 주기를 코드로 제공합니다.

- **논문 PDF**: [/mnt/data/TFL_CORAN.pdf](/mnt/data/TFL_CORAN.pdf)  
- **구조**: UE(DDQN, 슬롯) → gNB(수집/중계) → RIC(VGAE 추론·GMM·FL 집계, 라운드).

## Quickstart

```bash
# 0) 설치
pip install -e .[dev]  # 또는 pip install -r requirements.txt

# 1) VGAE 오프라인 프리트레이닝(식 (4))
python scripts/pretrain_vgae.py --snapshots data/historical --gen-if-missing --epochs 10 --out data/checkpoints/vgae_encoder.pt

# 2) 본 실험(표 II 기본: τE=5, τF=10) — 100 FL 라운드
python scripts/run_sim.py --config configs/default.yaml --config configs/env_sim.yaml --rounds 100

# 3) 테스트
pytest -q


configs/   envs/         ue/         ric/         gnb/        models/     data/
scripts/   utils/        tests/      docs/        .github/    README.md   pyproject.toml


If you use this code, please cite this repository. 
