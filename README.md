# TFL-CORAN

[English](README.en.md) | 한국어

첨부 논문 **“Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN”**의 수식과 실험 프로토콜을 실행 가능한 코드로 구현한 완성형 저장소입니다.

이 버전은 기존 `wsshinskku/TFL_CORAN` 프로토타입을 보완합니다. UE별 DDQN, 오프라인 VGAE, GMM soft membership, membership-weighted federated aggregation, handover/new-UE transfer initialization이 하나의 multi-timescale 실행 루프에 연결되어 있습니다. Heuristic/DRL/FDRL/CFDRL 기준선, ablation, 평가, 체크포인트, 테스트와 CI도 포함합니다.

> **재현 범위:** 논문은 채널/트래픽 trace, 신뢰도 목표, DDQN hidden layer, action rate grid, VGAE 역사 데이터 등 핵심 조건 일부를 공개하지 않습니다. 따라서 이 저장소는 **알고리즘·프로토콜 재현 구현**이며, 논문 Table 3/4의 수치를 그대로 생성했다고 주장하지 않습니다. 보고값과 결과에 영향을 주는 주요 구현 가정은 [`configs/paper.yaml`](configs/paper.yaml)과 [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md)에 표시했습니다. 상속된 기본값까지 포함한 실제 실행 설정의 기준은 각 run의 `resolved_config.yaml`입니다.

## 구현 범위

- UE local DDQN: online argmax + target evaluation, Eq. (3)-(5)
- QoS reward: reliability/target + throughput/target - latency/target, Eq. (1)
- standardized SITM context `[signal, interference, traffic, mobility]`
- symmetric-union kNN graph + Eq. (7) similarity weights (`k=6`은 각 노드의 directed 이웃 수이며, 대칭 합집합 뒤 실제 평균/최대 degree는 6을 넘을 수 있음)
- two-layer VGAE and deterministic posterior-mean inference, Eq. (8)-(12)
- GMM soft responsibilities, Eq. (13)-(14)
- cluster/global/personalized aggregation, Eq. (15)-(17)
- destination-cell cosine neighbor transfer, Eq. (18)-(19)
- 1 ms slot, episode/FL/clustering의 3중 시간척도
- 10% handover/reassignment와 3% activation, 고정 population
- 동일 simulator/config를 사용하는 5개 비교 방법 및 O/X ablation
- CSV/JSON metrics, adaptation 완료율/검열-aware 지표, model byte count, reproducibility metadata

## 설치

Python 3.10 이상이 필요합니다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
tfl-coran doctor
```

## 30초 내외 smoke 실행

```bash
tfl-coran run \
  --config configs/smoke.yaml \
  --output runs/smoke

pytest -q
```

Windows PowerShell에서는 줄바꿈 대신 한 줄로 실행하거나 backtick을 사용하면 됩니다.

결과:

```text
runs/smoke/
├── resolved_config.yaml
├── run_metadata.json
├── historical_contexts.npz
├── memberships_latest.npy
├── vgae_training.csv
├── training_metrics.csv
├── evaluation_by_group.csv
├── adaptation_events.csv
├── summary.json
└── checkpoints/
    ├── vgae.pt
    └── final_global.pt
```

## 논문 설정과 기준선

먼저 계산량을 확인하십시오. 375개 UE의 local DDQN을 학습하므로 CPU에서는 오래 걸릴 수 있습니다.

```bash
tfl-coran estimate --config configs/paper.yaml

tfl-coran pretrain-vgae \
  --config configs/paper.yaml \
  --output runs/shared/vgae.pt

tfl-coran benchmark \
  --config configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --vgae-checkpoint runs/shared/vgae.pt \
  --output runs/paper_seed42
```

빠른 기능 데모는 `configs/paper_fast.yaml`을 사용합니다. 논문 Table 4의 component toggle은 다음과 같이 실행합니다.

```bash
tfl-coran ablate --config configs/paper_fast.yaml --output runs/ablation
```

## 방법 정의

| 이름 | 구현 |
|---|---|
| `heuristic` | SINR 구간 기반 rate, service priority, 정적 round-robin frequency 분산을 사용하는 비학습 정책 |
| `drl` | 모든 UE transition을 pooled replay에 넣는 하나의 shared DDQN |
| `fdrl` | UE별 DDQN + uniform membership/FedAvg |
| `cfdrl` | GMM argmax one-hot hard cluster aggregation |
| `tfl_coran` | VGAE-GMM soft membership + personalized FL + transfer |

Ablation의 비활성 component 대체 동작은 논문에서 모두 정의하지 않으므로 다음처럼 명시적으로 고정했습니다. Variant A는 transfer만 끄고 VGAE+GMM을 유지합니다. Variant B는 transfer와 VGAE를 끄고 standardized raw context에 GMM을 적용합니다. Variant C는 transfer와 GMM을 끄고 VGAE embedding에 seed가 고정된 deterministic hard KMeans를 적용합니다. Variant D는 transfer/VGAE/GMM을 모두 끄고 uniform membership/FedAvg를 사용합니다. 따라서 특히 C의 hard KMeans는 논문 결과를 그대로 복원한 동작이 아니라, component toggle을 실행 가능하게 만든 구현 해석입니다.

## 기존 저장소에서 변경된 핵심

- config에 있던 150/120/105 UE가 실제 환경 생성에 반영됩니다.
- 무선 MDP를 continuing process로 유지하면서 episode를 최적화 구간으로 처리하고, target sync는 episode 기준입니다.
- transfer helper가 실제 handover/activation lifecycle에 연결됩니다.
- 신규 UE는 replay/optimizer를 초기화하고, handover UE는 이전+neighbor model을 혼합합니다.
- scheduler가 cell별로 6개 subband를 관리합니다.
- raw SITM scale collapse와 dense complete graph를 표준화+kNN으로 수정했습니다.
- cluster model뿐 아니라 shared global model도 계산합니다.
- CPU 자동 fallback, 정상 config loading, deterministic seeds, working tests/CI를 제공합니다.
- 평가 결과를 하드코딩하지 않으며 실제 simulator metric만 기록합니다.
- 중앙 DRL과 UE별 FL은 UE당 warmup 시점과 전체 optimizer update 예산을 맞춥니다.

자세한 수식 매핑은 [`docs/ALGORITHM.md`](docs/ALGORITHM.md), 재현성/한계는 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), 실제 릴리스 검증은 [`VALIDATION.md`](VALIDATION.md), 업그레이드 안내는 [`MIGRATION.md`](MIGRATION.md)를 참고하십시오.

## 외부 O-RAN 스택

기본 backend는 CI와 알고리즘 검증을 위한 self-contained Python simulator입니다. 논문이 언급한 UERANSIM/Open5GS/QuaDRiGa의 실험 config와 trace가 공개되지 않았으므로 실제 연동을 가장하지 않습니다. 별도 프로세스/trace adapter로 연결할 때는 [`docs/EXTERNAL_SIMULATORS.md`](docs/EXTERNAL_SIMULATORS.md)와 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)를 확인하십시오.

## 인용

```bibtex
@unpublished{shin2026tflcoran,
  title={Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN},
  author={Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Choi, Minseok and Shin, Jitae},
  note={Manuscript under revision at Computer Communications},
  year={2026}
}
```

## 라이선스

MIT License. 외부 simulator는 각자의 라이선스를 따르며 이 저장소에 포함되지 않습니다. 논문 PDF와 그림은 소프트웨어 MIT 라이선스 대상이라고 가정하지 않습니다.
