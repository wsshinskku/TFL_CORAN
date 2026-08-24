# TFL-CORAN

[English](README.en.md) | 한국어

TFL-CORAN은 5G O-RAN의 UE별 트래픽 제어를 위해 동적 클러스터링과 전이학습을 결합한 연합강화학습 프레임워크입니다. 각 UE는 로컬 DDQN을 학습하고, non-RT RIC은 VGAE 임베딩과 GMM 멤버십을 이용해 클러스터별 모델과 개인화 모델을 구성합니다. Handover 또는 신규 접속이 발생하면 같은 셀의 유사 UE 모델을 이용해 정책을 초기화합니다.

이 저장소에는 TFL-CORAN 알고리즘, 시뮬레이션 환경, 비교 방법, ablation, 다중 seed 평가 코드가 포함되어 있습니다.

## 주요 구성

- UE local DDQN: online network로 action을 선택하고 target network로 평가
- QoS reward: `reliability / target + throughput / target - latency / target`
- SITM context: signal, interference, traffic load, mobility
- symmetric kNN graph와 VGAE encoder
- GMM soft membership 기반 cluster/personalized aggregation
- handover 및 신규 UE를 위한 destination-cell model transfer
- slot, episode, FL round, cluster refresh를 분리한 multi-timescale loop
- Heuristic, DRL, FDRL, CFDRL, TFL-CORAN 비교 실험

수식과 코드의 대응 관계는 [`docs/ALGORITHM.md`](docs/ALGORITHM.md)에 정리되어 있습니다.

## 설치

Python 3.10 이상을 권장합니다.

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

## 빠른 실행

`smoke.yaml`은 CPU에서 전체 실행 경로를 확인하기 위한 소규모 설정입니다.

```bash
tfl-coran run \
  --config configs/smoke.yaml \
  --output runs/smoke

pytest -q
```

실행 결과는 다음 구조로 저장됩니다.

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

## 실험

`paper.yaml`은 논문에 제시된 topology와 학습 주기를 사용합니다. 375개 UE가 각각 DDQN을 학습하므로 먼저 예상 계산량을 확인하는 것이 좋습니다.

```bash
tfl-coran estimate --config configs/paper.yaml
```

VGAE를 먼저 학습한 뒤 모든 비교 방법을 같은 seed로 실행할 수 있습니다.

```bash
tfl-coran pretrain-vgae \
  --config configs/paper.yaml \
  --output runs/shared/vgae.pt

tfl-coran benchmark \
  --config configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --vgae-checkpoint runs/shared/vgae.pt \
  --output runs/paper_seed42
```

빠른 기능 확인에는 `configs/paper_fast.yaml`을 사용합니다.

```bash
tfl-coran ablate \
  --config configs/paper_fast.yaml \
  --output runs/ablation

tfl-coran reproduce \
  --config configs/paper.yaml \
  --methods heuristic drl fdrl cfdrl tfl_coran \
  --seeds 0 1 2 3 4 \
  --output runs/reproduction
```

## 비교 방법

| 이름 | 정의 |
|---|---|
| `heuristic` | SINR 기반 rate level, service priority, round-robin subband를 사용하는 비학습 정책 |
| `drl` | 모든 UE transition을 하나의 replay buffer에 모으는 shared DDQN |
| `fdrl` | UE별 DDQN과 FedAvg |
| `cfdrl` | GMM의 최대 posterior cluster를 사용하는 hard-cluster FL |
| `tfl_coran` | VGAE-GMM soft membership, personalized FL, model transfer |

Table 4의 component 조합은 다음과 같이 정의합니다.

- Variant A: transfer off, VGAE/GMM on
- Variant B: standardized raw context와 GMM 사용
- Variant C: VGAE embedding과 deterministic hard KMeans 사용
- Variant D: uniform membership과 FedAvg 사용

Variant B와 C에서 비활성화된 component를 대체하는 방식은 논문에 별도로 정의되어 있지 않으므로 위 설정을 실험 규칙으로 사용합니다.

## 재현성

논문은 주요 수식, topology, 학습 주기를 제시하지만 채널 trace, traffic trace, reliability target, DDQN layer, action grid 등 모든 실행 조건을 고정하지는 않습니다. 공개되지 않은 항목은 설정 파일에서 변경할 수 있으며, 기본값과 근거는 [`configs/paper.yaml`](configs/paper.yaml)과 [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md)에 기록했습니다. 실제 실행에 적용된 값은 각 결과 폴더의 `resolved_config.yaml`에서 확인할 수 있습니다.

`paper_reported/`의 값은 논문 표를 확인하기 위한 reference이며 실험 코드에서 입력으로 사용하지 않습니다. 생성 결과는 별도 `runs/` 경로에 저장합니다. UERANSIM, Open5GS, QuaDRiGa의 원본 설정과 trace는 저장소에 포함되어 있지 않으므로 기본 실행은 self-contained Python simulator를 사용합니다.

세부 실험 절차와 검증 범위는 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)와 [`VALIDATION.md`](VALIDATION.md)를 참고합니다.

## 외부 시스템 연동

외부 simulator 또는 testbed를 연결할 때 필요한 UE telemetry schema와 adapter 경계는 [`docs/EXTERNAL_SIMULATORS.md`](docs/EXTERNAL_SIMULATORS.md)에 정의되어 있습니다. UERANSIM, Open5GS, QuaDRiGa는 저장소에 포함하지 않으며 각 프로젝트의 라이선스를 따라야 합니다. 관련 내용은 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에 정리되어 있습니다.

## Citation

```bibtex
@unpublished{shin2026tflcoran,
  title={Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN},
  author={Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Choi, Minseok and Shin, Jitae},
  note={Manuscript under revision at Computer Communications},
  year={2026}
}
```

## License

The source code is released under the MIT License. External simulators and manuscript materials are governed by their respective licenses.
