# TFL-CORAN

**Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN** 논문의 **저자 공식 구현**입니다.

[English](README.md) | [한국어](README.ko.md)

**저자:** Wooseok Shin, Janghoon Yang, Zhiqiang Shen, Minseok Choi, Jitae Shin.

TFL-CORAN은 UE별 double deep Q-learning(DDQN), 그래프 표현 학습, 소프트 클러스터링, 모델 전이를 결합하여 서로 다른 특성을 가진 5G Open RAN 셀의 트래픽을 관리합니다.

## 방법

- **로컬 제어:** 각 UE가 DDQN으로 서브밴드, 전송률 단계, 우선순위를 선택합니다.
- **문맥 표현:** 변분 그래프 오토인코더(VGAE)가 신호·간섭·트래픽·이동성(SITM) 요약을 인코딩합니다.
- **동적 클러스터링:** Gaussian mixture의 소프트 소속도로 개인화된 연합 집계를 수행합니다.
- **모델 전이:** 핸드오버 또는 신규 활성화 UE가 목적지 셀의 유사한 활성 UE로부터 모델을 초기화합니다.
- **평가:** 학습 중 처리량, 지연, 신뢰도, QoS 만족도, 적응 과정을 측정한 뒤 학습을 멈춘 정책으로 별도 평가합니다.

기본 실행 환경은 채널, 이동성, 트래픽, 큐, 스케줄링을 설정할 수 있는 독립형 Python 시뮬레이터입니다.

## 설치

**Python 3.10 이상**을 사용합니다. 주요 의존성은 PyTorch, NumPy, SciPy, scikit-learn, PyYAML이며 CPU와 CUDA를 지원합니다.

```bash
git clone https://github.com/wsshinskku/TFL_CORAN.git
cd TFL_CORAN
python -m venv .venv
```

Linux/macOS에서는 `source .venv/bin/activate`, PowerShell에서는 `.venv\Scripts\Activate.ps1`로 가상환경을 활성화합니다.

```bash
python -m pip install -e ".[dev]"
tfl-coran doctor
```

## 빠른 시작

```bash
tfl-coran run -c configs/smoke.yaml -o runs/smoke
tfl-coran report runs/smoke
python -m pytest
```

Smoke 설정은 CPU에서 UE 12개와 짧은 학습 에피소드 2회를 실행합니다. DDQN, VGAE, 클러스터링, 집계, 전이 이벤트, 평가, 체크포인트 저장을 확인할 수 있습니다.

## 실험 설정

| 설정 | 용도 |
|---|---|
| [smoke.yaml](configs/smoke.yaml) | UE 12개를 이용한 빠른 통합 확인 |
| [paper_fast.yaml](configs/paper_fast.yaml) | UE 75개와 줄인 학습량을 사용하는 노트북 실험 |
| [paper.yaml](configs/paper.yaml) | 3개 셀의 UE 375개, 500에피소드, FL 100라운드 |

전체 설정은 셀별 UE 150/120/105개, 3.5 GHz 반송파, 100 MHz 대역폭, 서브밴드 6개, 1 ms 슬롯을 사용합니다. 서비스별 처리량/지연 목표는 eMBB 20 Mbps/15 ms, URLLC 10 Mbps/5 ms, mMTC 5 Mbps/10 ms입니다. 로컬 DDQN 에이전트 375개를 실행하기 전에 계산량을 확인할 수 있습니다.

```bash
tfl-coran estimate -c configs/paper.yaml
tfl-coran run -c configs/paper_fast.yaml -o runs/demo --seed 42
tfl-coran run -c configs/paper.yaml -o runs/full --device cuda
```

모든 실행은 상속된 기본값을 포함한 최종 설정을 `resolved_config.yaml`에 저장합니다. 상태·행동 인코딩과 모델 매개변수는 [실험 설정 문서](docs/ASSUMPTIONS.md)에 정리되어 있습니다.

## 비교 실험과 ablation

| 방법 | CLI 이름 | 동작 |
|---|---|---|
| Heuristic | `heuristic` | SINR 기반 전송률, 서비스 우선순위, 고정 주파수 분산 |
| DRL | `drl` | Replay를 모아 학습하는 중앙집중형 DDQN |
| FDRL | `fdrl` | UE별 DDQN과 FedAvg |
| CFDRL | `cfdrl` | Hard 클러스터 소속도에 따른 연합 집계 |
| TFL-CORAN | `tfl_coran` | 소프트 개인화 집계와 모델 전이 |

```bash
tfl-coran benchmark -c configs/paper_fast.yaml --methods all -o runs/benchmark
tfl-coran ablate -c configs/paper_fast.yaml -o runs/ablations
tfl-coran reproduce -c configs/paper.yaml --methods heuristic drl fdrl cfdrl tfl_coran --seeds 0 1 2 3 4 -o runs/reproduction
```

Ablation은 전체 방법과 네 가지 변형을 실행합니다. A는 전이를 끄고, B는 표준화한 원본 SITM에 GMM을 적용하며, C는 VGAE 임베딩에 hard KMeans를 사용하고, D는 균일 소속도/FedAvg를 사용합니다.

동일하게 사전 학습한 표현 모델을 공유하려면:

```bash
tfl-coran pretrain-vgae -c configs/paper_fast.yaml -o runs/shared/vgae.pt
tfl-coran benchmark -c configs/paper_fast.yaml --methods all --vgae-checkpoint runs/shared/vgae.pt -o runs/shared-benchmark
```

중앙집중형과 연합학습 방법은 전체 optimizer 업데이트 횟수와 replay 용량을 맞춥니다. 적응 지표는 완료된 이벤트의 평균, 완료율, 미완료 이벤트의 관찰 기간을 반영한 horizon-penalized 평균을 함께 제공합니다.

## 출력 파일

| 파일 | 내용 |
|---|---|
| `resolved_config.yaml` | 전체 실험 설정 |
| `run_metadata.json` | 시드, 패키지/플랫폼 버전, Git 상태 |
| `historical_contexts.npz` | 표현 학습에 사용한 SITM 관측값 |
| `memberships_latest.npy` | 최신 클라이언트별 클러스터 소속도 |
| `vgae_training.csv` | 표현 학습 손실 |
| `training_metrics.csv` | 학습 과정의 지표 |
| `evaluation_by_group.csv` | 고정 정책의 셀/서비스 그룹별 평가 |
| `adaptation_events.csv` | 핸드오버와 활성화 적응 기록 |
| `summary.json` | 실행 전체의 요약 지표 |
| `checkpoints/` | VGAE와 글로벌 모델 체크포인트 |

Benchmark, ablation, 다중 시드 명령은 각각 집계 JSON도 생성합니다. 다중 시드 요약에는 평균, 표본 표준편차, 정규근사 95% 신뢰구간이 포함됩니다.

## 저장소 구성

| 경로 | 역할 |
|---|---|
| [src/tfl_coran](src/tfl_coran) | 에이전트, 환경, VGAE, 클러스터링, FL, 전이, 실험 실행 |
| [configs](configs) | 실험 설정 |
| [scripts](scripts) | 호환용 진입점과 smoke 실행 스크립트 |
| [tests](tests) | 수식, 상태 전환, 통합 검증 |
| [알고리즘 가이드](docs/ALGORITHM.md) | 수식과 코드의 대응 및 갱신 순서 |
| [재현성 가이드](docs/REPRODUCIBILITY.md) | 시드, 비교 실험, 실행 정보 저장 |
| [외부 시스템 연동](docs/EXTERNAL_SIMULATORS.md) | 텔레메트리와 행동 인터페이스 |
| [논문 표](paper_reported/README.md) | 출처 정보가 포함된 표 3과 표 4 |
| [검증 기록](VALIDATION.md) | 수행한 검증과 테스트 범위 |

## 인용

```bibtex
@unpublished{shin2026tflcoran,
  title  = {Transfer-enhanced Federated Learning with Dynamic Clustering for Traffic Management in 5G Open RAN},
  author = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Choi, Minseok and Shin, Jitae},
  note   = {Manuscript under revision at Computer Communications},
  year   = {2026}
}
```

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 있습니다. 소스 코드는 [MIT License](LICENSE)로 배포하며, 외부 의존성의 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.
