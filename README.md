# Bio-Sync HPC
### A Cross-Layer Self-Optimising Federated Learning Infrastructure for Biomedical IoT

> **v2-stable** — Deployed on Azure Kubernetes Service · Southeast Asia

---

## What This Is

Bio-Sync HPC is a production-grade, event-driven Federated Learning (FL) infrastructure designed to process high-frequency biomedical EEG signals from distributed hospital silos without transmitting raw patient data to the cloud.

The system addresses a real clinical engineering problem: streaming continuous 256Hz EEG data to the cloud drains wearable batteries, incurs significant network egress costs, and violates data privacy laws (HIPAA/GDPR). Standard federated learning systems miss this because they run continuously on fixed schedules, failing to respond to rare acute events like seizures.

This project solves it through a cross-layer design that integrates Cloud Network Engineering (CNE), High-Performance Computing (HPC), and Edge Signal Processing (ECE).

---

## Architecture

```
┌─────────────────────┐    bidi gRPC stream    ┌─────────────────────────────────────┐
│   Hospital Silo A   │ ──────────────────────► │         HPC Aggregator (AKS)        │
│  (hospital-a ns)    │                         │                                     │
│                     │   ┌─────────────────┐   │  ┌─────────────┐  ┌─────────────┐  │
│  FFT Edge Filter    │   │ K8s NetworkPolicy│   │  │ ONNX Infer  │  │ Opacus DP   │  │
│  Importance Sampler │   │ (namespace iso.) │   │  │ (fast path) │  │ (training)  │  │
│  Force-burst (8%)   │   └─────────────────┘   │  └─────────────┘  └─────────────┘  │
└─────────────────────┘                         └────────────┬────────────────────────┘
                                                             │
┌─────────────────────┐    bidi gRPC stream    ┌────────────┴────────────────────────┐
│   Hospital Silo B   │ ──────────────────────► │     KEDA Autoscaler (1→4 pods)      │
│  (hospital-b ns)    │                         │     triggered on CPU burst          │
└─────────────────────┘                         └────────────┬────────────────────────┘
                                                             │
                              ┌──────────────────────────────┼──────────────────────┐
                              │                              │                      │
                    ┌─────────▼─────────┐        ┌──────────▼──────────┐  ┌────────▼───────┐
                    │  Azure Digital    │        │  Global Aggregator  │  │   MLflow v2.8  │
                    │  Twins (ADT)      │        │  FedAvg-Weighted    │  │   Experiment   │
                    │  Silo-Alpha       │        │  PVC weight buffer  │  │   Tracking     │
                    │  Silo-Beta        │        └─────────────────────┘  └────────────────┘
                    └───────────────────┘
                                                             │
                                                   ┌─────────▼──────────┐
                                                   │  React Dashboard   │
                                                   │  http://20.198.144 │
                                                   │  .247              │
                                                   └────────────────────┘
```

---

## Key Features

### CNE Layer — Network Engineering
- **Bidirectional gRPC streaming** over HTTP/2: replaces per-packet TCP handshakes with a single persistent stream per silo, reducing connection overhead from O(n) to O(1) during seizure bursts
- **Kubernetes NetworkPolicy** enforcing strict namespace isolation between hospital silos
- **ADT debounce** (5s cooldown) preventing API rate-limit exhaustion during burst events
- **Auto-reconnect** loop in silo clients with exponential backoff

### ECE Layer — Edge Signal Processing
- **FFT spectral seizure detection** using theta (4–8 Hz) and alpha (8–13 Hz) band energy analysis on a 256-sample rolling window at 100 Hz
- **Hann windowing** to reduce spectral leakage at buffer boundaries
- **Importance sampling / pruning** achieving ~87% bandwidth reduction by only transmitting packets with significant signal change or seizure risk
- **Spectral ratio metric** (theta+alpha energy / total energy) logged per packet, threshold 0.40

### HPC Layer — Compute
- **KEDA event-driven autoscaling**: hpc-aggregator scales from 1 to 4 pods on CPU utilisation spike during seizure bursts, then cools down after 15 seconds
- **ONNX Runtime inference**: 1D-CNN exported from PyTorch runs fast per-packet inference on the aggregator
- **Real PyTorch backpropagation** on burst packets using FFT-verified seizure events as pseudo-labels (weak supervision)

### MLOps Layer
- **Weighted FedAvg** (McMahan et al. 2017): global model aggregated from real `state_dict` tensors, weighted by number of training samples per silo
- **MLflow experiment tracking**: discrimination score, seizure/normal confidence, per-silo epsilon, and total training samples logged every 30 seconds
- **Global model checkpoint** saved to Azure File PVC and loaded by aggregator pods on startup, enabling true multi-round FL convergence
- **Real convergence metric**: discrimination score computed from actual model forward pass rather than a simulated exponential curve

### Security Layer
- **Formal (ε=0.5, δ=1e-5)-DP**: Opacus `PrivacyEngine` instruments PyTorch's backward pass with per-sample gradient clipping (L2 norm ≤ 1.0) and calibrated Gaussian noise
- **Rényi DP accountant**: epsilon spent tracked per silo per round and logged to MLflow as an auditable privacy trail
- **Multi-tenant isolation**: Kubernetes NetworkPolicy restricts hospital-a and hospital-b to egress-only access to the aggregator namespace

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud | Azure Kubernetes Service (AKS), Azure Container Registry (ACR) |
| Digital Twin | Azure Digital Twins (ADT), DTDL patient model |
| Storage | Azure File Storage (CSI driver, ReadWriteMany PVC) |
| Orchestration | Kubernetes, KEDA v2, Ansible, GNU Make |
| Networking | gRPC (bidirectional streaming), Protobuf, K8s NetworkPolicy |
| Inference | ONNX Runtime, PyTorch 1D-CNN |
| Privacy | Opacus v0.16.3, Rényi DP accountant |
| MLOps | MLflow v2.8.0 |
| Dashboard | React 18, Recharts, FastAPI, nginx |
| Data | CHB-MIT Scalp EEG Database (PhysioNet), resampled to 100 Hz |

---

## Project Structure

```
BioSync-HPC/
├── adt/
│   └── patient_model.json          # DTDL interface (HeartRate, IsCritical)
├── ansible/                        # Infrastructure provisioning playbooks
├── dashboard/
│   ├── src/Dashboard.jsx           # React SPA — 5 live panels
│   ├── nginx.conf                  # SPA routing + API proxy
│   ├── Dockerfile                  # Multi-stage Node build + nginx serve
│   └── package.json
├── dashboard-api/
│   ├── server.py                   # FastAPI bridge (ADT, K8s, MLflow, DP)
│   ├── requirements.txt
│   └── Dockerfile
├── hpc-server/
│   ├── aggregator.py               # gRPC server, ONNX inference, Opacus DP
│   ├── fed_avg.py                  # Weighted FedAvg + MLflow logging
│   ├── generate_model.py           # PyTorch → ONNX export
│   └── Dockerfile
├── k8s/
│   ├── infra.yaml                  # PV, PVC, MLflow, NetworkPolicy, deployments
│   └── hpc-scaler.yaml             # KEDA ScaledObject (CPU trigger)
├── k8s-dashboard-additions.yaml    # RBAC, dashboard-api, dashboard service
├── protos/
│   └── biosignal.proto             # BioNetService.StreamSignal (bidi)
├── silo-client/
│   ├── client.py                   # FFT filter, importance sampler, gRPC client
│   ├── eeg_data.npy                # CHB-MIT data (100 Hz, normalised 0–1)
│   └── Dockerfile
├── biosignal_pb2.py                # Generated protobuf stubs (root, for Docker)
├── biosignal_pb2_grpc.py           # Generated gRPC stubs (root, for Docker)
├── bio_logic.onnx                  # Exported 1D-CNN model
├── Makefile                        # Build, deploy, pause, resume targets
├── requirements.txt                # Pinned Python dependencies
├── RESTART_GUIDE.md                # Step-by-step session restart guide
└── IMPROVEMENTS_GUIDE.md           # Planned next upgrades with code
```

---

## Quick Start

### Prerequisites
- Azure CLI (`az`) with an active Azure for Students subscription
- Docker Desktop
- `kubectl`, `helm`
- Python 3.9+ with virtualenv

### First-Time Setup
Follow `REPROVISION.md` to create all Azure resources from scratch. This covers AKS cluster, ACR, ADT, storage account, KEDA, and K8s secrets.

### Every Session

```bash
# Start
make resume
source venv/bin/activate

# Re-export variables (lost on terminal close)


# Launch silos
kubectl run test-client -n hospital-a \
  --image=biosyncregistry1772554412.azurecr.io/silo-client:v10 \
  --env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
  --env="HOSPITAL_ID=Silo-Alpha"

kubectl run test-client -n hospital-b \
  --image=biosyncregistry1772554412.azurecr.io/silo-client:v10 \
  --env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
  --env="HOSPITAL_ID=Silo-Beta"

# Stop
make pause
```

### Image Version Bump (always follow this exact order)

```bash
docker build --no-cache --platform linux/amd64 \
  -t biosyncregistry1772554412.azurecr.io/hpc-aggregator:vNEW \
  -f hpc-server/Dockerfile .

docker push biosyncregistry1772554412.azurecr.io/hpc-aggregator:vNEW

sed -i '' 's/hpc-aggregator:vOLD/hpc-aggregator:vNEW/g' k8s/infra.yaml

kubectl apply -f k8s/infra.yaml          # Critical — never skip

kubectl rollout restart deployment/hpc-aggregator -n aggregator
```

---

## Observability

| What | Where |
|---|---|
| Live dashboard | `http://20.198.144.247` |
| MLflow experiments | `kubectl port-forward svc/mlflow-service -n aggregator 5001:5000` → `http://localhost:5001` |
| ADT Explorer | `https://explorer.digitaltwins.azure.net` → connect to `BioSyncTwin.api.sea.digitaltwins.azure.net` |
| Aggregator logs | `kubectl logs -n aggregator -l app=hpc-aggregator -f` |
| FedAvg logs | `kubectl logs -n aggregator -l app=global-aggregator -f` |
| KEDA status | `kubectl get scaledobject -n aggregator` |

---

## Verified Metrics (v2-stable)

| Metric | Value | How measured |
|---|---|---|
| Bandwidth reduction | ~87% | FFT pruning: packets saved / total attempts |
| gRPC TCP handshakes per burst | 1 (was 500) | Bidi stream vs. unary per-packet |
| KEDA scaling | 1 → 4 pods | Observed during seizure burst |
| FL rounds per session | 20+ in ~10 mins | MLflow run count |
| Privacy budget per round | ε=0.5, δ=1e-5 | Opacus Rényi DP accountant |
| Discrimination score trend | Monotonically increasing | MLflow metrics over 20 rounds |
| ADT update latency | <5s (debounced) | ADT Explorer live observation |

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)
- Patient chb01, recording chb01_03.edf
- Resampled to 100 Hz, normalised 0–1, stored as `silo-client/eeg_data.npy`
- 359,200 samples (~3,592 seconds of continuous EEG)
- Prepared using `prepare_data.py`

---

## Known Limitations

1. **Discrimination score is low (~0.03)**: The CNN receives a single scalar value, not a proper EEG time-series window. This is a proof-of-concept infrastructure project — the contribution is the distributed systems architecture, not clinical model performance.
2. **Opacus threading**: The threading.Lock workaround in v2-stable adds minor latency on burst packets. See `IMPROVEMENTS_GUIDE.md` for the clean manual DP replacement.
3. **Force-burst mechanism**: Real CHB-MIT segments at the current file position have low spectral energy. The 8% synthetic burst injection ensures KEDA and DP training are exercised during demos.

---

## Constraints

- **Platform**: M-series Mac (ARM64) — all Docker builds use `--platform linux/amd64`
- **Cloud**: Azure for Students ($100 credit) — always run `make pause` after each session
- **VM**: `standard_b2s_v2` in Southeast Asia (1 node, 4 vCPU, 8GB RAM)
- **Registry**: `biosyncregistry1772554412.azurecr.io` (Basic tier)

---

## Git Tags

| Tag | Description |
|---|---|
| `v1-stable` | Original V1: unary gRPC, amplitude threshold, simulated FedAvg |
| `v2-beta` | All upgrades deployed, three known issues |
| `v2-stable` | All issues fixed: threading, checkpoint key, KEDA metricType, OOM |
