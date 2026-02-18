
# Network-Aware Federated Learning for BioIoT Devices

### Bio-Sync HPC: A Cross-Layer Self-Optimizing Infrastructure

This project implements an event-driven, privacy-preserving infrastructure designed to handle bursty bio-medical signals (EEG) using Federated Learning. It bridges the gap between **Edge Signal Processing** and **Cloud High-Performance Computing (HPC)**.

The system features a **scale-to-zero** architecture using KEDA, high-performance **gRPC** networking, and a live state mirror using **Azure Digital Twins**.

---

## System Architecture

### Key Capabilities
1.  **Network-Aware Edge:** Implements signal pruning (Importance Sampling) at the edge to reduce network egress by ~30%.
2.  **Event-Driven HPC:** Uses **KEDA** to scale inference pods from 0 to $N$ based on real-time signal burst intensity.
3.  **Multi-Tenant Isolation:** Enforces strict namespace isolation between Hospital Silos using Kubernetes Network Policies.
4.  **Hybrid Cloud Mirroring:** Synchronizes local seizure events to **Azure Digital Twins** in real-time for remote observability.

---

## Tech Stack

*   **Orchestration:** Kubernetes (Minikube), KEDA, Ansible
*   **Networking:** gRPC (Protobuf), MQTT
*   **Compute/AI:** Python, ONNX Runtime, PyTorch (1D-CNN)
*   **Cloud:** Azure Digital Twins, Azure Identity
*   **Automation:** GNU Make, Docker

---

## Quick Start Guide

Follow these steps to spin up the entire distributed infrastructure locally.

### 1. Infrastructure Initialization
Start the local Kubernetes cluster with sufficient resources for HPC simulation.

```bash
# Start Minikube (Requires Docker)
minikube start --cpus 2 --memory 4096 --driver=docker

# CRITICAL: Point your shell to Minikube's internal Docker registry
# (You must run this command in every new terminal window)
eval $(minikube docker-env)

# Enable Metrics Server (Required for CPU-based scaling)
minikube addons disable metrics-server
minikube addons enable metrics-server
```

### 2. Environment Setup
Activate the Python environment and authenticate with the Cloud Control Plane.

```bash
# Navigate to project root and activate venv
source venv/bin/activate

# Login to Azure (Required for the Aggregator to sync with Digital Twins)
az login
```

### 3. Build & Deploy (Infrastructure as Code)
We use a `Makefile` to handle the container builds and Kubernetes manifest applications.

```bash
# 1. Build Docker images (Aggregator & Client) directly inside Minikube
make build

# 2. Provision Namespaces, Network Policies, KEDA Scalers, and Secrets
make deploy
```

### 4. Start the Simulation
Launch the multi-tenant simulation. This spins up two isolated "Silos" (Hospital A and Hospital B) that generate synthetic EEG data.

```bash
# Launches test-client pods in hospital-a and hospital-b namespaces
make run-hospitals
```

---

## Verification & Observability

### 1. Monitor Event-Driven Scaling (HPC Layer)
Check if KEDA is correctly monitoring the workload.
```bash
kubectl get scaledobject -n aggregator
```
*   **Expected Output:** `READY: True`, `ACTIVE: True` (when data is flowing).

### 2. Watch Real-Time Cloud Sync (Data Plane)
View the logs of the Aggregator to see gRPC signals arriving and Azure updates being sent.
```bash
kubectl logs -n aggregator -l app=hpc-aggregator -f --max-log-requests=10
```
*   **Look for:** `☁️ Azure Sync Success: Twin 'Silo-Alpha' updated.`

### 3. Azure Digital Twin Explorer
To visualize the patient state:
1.  Go to [Azure Digital Twins Explorer](https://explorer.digitaltwins.azure.net/).
2.  Connect to your instance URL.
3.  Run Query: `SELECT * FROM DIGITALTWINS`.
4.  Observe `HeartRate` and `IsCritical` updating live as the local simulation runs.

---

## 📂 Project Structure

```text
├── adt/               # Azure Digital Twin Models (DTDL)
├── ansible/           # Ansible Playbooks for Infra Provisioning
├── hpc-server/        # Aggregator Logic (gRPC Server + ONNX + Azure Bridge)
├── silo-client/       # Edge Logic (Signal Generator + Pruning + gRPC Client)
├── k8s/               # Kubernetes Manifests (NetPol, Services, Deployments)
├── protos/            # Protocol Buffer definitions
└── Makefile           # Automation entry point
```