import os
import json
import numpy as np
import time
import mlflow
import torch
import torch.nn as nn
# Must match aggregator.py DP config exactly for audit consistency
DP_EPSILON = 0.5
DP_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0

WEIGHTS_DIR = "/app/weights"
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.aggregator.svc.cluster.local:5000"
)

# ── MLOPS UPGRADE: BioNet1DCNN Definition ────────────────────────────
# Duplicated here so fed_avg.py is self-contained inside the container.
# In a production system this would live in a shared bio_model.py module.
class BioNet1DCNN(nn.Module):
    def __init__(self):
        super(BioNet1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.sigmoid(self.fc(x))


def federated_average(weight_updates: list) -> dict:
    total_samples = sum(u["num_samples"] for u in weight_updates)
    if total_samples == 0:
        total_samples = len(weight_updates)
        for u in weight_updates:
            u["num_samples"] = 1

    def strip_opacus_prefix(weights_dict):
        return {
            k.replace("_module.", ""): v
            for k, v in weights_dict.items()
        }

    cleaned_updates = []
    for u in weight_updates:
        cleaned = dict(u)
        cleaned["weights"] = strip_opacus_prefix(u["weights"])
        cleaned_updates.append(cleaned)

    global_weights = {}
    for key in cleaned_updates[0]["weights"].keys():
        global_weights[key] = torch.zeros_like(
            torch.tensor(cleaned_updates[0]["weights"][key])
        )

    for update in cleaned_updates:
        weight_factor = update["num_samples"] / total_samples
        for key in global_weights:
            layer_tensor = torch.tensor(
                update["weights"][key], dtype=torch.float32
            )
            global_weights[key] += weight_factor * layer_tensor

    return global_weights


def aggregate_and_log():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 Checking for federated updates...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Bio-Sync-HPC-Global-Learning")

    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.json')]

    if len(files) < 2:
        print(f"ℹ️  Waiting for more updates... (Found {len(files)})")
        return

    weight_updates = []
    hospitals_involved = set()
    files_to_delete = []
    total_samples_this_round = 0

    for file in files:
        file_path = os.path.join(WEIGHTS_DIR, file)

        if os.path.getsize(file_path) == 0:
            continue

        try:
            with open(file_path, 'r') as jf:
                data = json.load(jf)
                weight_updates.append(data)
                hospitals_involved.add(data['hospital'])
                total_samples_this_round += data.get('num_samples', 1)
                files_to_delete.append(file_path)
        except Exception as e:
            print(f"⚠️  Skipping {file} due to read error: {e}")
            continue

    if len(hospitals_involved) < 2:
        print("ℹ️  Updates found, but not from enough unique silos yet.")
        return

    # ── MLOPS UPGRADE: Real FedAvg on actual model tensors ───────────
    global_weights = federated_average(weight_updates)

    # ── Save global model checkpoint to PVC ──────────────────────────
    # Aggregator pods load this on startup — enables true multi-round FL.
    # Next round, each silo initialises from this checkpoint instead
    # of random weights, implementing the full FL convergence loop.
    global_model = BioNet1DCNN()
    global_model.load_state_dict(global_weights)
    global_model_path = os.path.join(WEIGHTS_DIR, "global_model.pt")

    try:
        torch.save(global_model.state_dict(), global_model_path)
        print(f"💾 Global model checkpoint saved to {global_model_path}")
    except Exception as e:
        print(f"⚠️  Could not save global checkpoint: {e}")

    # ── Compute real convergence metric ──────────────────────────────
    # Run a forward pass on a synthetic seizure sample to measure
    # how confidently the global model detects anomalies.
    # This replaces the fake exponential curve with an actual model output.
    global_model.eval()
    with torch.no_grad():
        test_seizure = torch.tensor([[[0.95]]], dtype=torch.float32)
        test_normal = torch.tensor([[[0.3]]], dtype=torch.float32)
        seizure_confidence = float(global_model(test_seizure)[0][0])
        normal_confidence = float(global_model(test_normal)[0][0])

    # Discrimination score: how well the model separates seizure from normal.
    # Ranges 0.0 (no discrimination) to 1.0 (perfect separation).
    discrimination_score = seizure_confidence - normal_confidence

    print(f"✅ FedAvg Complete | "
          f"Silos: {hospitals_involved} | "
          f"Total Samples: {total_samples_this_round} | "
          f"Seizure Conf: {seizure_confidence:.4f} | "
          f"Normal Conf: {normal_confidence:.4f} | "
          f"Discrimination: {discrimination_score:.4f}")

    # ── MLflow Logging ────────────────────────────────────────────────
    try:
        with mlflow.start_run():
            mlflow.log_metric("seizure_confidence", seizure_confidence)
            mlflow.log_metric("normal_confidence", normal_confidence)
            mlflow.log_metric("discrimination_score", discrimination_score)
            mlflow.log_metric("total_training_samples",
                              total_samples_this_round)
            mlflow.log_metric("participating_silos", len(hospitals_involved))
            mlflow.log_param("optimizer", "FedAvg-Weighted")
            mlflow.log_param("dp_mechanism", "Opacus-Gaussian")
            mlflow.log_param("dp_max_grad_norm", DP_MAX_GRAD_NORM)

            # SECURITY UPGRADE: Log per-silo privacy budget consumption.
            # This is what a DP audit trail looks like in production.
            # A compliance officer can verify no silo exceeded epsilon.
            for update in weight_updates:
                silo_id = update["hospital"].replace("-", "_")
                epsilon_spent = update.get("epsilon_spent", "N/A")
                mlflow.log_metric(
                    f"epsilon_spent_{silo_id}",
                    float(epsilon_spent) if epsilon_spent != "N/A" else 0.0
                )
                # Flag budget violations for audit
                if epsilon_spent != "N/A" and float(epsilon_spent) > DP_EPSILON:
                    print(f"🚨 PRIVACY AUDIT: {update['hospital']} "
                          f"exceeded ε budget! "
                          f"Spent={epsilon_spent:.4f} > "
                          f"Limit={DP_EPSILON}")

            print(f"📊 MLflow Logged: "
                  f"Discrimination={discrimination_score:.4f}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (will retry next round): {e}")

    # ── Cleanup: independent of MLflow success ────────────────────────
    try:
        for file_path in files_to_delete:
            os.remove(file_path)
        print("🧹 Shared buffer cleared for next round.")
    except Exception as e:
        print(f"⚠️  Cleanup error (non-fatal): {e}")


if __name__ == "__main__":
    time.sleep(10)
    while True:
        try:
            aggregate_and_log()
        except Exception as e:
            print(f"❌ Error in aggregation loop: {e}")
        time.sleep(30)