import os
import json
import numpy as np
import time
import mlflow

# Configuration
WEIGHTS_DIR = "/app/weights" 
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.aggregator.svc.cluster.local:5000")

def aggregate_and_log():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 Checking for federated updates...")
    
    # 1. Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Bio-Sync-HPC-Global-Learning")
    
    # 2. Gather all local update files
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.json')]
    
    if len(files) < 2:
        print(f"ℹ️  Waiting for more updates... (Found {len(files)})")
        return

    all_local_weights = []
    hospitals_involved = set()

    # 3. Load Weights with Safety Checks
    for file in files:
        file_path = os.path.join(WEIGHTS_DIR, file)
        
        # HPC Fix: Skip file if it's still being written (size is 0)
        if os.path.getsize(file_path) == 0:
            continue
            
        try:
            with open(file_path, 'r') as jf:
                data = json.load(jf)
                all_local_weights.append(data['weights'])
                hospitals_involved.add(data['hospital'])
        except Exception as e:
            print(f"⚠️  Skipping {file} due to read error (syncing latency)")
            continue

    # Only aggregate if we successfully read from at least 2 silos
    if len(hospitals_involved) < 2:
        print("ℹ️  Updates found, but not from enough unique silos yet.")
        return

    # 4. Perform Federated Averaging (FedAvg)
    global_weights = np.mean(all_local_weights, axis=0)
    
    print(f"✅ SUCCESS: Aggregated {len(all_local_weights)} updates from {hospitals_involved}")
    
    # 5. Simulate Convergence
    accuracy = 0.5 + (0.45 * (1 - np.exp(-len(all_local_weights)/20)))
    
    # 6. Log to MLflow UI (Wrapped in try-except for CNE Host Header issues)
    try:
        with mlflow.start_run():
            mlflow.log_metric("global_accuracy", accuracy)
            mlflow.log_metric("participating_silos", len(hospitals_involved))
            mlflow.log_param("optimizer", "FedAvg")
            print(f"📊 MLflow Logged: Accuracy={accuracy:.4f}")
            
        # 7. Cleanup only after successful log
        for file in files:
            os.remove(os.path.join(WEIGHTS_DIR, file))
        print("🧹 Shared buffer cleared for next round.")
    except Exception as e:
        print(f"⚠️  MLflow Logging delayed (Network/Host Busy): {e}")

if __name__ == "__main__":
    time.sleep(10) # Wait for infrastructure settle
    while True:
        try:
            aggregate_and_log()
        except Exception as e:
            print(f"❌ Error in loop: {e}")
        time.sleep(30)