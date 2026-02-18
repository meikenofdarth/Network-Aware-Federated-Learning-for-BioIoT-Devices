import os
import json
import numpy as np
import time

# Directory where aggregator pods save their local updates
WEIGHTS_DIR = "/app/weights" 

def aggregate_weights():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 Initializing Global Aggregation Round...")
    
    # 1. Gather all local update files
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.json')]
    
    # CNE/HPC Logic: We need at least one update from each Silo to ensure a fair global model
    if len(files) < 2:
        print(f"ℹ️  Waiting for more updates... (Found {len(files)} updates, need at least 2).")
        return

    all_local_weights = []
    hospitals_involved = set()

    for file in files:
        file_path = os.path.join(WEIGHTS_DIR, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_local_weights.append(data['weights'])
                hospitals_involved.add(data['hospital'])
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    # 2. Perform Federated Averaging (FedAvg)
    # Using NumPy to calculate the mean across the columns of our weight matrix
    global_weights = np.mean(all_local_weights, axis=0)

    print(f"✅ SUCCESS: Aggregated {len(files)} updates from {hospitals_involved}")
    print(f"📈 Global Model State (Weights Vector): {global_weights.tolist()}")
    
    # 3. Model Versioning (CNE/HPC Skill)
    # In a production system, we would now overwrite the 'bio_logic.onnx' 
    # or save a new version. For this demo, we simulate the update.
    print(f"✨ Global Model Registry updated to Version: {int(time.time())}")

    # 4. Cleanup: Remove processed files to prepare for the next round
    for file in files:
        os.remove(os.path.join(WEIGHTS_DIR, file))
    print("🧹 Shared buffer cleared for next round.")

if __name__ == "__main__":
    while True:
        aggregate_weights()
        time.sleep(30) # Run aggregation window every 30 seconds