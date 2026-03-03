import grpc
import time
import random
import os
import numpy as np # Required for real data
import biosignal_pb2
import biosignal_pb2_grpc

# Environment Configuration
AGGREGATOR_ADDR = os.getenv("AGGREGATOR_ADDR", "localhost:50051")
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "Hospital-Alpha")

# Optimization Parameters
PRUNING_THRESHOLD = 0.05  # Lower threshold because normalized data is 0.0-1.0
KEEP_ALIVE_INTERVAL = 10  

def run():
    print(f"Connecting to Aggregator at {AGGREGATOR_ADDR}")
    
    # --- REAL DATA LOADING ---
    try:
        # Load the pre-processed CHB-MIT data
        eeg_trace = np.load("eeg_data.npy")
        print(f"✅ Loaded {len(eeg_trace)} real EEG samples from CHB-MIT.")
    except Exception as e:
        print(f"⚠️ Real Data Error: {e}. Falling back to synthetic.")
        eeg_trace = None

    with grpc.insecure_channel(AGGREGATOR_ADDR) as channel:
        stub = biosignal_pb2_grpc.BioNetServiceStub(channel)
        
        last_sent_val = 0
        last_sent_time = 0
        packets_saved = 0
        total_attempts = 0
        idx = 0

        while True:
            total_attempts += 1
            
            # --- 1. SIGNAL GENERATION (Real vs Synthetic) ---
            if eeg_trace is not None:
                # Loop through the real file
                val = float(eeg_trace[idx % len(eeg_trace)])
                # In CHB-MIT, seizures are high amplitude. 
                # We simulate a "Detected Event" if signal > 0.8 (Normalized)
                is_burst = val > 0.8 
                idx += 1
            else:
                # Fallback
                val = 60 + random.uniform(-5, 5)
                is_burst = random.random() > 0.94
                if is_burst: val += 40

            # --- 2. PRUNING LOGIC ---
            current_time = time.time()
            delta = abs(val - last_sent_val)
            time_since_last = current_time - last_sent_time
            
            should_send = is_burst or (delta > PRUNING_THRESHOLD) or (time_since_last > KEEP_ALIVE_INTERVAL)

            if should_send:
                # --- 3. HPC STRESS TEST (KEDA Trigger) ---
                # Real seizures last seconds. We simulate the network load of that.
                iterations = 1000 if is_burst else 1
                
                if is_burst:
                    print(f"!!! SEIZURE DETECTED (Val: {val:.2f}) - HPC BURST STARTED !!!")

                for i in range(iterations):
                    # Add tiny jitter to burst values to mimic sensor noise
                    burst_val = val + random.uniform(-0.01, 0.01)
                    request = biosignal_pb2.SignalRequest(
                        hospital_id=HOSPITAL_ID,
                        value=burst_val,
                        is_burst=is_burst,
                        timestamp=time.time()
                    )
                    try:
                        stub.SendSignal(request)
                    except grpc.RpcError:
                        # If network is congested, we drop packets (Standard UDP/IoT behavior)
                        break 
                
                last_sent_val = val
                last_sent_time = time.time()
                if not is_burst:
                    print(f"[SEND] Val: {val:.4f} | Delta: {delta:.4f}")
            else:
                packets_saved += 1
                if packets_saved % 20 == 0:
                    efficiency = (packets_saved / total_attempts) * 100
                    print(f"[PRUNE] Bandwidth Efficiency: {efficiency:.1f}%")
            
            # 10Hz sampling rate (Adjust this to 0.5 or 1.0 to save Laptop CPU)
            time.sleep(0.1) 

if __name__ == '__main__':
    run()