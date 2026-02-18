import grpc
import time
import random
import os
import biosignal_pb2
import biosignal_pb2_grpc

# Environment Configuration
AGGREGATOR_ADDR = os.getenv("AGGREGATOR_ADDR", "localhost:50051")
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "Hospital-Alpha")

# Optimization Parameters (ECE Focus)
PRUNING_THRESHOLD = 2.0  
KEEP_ALIVE_INTERVAL = 10  

def run():
    print(f"Connecting to Aggregator at {AGGREGATOR_ADDR}")
    print(f"Mode: Adaptive Pruning + HPC Stress Testing active...")
    
    with grpc.insecure_channel(AGGREGATOR_ADDR) as channel:
        stub = biosignal_pb2_grpc.BioNetServiceStub(channel)
        
        last_sent_val = 0
        last_sent_time = 0
        packets_saved = 0
        total_attempts = 0

        while True:
            total_attempts += 1
            # 1. Generate Signal (ECE Logic)
            val = 60 + random.uniform(-5, 5)
            is_burst = random.random() > 0.94 # ~6% chance of a seizure burst
            if is_burst: val += 40
            
            # 2. Pruning Logic (Network Awareness)
            current_time = time.time()
            delta = abs(val - last_sent_val)
            time_since_last = current_time - last_sent_time
            
            should_send = is_burst or (delta > PRUNING_THRESHOLD) or (time_since_last > KEEP_ALIVE_INTERVAL)

            if should_send:
                # CNE/HPC Skill: Burst Simulation
                # If it's a medical burst, we flood the network to test HPC scaling
                iterations = 1000 if is_burst else 1
                if is_burst:
                    print(f"!!! SEIZURE DETECTED - INITIATING HPC STRESS TEST (50 packets) !!!")

                for i in range(iterations):
                    request = biosignal_pb2.SignalRequest(
                        hospital_id=HOSPITAL_ID,
                        value=val + (random.uniform(-1,1) if is_burst else 0),
                        is_burst=is_burst,
                        timestamp=time.time()
                    )
                    try:
                        # High-speed gRPC call
                        stub.SendSignal(request)
                    except grpc.RpcError as e:
                        print(f"Network Congestion/Error: {e.code()}")
                        break # Stop burst if network is failing
                
                # Update tracking after the send/burst
                last_sent_val = val
                last_sent_time = time.time()
                if not is_burst:
                    print(f"[SEND] Val: {val:.2f} | Delta: {delta:.2f}")
            else:
                packets_saved += 1
                if packets_saved % 10 == 0:
                    efficiency = (packets_saved / total_attempts) * 100
                    print(f"[PRUNE] Bandwidth Saved. Efficiency: {efficiency:.1f}%")
            
            time.sleep(1)

if __name__ == '__main__':
    run()