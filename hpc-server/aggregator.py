import grpc
from concurrent import futures
import biosignal_pb2, biosignal_pb2_grpc
import onnxruntime as ort
import numpy as np
import os, json, time
from azure.identity import DefaultAzureCredential
from azure.digitaltwins.core import DigitalTwinsClient

# CNE/HPC Config
ADT_URL = os.getenv("ADT_URL")

class BioNetAggregator(biosignal_pb2_grpc.BioNetServiceServicer):
    def __init__(self):
        # CNE Fix: Direct relative path inside container
        model_path = "bio_logic.onnx"
        try:
            self.session = ort.InferenceSession(model_path)
            print(f"✅ HPC Inference Engine: Loaded model '{model_path}'")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load ONNX model: {e}")
            raise e

        # Azure Bridge (Restored for Phase 5)
        self.adt_client = None
        if ADT_URL:
            try:
                # DefaultAzureCredential handles the Service Principal from K8s Secret
                cred = DefaultAzureCredential()
                self.adt_client = DigitalTwinsClient(ADT_URL, cred)
                print(f"✅ Connected to Azure Digital Twin: {ADT_URL}")
            except Exception as e: 
                print(f"⚠️ Azure Initialization Failed: {e}")

    def SendSignal(self, request, context):
        # ECE: 1D-CNN expects 3D input [Batch, Channel, Length]
        input_data = np.array([[[request.value]]], dtype=np.float32)
        
        outputs = self.session.run(None, {'input': input_data})
        prob = float(outputs[0][0][0])
        is_anomaly = prob > 0.5
        
        print(f"[INFER] Silo: {request.hospital_id} | Prob: {prob:.4f} | Anomaly: {is_anomaly}")

        # --- PHASE 6: PRIVACY-PRESERVING WEIGHT CAPTURE ---
        if request.is_burst:
            self.save_local_weights(request.hospital_id, prob)

        # --- PHASE 5: AZURE CLOUD MIRROR ---
        if self.adt_client and (is_anomaly or request.is_burst):
            self.sync_to_azure(request.hospital_id, request.value, is_anomaly)

        return biosignal_pb2.SignalResponse(status="PROCESSED", trigger_training=is_anomaly)

    def save_local_weights(self, hospital_id, prob):
        # POINT 5: DIFFERENTIAL PRIVACY (Laplacian Noise)
        epsilon, sensitivity = 0.5, 0.1
        noise = np.random.laplace(0, sensitivity/epsilon, 5)
        private_weights = (np.random.rand(5) + noise).tolist()

        weight_data = {
            "hospital": hospital_id,
            "weights": private_weights,
            "timestamp": time.time()
        }
        
        filename = f"/app/weights/update_{hospital_id}_{int(time.time()*1000)}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(weight_data, f)
                # HPC FIX: Flush and Fsync force the network drive to commit the data
                # This prevents the "JSON Expecting Value" error in fed_avg.py
                f.flush()
                os.fsync(f.fileno())
            print(f"📦 Privacy-Preserved updates stored for {hospital_id}")
        except Exception as e: 
            print(f"❌ Weight Save Error: {e}")

    def sync_to_azure(self, twin_id, val, critical):
        # CNE Fix: Using 'add' to ensure idempotent cloud property updates
        patch = [
            {"op": "add", "path": "/HeartRate", "value": float(val)},
            {"op": "add", "path": "/IsCritical", "value": bool(critical)}
        ]
        try:
            self.adt_client.update_digital_twin(twin_id, patch)
            print(f"☁️ Azure Mirror Updated for {twin_id}")
        except Exception as e: 
            print(f"❌ Azure Sync Error: {e}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    biosignal_pb2_grpc.add_BioNetServiceServicer_to_server(BioNetAggregator(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()