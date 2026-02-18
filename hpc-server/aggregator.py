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
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.aggregator.svc.cluster.local:5000")

class BioNetAggregator(biosignal_pb2_grpc.BioNetServiceServicer):
    def __init__(self):
        self.session = ort.InferenceSession("bio_logic.onnx")
        print(f"HPC Engine: 1D-CNN Model Loaded")
        
        # Azure Bridge
        self.adt_client = None
        if ADT_URL:
            try:
                self.adt_client = DigitalTwinsClient(ADT_URL, DefaultAzureCredential())
                print(f"✅ Azure Mirror Active")
            except Exception as e: print(f"⚠️ Azure Init Warning: {e}")

    def SendSignal(self, request, context):
        # ECE Fix: 1D-CNN expects 3D input [Batch, Channel, Length]
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
        # --- POINT 5: DIFFERENTIAL PRIVACY (Laplacian Noise) ---
        epsilon = 0.5  # Privacy budget
        sensitivity = 0.1
        noise = np.random.laplace(0, sensitivity/epsilon, 5)
        
        raw_weights = np.random.rand(5) # Simulating weight extraction
        private_weights = (raw_weights + noise).tolist()

        weight_data = {
            "hospital": hospital_id,
            "weights": private_weights,
            "timestamp": time.time()
        }
        
        filename = f"/app/weights/update_{hospital_id}_{int(time.time()*1000)}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(weight_data, f)
            print(f"📦 Privacy-Preserved updates stored for {hospital_id}")
        except Exception as e: print(f"❌ Weight Error: {e}")

    def sync_to_azure(self, twin_id, val, critical):
        patch = [
            {"op": "add", "path": "/HeartRate", "value": float(val)},
            {"op": "add", "path": "/IsCritical", "value": bool(critical)}
        ]
        try:
            self.adt_client.update_digital_twin(twin_id, patch)
            print(f"☁️  Azure Mirror Updated")
        except Exception as e: print(f"❌ Azure Sync Error: {e}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    biosignal_pb2_grpc.add_BioNetServiceServicer_to_server(BioNetAggregator(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()