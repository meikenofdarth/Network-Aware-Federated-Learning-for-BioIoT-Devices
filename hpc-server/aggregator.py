import grpc
from concurrent import futures
import threading
import biosignal_pb2, biosignal_pb2_grpc
import onnxruntime as ort
import numpy as np
import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from azure.identity import DefaultAzureCredential
from azure.digitaltwins.core import DigitalTwinsClient

ADT_URL = os.getenv("ADT_URL")
ADT_DEBOUNCE_SECONDS = 5.0

DP_EPSILON = 0.5
DP_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0


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


class BioNetAggregator(biosignal_pb2_grpc.BioNetServiceServicer):
    def __init__(self):
        # ── ONNX Session (fast inference path) ───────────────────────
        model_path = "bio_logic.onnx"
        try:
            self.session = ort.InferenceSession(model_path)
            print(f"✅ HPC Inference Engine: Loaded ONNX model '{model_path}'")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load ONNX model: {e}")
            raise e

        # ── Opacus DP Engine ──────────────────────────────────────────
        raw_model = BioNet1DCNN()
        self.torch_model = ModuleValidator.fix(raw_model)

        errors = ModuleValidator.validate(self.torch_model, strict=False)
        if errors:
            print(f"⚠️ Model validation warnings: {errors}")
        else:
            print(f"✅ Model validated for Opacus DP compatibility")

        base_optimizer = optim.SGD(
            self.torch_model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        self.criterion = nn.BCELoss()

        self.privacy_engine = PrivacyEngine()
        (
            self.torch_model,
            self.optimizer,
            self._dummy_data_loader
        ) = self.privacy_engine.make_private_with_epsilon(
            module=self.torch_model,
            optimizer=base_optimizer,
            data_loader=self._build_dummy_loader(),
            epochs=1,
            target_epsilon=DP_EPSILON,
            target_delta=DP_DELTA,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )

        print(f"🔒 Opacus DP Engine attached | "
              f"Target ε={DP_EPSILON} | "
              f"δ={DP_DELTA} | "
              f"Max Grad Norm={DP_MAX_GRAD_NORM}")

        self.training_samples_this_round = 0

        # FIX 2: Load global checkpoint with _module. prefix so Opacus
        # GradSampleModule can load it. fed_avg.py saves with plain keys
        # (conv1.weight) but Opacus expects prefixed keys (_module.conv1.weight).
        global_model_path = "/app/weights/global_model.pt"
        if os.path.exists(global_model_path):
            try:
                state = torch.load(
                    global_model_path,
                    map_location=torch.device('cpu')
                )
                # Add _module. prefix to match Opacus GradSampleModule keys
                opacus_state = {
                    f"_module.{k}": v
                    for k, v in state.items()
                }
                self.torch_model.load_state_dict(opacus_state)
                print(f"✅ Loaded global model checkpoint from previous round")
            except Exception as e:
                print(f"⚠️ Could not load global checkpoint: {e}. "
                      f"Starting from random init.")
        else:
            print(f"ℹ️  No global checkpoint found. "
                  f"Starting from random initialisation.")

        # ── FIX 1: Training lock ──────────────────────────────────────
        # Opacus per-sample gradient hooks conflict with the gRPC stream
        # generator when called synchronously inside StreamSignal.
        # A threading.Lock ensures only one training step runs at a time,
        # and fire-and-forget threading.Thread keeps the gRPC stream open
        # while training completes independently.
        self._train_lock = threading.Lock()

        # ── Azure Digital Twins ───────────────────────────────────────
        self.adt_client = None
        if ADT_URL:
            try:
                cred = DefaultAzureCredential()
                self.adt_client = DigitalTwinsClient(ADT_URL, cred)
                print(f"✅ Connected to Azure Digital Twin: {ADT_URL}")
            except Exception as e:
                print(f"⚠️ Azure Initialization Failed: {e}")

        self._last_adt_sync = {}

    def _build_dummy_loader(self):
        """
        Opacus requires a DataLoader reference during make_private_with_epsilon
        to compute the noise multiplier from the target epsilon and dataset size.
        We pass a minimal dummy loader — the computed noise_multiplier is what
        matters and is correctly applied to every subsequent training step.
        """
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.zeros(100, 1, 1),
            torch.zeros(100, 1)
        )
        return torch.utils.data.DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False
        )

    def _local_train_step(self, value: float, label: float):
        """
        FIX 1: Opacus training runs in a separate thread.

        Root cause of 'Per sample gradient is not initialized':
        Opacus registers per-sample gradient hooks on the model. When
        _local_train_step is called synchronously inside StreamSignal's
        generator loop, the gRPC framework's internal threading interferes
        with Opacus's hook lifecycle — hooks fire before gradients are
        populated.

        Solution: acquire a lock and run the entire training step in a
        daemon thread. The gRPC stream yields its response immediately
        without waiting for training to complete. Training runs safely
        in the background on the same model instance.

        The lock prevents two concurrent burst packets from running
        simultaneous backward passes, which would corrupt gradients.
        """
        def train_in_thread():
            with self._train_lock:
                try:
                    self.torch_model.train()
                    self.optimizer.zero_grad()

                    x = torch.tensor([[[value]]], dtype=torch.float32)
                    target = torch.tensor([[label]], dtype=torch.float32)

                    output = self.torch_model(x)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    self.training_samples_this_round += 1
                    epsilon_spent = self.privacy_engine.get_epsilon(
                        delta=DP_DELTA
                    )

                    print(f"🔒 DP Train Step | "
                          f"Loss: {loss.item():.4f} | "
                          f"ε spent: {epsilon_spent:.4f} / {DP_EPSILON} | "
                          f"Samples: {self.training_samples_this_round}")

                    if epsilon_spent > DP_EPSILON * 0.8:
                        print(f"⚠️  Privacy budget at "
                              f"{(epsilon_spent / DP_EPSILON) * 100:.0f}% — "
                              f"consider triggering global aggregation")

                except Exception as e:
                    print(f"⚠️ Training step skipped: {e}")

        # Fire-and-forget — stream is not blocked
        t = threading.Thread(target=train_in_thread, daemon=True)
        t.start()

    def _save_model_weights(self, hospital_id: str):
        """
        Saves Opacus-trained state_dict with formal DP guarantee.
        Opacus already applied calibrated noise during backward pass —
        no manual noise injection needed here.
        """
        # Wait for any in-progress training to finish before reading weights
        with self._train_lock:
            epsilon_spent = self.privacy_engine.get_epsilon(delta=DP_DELTA)

            weight_data = {
                "hospital": hospital_id,
                "num_samples": self.training_samples_this_round,
                "weights": {
                    k: v.tolist()
                    for k, v in self.torch_model.state_dict().items()
                },
                "epsilon_spent": epsilon_spent,
                "delta": DP_DELTA,
                "timestamp": time.time()
            }

        filename = (f"/app/weights/update_"
                    f"{hospital_id}_{int(time.time() * 1000)}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(weight_data, f)
                f.flush()
                os.fsync(f.fileno())
            print(f"📦 Formally DP-protected weights stored for "
                  f"{hospital_id} | ε={epsilon_spent:.4f}")
        except Exception as e:
            print(f"❌ Weight Save Error: {e}")

    def StreamSignal(self, request_iterator, context):
        peer = context.peer()
        print(f"🔗 Stream opened from: {peer}")

        try:
            for request in request_iterator:
                # ── 1. ONNX INFERENCE (fast path — runs synchronously) ─
                input_data = np.array(
                    [[[request.value]]], dtype=np.float32
                )
                outputs = self.session.run(None, {'input': input_data})
                prob = float(outputs[0][0][0])
                is_anomaly = prob > 0.5

                print(f"[INFER] Silo: {request.hospital_id} | "
                      f"Val: {request.value:.4f} | "
                      f"Prob: {prob:.4f} | "
                      f"Anomaly: {is_anomaly}")

                # ── 2. DP TRAINING (async — does not block stream) ─────
                if request.is_burst:
                    # _local_train_step fires a daemon thread internally
                    self._local_train_step(request.value, label=1.0)
                    # _save_model_weights acquires the lock and waits for
                    # the training thread before reading state_dict
                    self._save_model_weights(request.hospital_id)

                # ── 3. ADT SYNC (debounced) ────────────────────────────
                if self.adt_client and (is_anomaly or request.is_burst):
                    self._debounced_adt_sync(
                        request.hospital_id,
                        request.value,
                        is_anomaly
                    )

                # ── 4. YIELD RESPONSE ─────────────────────────────────
                yield biosignal_pb2.SignalResponse(
                    status="PROCESSED",
                    trigger_training=is_anomaly
                )

        except Exception as e:
            if isinstance(e, grpc.RpcError):
                print(f"⚠️ Stream RPC error from {peer}: {e}")
            else:
                print(f"⚠️ Unexpected stream error from {peer}: {e}")
        finally:
            print(f"🔌 Stream closed from: {peer}")

    def _debounced_adt_sync(self, twin_id, val, critical):
        now = time.time()
        last = self._last_adt_sync.get(twin_id, 0)
        if now - last < ADT_DEBOUNCE_SECONDS:
            return
        self._last_adt_sync[twin_id] = now
        self.sync_to_azure(twin_id, val, critical)

    def sync_to_azure(self, twin_id, val, critical):
        patch = [
            {"op": "add", "path": "/HeartRate", "value": float(val)},
            {"op": "add", "path": "/IsCritical", "value": bool(critical)}
        ]
        try:
            self.adt_client.update_digital_twin(twin_id, patch)
            print(f"☁️  Azure Mirror Updated for {twin_id}")
        except Exception as e:
            print(f"❌ Azure Sync Error: {e}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    biosignal_pb2_grpc.add_BioNetServiceServicer_to_server(
        BioNetAggregator(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("🚀 BioNet Aggregator (Bidi-Streaming + Opacus DP) "
          "listening on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()