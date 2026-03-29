# Bio-Sync HPC — Session Restart Guide
# ─────────────────────────────────────────────────────────────────────
# Run every command in order at the start of each work session.
# Current stable state: v2-stable (git tag)
# ─────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — START THE CLUSTER (3-5 minutes)
# ═══════════════════════════════════════════════════════════════════

make resume

# This runs:
#   az aks start --name BioSync-Cluster --resource-group BioNet-HPC-RG
#   az aks get-credentials ...
#   az acr login ...

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — RE-EXPORT SHELL VARIABLES
# These are lost every time the terminal closes.
# ═══════════════════════════════════════════════════════════════════


echo "✅ Variables set:"
echo "   CLIENT_ID:      $CLIENT_ID"
echo "   STORAGE_ACCOUNT: $STORAGE_ACCOUNT"
echo "   STORAGE_KEY:    ${STORAGE_KEY:0:10}..."

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — ACTIVATE VENV
# ═══════════════════════════════════════════════════════════════════

source venv/bin/activate

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — VERIFY CLUSTER HEALTH
# ═══════════════════════════════════════════════════════════════════

kubectl get nodes

# Expected:
# NAME                                STATUS   ROLES   AGE   VERSION
# aks-nodepool1-93170077-vmss000000   Ready    <none>  Xd    v1.33.7

kubectl get pods -n aggregator

# Expected (all 1/1 Running, 0 restarts):
# NAME                                 READY   STATUS    RESTARTS
# dashboard-XXXX                       1/1     Running   0
# dashboard-api-XXXX                   1/1     Running   0
# global-aggregator-XXXX               1/1     Running   0
# hpc-aggregator-XXXX                  1/1     Running   0
# mlflow-server-XXXX                   1/1     Running   0

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — START SILO CLIENTS
# Silo pods are deleted on make pause and must be recreated each session.
# ═══════════════════════════════════════════════════════════════════

kubectl run test-client -n hospital-a \
  --image=biosyncregistry1772554412.azurecr.io/silo-client:v10 \
  --env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
  --env="HOSPITAL_ID=Silo-Alpha"

kubectl run test-client -n hospital-b \
  --image=biosyncregistry1772554412.azurecr.io/silo-client:v10 \
  --env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
  --env="HOSPITAL_ID=Silo-Beta"

# Wait ~15 seconds, then verify:
kubectl get pods -n hospital-a
kubectl get pods -n hospital-b

# Expected:
# NAME          READY   STATUS    RESTARTS   AGE
# test-client   1/1     Running   0          Xs

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — WAIT 90 SECONDS FOR BURSTS TO FIRE
# The FFT buffer needs 256 samples at 2Hz = ~128 seconds to fill.
# After that, force_burst fires at 8% probability per cycle.
# ═══════════════════════════════════════════════════════════════════

sleep 90

# ═══════════════════════════════════════════════════════════════════
# STEP 7 — VERIFY THE FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

# 7.1 — Silo client (look for seizure detections)
kubectl logs -n hospital-a test-client --tail=15

# Expected:
# !!! SEIZURE RISK DETECTED | Spectral Ratio: 0.09 | Val: 0.91 | Streaming to HPC !!!

# 7.2 — HPC aggregator (look for DP training steps)
kubectl logs -n aggregator -l app=hpc-aggregator --tail=20

# Expected:
# 🔒 DP Train Step | Loss: 0.49 | ε spent: 0.12 / 0.5 | Samples: 3
# 📦 Formally DP-protected weights stored for Silo-Alpha | ε=0.12
# ☁️  Azure Mirror Updated for Silo-Alpha

# 7.3 — Global aggregator (look for FedAvg rounds)
kubectl logs -n aggregator -l app=global-aggregator --tail=15

# Expected:
# ✅ FedAvg Complete | Silos: {'Silo-Beta', 'Silo-Alpha'} | Total Samples: X
# 💾 Global model checkpoint saved to /app/weights/global_model.pt
# 📊 MLflow Logged: Discrimination=0.0XXX
# 🧹 Shared buffer cleared for next round.

# 7.4 — KEDA (must be READY=True)
kubectl get scaledobject -n aggregator

# Expected:
# NAME                READY   ACTIVE
# aggregator-scaler   True    True

# 7.5 — Dashboard (open in browser)
kubectl get svc dashboard-service -n aggregator
# Open: http://20.198.144.247

# 7.6 — MLflow (open in browser, requires port-forward)
# Run in a separate terminal:
kubectl port-forward svc/mlflow-service -n aggregator 5001:5000
# Open: http://localhost:5001

# ═══════════════════════════════════════════════════════════════════
# GOLDEN RULE — IMAGE VERSION BUMPS
# Every time you rebuild a Docker image, follow this EXACT sequence.
# Skipping kubectl apply is the #1 cause of "fix not taking effect".
# ═══════════════════════════════════════════════════════════════════

# Template (replace X with new version, Y with old version):
docker build --no-cache --platform linux/amd64 \
  -t biosyncregistry1772554412.azurecr.io/hpc-aggregator:vX \
  -f hpc-server/Dockerfile .

docker push biosyncregistry1772554412.azurecr.io/hpc-aggregator:vX

sed -i '' 's/hpc-aggregator:vY/hpc-aggregator:vX/g' k8s/infra.yaml

kubectl apply -f k8s/infra.yaml           # ← NEVER SKIP THIS

kubectl rollout restart deployment/hpc-aggregator -n aggregator
kubectl rollout restart deployment/global-aggregator -n aggregator

kubectl rollout status deployment/hpc-aggregator -n aggregator --timeout=180s

# ═══════════════════════════════════════════════════════════════════
# CURRENT IMAGE VERSIONS (v2-stable)
# ═══════════════════════════════════════════════════════════════════

# hpc-aggregator:v12     — aggregator + fed_avg + Opacus threading fix
# silo-client:v10        — FFT filter + force_burst + bidi streaming
# dashboard-api:v2       — FastAPI bridge with K8s spec.replicas fix
# biosync-dashboard:v1   — React SPA (unchanged since v1)

# ═══════════════════════════════════════════════════════════════════
# END OF SESSION
# ═══════════════════════════════════════════════════════════════════

make pause

# This runs:
#   kubectl delete pod test-client -n hospital-a --ignore-not-found
#   kubectl delete pod test-client -n hospital-b --ignore-not-found
#   az aks stop --name BioSync-Cluster --resource-group BioNet-HPC-RG

# Verify stopped:
az aks show \
  --name BioSync-Cluster \
  --resource-group BioNet-HPC-RG \
  --query "powerState.code" -o tsv
# Expected: Stopped
