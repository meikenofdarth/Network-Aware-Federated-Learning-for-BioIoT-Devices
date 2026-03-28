"""
dashboard-api/server.py

FastAPI bridge that exposes 4 endpoints consumed by the React dashboard:
  GET /adt    → Azure Digital Twins twin properties (HeartRate, IsCritical)
  GET /pods   → Live hpc-aggregator pod count via K8s API
  GET /mlflow → Last N FL runs from MLflow tracking server
  GET /dp     → Per-silo epsilon spent from latest MLflow run metrics

All endpoints return gracefully degraded responses on error so the
React dashboard can fall back to simulation without crashing.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
from azure.identity import DefaultAzureCredential
from azure.digitaltwins.core import DigitalTwinsClient
from kubernetes import client as k8s_client, config as k8s_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard-api")

# ── Config ────────────────────────────────────────────────────────────
ADT_URL      = os.getenv("ADT_URL", "")
MLFLOW_URI   = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.aggregator.svc.cluster.local:5000"
)
NAMESPACE    = os.getenv("AGGREGATOR_NAMESPACE", "aggregator")
DEPLOYMENT   = os.getenv("AGGREGATOR_DEPLOYMENT", "hpc-aggregator")
TWIN_IDS     = ["Silo-Alpha", "Silo-Beta"]
MLFLOW_RUNS  = int(os.getenv("MLFLOW_MAX_RUNS", "20"))
EXPERIMENT   = "Bio-Sync-HPC-Global-Learning"

app = FastAPI(title="Bio-Sync Dashboard API", version="2.0.0")

# Allow the nginx container to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Azure Digital Twins client ────────────────────────────────────────
_adt_client = None

def get_adt_client():
    global _adt_client
    if _adt_client is None and ADT_URL:
        try:
            cred = DefaultAzureCredential()
            _adt_client = DigitalTwinsClient(ADT_URL, cred)
            log.info(f"ADT client connected: {ADT_URL}")
        except Exception as e:
            log.warning(f"ADT client init failed: {e}")
    return _adt_client


# ── Kubernetes client ─────────────────────────────────────────────────
_k8s_ready = False

def get_k8s():
    global _k8s_ready
    if not _k8s_ready:
        try:
            # Loads in-cluster service account credentials automatically
            k8s_config.load_incluster_config()
            _k8s_ready = True
            log.info("K8s in-cluster config loaded")
        except Exception:
            try:
                k8s_config.load_kube_config()
                _k8s_ready = True
                log.info("K8s kubeconfig loaded (local dev)")
            except Exception as e:
                log.warning(f"K8s config failed: {e}")
    return _k8s_ready


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/adt")
def get_adt():
    """
    Returns HeartRate and IsCritical for each Digital Twin.
    Response shape:
      { "Silo-Alpha": { "HeartRate": 0.95, "IsCritical": true }, ... }
    """
    result = {}
    adt = get_adt_client()
    if not adt:
        return {}

    for twin_id in TWIN_IDS:
        try:
            twin = adt.get_digital_twin(twin_id)
            result[twin_id] = {
                "HeartRate":  twin.get("HeartRate",  0.0),
                "IsCritical": twin.get("IsCritical", False),
            }
        except Exception as e:
            log.warning(f"ADT fetch failed for {twin_id}: {e}")
            result[twin_id] = {"HeartRate": 0.0, "IsCritical": False}

    return result


# @app.get("/pods")
# def get_pods():
#     """
#     Returns the current ready pod count for the hpc-aggregator deployment.
#     Response shape: { "count": 3, "desired": 4 }
#     """
#     if not get_k8s():
#         return {"count": 1, "desired": 1}

#     try:
#         apps_v1 = k8s_client.AppsV1Api()
#         dep = apps_v1.read_namespaced_deployment(
#             name=DEPLOYMENT,
#             namespace=NAMESPACE
#         )
#         ready   = dep.status.ready_replicas   or 0
#         desired = dep.status.desired_replicas or dep.spec.replicas or 1
#         return {"count": ready, "desired": desired}
#     except Exception as e:
#         log.warning(f"K8s pods fetch failed: {e}")
#         return {"count": 1, "desired": 1}
@app.get("/pods")
def get_pods():
    if not get_k8s():
        return {"count": 1, "desired": 1}
    try:
        apps_v1 = k8s_client.AppsV1Api()
        dep = apps_v1.read_namespaced_deployment(
            name=DEPLOYMENT,
            namespace=NAMESPACE
        )
        ready   = dep.status.ready_replicas or 0
        # desired comes from spec, not status
        desired = dep.spec.replicas or 1
        return {"count": ready, "desired": desired}
    except Exception as e:
        log.warning(f"K8s pods fetch failed: {e}")
        return {"count": 1, "desired": 1}

@app.get("/mlflow")
async def get_mlflow():
    """
    Fetches the last MLFLOW_RUNS completed runs from MLflow REST API.
    Response shape:
      { "runs": [ { "discrimination_score": 0.7, "seizure_confidence": 0.9,
                    "normal_confidence": 0.2, "round": 5 }, ... ] }
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as http:
            # Step 1: get experiment ID by name
            exp_resp = await http.get(
                f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": EXPERIMENT}
            )
            if exp_resp.status_code != 200:
                return {"runs": []}

            exp_id = exp_resp.json()["experiment"]["experiment_id"]

            # Step 2: search runs, ordered by start time descending
            runs_resp = await http.post(
                f"{MLFLOW_URI}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [exp_id],
                    "max_results": MLFLOW_RUNS,
                    "order_by": ["start_time ASC"],
                    "filter": "status = 'FINISHED'",
                }
            )
            if runs_resp.status_code != 200:
                return {"runs": []}

            raw_runs = runs_resp.json().get("runs", [])
            parsed = []
            for i, r in enumerate(raw_runs):
                metrics = {m["key"]: m["value"] for m in r.get("data", {}).get("metrics", [])}
                parsed.append({
                    "round":                 i + 1,
                    "discrimination_score":  metrics.get("discrimination_score",  0.0),
                    "seizure_confidence":    metrics.get("seizure_confidence",    0.0),
                    "normal_confidence":     metrics.get("normal_confidence",     0.0),
                    "total_training_samples":metrics.get("total_training_samples",0),
                    "participating_silos":   metrics.get("participating_silos",   0),
                })
            return {"runs": parsed}

    except Exception as e:
        log.warning(f"MLflow fetch failed: {e}")
        return {"runs": []}


@app.get("/dp")
async def get_dp():
    """
    Returns per-silo epsilon spent from the most recent MLflow run.
    Response shape: { "Silo_Alpha": 0.31, "Silo_Beta": 0.28 }

    Keys use underscore because MLflow metric names cannot contain hyphens.
    The dashboard normalises both formats.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as http:
            exp_resp = await http.get(
                f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": EXPERIMENT}
            )
            if exp_resp.status_code != 200:
                return {}

            exp_id = exp_resp.json()["experiment"]["experiment_id"]

            runs_resp = await http.post(
                f"{MLFLOW_URI}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [exp_id],
                    "max_results": 1,
                    "order_by": ["start_time DESC"],
                    "filter": "status = 'FINISHED'",
                }
            )
            if runs_resp.status_code != 200:
                return {}

            runs = runs_resp.json().get("runs", [])
            if not runs:
                return {}

            metrics = {
                m["key"]: m["value"]
                for m in runs[0].get("data", {}).get("metrics", [])
            }

            return {
                "Silo_Alpha": metrics.get("epsilon_spent_Silo_Alpha", 0.0),
                "Silo_Beta":  metrics.get("epsilon_spent_Silo_Beta",  0.0),
            }

    except Exception as e:
        log.warning(f"DP metrics fetch failed: {e}")
        return {}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)