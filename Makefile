.PHONY: login build deploy run-hospitals clean build-dashboard deploy-dashboard dashboard-ip pause resume

PYTHON := ./venv/bin/python3
REGISTRY := biosyncregistry1772554412.azurecr.io
# --- VERSION 5 (Demo Mode) ---
VERSION := v7
DASHBOARD_VERSION := v1

login:
	az acr login --name biosyncregistry1772554412

build:
	# Regenerate gRPC stubs into root (where Dockerfiles expect them)
	python -m grpc_tools.protoc \
		-I protos \
		--python_out=. \
		--grpc_python_out=. \
		protos/biosignal.proto
	# Generate Model
	$(PYTHON) hpc-server/generate_model.py
	
	# Build & Push Aggregator (No-cache ensures new ONNX model is copied)
	docker build --no-cache --platform linux/amd64 -t $(REGISTRY)/hpc-aggregator:$(VERSION) -f hpc-server/Dockerfile .
	docker push $(REGISTRY)/hpc-aggregator:$(VERSION)
	
	# Build & Push Client
	docker build --platform linux/amd64 -t $(REGISTRY)/silo-client:$(VERSION) -f silo-client/Dockerfile .
	docker push $(REGISTRY)/silo-client:$(VERSION)

deploy:
	# Create Namespaces
	kubectl create namespace hospital-a --dry-run=client -o yaml | kubectl apply -f -
	kubectl create namespace hospital-b --dry-run=client -o yaml | kubectl apply -f -
	kubectl create namespace aggregator --dry-run=client -o yaml | kubectl apply -f -
	
	# Label Namespaces
	kubectl label ns hospital-a kubernetes.io/metadata.name=hospital-a --overwrite
	kubectl label ns hospital-b kubernetes.io/metadata.name=hospital-b --overwrite
	kubectl label ns aggregator kubernetes.io/metadata.name=aggregator --overwrite
	
	# Deploy Infra & KEDA
	kubectl apply -f k8s/infra.yaml
	kubectl apply -f k8s/hpc-scaler.yaml

run-hospitals:
	# Hospital Alpha
	kubectl delete pod test-client -n hospital-a --ignore-not-found
	kubectl run test-client -n hospital-a --image=$(REGISTRY)/silo-client:$(VERSION) \
		--env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
		--env="HOSPITAL_ID=Silo-Alpha"
	
	# Hospital Beta
	kubectl delete pod test-client -n hospital-b --ignore-not-found
	kubectl run test-client -n hospital-b --image=$(REGISTRY)/silo-client:$(VERSION) \
		--env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
		--env="HOSPITAL_ID=Silo-Beta"

clean:
	kubectl delete pod test-client -n hospital-a --ignore-not-found
	kubectl delete pod test-client -n hospital-b --ignore-not-found

build-dashboard:
	# Build and push FastAPI bridge
	docker build --platform linux/amd64 \
		-t $(REGISTRY)/dashboard-api:$(DASHBOARD_VERSION) \
		-f dashboard-api/Dockerfile .
	docker push $(REGISTRY)/dashboard-api:$(DASHBOARD_VERSION)

	# Build and push React + nginx frontend
	docker build --platform linux/amd64 \
		-t $(REGISTRY)/biosync-dashboard:$(DASHBOARD_VERSION) \
		-f dashboard/Dockerfile .
	docker push $(REGISTRY)/biosync-dashboard:$(DASHBOARD_VERSION)

deploy-dashboard:
	# Apply the dashboard additions (RBAC, API, frontend)
	kubectl apply -f k8s-dashboard-additions.yaml
	@echo "Waiting for dashboard LoadBalancer IP..."
	kubectl rollout status deployment/dashboard     -n aggregator --timeout=120s
	kubectl rollout status deployment/dashboard-api -n aggregator --timeout=120s

dashboard-ip:
	# Prints the external IP of the dashboard LoadBalancer service.
	# Run this after deploy-dashboard — may take 60-90s to provision.
	kubectl get svc dashboard-service -n aggregator \
		-o jsonpath='{.status.loadBalancer.ingress[0].ip}'
	@echo ""


pause:
	kubectl delete pod test-client -n hospital-a --ignore-not-found
	kubectl delete pod test-client -n hospital-b --ignore-not-found
	az aks stop --name BioSync-Cluster --resource-group BioNet-HPC-RG
	@echo "✅ Cluster stopped. Credits preserved."

resume:
	az aks start --name BioSync-Cluster --resource-group BioNet-HPC-RG
	az aks get-credentials --name BioSync-Cluster --resource-group BioNet-HPC-RG --overwrite-existing
	az acr login --name biosyncregistry1772554412
	@echo "✅ Cluster running. Now re-export your shell variables."