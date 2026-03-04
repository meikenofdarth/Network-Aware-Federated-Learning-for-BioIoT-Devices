.PHONY: login build deploy run-hospitals clean

PYTHON := ./venv/bin/python3
REGISTRY := biosyncregistry1772554412.azurecr.io
# --- VERSION 5 (Demo Mode) ---
VERSION := v5

login:
	az acr login --name biosyncregistry1772554412

build:
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