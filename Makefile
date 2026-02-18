.PHONY: build deploy run-hospitals clean

# CNE Focus: Ensure we use the Python from our virtual environment
PYTHON := ./venv/bin/python3

build:
	eval $$(minikube docker-env) && \
	$(PYTHON) hpc-server/generate_model.py && \
	docker build -t hpc-aggregator:v1 -f hpc-server/Dockerfile . && \
	docker build -t silo-client:v1 -f silo-client/Dockerfile .

deploy:
	# 1. Create namespaces first
	kubectl create namespace hospital-a --dry-run=client -o yaml | kubectl apply -f -
	kubectl create namespace hospital-b --dry-run=client -o yaml | kubectl apply -f -
	kubectl create namespace aggregator --dry-run=client -o yaml | kubectl apply -f -
	# 2. Apply CNE labels for Network Isolation
	kubectl label ns hospital-a kubernetes.io/metadata.name=hospital-a --overwrite
	kubectl label ns hospital-b kubernetes.io/metadata.name=hospital-b --overwrite
	kubectl label ns aggregator kubernetes.io/metadata.name=aggregator --overwrite
	# 3. Apply the infrastructure
	kubectl apply -f k8s/infra.yaml
	kubectl apply -f k8s/hpc-scaler.yaml

run-hospitals:
	# Launch Hospital Alpha
	kubectl delete pod test-client -n hospital-a --ignore-not-found
	kubectl run test-client -n hospital-a --image=silo-client:v1 \
		--env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
		--env="HOSPITAL_ID=Silo-Alpha"
	# Launch Hospital Beta (Multi-tenant proof)
	kubectl delete pod test-client -n hospital-b --ignore-not-found
	kubectl run test-client -n hospital-b --image=silo-client:v1 \
		--env="AGGREGATOR_ADDR=aggregator-service.aggregator.svc.cluster.local:50051" \
		--env="HOSPITAL_ID=Silo-Beta"

clean:
	kubectl delete pod test-client -n hospital-a --ignore-not-found
	kubectl delete pod test-client -n hospital-b --ignore-not-found