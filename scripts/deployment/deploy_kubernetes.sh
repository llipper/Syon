#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/api/
kubectl apply -f kubernetes/inference/