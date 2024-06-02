#!/bin/bash

export INGRESS_HOST=localhost
export INGRESS_PORT=8080

SERVICE_HOSTNAME=$(kubectl get inferenceservice test-rotate-image -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/test-rotate-image:predict" -d @./input.json
curl -v -H "Host: ${SERVICE_HOSTNAME}" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/docs/test-rotate-image"
