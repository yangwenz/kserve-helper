#!/bin/bash

curl localhost:8080/v1/models/test-rotate-image:predict -d @./input.json
curl -X GET localhost:8080/v1/docs/test-rotate-image