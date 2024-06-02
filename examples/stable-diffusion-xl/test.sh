#!/bin/bash

curl localhost:8080/v1/models/stable-diffusion-xl:predict -d @./input.json
curl -X GET localhost:8080/v1/docs/stable-diffusion-xl