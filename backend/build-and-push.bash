#!/bin/bash

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 275838112029.dkr.ecr.ap-south-1.amazonaws.com
docker build -t tourist-forecast-model .
docker tag tourist-forecast-model 275838112029.dkr.ecr.ap-south-1.amazonaws.com/tourist-forecast-model:latest
docker push 275838112029.dkr.ecr.ap-south-1.amazonaws.com/tourist-forecast-model:latest
