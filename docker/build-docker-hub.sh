#!/bin/sh
#This version build the docker hub containers

VERSION="1.1"

docker build -t easynmt/api:${VERSION}-cpu -f api/cpu.dockerfile api/
docker push easynmt/api:${VERSION}-cpu

docker build -t easynmt/api:${VERSION}-cuda10.1 -f api/cuda10.1.dockerfile api/
docker push easynmt/api:${VERSION}-cuda10.1

docker build -t easynmt/api:${VERSION}-cuda11.0 -f api/cuda11.0.dockerfile api/
docker push easynmt/api:${VERSION}-cuda11.0

docker build -t easynmt/api:${VERSION}-cuda11.1 -f api/cuda11.1.dockerfile api/
docker push easynmt/api:${VERSION}-cuda11.1