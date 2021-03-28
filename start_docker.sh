#!/bin/sh
docker run \
    --gpus device=$1 \
    --runtime=nvidia \
    --rm \
    -it \
    -u $(id -u):$(id -g) \
    -v /data:/data \
    -v /data2:/data2 \
    -e USER=$USER \
    -e HOME=/data/$USER \
    -w $PWD \
    urban-forest/latest \
    bash
