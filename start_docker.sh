docker run \
  -u $(id -u):$(id -g) \
  -e --runtime=nvidia \
  -it \
  -v /data2:/data2 \
  -v "$(pwd)":/noses \
  68d67bb8e289
