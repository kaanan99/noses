docker run \
  -u 3221772:$(id -g) \
  --env="DISPLAY" \
  -e --runtime=nvidia \
  -it \
  -v /data2:/data2 \
  -v "$(pwd)":/noses \
  68d67bb8e289
