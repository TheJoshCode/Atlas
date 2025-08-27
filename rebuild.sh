docker build -t atlas .
docker run --rm -it --gpus all --device /dev/snd atlas
