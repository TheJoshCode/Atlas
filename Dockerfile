FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update
RUN apt-get install -y ffmpeg libsndfile1 build-essential git wget cmake ninja-build build-essential vim curl libcurl4-openssl-dev python3-pip pulseaudio libportaudio2 libpulse0 alsa-utils tzdata portaudio19-dev espeak pciutils
RUN rm -rf /var/lib/apt/lists/* 
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
COPY . .
CMD ["python", "Atlas.py"]